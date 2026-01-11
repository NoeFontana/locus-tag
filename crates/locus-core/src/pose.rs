use nalgebra::{Matrix3, Vector3};

/// Camera intrinsics parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
}

impl CameraIntrinsics {
    /// Create new intrinsics.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Convert to a 3x3 matrix.
    pub fn as_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    /// Get inverse matrix.
    pub fn inv_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            1.0 / self.fx,
            0.0,
            -self.cx / self.fx,
            0.0,
            1.0 / self.fy,
            -self.cy / self.fy,
            0.0,
            0.0,
            1.0,
        )
    }
}

/// A 3D pose representing rotation and translation.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    /// 3x3 Rotation matrix.
    pub rotation: Matrix3<f64>,
    /// 3x1 Translation vector.
    pub translation: Vector3<f64>,
}

impl Pose {
    /// Create a new pose.
    pub fn new(rotation: Matrix3<f64>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Project a 3D point into the image using this pose and intrinsics.
    pub fn project(&self, point: &Vector3<f64>, intrinsics: &CameraIntrinsics) -> [f64; 2] {
        let p_cam = self.rotation * point + self.translation;
        let x = (p_cam.x / p_cam.z) * intrinsics.fx + intrinsics.cx;
        let y = (p_cam.y / p_cam.z) * intrinsics.fy + intrinsics.cy;
        [x, y]
    }
}

/// Estimate pose from tag detection using homography decomposition and refinement.
///
/// # Arguments
/// * `intrinsics` - Camera intrinsics.
/// * `corners` - Detected corners in image coordinates [[x, y]; 4].
/// * `tag_size` - Physical size of the tag in world units (e.g., meters).
pub fn estimate_tag_pose(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
) -> Option<Pose> {
    // 1. DLT Initialization from Homography
    // Tag corners in object space: centered at (0,0) on Z=0 plane.
    let s = tag_size * 0.5;
    let obj_pts = [
        [-s, -s], // TL
        [s, -s],  // TR
        [s, s],   // BR
        [-s, s],  // BL
    ];

    let h_struct = crate::decoder::Homography::from_pairs(&obj_pts, corners)?;
    let h = h_struct.h;

    // Decompose Homography: H = K * [R1 R2 t]
    // [R1 R2 t] = K^-1 * H
    let k_inv = intrinsics.inv_matrix();
    let m = k_inv * h;

    // Normalize columns to get rotation and translation
    let mut r1 = m.column(0).into_owned();
    let mut r2 = m.column(1).into_owned();
    let mut t = m.column(2).into_owned();

    let scale = 1.0 / (r1.norm() * r2.norm()).sqrt();
    r1 *= scale;
    r2 *= scale;
    t *= scale;

    // Orthogonalize R1 and R2 to find R3
    let r3 = r1.cross(&r2);

    // Polar decomposition of [R1 R2 R3] to ensure it's a valid SO(3) matrix
    let rot_raw = Matrix3::from_columns(&[r1, r2, r3]);
    let svd = rot_raw.svd(true, true);
    let u = svd.u.expect("SVD U failed");
    let vt = svd.v_t.expect("SVD Vt failed");
    let mut rotation = u * vt;

    // Ensure determinant is 1 (avoid reflection)
    if rotation.determinant() < 0.0 {
        rotation = u * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0) * vt;
    }

    // Ensure camera is in front of the tag (t.z > 0)
    if t.z < 0.0 {
        t = -t;
        // Adjust rotation? (Actually t.z < 0 often means the tag is upside down or flipped homography)
        // For AprilTags, we usually just need to flip the pose if the tag is "behind" the camera.
        // But better is to just return and refine.
    }

    let initial_pose = Pose::new(rotation, t);

    // 2. Refinement via Orthogonal Iteration (OI)
    // For 4 points, OI is very fast and robust.
    Some(refine_pose_oi(intrinsics, corners, &obj_pts, initial_pose))
}

/// Refine pose using Orthogonal Iteration.
fn refine_pose_oi(
    intrinsics: &CameraIntrinsics,
    img_pts: &[[f64; 2]; 4],
    obj_pts: &[[f64; 2]; 4],
    initial_pose: Pose,
) -> Pose {
    let mut r = initial_pose.rotation;
    let mut t = initial_pose.translation;

    // Normalized image coordinates (v_i)
    let k_inv = intrinsics.inv_matrix();
    let v: Vec<Vector3<f64>> = img_pts
        .iter()
        .map(|p| {
            let n = k_inv * Vector3::new(p[0], p[1], 1.0);
            n.normalize()
        })
        .collect();

    // Projector matrices P_i = v_i * v_i^T / (v_i^T * v_i)
    // Since v_i is normalized, P_i = v_i * v_i^T
    let p_mats: Vec<Matrix3<f64>> = v.iter().map(|vi| vi * vi.transpose()).collect();

    // Object points in 3D (Z=0)
    let p: Vec<Vector3<f64>> = obj_pts
        .iter()
        .map(|p| Vector3::new(p[0], p[1], 0.0))
        .collect();

    let n = p.len() as f64;
    // Mean of object points
    let p_bar = p.iter().sum::<Vector3<f64>>() / n;

    // I - (1/n) * sum(P_i)
    let sum_p = p_mats.iter().sum::<Matrix3<f64>>();
    let m_inv = (Matrix3::identity() - (1.0 / n) * sum_p)
        .try_inverse()
        .expect("M matrix not invertible, check camera geometry");

    for _ in 0..10 {
        // Compute optimal translation given R
        // t(R) = M^-1 * (1/n) * sum((P_i - I) * R * p_i)
        let mut sum_tp = Vector3::zeros();
        for i in 0..p.len() {
            sum_tp += (p_mats[i] - Matrix3::identity()) * (r * p[i]);
        }
        t = m_inv * (1.0 / n) * sum_tp;

        // Compute optimal R given t
        // This is a Procrustes problem: R = argmin sum || (I - P_i) * (R*p_i + t) ||^2
        // Find R that aligns q_i = (I-P_i)*t with p_i? No, it's slightly different.
        // Actually, we align p_i' = R*p_i + t with projected points.

        let mut b = Matrix3::zeros();
        for i in 0..p.len() {
            let p_centered = p[i] - p_bar;
            let vi_prime = p_mats[i] * (r * p[i] + t);
            b += vi_prime * p_centered.transpose();
        }

        let svd = b.svd(true, true);
        let u = svd.u.expect("SVD U failed in OI");
        let vt = svd.v_t.expect("SVD Vt failed in OI");
        r = u * vt;
        if r.determinant() < 0.0 {
            r = u * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0) * vt;
        }
    }

    Pose::new(r, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_pose_projection() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        let rotation = Matrix3::identity();
        let translation = Vector3::new(0.0, 0.0, 2.0); // 2 meters away
        let pose = Pose::new(rotation, translation);

        let p_world = Vector3::new(0.1, 0.1, 0.0);
        let p_img = pose.project(&p_world, &intrinsics);

        // x = (0.1 / 2.0) * 500 + 320 = 0.05 * 500 + 320 = 25 + 320 = 345
        // y = (0.1 / 2.0) * 500 + 240 = 265
        assert!((p_img[0] - 345.0).abs() < 1e-6);
        assert!((p_img[1] - 265.0).abs() < 1e-6);
    }

    #[test]
    fn test_perfect_pose_estimation() {
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
        let gt_rot = Matrix3::identity(); // Facing straight
        let gt_t = Vector3::new(0.1, -0.2, 1.5);
        let gt_pose = Pose::new(gt_rot, gt_t);

        let tag_size = 0.16; // 16cm
        let s = tag_size * 0.5;
        let obj_pts = [
            Vector3::new(-s, -s, 0.0),
            Vector3::new(s, -s, 0.0),
            Vector3::new(s, s, 0.0),
            Vector3::new(-s, s, 0.0),
        ];

        let mut img_pts = [[0.0, 0.0]; 4];
        for i in 0..4 {
            img_pts[i] = gt_pose.project(&obj_pts[i], &intrinsics);
        }

        let est_pose =
            estimate_tag_pose(&intrinsics, &img_pts, tag_size).expect("Pose estimation failed");

        // Check translation
        assert!((est_pose.translation.x - gt_t.x).abs() < 1e-3);
        assert!((est_pose.translation.y - gt_t.y).abs() < 1e-3);
        assert!((est_pose.translation.z - gt_t.z).abs() < 1e-3);

        // Check rotation (identity)
        let diff_rot = est_pose.rotation - gt_rot;
        assert!(diff_rot.norm() < 1e-3);
    }

    proptest! {
        #[test]
        fn prop_intrinsics_inversion(
            fx in 100.0..2000.0f64,
            fy in 100.0..2000.0f64,
            cx in 0.0..1000.0f64,
            cy in 0.0..1000.0f64
        ) {
            let intrinsics = CameraIntrinsics::new(fx, fy, cx, cy);
            let k = intrinsics.as_matrix();
            let k_inv = intrinsics.inv_matrix();
            let identity = k * k_inv;

            let expected = Matrix3::<f64>::identity();
            for i in 0..3 {
                for j in 0..3 {
                    prop_assert!((identity[(i, j)] - expected[(i, j)]).abs() < 1e-9);
                }
            }
        }

        #[test]
        fn prop_pose_recovery_stability(
            tx in -0.5..0.5f64,
            ty in -0.5..0.5f64,
            tz in 0.5..5.0f64, // Tag must be in front of camera
            roll in -0.5..0.5f64,
            pitch in -0.5..0.5f64,
            yaw in -0.5..0.5f64,
            noise in 0.0..0.1f64 // pixels of noise
        ) {
            let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
            let translation = Vector3::new(tx, ty, tz);

            // Create rotation from Euler angles using Rotation3
            let r_obj = nalgebra::Rotation3::from_euler_angles(roll, pitch, yaw);
            let rotation = r_obj.matrix().into_owned();
            let gt_pose = Pose::new(rotation, translation);

            let tag_size = 0.16;
            let s = tag_size * 0.5;
            let obj_pts = [
                Vector3::new(-s, -s, 0.0),
                Vector3::new(s, -s, 0.0),
                Vector3::new(s, s, 0.0),
                Vector3::new(-s, s, 0.0),
            ];

            let mut img_pts = [[0.0, 0.0]; 4];
            for i in 0..4 {
                let p = gt_pose.project(&obj_pts[i], &intrinsics);
                // Add tiny bit of noise
                img_pts[i] = [p[0] + noise, p[1] + noise];
            }

            if let Some(est_pose) = estimate_tag_pose(&intrinsics, &img_pts, tag_size) {
                // Check if recovered pose is reasonably close
                // Note: noise decreases accuracy, so we use a loose threshold
                let t_err = (est_pose.translation - translation).norm();
                prop_assert!(t_err < 0.1 + noise * 0.1, "Translation error {} too high for noise {}", t_err, noise);

                let r_err = (est_pose.rotation - rotation).norm();
                prop_assert!(r_err < 0.1 + noise * 0.1, "Rotation error {} too high for noise {}", r_err, noise);
            }
        }
    }
}
