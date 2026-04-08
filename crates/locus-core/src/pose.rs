//! 3D Pose Estimation (PnP) for fiducial markers.
//!
//! This module recovers the 6-DOF transformation between the camera and the tag.
//! It supports:
//! - **Fast Mode (IPPE)**: Infinitesimal Plane-Based Pose Estimation for low latency.
//! - **Accurate Mode (Weighted LM)**: Iterative refinement weighted by sub-pixel uncertainty.

#![allow(clippy::many_single_char_names, clippy::similar_names)]
use crate::batch::{DetectionBatch, Pose6D};
use crate::config::PoseEstimationMode;
use crate::image::ImageView;
use nalgebra::{Matrix2, Matrix3, Matrix6, Rotation3, UnitQuaternion, Vector3, Vector6};

/// Camera intrinsics parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    #[must_use]
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Convert to a 3x3 matrix.
    #[must_use]
    pub fn as_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    /// Get inverse matrix.
    #[must_use]
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

// 3D Pose Estimation using PnP (Perspective-n-Point).
/// A 3D pose representing rotation and translation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pose {
    /// 3x3 Rotation matrix.
    pub rotation: Matrix3<f64>,
    /// 3x1 Translation vector.
    pub translation: Vector3<f64>,
}

impl Pose {
    /// Create a new pose.
    #[must_use]
    pub fn new(rotation: Matrix3<f64>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Project a 3D point into the image using this pose and intrinsics.
    #[must_use]
    pub fn project(&self, point: &Vector3<f64>, intrinsics: &CameraIntrinsics) -> [f64; 2] {
        let p_cam = self.rotation * point + self.translation;
        let z = p_cam.z.max(1e-4);
        let x = (p_cam.x / z) * intrinsics.fx + intrinsics.cx;
        let y = (p_cam.y / z) * intrinsics.fy + intrinsics.cy;
        [x, y]
    }
}

/// Fills the lower triangle of a 6×6 normal-equations matrix from its upper triangle.
#[inline]
pub(crate) fn symmetrize_jtj6(jtj: &mut Matrix6<f64>) {
    for r in 1..6 {
        for c in 0..r {
            jtj[(r, c)] = jtj[(c, r)];
        }
    }
}

/// Computes the 10 non-zero scalar entries of the 2×6 left-perturbation SE(3) Jacobian
/// for a calibrated projection.
///
/// The full 2×6 Jacobian has zeros at column (0,1) and (1,0); those are omitted.
///
/// # Returns
/// `(ju0, ju2, ju3, ju4, ju5, jv1, jv2, jv3, jv4, jv5)` where
/// `du/dξ = [ju0, 0, ju2, ju3, ju4, ju5]` and `dv/dξ = [0, jv1, jv2, jv3, jv4, jv5]`.
#[inline]
pub(crate) fn projection_jacobian(
    x_z: f64,
    y_z: f64,
    z_inv: f64,
    intrinsics: &CameraIntrinsics,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let fx = intrinsics.fx;
    let fy = intrinsics.fy;
    (
        fx * z_inv,
        -fx * x_z * z_inv,
        -fx * x_z * y_z,
        fx * (x_z * x_z + 1.0),
        -fx * y_z,
        fy * z_inv,
        -fy * y_z * z_inv,
        -fy * (y_z * y_z + 1.0),
        fy * y_z * x_z,
        fy * x_z,
    )
}

/// Estimate pose from tag detection using homography decomposition and refinement.
///
/// # Arguments
/// * `intrinsics` - Camera intrinsics.
/// * `corners` - Detected corners in image coordinates [[x, y]; 4].
/// * `tag_size` - Physical size of the tag in world units (e.g., meters).
/// * `img` - Optional image view (required for Accurate mode).
/// * `mode` - Pose estimation mode (Fast vs Accurate).
///
/// # Returns
/// A tuple containing:
/// * `Option<Pose>`: The estimated pose (if successful).
/// * `Option<[[f64; 6]; 6]>`: The estimated covariance matrix (if Accurate mode enabled).
///
/// # Panics
/// Panics if SVD decomposition fails during orthogonalization (extremely rare).
#[must_use]
#[allow(clippy::missing_panics_doc)]
#[tracing::instrument(skip_all, name = "pipeline::estimate_tag_pose")]
pub fn estimate_tag_pose(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    img: Option<&ImageView>,
    mode: PoseEstimationMode,
) -> (Option<Pose>, Option<[[f64; 6]; 6]>) {
    estimate_tag_pose_with_config(
        intrinsics,
        corners,
        tag_size,
        img,
        mode,
        &crate::config::DetectorConfig::default(),
        None,
    )
}

/// Estimate pose with explicit configuration for tuning parameters.
#[must_use]
#[allow(clippy::missing_panics_doc)]
#[tracing::instrument(skip_all, name = "pipeline::estimate_tag_pose")]
pub fn estimate_tag_pose_with_config(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    img: Option<&ImageView>,
    mode: PoseEstimationMode,
    config: &crate::config::DetectorConfig,
    external_covariances: Option<&[Matrix2<f64>; 4]>,
) -> (Option<Pose>, Option<[[f64; 6]; 6]>) {
    // 1. Canonical Homography: Map canonical square [-1,1]x[-1,1] to image pixels.
    let Some(h_poly) = crate::decoder::Homography::square_to_quad(corners) else {
        return (None, None);
    };
    let h_pixel = h_poly.h;

    // 2. Normalize Homography: H_norm = K_inv * H_pixel
    let k_inv = intrinsics.inv_matrix();
    let h_norm = k_inv * h_pixel;

    // 3. Scale to Physical Model (Center Origin Convention)
    // Object points are centered: x_obj in [-s/2, s/2], y_obj in [-s/2, s/2]
    // Canonical mapping: xc = (2/s)*x_obj, yc = (2/s)*y_obj  (no offset)
    // P_cam = H_norm * [ (2/s)*x_obj, (2/s)*y_obj, 1 ]^T
    // P_cam = [ (2/s)*h1, (2/s)*h2, h3 ] * [x_obj, y_obj, 1 ]^T
    let scaler = 2.0 / tag_size;
    let mut h_metric = h_norm;
    h_metric.column_mut(0).scale_mut(scaler);
    h_metric.column_mut(1).scale_mut(scaler);

    // 4. IPPE Core: Decompose Jacobian (first 2 image cols) into 2 potential poses.
    let Some(candidates) = solve_ippe_square(&h_metric) else {
        return (None, None);
    };

    // 5. Disambiguation: Choose the pose with lower reprojection error.
    let best_pose = find_best_pose(intrinsics, corners, tag_size, &candidates);

    // 6. Refinement: Levenberg-Marquardt (LM)
    match (mode, img, external_covariances) {
        (PoseEstimationMode::Accurate, _, Some(ext_covs)) => {
            // Use provided covariances (e.g. from GWLF)
            let (refined_pose, covariance) = crate::pose_weighted::refine_pose_lm_weighted(
                intrinsics, corners, tag_size, best_pose, ext_covs,
            );
            (Some(refined_pose), Some(covariance))
        },
        (PoseEstimationMode::Accurate, Some(image), None) => {
            // Compute corner uncertainty from structure tensor
            let uncertainty = crate::pose_weighted::compute_framework_uncertainty(
                image,
                corners,
                &h_poly,
                config.tikhonov_alpha_max,
                config.sigma_n_sq,
                config.structure_tensor_radius,
            );
            let (refined_pose, covariance) = crate::pose_weighted::refine_pose_lm_weighted(
                intrinsics,
                corners,
                tag_size,
                best_pose,
                &uncertainty,
            );
            (Some(refined_pose), Some(covariance))
        },
        _ => {
            // Fast mode (Identity weights)
            (
                Some(refine_pose_lm(
                    intrinsics,
                    corners,
                    tag_size,
                    best_pose,
                    config.huber_delta_px,
                )),
                None,
            )
        },
    }
}

/// Solves the IPPE-Square problem.
/// Returns two possible poses ($R_a, t$) and ($R_b, t$) corresponding to the two minima of the PnP error function.
///
/// This uses an analytical approach derived from the homography Jacobian's SVD.
/// The second solution handles the "Necker reversal" ambiguity inherent in planar pose estimation.
fn solve_ippe_square(h: &Matrix3<f64>) -> Option<[Pose; 2]> {
    // IpPE-Square Analytical Solution (Zero Alloc)
    // Jacobian J = [h1, h2]
    let h1 = h.column(0);
    let h2 = h.column(1);

    // 1. Compute B = J^T J (2x2 symmetric matrix)
    //    [ a  c ]
    //    [ c  b ]
    let a = h1.dot(&h1);
    let b = h2.dot(&h2);
    let c = h1.dot(&h2);

    // 2. Eigen Analysis of 2x2 Matrix B
    //    Characteristic eq: lambda^2 - Tr(B)lambda + Det(B) = 0
    let trace = a + b;
    let det = a * b - c * c;
    let delta = (trace * trace - 4.0 * det).max(0.0).sqrt();

    // lambda1 >= lambda2
    let s1_sq = (trace + delta) * 0.5;
    let s2_sq = (trace - delta) * 0.5;

    // Singular values sigma = sqrt(lambda)
    let s1 = s1_sq.sqrt();
    let s2 = s2_sq.sqrt();

    // Check for Frontal View (Degeneracy: s1 ~= s2)
    // We use a safe threshold relative to the max singular value.
    if (s1 - s2).abs() < 1e-4 * s1 {
        // Degenerate Case: Frontal View (J columns orthogonal & equal length)
        // R = Gram-Schmidt orthonormalization of [h1, h2, h1xh2]

        let mut r1 = h1.clone_owned();
        let scale = 1.0 / r1.norm();
        r1 *= scale;

        // Orthogonalize r2 w.r.t r1
        let mut r2 = h2 - r1 * (h2.dot(&r1));
        r2 = r2.normalize();

        let r3 = r1.cross(&r2);
        let rot = Matrix3::from_columns(&[r1, r2, r3]);

        // Translation: t = h3 * scale.
        // Gamma (homography scale) is recovered from J.
        // gamma * R = J => gamma = ||h1|| (roughly).
        // We use average of singular values for robustness.
        let gamma = (s1 + s2) * 0.5;
        if gamma < 1e-8 {
            return None;
        } // Avoid div/0
        let tz = 1.0 / gamma;
        let t = h.column(2) * tz;

        let pose = Pose::new(rot, t);
        return Some([pose, pose]);
    }

    // 3. Recover Rotation A (Primary)
    // We basically want R such that J ~ R * S_prj.
    // Standard approach: R = U * V^T where J = U S V^T.
    //
    // Analytical 3x2 SVD reconstruction:
    // U = [u1, u2], V = [v1, v2]
    // U_i = J * v_i / s_i

    // Eigenvectors of B (columns of V)
    // For 2x2 matrix [a c; c b]:
    // If c != 0:
    //   v1 = [s1^2 - b, c], normalized
    // else:
    //   v1 = [1, 0] if a > b else [0, 1]

    let v1 = if c.abs() > 1e-8 {
        let v = nalgebra::Vector2::new(s1_sq - b, c);
        v.normalize()
    } else if a >= b {
        nalgebra::Vector2::new(1.0, 0.0)
    } else {
        nalgebra::Vector2::new(0.0, 1.0)
    };

    // v2 is orthogonal to v1. For 2D, [-v1.y, v1.x]
    let v2 = nalgebra::Vector2::new(-v1.y, v1.x);

    // Compute Left Singular Vectors u1, u2 inside the 3D space
    // u1 = J * v1 / s1
    // u2 = J * v2 / s2
    let j_v1 = h1 * v1.x + h2 * v1.y;
    let j_v2 = h1 * v2.x + h2 * v2.y;

    // Safe division check
    if s1 < 1e-8 {
        return None;
    }
    let u1 = j_v1 / s1;
    let u2 = j_v2 / s2.max(1e-8); // Avoid div/0 for s2

    // Reconstruct Rotation A
    // R = U * V^T (extended to 3x3)
    // The columns of R are r1, r2, r3.
    // In SVD terms:
    // [r1 r2] = [u1 u2] * [v1 v2]^T
    // r1 = u1 * v1.x + u2 * v2.x
    // r2 = u1 * v1.y + u2 * v2.y
    let r1_a = u1 * v1.x + u2 * v2.x;
    let r2_a = u1 * v1.y + u2 * v2.y;
    let r3_a = r1_a.cross(&r2_a);
    let rot_a = Matrix3::from_columns(&[r1_a, r2_a, r3_a]);

    // Translation A
    let gamma = (s1 + s2) * 0.5;
    let tz = 1.0 / gamma;
    let t_a = h.column(2) * tz;
    let pose_a = Pose::new(rot_a, t_a);

    // 4. Recover Rotation B (Second Solution)
    // Necker Reversal: Reflect normal across line of sight.
    // n_b = [-nx, -ny, nz] (approx).
    // Better analytical dual from IPPE:
    // The second solution corresponds to rotating U's second column?
    //
    // Let's stick to the robust normal reflection method which works well.
    let n_a = rot_a.column(2);
    // Safe normalize for n_b
    let n_b_raw = Vector3::new(-n_a.x, -n_a.y, n_a.z);
    let n_b = if n_b_raw.norm_squared() > 1e-8 {
        n_b_raw.normalize()
    } else {
        // Fallback (should be impossible for unit vector n_a)
        Vector3::z_axis().into_inner()
    };

    // Construct R_b using Gram-Schmidt from (h1, n_b)
    // x_axis projection is h1 (roughly).
    // x_b = (h1 - (h1.n)n).normalize
    let h1_norm = h1.normalize();
    let x_b_raw = h1_norm - n_b * h1_norm.dot(&n_b);
    let x_b = if x_b_raw.norm_squared() > 1e-8 {
        x_b_raw.normalize()
    } else {
        // Fallback: if h1 is parallel to n_b (degenerate), pick any orthogonal
        let tangent = if n_b.x.abs() > 0.9 {
            Vector3::y_axis().into_inner()
        } else {
            Vector3::x_axis().into_inner()
        };
        tangent.cross(&n_b).normalize()
    };

    let y_b = n_b.cross(&x_b);
    let rot_b = Matrix3::from_columns(&[x_b, y_b, n_b]);
    let pose_b = Pose::new(rot_b, t_a);

    Some([pose_a, pose_b])
}

/// Returns the 4 tag corners in object space with origin at the geometric center of the tag.
/// Corner order: TL, TR, BR, BL (clockwise). All Z=0 (tag lies on the object plane).
pub(crate) fn centered_tag_corners(tag_size: f64) -> [Vector3<f64>; 4] {
    let h = tag_size * 0.5;
    [
        Vector3::new(-h, -h, 0.0),
        Vector3::new(h, -h, 0.0),
        Vector3::new(h, h, 0.0),
        Vector3::new(-h, h, 0.0),
    ]
}

/// Use a Manifold-Aware Trust-Region Levenberg-Marquardt solver to refine the pose.
///
/// This upgrades the classic LM recipe to a SOTA production solver with three key improvements:
///
/// 1. **Huber M-Estimator (IRLS):** Wraps the reprojection residual in a Huber loss to
///    dynamically down-weight corners with large errors (e.g. from motion blur or occlusion),
///    preventing a single bad corner from corrupting the entire solution.
///
/// 2. **Marquardt Diagonal Scaling:** Damps each parameter proportionally to its own curvature
///    via `D = diag(J^T W J)`, instead of a uniform identity matrix. This correctly handles the
///    scale mismatch between rotational (radians) and translational (meters) parameters.
///
/// 3. **Nielsen's Gain Ratio Control:** Evaluates step quality using the ratio of actual vs.
///    predicted cost reduction. Good steps shrink `lambda` aggressively (Gauss-Newton speed),
///    while bad steps grow it with doubling-`nu` backoff (gradient descent safety). This is
///    strictly superior to the heuristic `lambda *= 10 / 0.1` approach.
#[allow(clippy::too_many_lines)]
fn refine_pose_lm(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    initial_pose: Pose,
    huber_delta_px: f64,
) -> Pose {
    let huber_delta = huber_delta_px;

    let mut pose = initial_pose;
    let obj_pts = centered_tag_corners(tag_size);

    // Nielsen's trust-region state. Start with small damping to encourage Gauss-Newton steps.
    let mut lambda = 1e-3_f64;
    let mut nu = 2.0_f64;
    let mut current_cost = huber_cost(intrinsics, corners, &obj_pts, &pose, huber_delta);

    // Allow up to 20 iterations; typically exits in 3-6 via the convergence gates below.
    for _ in 0..20 {
        // Build J^T W J (6x6) and J^T W r (6x1) with Huber IRLS weights.
        // Layout: delta = [tx, ty, tz, rx, ry, rz] (translation then rotation in se(3)).
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();

        for i in 0..4 {
            let p_world = obj_pts[i];
            let p_cam = pose.rotation * p_world + pose.translation;
            let z_inv = 1.0 / p_cam.z;
            let z_inv2 = z_inv * z_inv;

            let u_est = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
            let v_est = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;

            let res_u = corners[i][0] - u_est;
            let res_v = corners[i][1] - v_est;

            // Huber IRLS weight: w=1 inside the trust region, w=delta/r outside.
            // Applied per 2D point (geometric pixel distance).
            let r_norm = (res_u * res_u + res_v * res_v).sqrt();
            let w = if r_norm <= huber_delta {
                1.0
            } else {
                huber_delta / r_norm
            };

            // Jacobian of projection wrt Camera Point (du/dP_cam) (2x3):
            // [ fx/z   0     -fx*x/z^2 ]
            // [ 0      fy/z  -fy*y/z^2 ]
            let du_dp = Vector3::new(
                intrinsics.fx * z_inv,
                0.0,
                -intrinsics.fx * p_cam.x * z_inv2,
            );
            let dv_dp = Vector3::new(
                0.0,
                intrinsics.fy * z_inv,
                -intrinsics.fy * p_cam.y * z_inv2,
            );

            // Jacobian of Camera Point wrt Pose Update (Lie Algebra) (3x6):
            // d(exp(w)*P)/d(xi) at xi=0 = [ I | -[P]_x ]
            let mut row_u = Vector6::zeros();
            row_u[0] = du_dp[0];
            row_u[1] = du_dp[1];
            row_u[2] = du_dp[2];
            row_u[3] = du_dp[1] * p_cam.z - du_dp[2] * p_cam.y;
            row_u[4] = du_dp[2] * p_cam.x - du_dp[0] * p_cam.z;
            row_u[5] = du_dp[0] * p_cam.y - du_dp[1] * p_cam.x;

            let mut row_v = Vector6::zeros();
            row_v[0] = dv_dp[0];
            row_v[1] = dv_dp[1];
            row_v[2] = dv_dp[2];
            row_v[3] = dv_dp[1] * p_cam.z - dv_dp[2] * p_cam.y;
            row_v[4] = dv_dp[2] * p_cam.x - dv_dp[0] * p_cam.z;
            row_v[5] = dv_dp[0] * p_cam.y - dv_dp[1] * p_cam.x;

            // Accumulate J^T W J and J^T W r with Huber weight.
            jtj += w * (row_u * row_u.transpose() + row_v * row_v.transpose());
            jtr += w * (row_u * res_u + row_v * res_v);
        }

        // Gate 1: Gradient convergence — solver is at a stationary point.
        if jtr.amax() < 1e-8 {
            break;
        }

        // Marquardt diagonal scaling: D = diag(J^T W J).
        // Damps each DOF proportionally to its own curvature, correcting for the
        // scale mismatch between rotational and translational gradient magnitudes.
        let d_diag = Vector6::new(
            jtj[(0, 0)].max(1e-8),
            jtj[(1, 1)].max(1e-8),
            jtj[(2, 2)].max(1e-8),
            jtj[(3, 3)].max(1e-8),
            jtj[(4, 4)].max(1e-8),
            jtj[(5, 5)].max(1e-8),
        );

        // Solve (J^T W J + lambda * D) delta = J^T W r
        let mut jtj_damped = jtj;
        for k in 0..6 {
            jtj_damped[(k, k)] += lambda * d_diag[k];
        }

        let delta = if let Some(chol) = jtj_damped.cholesky() {
            chol.solve(&jtr)
        } else {
            // System is ill-conditioned; increase damping and retry.
            lambda *= 10.0;
            nu = 2.0;
            continue;
        };

        // Nielsen's gain ratio: rho = actual_reduction / predicted_reduction.
        // Predicted reduction from the quadratic model (Madsen et al. eq 3.9):
        // L(0) - L(delta) = 0.5 * delta^T (lambda * D * delta + J^T W r)
        let predicted_reduction = 0.5 * delta.dot(&(lambda * d_diag.component_mul(&delta) + jtr));

        // Evaluate new pose via SE(3) exponential map (manifold-safe update).
        let twist = Vector3::new(delta[3], delta[4], delta[5]);
        let trans_update = Vector3::new(delta[0], delta[1], delta[2]);
        let rot_update = nalgebra::Rotation3::new(twist).matrix().into_owned();
        let new_pose = Pose::new(
            rot_update * pose.rotation,
            rot_update * pose.translation + trans_update,
        );

        let new_cost = huber_cost(intrinsics, corners, &obj_pts, &new_pose, huber_delta);
        let actual_reduction = current_cost - new_cost;

        let rho = if predicted_reduction > 1e-12 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        if rho > 0.0 {
            // Accept step: update state and shrink lambda toward Gauss-Newton regime.
            pose = new_pose;
            current_cost = new_cost;
            // Nielsen's update rule: lambda scales by max(1/3, 1 - (2*rho - 1)^3).
            lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            nu = 2.0;

            // Gate 2: Step size convergence — pose has stopped moving.
            if delta.norm() < 1e-7 {
                break;
            }
        } else {
            // Reject step: grow lambda with doubling backoff to stay within trust region.
            lambda *= nu;
            nu *= 2.0;
        }
    }

    pose
}

/// Computes the total Huber robust cost over all four corner reprojection residuals.
///
/// The Huber function is quadratic for `|r| <= delta` (L2 regime) and linear beyond
/// (L1 regime), providing continuous differentiability at the transition point.
fn huber_cost(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
    delta: f64,
) -> f64 {
    let mut cost = 0.0;
    for i in 0..4 {
        let p = pose.project(&obj_pts[i], intrinsics);
        let r_u = corners[i][0] - p[0];
        let r_v = corners[i][1] - p[1];
        // Huber on the 2D geometric distance, consistent with the IRLS weight computation.
        let r_norm = (r_u * r_u + r_v * r_v).sqrt();
        if r_norm <= delta {
            cost += 0.5 * r_norm * r_norm;
        } else {
            cost += delta * (r_norm - 0.5 * delta);
        }
    }
    cost
}

fn reprojection_error(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
) -> f64 {
    let mut err_sq = 0.0;
    for i in 0..4 {
        let p = pose.project(&obj_pts[i], intrinsics);
        err_sq += (p[0] - corners[i][0]).powi(2) + (p[1] - corners[i][1]).powi(2);
    }
    err_sq
}

fn find_best_pose(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    candidates: &[Pose; 2],
) -> Pose {
    let obj_pts = centered_tag_corners(tag_size);

    let err0 = reprojection_error(intrinsics, corners, &obj_pts, &candidates[0]);
    let err1 = reprojection_error(intrinsics, corners, &obj_pts, &candidates[1]);

    // Choose the candidate with lower reprojection error.
    if err1 < err0 {
        candidates[1]
    } else {
        candidates[0]
    }
}

/// Refine poses for all valid candidates in the batch using the Structure of Arrays (SoA) layout.
///
/// This function operates only on the first `v` candidates in the batch, which must have been
/// partitioned such that all valid candidates are in the range `[0..v]`.
#[tracing::instrument(skip_all, name = "pipeline::pose_refinement")]
pub fn refine_poses_soa(
    batch: &mut DetectionBatch,
    v: usize,
    intrinsics: &CameraIntrinsics,
    tag_size: f64,
    img: Option<&ImageView>,
    mode: PoseEstimationMode,
) {
    refine_poses_soa_with_config(
        batch,
        v,
        intrinsics,
        tag_size,
        img,
        mode,
        &crate::config::DetectorConfig::default(),
    );
}

/// Refine poses for all valid candidates with explicit config for tuning parameters.
#[tracing::instrument(skip_all, name = "pipeline::pose_refinement")]
pub fn refine_poses_soa_with_config(
    batch: &mut DetectionBatch,
    v: usize,
    intrinsics: &CameraIntrinsics,
    tag_size: f64,
    img: Option<&ImageView>,
    mode: PoseEstimationMode,
    config: &crate::config::DetectorConfig,
) {
    use rayon::prelude::*;

    // Process valid candidates in parallel.
    // We collect into a temporary Vec to avoid unsafe parallel writes to the batch.
    let poses: Vec<_> = (0..v)
        .into_par_iter()
        .map(|i| {
            let corners = [
                [
                    f64::from(batch.corners[i][0].x),
                    f64::from(batch.corners[i][0].y),
                ],
                [
                    f64::from(batch.corners[i][1].x),
                    f64::from(batch.corners[i][1].y),
                ],
                [
                    f64::from(batch.corners[i][2].x),
                    f64::from(batch.corners[i][2].y),
                ],
                [
                    f64::from(batch.corners[i][3].x),
                    f64::from(batch.corners[i][3].y),
                ],
            ];

            // If GWLF is enabled, we should have corner covariances in the batch.
            // Check if any covariance is non-zero.
            let mut ext_covs = [Matrix2::zeros(); 4];
            let mut has_ext_covs = false;
            #[allow(clippy::needless_range_loop)]
            for j in 0..4 {
                ext_covs[j] = Matrix2::new(
                    f64::from(batch.corner_covariances[i][j * 4]),
                    f64::from(batch.corner_covariances[i][j * 4 + 1]),
                    f64::from(batch.corner_covariances[i][j * 4 + 2]),
                    f64::from(batch.corner_covariances[i][j * 4 + 3]),
                );
                if ext_covs[j].norm_squared() > 1e-12 {
                    has_ext_covs = true;
                }
            }

            let (pose_opt, _) = estimate_tag_pose_with_config(
                intrinsics,
                &corners,
                tag_size,
                img,
                mode,
                config,
                if has_ext_covs { Some(&ext_covs) } else { None },
            );

            if let Some(pose) = pose_opt {
                // `UnitQuaternion::from_matrix` delegates to `Rotation3::from_matrix_eps` with
                // `max_iter = 0`, which nalgebra treats as usize::MAX ("loop until convergence").
                // For degenerate IPPE outputs (near-singular homographies at extreme tag angles),
                // the Müller iterative algorithm never converges — infinite loop.
                //
                // Use the closed-form Shepperd method instead: wrap the matrix as a `Rotation3`
                // (no orthogonalization) then extract the quaternion analytically. The LM solver
                // maintains pose on SO(3) via the exponential map, so `pose.rotation` is always
                // sufficiently close to orthogonal for this to be accurate.
                let rot = Rotation3::from_matrix_unchecked(pose.rotation);
                let q = UnitQuaternion::from_rotation_matrix(&rot);
                let t = pose.translation;

                // Data layout: [tx, ty, tz, qx, qy, qz, qw]
                let mut data = [0.0f32; 7];
                data[0] = t.x as f32;
                data[1] = t.y as f32;
                data[2] = t.z as f32;
                data[3] = q.coords.x as f32;
                data[4] = q.coords.y as f32;
                data[5] = q.coords.z as f32;
                data[6] = q.coords.w as f32;
                Some(data)
            } else {
                None
            }
        })
        .collect();

    for (i, pose_data) in poses.into_iter().enumerate() {
        if let Some(data) = pose_data {
            batch.poses[i] = Pose6D { data, padding: 0.0 };
        } else {
            batch.poses[i] = Pose6D {
                data: [0.0; 7],
                padding: 0.0,
            };
        }
    }
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
        let obj_pts = centered_tag_corners(tag_size);

        let mut img_pts = [[0.0, 0.0]; 4];
        for i in 0..4 {
            img_pts[i] = gt_pose.project(&obj_pts[i], &intrinsics);
        }

        let (est_pose, _) = estimate_tag_pose(
            &intrinsics,
            &img_pts,
            tag_size,
            None,
            PoseEstimationMode::Fast,
        );
        let est_pose = est_pose.expect("Pose estimation failed");

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
            let obj_pts = centered_tag_corners(tag_size);

            let mut img_pts = [[0.0, 0.0]; 4];
            for i in 0..4 {
                let p = gt_pose.project(&obj_pts[i], &intrinsics);
                // Add tiny bit of noise
                img_pts[i] = [p[0] + noise, p[1] + noise];
            }

            if let (Some(est_pose), _) = estimate_tag_pose(&intrinsics, &img_pts, tag_size, None, PoseEstimationMode::Fast) {
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
