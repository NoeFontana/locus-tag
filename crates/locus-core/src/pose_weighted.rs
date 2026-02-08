use nalgebra::{Matrix2, Matrix6, Vector2, Vector3, Vector6};
use crate::image::ImageView;
use crate::pose::{CameraIntrinsics, Pose};

/// Compute the covariance of the corner position estimation error based on the Structure Tensor.
///
/// The covariance $\Sigma_c$ is approximated as the inverse of the Structure Tensor $S$:
/// $$ \Sigma_c \approx \sigma_n^2 S^{-1} $$
/// where $\sigma_n^2$ is the pixel noise variance (assumed around 2.0 for typical webcams).
///
/// The Structure Tensor is computed over a small window around the corner.
fn compute_corner_covariance(img: &ImageView, center: [f64; 2]) -> Matrix2<f64> {
    let radius = 2; // 5x5 window
    let cx = center[0].round() as isize;
    let cy = center[1].round() as isize;

    let mut sum_gx2 = 0.0;
    let mut sum_gy2 = 0.0;
    let mut sum_gxgy = 0.0;

    let w = img.width as isize;
    let h = img.height as isize;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = cx + dx;
            let py = cy + dy;

            if px < 1 || px >= w - 1 || py < 1 || py >= h - 1 {
                continue;
            }

            let idx = (py * img.stride as isize + px) as usize;
            let stride = img.stride;
            
            let p00 = i16::from(img.data[idx - stride - 1]);
            let p01 = i16::from(img.data[idx - stride]);
            let p02 = i16::from(img.data[idx - stride + 1]);
            let p10 = i16::from(img.data[idx - 1]);
            let p12 = i16::from(img.data[idx + 1]);
            let p20 = i16::from(img.data[idx + stride - 1]);
            let p21 = i16::from(img.data[idx + stride]);
            let p22 = i16::from(img.data[idx + stride + 1]);

            let gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
            let gy = (p22 + 2 * p21 + p20) - (p02 + 2 * p01 + p00);
            
            let gx = f64::from(gx);
            let gy = f64::from(gy);

            let weight = 1.0; 

            sum_gx2 += weight * gx * gx;
            sum_gy2 += weight * gy * gy;
            sum_gxgy += weight * gx * gy;
        }
    }

    let epsilon = 1.0; 
    let s = Matrix2::new(sum_gx2 + epsilon, sum_gxgy, sum_gxgy, sum_gy2 + epsilon);

    let sigma_n_sq = 4.0;
    
    let det = s[(0,0)] * s[(1,1)] - s[(0,1)] * s[(1,0)];
    if det.abs() < 1e-6 {
         return Matrix2::identity().scale(100.0);
    }
    
    let inv_det = 1.0 / det;
    let s_inv = Matrix2::new(s[(1,1)], -s[(0,1)], -s[(1,0)], s[(0,0)]).scale(inv_det);
    
    s_inv.scale(sigma_n_sq)
}

/// Compute framework uncertainty for all 4 corners.
///
/// This serves as a batch wrapper around `compute_corner_covariance`.
pub fn compute_framework_uncertainty(
    img: &ImageView,
    corners: &[[f64; 2]; 4],
    _h_poly: &crate::decoder::Homography,
) -> [Matrix2<f64>; 4] {
    let mut covariances = [Matrix2::zeros(); 4];
    for i in 0..4 {
        covariances[i] = compute_corner_covariance(img, corners[i]);
    }
    covariances
}

/// Use Levenberg-Marquardt to refine the pose by minimizing Mahalanobis distance.
pub fn refine_pose_lm_weighted(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    initial_pose: Pose,
    corner_covariances: &[Matrix2<f64>; 4],
) -> (Pose, [[f64; 6]; 6]) {
    let mut pose = initial_pose;
    let s = tag_size * 0.5;
    let obj_pts = [
        Vector3::new(-s, -s, 0.0),
        Vector3::new(s, -s, 0.0),
        Vector3::new(s, s, 0.0),
        Vector3::new(-s, s, 0.0),
    ];
    
    let mut info_matrices = [Matrix2::zeros(); 4];
    for i in 0..4 {
        match corner_covariances[i].try_inverse() {
            Some(inv) => info_matrices[i] = inv,
            None => info_matrices[i] = Matrix2::identity(),
        }
    }

    let mut lambda = 0.01;
    let mut current_err = weighted_reprojection_error(intrinsics, corners, &obj_pts, &pose, &info_matrices);

    let mut jtj = Matrix6::<f64>::zeros();

    for _ in 0..5 {
        jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();

        for i in 0..4 {
            let p_world = obj_pts[i];
            let p_cam = pose.rotation * p_world + pose.translation;
            let z_inv = 1.0 / p_cam.z;
            let z_inv2 = z_inv * z_inv;

            let u_est = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
            let v_est = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;

            let res = Vector2::new(corners[i][0] - u_est, corners[i][1] - v_est);

            let du_dp = Vector3::new(intrinsics.fx * z_inv, 0.0, -intrinsics.fx * p_cam.x * z_inv2);
            let dv_dp = Vector3::new(0.0, intrinsics.fy * z_inv, -intrinsics.fy * p_cam.y * z_inv2);
            
            let mut jac_i = nalgebra::SMatrix::<f64, 2, 6>::zeros();
            
            // Row 0 (u)
            jac_i[(0,0)] = du_dp[0];
            jac_i[(0,1)] = du_dp[1];
            jac_i[(0,2)] = du_dp[2];
            jac_i[(0,3)] = du_dp[1] * p_cam.z - du_dp[2] * p_cam.y;
            jac_i[(0,4)] = du_dp[2] * p_cam.x - du_dp[0] * p_cam.z;
            jac_i[(0,5)] = du_dp[0] * p_cam.y - du_dp[1] * p_cam.x;
            
            // Row 1 (v)
            jac_i[(1,0)] = dv_dp[0];
            jac_i[(1,1)] = dv_dp[1];
            jac_i[(1,2)] = dv_dp[2];
            jac_i[(1,3)] = dv_dp[1] * p_cam.z - dv_dp[2] * p_cam.y;
            jac_i[(1,4)] = dv_dp[2] * p_cam.x - dv_dp[0] * p_cam.z;
            jac_i[(1,5)] = dv_dp[0] * p_cam.y - dv_dp[1] * p_cam.x;
            
            let weighted_jac = jac_i.transpose() * info_matrices[i]; 
            
            jtj += weighted_jac * jac_i;
            jtr += weighted_jac * res;
        }

        let mut jtj_damped = jtj;
        for k in 0..6 {
            jtj_damped[(k, k)] += lambda;
        }

        let decomposition = jtj_damped.cholesky();
        let delta = if let Some(chol) = decomposition {
            chol.solve(&jtr)
        } else {
            lambda *= 10.0;
            continue;
        };

        let update_twist = Vector3::new(delta[3], delta[4], delta[5]);
        let update_trans = Vector3::new(delta[0], delta[1], delta[2]);
        let update_rot = nalgebra::Rotation3::new(update_twist).matrix().into_owned();

        let new_rot = update_rot * pose.rotation;
        let new_trans = update_rot * pose.translation + update_trans;
        let new_pose = Pose::new(new_rot, new_trans);

        let new_err = weighted_reprojection_error(intrinsics, corners, &obj_pts, &new_pose, &info_matrices);

        if new_err < current_err {
            pose = new_pose;
            current_err = new_err;
            lambda *= 0.1;
        } else {
            lambda *= 10.0;
        }

        if delta.norm() < 1e-6 {
            break;
        }
    }
    
    let mse = current_err.max(1e-9) / 2.0; // Reduced dof?
    
    let covariance = jtj.try_inverse().unwrap_or(Matrix6::identity()).scale(mse);
    
    (pose, covariance.into())
}

fn weighted_reprojection_error(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
    info_matrices: &[Matrix2<f64>; 4],
) -> f64 {
    let mut err = 0.0;
    for i in 0..4 {
        let p = pose.project(&obj_pts[i], intrinsics);
        let res = Vector2::new(corners[i][0] - p[0], corners[i][1] - p[1]);
        err += res.dot(&(info_matrices[i] * res));
    }
    err
}
