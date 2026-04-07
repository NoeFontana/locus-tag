#![allow(clippy::similar_names)]
use crate::image::ImageView;
use crate::pose::{CameraIntrinsics, Pose, projection_jacobian, symmetrize_jtj6};
use nalgebra::{Matrix2, Matrix6, Vector3, Vector6};

const OFF_IMAGE_CORNER_VARIANCE: f64 = 100.0;

fn corner_has_structure_tensor_support(img: &ImageView, center: [f64; 2], radius: i32) -> bool {
    let cx = center[0].floor() as isize;
    let cy = center[1].floor() as isize;
    let radius = radius as isize;
    let w = img.width as isize;
    let h = img.height as isize;

    cx - radius >= 1 && cx + radius <= w - 2 && cy - radius >= 1 && cy + radius <= h - 2
}

/// Compute the covariance of the corner position estimation error based on the Structure Tensor.
///
/// The covariance $\Sigma_c$ is approximated as the inverse of the Structure Tensor $S$:
/// $$ \Sigma_c \approx \sigma_n^2 S^{-1} $$
/// where $\sigma_n^2$ is the pixel noise variance (assumed around 2.0 for typical webcams).
///
/// The Structure Tensor is computed over a small window around the corner.
pub(crate) fn compute_corner_covariance(
    img: &ImageView,
    center: [f64; 2],
    alpha_max: f64,
    sigma_n_sq: f64,
    radius: i32,
) -> Matrix2<f64> {
    let cx = center[0].floor() as isize;
    let cy = center[1].floor() as isize;

    let mut sum_gx2 = 0.0;
    let mut sum_gy2 = 0.0;
    let mut sum_gxgy = 0.0;

    let w = img.width.cast_signed();
    let h = img.height.cast_signed();
    let stride = img.stride.cast_signed();

    let x_start = (cx - radius as isize).max(1);
    let x_end = (cx + radius as isize).min(w - 2);
    let y_start = (cy - radius as isize).max(1);
    let y_end = (cy + radius as isize).min(h - 2);

    debug_assert!(x_start <= x_end && y_start <= y_end);

    // Pre-slice rows to exact bounds so LLVM can prove k, k+1, k+2 ∈ [0, len)
    // and elide bounds checks, enabling inner-loop auto-vectorisation.
    let x_min = (x_start - 1).cast_unsigned();
    let x_count = (x_end - x_start + 1).cast_unsigned();
    let row_slice_len = x_count + 2; // left neighbor + x_count pixels + right neighbor

    for py in y_start..=y_end {
        let offset = (py * stride).cast_unsigned();
        let su = stride.cast_unsigned();

        let base0 = offset - su + x_min;
        let base1 = offset + x_min;
        let base2 = offset + su + x_min;

        let row0 = &img.data[base0..base0 + row_slice_len];
        let row1 = &img.data[base1..base1 + row_slice_len];
        let row2 = &img.data[base2..base2 + row_slice_len];

        for k in 0..x_count {
            let p00 = i32::from(row0[k]);
            let p01 = i32::from(row0[k + 1]);
            let p02 = i32::from(row0[k + 2]);
            let p10 = i32::from(row1[k]);
            let p12 = i32::from(row1[k + 2]);
            let p20 = i32::from(row2[k]);
            let p21 = i32::from(row2[k + 1]);
            let p22 = i32::from(row2[k + 2]);

            let gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
            let gy = (p22 + 2 * p21 + p20) - (p02 + 2 * p01 + p00);

            let gx_f = f64::from(gx);
            let gy_f = f64::from(gy);

            sum_gx2 += gx_f * gx_f;
            sum_gy2 += gy_f * gy_f;
            sum_gxgy += gx_f * gy_f;
        }
    }

    // Tikhonov regularization on the structure tensor (ε=1.0)
    let s00 = sum_gx2 + 1.0;
    let s11 = sum_gy2 + 1.0;
    let s01 = sum_gxgy;

    let det = s00 * s11 - s01 * s01;
    // Det check: if the window is flat, return high uncertainty.
    if det < 1e-4 {
        return Matrix2::identity().scale(OFF_IMAGE_CORNER_VARIANCE);
    }

    let inv_det = 1.0 / det;
    let s_inv = Matrix2::new(s11, -s01, -s01, s00).scale(inv_det);

    // Anisotropy ratio R = λ_min / λ_max.
    let trace = s00 + s11;
    let discriminant = ((s00 - s11).powi(2) + 4.0 * s01 * s01).sqrt();
    let r = ((trace - discriminant) / (trace + discriminant)).max(0.0);
    let alpha = alpha_max * (1.0 - r).powi(2);

    s_inv.scale(sigma_n_sq) + Matrix2::identity().scale(alpha)
}

/// Compute framework uncertainty for all 4 corners.
#[must_use]
pub(crate) fn compute_framework_uncertainty(
    img: &ImageView,
    corners: &[[f64; 2]; 4],
    _h_poly: &crate::decoder::Homography,
    tikhonov_alpha_max: f64,
    sigma_n_sq: f64,
    structure_tensor_radius: u8,
) -> [Matrix2<f64>; 4] {
    let mut covariances = [Matrix2::zeros(); 4];
    let radius = i32::from(structure_tensor_radius);
    for i in 0..4 {
        covariances[i] = if corner_has_structure_tensor_support(img, corners[i], radius) {
            compute_corner_covariance(img, corners[i], tikhonov_alpha_max, sigma_n_sq, radius)
        } else {
            Matrix2::identity().scale(OFF_IMAGE_CORNER_VARIANCE)
        };
    }
    covariances
}

/// Accumulates the normal equations J^T W̃ J and J^T W̃ r, and returns the total Huber cost.
fn build_normal_equations(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
    info_matrices: &[Matrix2<f64>; 4],
    huber_k: f64,
) -> (Matrix6<f64>, Vector6<f64>, f64) {
    let mut jtj = Matrix6::<f64>::zeros();
    let mut jtr = Vector6::<f64>::zeros();
    let mut total_cost = 0.0;

    for i in 0..4 {
        let p_cam = pose.rotation * obj_pts[i] + pose.translation;
        if p_cam.z < 1e-4 {
            total_cost += 1e6;
            continue;
        }

        let z_inv = 1.0 / p_cam.z;
        let x_z = p_cam.x * z_inv;
        let y_z = p_cam.y * z_inv;

        let u_est = intrinsics.fx * x_z + intrinsics.cx;
        let v_est = intrinsics.fy * y_z + intrinsics.cy;
        let res_u = corners[i][0] - u_est;
        let res_v = corners[i][1] - v_est;

        let info = &info_matrices[i];
        let s_i_sq = res_u * (info[(0, 0)] * res_u + info[(0, 1)] * res_v)
            + res_v * (info[(1, 0)] * res_u + info[(1, 1)] * res_v);
        let s_i = s_i_sq.sqrt();

        if s_i <= huber_k {
            total_cost += 0.5 * s_i_sq;
        } else {
            total_cost += huber_k * (s_i - 0.5 * huber_k);
        }

        let w = if s_i <= huber_k { 1.0 } else { huber_k / s_i };
        let w00 = info[(0, 0)] * w;
        let w01 = info[(0, 1)] * w;
        let w10 = info[(1, 0)] * w;
        let w11 = info[(1, 1)] * w;

        let (ju0, ju2, ju3, ju4, ju5, jv1, jv2, jv3, jv4, jv5) =
            projection_jacobian(x_z, y_z, z_inv, intrinsics);

        let k00 = ju0 * w00;
        let k01 = ju0 * w01;
        let k10 = jv1 * w10;
        let k11 = jv1 * w11;
        let k20 = ju2 * w00 + jv2 * w10;
        let k21 = ju2 * w01 + jv2 * w11;
        let k30 = ju3 * w00 + jv3 * w10;
        let k31 = ju3 * w01 + jv3 * w11;
        let k40 = ju4 * w00 + jv4 * w10;
        let k41 = ju4 * w01 + jv4 * w11;
        let k50 = ju5 * w00 + jv5 * w10;
        let k51 = ju5 * w01 + jv5 * w11;

        jtr[0] += k00 * res_u + k01 * res_v;
        jtr[1] += k10 * res_u + k11 * res_v;
        jtr[2] += k20 * res_u + k21 * res_v;
        jtr[3] += k30 * res_u + k31 * res_v;
        jtr[4] += k40 * res_u + k41 * res_v;
        jtr[5] += k50 * res_u + k51 * res_v;

        jtj[(0, 0)] += k00 * ju0;
        jtj[(0, 1)] += k01 * jv1;
        jtj[(0, 2)] += k00 * ju2 + k01 * jv2;
        jtj[(0, 3)] += k00 * ju3 + k01 * jv3;
        jtj[(0, 4)] += k00 * ju4 + k01 * jv4;
        jtj[(0, 5)] += k00 * ju5 + k01 * jv5;

        jtj[(1, 1)] += k11 * jv1;
        jtj[(1, 2)] += k10 * ju2 + k11 * jv2;
        jtj[(1, 3)] += k10 * ju3 + k11 * jv3;
        jtj[(1, 4)] += k10 * ju4 + k11 * jv4;
        jtj[(1, 5)] += k10 * ju5 + k11 * jv5;

        jtj[(2, 2)] += k20 * ju2 + k21 * jv2;
        jtj[(2, 3)] += k20 * ju3 + k21 * jv3;
        jtj[(2, 4)] += k20 * ju4 + k21 * jv4;
        jtj[(2, 5)] += k20 * ju5 + k21 * jv5;

        jtj[(3, 3)] += k30 * ju3 + k31 * jv3;
        jtj[(3, 4)] += k30 * ju4 + k31 * jv4;
        jtj[(3, 5)] += k30 * ju5 + k31 * jv5;

        jtj[(4, 4)] += k40 * ju4 + k41 * jv4;
        jtj[(4, 5)] += k40 * ju5 + k41 * jv5;

        jtj[(5, 5)] += k50 * ju5 + k51 * jv5;
    }

    symmetrize_jtj6(&mut jtj);

    (jtj, jtr, total_cost)
}

/// Huber robust cost over Mahalanobis distances.
fn huber_mahalanobis_cost(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
    info_matrices: &[Matrix2<f64>; 4],
    k: f64,
) -> f64 {
    let mut cost = 0.0;
    let fx = intrinsics.fx;
    let fy = intrinsics.fy;
    let cx = intrinsics.cx;
    let cy = intrinsics.cy;

    for i in 0..4 {
        let p_cam = pose.rotation * obj_pts[i] + pose.translation;
        if p_cam.z < 1e-4 {
            cost += 1e6;
            continue;
        }
        let z_inv = 1.0 / p_cam.z;
        let u_est = fx * p_cam.x * z_inv + cx;
        let v_est = fy * p_cam.y * z_inv + cy;
        let res_u = corners[i][0] - u_est;
        let res_v = corners[i][1] - v_est;

        let info = &info_matrices[i];
        let s_sq = res_u * (info[(0, 0)] * res_u + info[(0, 1)] * res_v)
            + res_v * (info[(1, 0)] * res_u + info[(1, 1)] * res_v);
        let s = s_sq.sqrt();

        if s <= k {
            cost += 0.5 * s_sq;
        } else {
            cost += k * (s - 0.5 * k);
        }
    }
    cost
}

/// Refine the pose by minimizing a Huber-robust Mahalanobis objective.
#[must_use]
pub(crate) fn refine_pose_lm_weighted(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    initial_pose: Pose,
    corner_covariances: &[Matrix2<f64>; 4],
) -> (Pose, [[f64; 6]; 6]) {
    const HUBER_K: f64 = 1.345;
    const MAX_ITERS: usize = 20;

    let mut pose = initial_pose;
    let obj_pts = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(tag_size, 0.0, 0.0),
        Vector3::new(tag_size, tag_size, 0.0),
        Vector3::new(0.0, tag_size, 0.0),
    ];

    let mut info_matrices = [Matrix2::<f64>::zeros(); 4];
    for i in 0..4 {
        info_matrices[i] = corner_covariances[i]
            .try_inverse()
            .unwrap_or_else(Matrix2::identity);
    }

    let mut lambda = 1e-3_f64;
    let mut nu = 2.0_f64;

    // Cache for normal equations at the current accepted pose.
    let mut current_jtj = Matrix6::zeros();
    let mut current_jtr = Vector6::zeros();
    let mut current_cost = f64::MAX;
    let mut needs_rebuild = true;

    for _ in 0..MAX_ITERS {
        if needs_rebuild {
            let (jtj, jtr, cost) = build_normal_equations(
                intrinsics,
                corners,
                &obj_pts,
                &pose,
                &info_matrices,
                HUBER_K,
            );
            current_jtj = jtj;
            current_jtr = jtr;
            current_cost = cost;
            needs_rebuild = false;
        }

        if current_jtr.amax() < 1e-8 {
            break;
        }

        let mut jtj_damped = current_jtj;
        for k in 0..6 {
            // Marquardt damping with a safety epsilon to prevent singularity.
            jtj_damped[(k, k)] += lambda * current_jtj[(k, k)].max(1e-6);
        }

        let delta = if let Some(chol) = jtj_damped.cholesky() {
            chol.solve(&current_jtr)
        } else {
            lambda *= 10.0;
            nu = 2.0;
            continue;
        };

        // Gain ratio numerator: Actual cost reduction
        let twist = Vector3::new(delta[3], delta[4], delta[5]);
        let trans_update = Vector3::new(delta[0], delta[1], delta[2]);
        let rot_update = nalgebra::Rotation3::new(twist).matrix().into_owned();
        let new_pose = Pose::new(
            rot_update * pose.rotation,
            rot_update * pose.translation + trans_update,
        );

        let new_cost = huber_mahalanobis_cost(
            intrinsics,
            corners,
            &obj_pts,
            &new_pose,
            &info_matrices,
            HUBER_K,
        );

        // Gain ratio denominator: Predicted reduction based on quadratic model
        let predicted_reduction =
            0.5 * delta.dot(&(lambda * delta.component_mul(&current_jtj.diagonal()) + current_jtr));
        let actual_reduction = current_cost - new_cost;
        let rho = if predicted_reduction > 1e-12 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        if rho > 0.0 {
            // Accept step: update pose and trigger Hessian rebuild for next iteration.
            pose = new_pose;
            current_cost = new_cost;
            needs_rebuild = true;
            lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            nu = 2.0;
            if delta.norm() < 1e-7 {
                break;
            }
        } else {
            // Reject step: increase damping and retry with same Hessian.
            lambda *= nu;
            nu *= 2.0;
        }
    }

    // Final covariance extraction from the last cached Hessian.
    if needs_rebuild {
        let (jtj, _, _) = build_normal_equations(
            intrinsics,
            corners,
            &obj_pts,
            &pose,
            &info_matrices,
            HUBER_K,
        );
        current_jtj = jtj;
    }
    let covariance = current_jtj.try_inverse().unwrap_or_else(Matrix6::identity);

    (pose, covariance.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{CameraIntrinsics, Pose};
    use crate::{decoder::Homography, image::ImageView};

    #[test]
    fn test_jacobian_rotation_rows() {
        // Verify rotation Jacobian rows for left SE(3) perturbation.
        // Setup: p_cam = [1,1,1], fx=fy=800, cx=cy=0, z=1.
        //
        // Left perturbation: T_new = exp(δξ)·T  →  ∂p_cam/∂δω = −[p_cam]×
        // so ∂u/∂δω = du_dp · (−[p_cam]×) = p_cam × du_dp.
        //
        // Expected (via cross product with p_cam=[1,1,1], du_dp=[800,0,-800]):
        //   δω_x: p × j = (1·(-800) - 1·0,  …) → ∂u/∂δω_x = 1·(-800) - 1·0 = -800
        //   δω_y: → ∂u/∂δω_y = 1·800 - 1·(-800) = 1600
        //   δω_z: → ∂u/∂δω_z = 1·0 - 1·800 = -800
        let fx = 800.0_f64;
        let z_inv = 1.0_f64;
        let z_inv2 = 1.0_f64;
        let p_cam_x = 1.0_f64;
        let p_cam_y = 1.0_f64;
        let p_cam_z = 1.0_f64;

        let du_dp = nalgebra::Vector3::new(fx * z_inv, 0.0, -fx * p_cam_x * z_inv2);

        // Implementation formula (p_cam × du_dp):
        let row_u_3 = p_cam_y * du_dp[2] - p_cam_z * du_dp[1];
        let row_u_4 = p_cam_z * du_dp[0] - p_cam_x * du_dp[2];
        let row_u_5 = p_cam_x * du_dp[1] - p_cam_y * du_dp[0];

        assert!((row_u_3 - (-800.0)).abs() < 1e-9, "∂u/∂δω_x: got {row_u_3}");
        assert!((row_u_4 - 1600.0).abs() < 1e-9, "∂u/∂δω_y: got {row_u_4}");
        assert!((row_u_5 - (-800.0)).abs() < 1e-9, "∂u/∂δω_z: got {row_u_5}");
    }

    #[test]
    fn test_convergence_from_rotated_initial_pose() {
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);

        // Ground truth pose: board at ~0.5m, slightly tilted
        let gt_rot = nalgebra::Rotation3::from_euler_angles(0.1, 0.15, 0.05)
            .matrix()
            .into_owned();
        let gt_t = nalgebra::Vector3::new(0.02, -0.01, 0.5);
        let gt_pose = Pose::new(gt_rot, gt_t);

        // tag_size = 0.02m; obj_pts match refine_pose_lm_weighted internal layout:
        // [0,0,0], [s,0,0], [s,s,0], [0,s,0]
        let s = 0.02_f64;
        let obj_pts = [
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
            nalgebra::Vector3::new(s, 0.0, 0.0),
            nalgebra::Vector3::new(s, s, 0.0),
            nalgebra::Vector3::new(0.0, s, 0.0),
        ];

        // Perfect observed corners (no noise)
        let corners: [[f64; 2]; 4] =
            core::array::from_fn(|i| gt_pose.project(&obj_pts[i], &intrinsics));

        // Initial pose: same translation, rotation perturbed by ~15° about y-axis
        let perturb = nalgebra::Rotation3::from_euler_angles(0.0, 0.26, 0.0)
            .matrix()
            .into_owned(); // 0.26 rad ≈ 14.9°
        let init_rot = perturb * gt_rot;
        let init_pose = Pose::new(init_rot, gt_t);

        // Use identity covariances (isotropic, unit weight)
        let identity_covs = [Matrix2::<f64>::identity(); 4];

        let (result, _cov) =
            refine_pose_lm_weighted(&intrinsics, &corners, s, init_pose, &identity_covs);

        let t_err = (result.translation - gt_t).norm() * 1000.0; // mm
        let q_gt = nalgebra::UnitQuaternion::from_matrix(&gt_rot);
        let q_est = nalgebra::UnitQuaternion::from_matrix(&result.rotation);
        let r_err_deg = q_gt.angle_to(&q_est).to_degrees();

        assert!(t_err < 0.01, "translation error too large: {t_err:.4} mm");
        assert!(
            r_err_deg < 0.001,
            "rotation error too large: {r_err_deg:.5} deg"
        );
    }

    /// Analytical gradient (from `jtr`) must equal the numerical gradient of
    /// `huber_mahalanobis_cost` via central differences.
    ///
    /// Derivation: C = Σ ρ(s_i), r_i = obs - pred, ∂r/∂ξ = -J
    ///   → ∂C/∂ξ_k = -Σ w_i · (J_i^T W_i r_i)_k = -jtr[k]
    ///
    /// A sign error in J flips jtr for those DOFs while leaving ∂C/∂ξ unchanged,
    /// so the assertion `jtr[k] ≈ -numerical_grad[k]` exposes the bug directly.
    #[test]
    fn test_gradient_matches_finite_differences() {
        const K: f64 = 1.345;
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);

        // Off-center pose with non-trivial rotation to make coupling terms large.
        let rot = nalgebra::Rotation3::from_euler_angles(0.3, 0.2, 0.1)
            .matrix()
            .into_owned();
        let t = nalgebra::Vector3::new(0.15, 0.12, 0.5);
        let pose = Pose::new(rot, t);

        let s = 0.04_f64;
        let obj_pts = [
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
            nalgebra::Vector3::new(s, 0.0, 0.0),
            nalgebra::Vector3::new(s, s, 0.0),
            nalgebra::Vector3::new(0.0, s, 0.0),
        ];

        // Perfect projections with deliberate noise → non-zero residuals required
        // for the gradient to expose the sign error.
        let mut corners: [[f64; 2]; 4] =
            core::array::from_fn(|i| pose.project(&obj_pts[i], &intrinsics));
        corners[0][0] += 3.0;
        corners[0][1] -= 2.0;
        corners[1][0] -= 1.5;
        corners[1][1] += 2.5;
        corners[2][0] += 0.8;
        corners[2][1] += 1.2;

        let info = [Matrix2::<f64>::identity(); 4];

        // Analytical gradient: jtr[k] = (J^T W̃ r)[k], so ∂C/∂ξ_k = -jtr[k].
        let (_jtj, jtr, _cost) =
            build_normal_equations(&intrinsics, &corners, &obj_pts, &pose, &info, K);

        let eps = 1e-6;
        for dof in 0..6 {
            let cost_fwd = huber_mahalanobis_cost(
                &intrinsics,
                &corners,
                &obj_pts,
                &perturb_pose(&pose, dof, eps),
                &info,
                K,
            );
            let cost_bwd = huber_mahalanobis_cost(
                &intrinsics,
                &corners,
                &obj_pts,
                &perturb_pose(&pose, dof, -eps),
                &info,
                K,
            );
            let numerical_grad = (cost_fwd - cost_bwd) / (2.0 * eps);

            // A sign error in J flips jtr[dof] relative to the true gradient, so
            // |jtr + grad| ≈ 2·|grad|, giving relative_err ≈ 2.0.
            // Numerical finite-difference error gives relative_err ≪ 1e-3.
            let scale = jtr[dof].abs().max(numerical_grad.abs()).max(1.0);
            let relative_err = (jtr[dof] + numerical_grad).abs() / scale;
            assert!(
                relative_err < 1e-3,
                "DOF {dof}: jtr={:.6} -numerical_grad={:.6} relative_err={:.2e}",
                jtr[dof],
                -numerical_grad,
                relative_err
            );
        }
    }

    fn perturb_pose(pose: &Pose, dof: usize, eps: f64) -> Pose {
        let mut delta = nalgebra::Vector6::<f64>::zeros();
        delta[dof] = eps;
        let twist = nalgebra::Vector3::new(delta[3], delta[4], delta[5]);
        let t_update = nalgebra::Vector3::new(delta[0], delta[1], delta[2]);
        let r_update = nalgebra::Rotation3::new(twist).matrix().into_owned();
        Pose::new(
            r_update * pose.rotation,
            r_update * pose.translation + t_update,
        )
    }

    #[test]
    fn test_framework_uncertainty_culls_off_image_corners() {
        let mut pixels = vec![0_u8; 9 * 9];
        for y in 0..9 {
            for x in 0..9 {
                pixels[y * 9 + x] = match (x >= 4, y >= 4) {
                    (false, false) | (true, true) => 0,
                    (true, false) | (false, true) => 255,
                };
            }
        }
        let img = ImageView::new(&pixels, 9, 9, 9).expect("valid test image");
        let corners = [[4.0, 4.0], [0.2, 4.0], [4.0, 8.9], [4.0, 0.1]];
        let h = Homography {
            h: nalgebra::Matrix3::identity(),
        };

        let covariances = compute_framework_uncertainty(&img, &corners, &h, 0.1, 2.0, 2);

        assert_ne!(
            covariances[0],
            Matrix2::identity().scale(OFF_IMAGE_CORNER_VARIANCE)
        );
        assert_eq!(
            covariances[1],
            Matrix2::identity().scale(OFF_IMAGE_CORNER_VARIANCE)
        );
        assert_eq!(
            covariances[2],
            Matrix2::identity().scale(OFF_IMAGE_CORNER_VARIANCE)
        );
        assert_eq!(
            covariances[3],
            Matrix2::identity().scale(OFF_IMAGE_CORNER_VARIANCE)
        );
    }
}
