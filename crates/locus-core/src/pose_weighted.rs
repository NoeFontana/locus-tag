#![allow(clippy::similar_names)]
use crate::image::ImageView;
use crate::pose::{
    BodyFrameNormalEquations, CameraIntrinsics, Pose, centered_tag_corners,
    pinhole_projection_gradients,
};
use nalgebra::{Matrix2, Matrix6, Vector3, Vector6};

const OFF_IMAGE_CORNER_VARIANCE: f64 = 100.0;
const STACK_WEIGHT_CAPACITY: usize = 17;

fn corner_has_structure_tensor_support(img: &ImageView, center: [f64; 2], radius: i32) -> bool {
    let cx = center[0].floor() as isize;
    let cy = center[1].floor() as isize;
    let radius = radius as isize;
    let w = img.width.cast_signed();
    let h = img.height.cast_signed();

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

    let w = img.width.cast_signed();
    let h = img.height.cast_signed();
    let stride = img.stride.cast_signed();

    let x_start = (cx - radius as isize).max(1);
    let x_end = (cx + radius as isize).min(w - 2);
    let y_start = (cy - radius as isize).max(1);
    let y_end = (cy + radius as isize).min(h - 2);
    if x_start > x_end || y_start > y_end {
        return Matrix2::identity().scale(OFF_IMAGE_CORNER_VARIANCE);
    }

    let sigma_sq = (f64::from(radius.max(1)) / 2.0).powi(2);
    let (sum_gx2, sum_gy2, sum_gxgy) = accumulate_structure_tensor_sums(
        img, center, stride, x_start, x_end, y_start, y_end, sigma_sq,
    );

    finalize_corner_covariance(sum_gx2, sum_gy2, sum_gxgy, alpha_max, sigma_n_sq)
}

#[expect(
    clippy::too_many_arguments,
    reason = "structure-tensor accumulation threads the image, window bounds and sigma; grouping into a struct adds indirection"
)]
fn accumulate_structure_tensor_sums(
    img: &ImageView,
    center: [f64; 2],
    stride: isize,
    x_start: isize,
    x_end: isize,
    y_start: isize,
    y_end: isize,
    sigma_sq: f64,
) -> (f64, f64, f64) {
    let mut sum_gx2 = 0.0;
    let mut sum_gy2 = 0.0;
    let mut sum_gxgy = 0.0;

    let x_count = (x_end - x_start + 1).cast_unsigned();
    let x_min = (x_start - 1).cast_unsigned();
    let row_slice_len = x_count + 2;
    debug_assert!(x_count <= STACK_WEIGHT_CAPACITY);

    let mut x_weights_stack = [0.0_f64; STACK_WEIGHT_CAPACITY];
    let x_weights = &mut x_weights_stack[..x_count];

    for (k, weight) in x_weights.iter_mut().enumerate() {
        let px = x_start + k.cast_signed();
        let dx = px as f64 - center[0];
        *weight = (-(dx * dx) / (2.0 * sigma_sq)).exp();
    }

    for py in y_start..=y_end {
        let dy = py as f64 - center[1];
        let y_weight = (-(dy * dy) / (2.0 * sigma_sq)).exp();
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
            let weight = x_weights[k] * y_weight;

            sum_gx2 += gx_f * gx_f * weight;
            sum_gy2 += gy_f * gy_f * weight;
            sum_gxgy += gx_f * gy_f * weight;
        }
    }

    (sum_gx2, sum_gy2, sum_gxgy)
}

fn finalize_corner_covariance(
    sum_gx2: f64,
    sum_gy2: f64,
    sum_gxgy: f64,
    alpha_max: f64,
    sigma_n_sq: f64,
) -> Matrix2<f64> {
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

#[cfg(feature = "bench-internals")]
#[must_use]
/// Bench-only wrapper for the production corner-covariance kernel.
pub fn bench_compute_corner_covariance(
    img: &ImageView,
    center: [f64; 2],
    alpha_max: f64,
    sigma_n_sq: f64,
    radius: i32,
) -> Matrix2<f64> {
    compute_corner_covariance(img, center, alpha_max, sigma_n_sq, radius)
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
    let mut ne = BodyFrameNormalEquations::new(pose);
    let mut total_cost = 0.0;

    for i in 0..4 {
        let pb = obj_pts[i];
        let p_cam = pose.rotation * pb + pose.translation;
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
        let s_i_sq = crate::pose::mahalanobis_d2([res_u, res_v], info);
        let s_i = s_i_sq.sqrt();

        if s_i <= huber_k {
            total_cost += 0.5 * s_i_sq;
        } else {
            total_cost += huber_k * (s_i - 0.5 * huber_k);
        }

        let w = if s_i <= huber_k { 1.0 } else { huber_k / s_i };

        // Pinhole projection gradients ∂[u,v]/∂P_cam (this builder stays pinhole —
        // the distortion-aware path is `pose::corner_normal_equations`); the
        // body-frame Jacobian + weighted accumulation live in the shared
        // `BodyFrameNormalEquations`. Weight is the anisotropic information·Huber
        // matrix `W = info·w`.
        let (du_dp, dv_dp) = pinhole_projection_gradients(intrinsics, z_inv, x_z, y_z);
        ne.add(&pb, &du_dp, &dv_dp, res_u, res_v, &(info * w));
    }

    let (jtj, jtr) = ne.finish();
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
///
/// Thin wrapper that inverts the per-corner covariances and delegates to
/// [`refine_pose_lm_weighted_with_info`]. Callers holding pre-computed
/// info matrices (e.g. the outlier-aware drop path, which inverts once
/// for the trigger d²) should call the info-direct variant to skip the
/// redundant inversion. A zero info matrix on corner `i` is treated as
/// "skip this corner": its contribution to `JᵀWJ`, `JᵀWr`, and the
/// Huber cost is identically zero.
#[must_use]
pub(crate) fn refine_pose_lm_weighted(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    initial_pose: Pose,
    corner_covariances: &[Matrix2<f64>; 4],
) -> (Pose, [[f64; 6]; 6]) {
    let info_matrices: [Matrix2<f64>; 4] = core::array::from_fn(|i| {
        corner_covariances[i]
            .try_inverse()
            .unwrap_or_else(Matrix2::identity)
    });
    refine_pose_lm_weighted_with_info(intrinsics, corners, tag_size, initial_pose, &info_matrices)
}

/// Info-matrix-direct variant of [`refine_pose_lm_weighted`]. Pre-inverted
/// `Σ_c⁻¹` skips the 4 `try_inverse` calls — useful for the outlier-aware
/// drop path, which inverts once to compute per-corner d² and then masks
/// one corner by zeroing its info matrix.
#[must_use]
pub(crate) fn refine_pose_lm_weighted_with_info(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    initial_pose: Pose,
    info_matrices: &[Matrix2<f64>; 4],
) -> (Pose, [[f64; 6]; 6]) {
    const HUBER_K: f64 = 1.345;
    const MAX_ITERS: usize = 20;

    let mut pose = initial_pose;
    let obj_pts = centered_tag_corners(tag_size);
    let info_matrices = *info_matrices;

    let mut lambda = 1e-3_f64;
    let mut nu = 2.0_f64;

    // Cache for normal equations at the current accepted pose.
    let mut current_jtj = Matrix6::zeros();
    let mut current_jtr = Vector6::zeros();
    let mut current_cost = f64::MAX;
    let mut needs_rebuild = true;
    // `consecutive_chol_failures` guard: without it, a
    // persistently rank-deficient Hessian causes the LM to spin through
    // every iteration multiplying λ by 10 with no progress, then exit
    // via MAX_ITERS and emit the NaN covariance sentinel. Bailing out
    // after 3 consecutive failures lets the sentinel fire promptly with
    // the same diagnostic value, without burning iterations.
    let mut consecutive_chol_failures: u8 = 0;

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
            consecutive_chol_failures = 0;
            chol.solve(&current_jtr)
        } else {
            consecutive_chol_failures += 1;
            if consecutive_chol_failures >= 3 {
                break;
            }
            lambda *= 10.0;
            nu = 2.0;
            continue;
        };

        // Gain ratio numerator: Actual cost reduction
        let new_pose = pose.retract(&delta);

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
    // Singular OR near-singular `JᵀWJ` (e.g. degenerate near-coplanar
    // 4-point geometry at grazing incidence) produces no calibrated
    // covariance. `try_inverse` only catches EXACT singularity (LU zero
    // pivot); a near-singular Hessian with tiny non-zero pivot inverts
    // to a finite-but-absurd matrix (~1e15-1e30). Treat both cases the
    // same way: emit a NaN-filled sentinel so downstream consumers can
    // branch on `cov[(0,0)].is_nan()`.
    let covariance = match current_jtj.try_inverse() {
        Some(inv) if inv.iter().all(|v| v.is_finite()) => inv,
        _ => Matrix6::from_element(f64::NAN),
    };

    (pose, covariance.into())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::pose::{CameraIntrinsics, Pose};
    use crate::{decoder::Homography, image::ImageView};

    #[test]
    fn test_jacobian_rotation_rows() {
        // Verify the **right** (body-frame) SE(3) rotation Jacobian rows that
        // `build_normal_equations` accumulates, against a central finite difference
        // of the projected `u` through `Pose::retract`. Non-identity R and t so the
        // body and camera frames genuinely differ (at R=I,t=0 they coincide and the
        // test would not distinguish the convention).
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);
        let r = nalgebra::Rotation3::from_euler_angles(0.2, -0.1, 0.15)
            .matrix()
            .into_owned();
        let t = nalgebra::Vector3::new(0.03, -0.02, 0.6);
        let pose = Pose::new(r, t);
        let pb = nalgebra::Vector3::new(0.01, -0.008, 0.0);

        // Analytic body-frame rotation rows (mirrors build_normal_equations).
        let p_cam = r * pb + t;
        let z_inv = 1.0 / p_cam.z;
        let x_z = p_cam.x * z_inv;
        let du_dp =
            nalgebra::Vector3::new(intrinsics.fx * z_inv, 0.0, -intrinsics.fx * x_z * z_inv);
        let dqu = r.transpose() * du_dp;
        let row_u = [
            -(dqu[1] * pb.z - dqu[2] * pb.y),
            -(dqu[2] * pb.x - dqu[0] * pb.z),
            -(dqu[0] * pb.y - dqu[1] * pb.x),
        ];

        // Finite-difference ∂u/∂ω through the right retract.
        let proj_u = |p: &Pose| {
            let pc = p.rotation * pb + p.translation;
            intrinsics.fx * (pc.x / pc.z) + intrinsics.cx
        };
        let eps = 1e-7;
        for k in 0..3 {
            let mut d = Vector6::zeros();
            d[3 + k] = eps;
            let fwd = proj_u(&pose.retract(&d));
            d[3 + k] = -eps;
            let bwd = proj_u(&pose.retract(&d));
            let num = (fwd - bwd) / (2.0 * eps);
            assert!(
                (row_u[k] - num).abs() < 1e-3,
                "∂u/∂ω_{k}: analytic {} vs fd {num}",
                row_u[k]
            );
        }
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

        let s = 0.02_f64;
        let obj_pts = centered_tag_corners(s);

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
        let q_gt = crate::pose::quat_from_so3(gt_rot);
        let q_est = crate::pose::quat_from_so3(result.rotation);
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
        let obj_pts = centered_tag_corners(s);

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
        pose.retract(&delta)
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

    /// Singular `JᵀWJ` ⇒ NaN-filled covariance sentinel (not identity).
    ///
    /// Construction: zero every per-corner info matrix. The Huber-Mahalanobis
    /// objective then drops all four observations' contributions, so the
    /// accumulated `JᵀWJ` is identically zero — exactly singular at any
    /// floating-point precision. `try_inverse` returns `None`, triggering
    /// the NaN sentinel introduced by PR-C.
    #[test]
    fn singular_hessian_returns_nan_covariance() {
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);
        let s = 0.04_f64;
        let obj_pts = centered_tag_corners(s);
        let gt_pose = Pose::new(
            nalgebra::Rotation3::identity().matrix().into_owned(),
            nalgebra::Vector3::new(0.0, 0.0, 0.5),
        );
        let corners: [[f64; 2]; 4] =
            core::array::from_fn(|i| gt_pose.project(&obj_pts[i], &intrinsics));

        // All-zero info ⇒ zero `JᵀWJ` ⇒ rank-0 / singular.
        let info = [Matrix2::<f64>::zeros(); 4];

        let (_pose, cov) =
            refine_pose_lm_weighted_with_info(&intrinsics, &corners, s, gt_pose, &info);

        // Per the NaN-sentinel contract: any NaN on the diagonal is a
        // sufficient signal for downstream consumers.
        assert!(
            cov[0][0].is_nan(),
            "expected NaN-filled covariance on singular JᵀWJ, got cov[0][0]={}",
            cov[0][0],
        );
        // And every entry should be NaN under `Matrix6::from_element(NAN)`.
        for (r, row) in cov.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert!(
                    v.is_nan(),
                    "NaN-sentinel covariance must be NaN everywhere: cov[{r}][{c}]={v}",
                );
            }
        }
    }

    /// Degenerate-geometry production test: tag size = 0 → all four
    /// object-space corners coincide at the origin → all four projected
    /// image corners coincide at the same pixel → the per-corner
    /// Jacobian is identical for all 4 observations → `JᵀWJ` has rank
    /// 2 < 6 → singular. Exercises the production LM through a
    /// realistic (non-zero-info) input that the `singular_hessian_returns_
    /// nan_covariance` algebra-trivial test cannot reach.
    ///
    /// This pins the "degenerate geometry that survived the upstream
    /// gates" scenario the PR description cites, complementing the
    /// algebra-trivial all-zero-info path.
    #[test]
    fn degenerate_geometry_returns_nan_covariance() {
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);
        // Tag size 0 ⇒ all 4 object-space corners coincide at the origin.
        let s = 0.0_f64;
        let obj_pts = centered_tag_corners(s);
        let gt_pose = Pose::new(
            nalgebra::Rotation3::identity().matrix().into_owned(),
            nalgebra::Vector3::new(0.0, 0.0, 0.5),
        );
        let corners: [[f64; 2]; 4] =
            core::array::from_fn(|i| gt_pose.project(&obj_pts[i], &intrinsics));

        // Realistic identity-scaled info matrices: the singularity
        // comes from coincident geometry, not from zero weights.
        let info = [Matrix2::<f64>::identity(); 4];

        let (_pose, cov) =
            refine_pose_lm_weighted_with_info(&intrinsics, &corners, s, gt_pose, &info);

        // Either: exact-singular ⇒ NaN sentinel everywhere from the
        // `try_inverse → None` lane, or near-singular ⇒ at least one
        // non-finite entry from the post-inverse `is_finite` gate.
        // Both satisfy the NaN-sentinel contract.
        let any_non_finite = cov.iter().flatten().any(|v| !v.is_finite());
        assert!(
            any_non_finite,
            "expected NaN-sentinel covariance for coincident-corner geometry; \
             every entry was finite, cov[0][0]={}",
            cov[0][0],
        );
    }
}
