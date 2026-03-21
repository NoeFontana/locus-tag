#![allow(clippy::similar_names)]
use crate::image::ImageView;
use crate::pose::{CameraIntrinsics, Pose};
use nalgebra::{Matrix2, Matrix6, Vector2, Vector3, Vector6};

/// Compute the covariance of the corner position estimation error based on the Structure Tensor.
///
/// The covariance $\Sigma_c$ is approximated as the inverse of the Structure Tensor $S$:
/// $$ \Sigma_c \approx \sigma_n^2 S^{-1} $$
/// where $\sigma_n^2$ is the pixel noise variance (assumed around 2.0 for typical webcams).
///
/// The Structure Tensor is computed over a small window around the corner.
fn compute_corner_covariance(
    img: &ImageView,
    center: [f64; 2],
    alpha_max: f64,
    sigma_n_sq: f64,
    radius: i32,
) -> Matrix2<f64> {
    // Gain-scheduled Tikhonov regularization: Σ_reg = σ_n² M⁻¹ + α(R)·I
    //
    // Without regularization, a foreshortened tag drives λ_max(M) → ∞. The Mahalanobis
    // distance s_i = √(rᵢᵀWᵢrᵢ) then explodes even for small residuals, causing Huber to
    // zero-out that corner and sever the rotational constraint.
    //
    // Rather than a static α, we schedule it from the anisotropy ratio of M:
    //   R = λ_min / λ_max  ∈ [0, 1]   (0 = pure edge / grazing, 1 = isotropic / frontal)
    //   α(R) = α_max · (1 − R)²
    //
    // This keeps α ≈ 0 for well-conditioned frontal tags (preserving close-range precision)
    // and smoothly ramps to α_max for severely foreshortened tags (bounding observer gain).
    // The quadratic transfer function keeps regularization inactive until R < ~0.3, only
    // engaging at geometrically severe angles.
    //
    // α_max controls maximum information per corner; bounded at 1/α (px⁻²).
    let alpha_max_val = alpha_max;
    let cx = center[0].floor() as isize;
    let cy = center[1].floor() as isize;

    let mut sum_gx2 = 0.0;
    let mut sum_gy2 = 0.0;
    let mut sum_gxgy = 0.0;

    let w = img.width.cast_signed();
    let h = img.height.cast_signed();

    for dy in -radius as isize..=radius as isize {
        for dx in -radius as isize..=radius as isize {
            let px = cx + dx;
            let py = cy + dy;

            if px < 1 || px >= w - 1 || py < 1 || py >= h - 1 {
                continue;
            }

            let idx = (py * img.stride.cast_signed() + px).cast_unsigned();
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

    // sigma_n_sq is now passed as a parameter

    let det = s[(0, 0)] * s[(1, 1)] - s[(0, 1)] * s[(1, 0)];
    if det.abs() < 1e-6 {
        return Matrix2::identity().scale(100.0);
    }

    let inv_det = 1.0 / det;
    let s_inv = Matrix2::new(s[(1, 1)], -s[(0, 1)], -s[(1, 0)], s[(0, 0)]).scale(inv_det);

    // Anisotropy ratio R = λ_min / λ_max derived analytically from M's eigenvalues.
    // trace + discriminant > 0 always (trace = a+b ≥ 2·epsilon > 0).
    // (trace − discriminant) is clamped to 0 to absorb f64 rounding on degenerate inputs.
    let trace = s[(0, 0)] + s[(1, 1)];
    let diff = s[(0, 0)] - s[(1, 1)];
    let discriminant = (diff * diff + 4.0 * s[(0, 1)] * s[(0, 1)]).sqrt();
    let r = ((trace - discriminant) / (trace + discriminant)).max(0.0);
    let alpha = alpha_max_val * (1.0 - r) * (1.0 - r);

    s_inv.scale(sigma_n_sq) + Matrix2::identity().scale(alpha)
}

/// Compute framework uncertainty for all 4 corners.
///
/// This serves as a batch wrapper around `compute_corner_covariance`.
#[must_use]
pub(crate) fn compute_framework_uncertainty(
    img: &ImageView,
    corners: &[[f64; 2]; 4],
    _h_poly: &crate::homography::Homography,
    tikhonov_alpha_max: f64,
    sigma_n_sq: f64,
    structure_tensor_radius: u8,
) -> [Matrix2<f64>; 4] {
    let mut covariances = [Matrix2::zeros(); 4];
    for i in 0..4 {
        covariances[i] = compute_corner_covariance(
            img,
            corners[i],
            tikhonov_alpha_max,
            sigma_n_sq,
            i32::from(structure_tensor_radius),
        );
    }
    covariances
}

/// Accumulates the Huber-augmented normal equations J^T W̃ J and J^T W̃ r at the given pose.
///
/// For each corner i, the augmented information matrix is:
/// `W̃_i = w(s_i) · W_i`
/// where `s_i = √(rᵢᵀ Wᵢ rᵢ)` is the Mahalanobis distance and `w(s_i)` is the Huber
/// IRLS weight. This is the core computation shared between the LM inner loop and the
/// final post-convergence covariance extraction.
fn build_normal_equations(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
    info_matrices: &[Matrix2<f64>; 4],
    huber_k: f64,
) -> (Matrix6<f64>, Vector6<f64>) {
    let mut jtj = Matrix6::<f64>::zeros();
    let mut jtr = Vector6::<f64>::zeros();

    for i in 0..4 {
        let p_world = obj_pts[i];
        let p_cam = pose.rotation * p_world + pose.translation;
        let z_inv = 1.0 / p_cam.z;
        let z_inv2 = z_inv * z_inv;

        let u_est = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
        let v_est = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;
        let res = Vector2::new(corners[i][0] - u_est, corners[i][1] - v_est);

        // Mahalanobis distance: s_i = √(rᵢᵀ Wᵢ rᵢ).
        let s_i = res.dot(&(info_matrices[i] * res)).sqrt();
        // Huber IRLS weight: w(s) = 1 inside the inlier ball, k/s outside.
        let w = if s_i <= huber_k { 1.0 } else { huber_k / s_i };
        // Augmented information matrix: W̃_i = w(s_i) · Wᵢ.
        let w_tilde = info_matrices[i] * w;

        // Jacobian of projection wrt Camera Point (2x3):
        // [ fx/z   0     -fx·x/z² ]
        // [ 0      fy/z  -fy·y/z² ]
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

        // Jacobian of Camera Point wrt se(3) twist δξ = [δt | δω] (3x6):
        // d(exp(δω)·P + δt)/dδξ |_{δξ=0}  =  [ I | -[P]× ]
        let mut jac_i = nalgebra::SMatrix::<f64, 2, 6>::zeros();
        jac_i[(0, 0)] = du_dp[0];
        jac_i[(0, 1)] = du_dp[1];
        jac_i[(0, 2)] = du_dp[2];
        jac_i[(0, 3)] = du_dp[1] * p_cam.z - du_dp[2] * p_cam.y;
        jac_i[(0, 4)] = du_dp[2] * p_cam.x - du_dp[0] * p_cam.z;
        jac_i[(0, 5)] = du_dp[0] * p_cam.y - du_dp[1] * p_cam.x;
        jac_i[(1, 0)] = dv_dp[0];
        jac_i[(1, 1)] = dv_dp[1];
        jac_i[(1, 2)] = dv_dp[2];
        jac_i[(1, 3)] = dv_dp[1] * p_cam.z - dv_dp[2] * p_cam.y;
        jac_i[(1, 4)] = dv_dp[2] * p_cam.x - dv_dp[0] * p_cam.z;
        jac_i[(1, 5)] = dv_dp[0] * p_cam.y - dv_dp[1] * p_cam.x;

        let weighted_jac = jac_i.transpose() * w_tilde;
        jtj += weighted_jac * jac_i;
        jtr += weighted_jac * res;
    }

    (jtj, jtr)
}

/// Huber robust cost over Mahalanobis distances for all four corners.
///
/// `ρ(s) = ½s²` for `s ≤ k`, and `k(s − ½k)` for `s > k`.
/// Using the Mahalanobis distance as the argument gives an anisotropic robust loss
/// that respects the per-corner uncertainty from the Structure Tensor.
fn huber_mahalanobis_cost(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
    info_matrices: &[Matrix2<f64>; 4],
    k: f64,
) -> f64 {
    let mut cost = 0.0;
    for i in 0..4 {
        let p = pose.project(&obj_pts[i], intrinsics);
        let res = Vector2::new(corners[i][0] - p[0], corners[i][1] - p[1]);
        let s = res.dot(&(info_matrices[i] * res)).sqrt();
        if s <= k {
            cost += 0.5 * s * s;
        } else {
            cost += k * (s - 0.5 * k);
        }
    }
    cost
}

/// Refine the pose by minimizing a Huber-robust Mahalanobis objective.
///
/// This upgrades the previous pure-L2 weighted LM with three improvements that mirror the
/// Fast mode solver, adapted to the anisotropic information-matrix setting:
///
/// 1. **Huber-on-Mahalanobis M-Estimator (IRLS):** Computes the Mahalanobis distance
///    `s_i = √(rᵢᵀ Σᵢ⁻¹ rᵢ)` and applies weight `w(s_i) = min(1, k/s_i)` (`k = 1.345`,
///    covering 95% of Gaussian noise). The augmented matrix `W̃_i = w(s_i) · Σᵢ⁻¹` replaces
///    the pure information matrix, capping the gradient a single outlier corner can exert.
///
/// 2. **Marquardt Diagonal Scaling + Nielsen Trust-Region:** `D = diag(JᵀW̃J)` damps each
///    DOF by its own curvature. Nielsen's gain ratio `ρ = actual/predicted` controls lambda
///    with a smooth cubic schedule on accept and nu-doubling backoff on reject.
///
/// 3. **Correct Covariance Extraction:** Returns `(JᵀW̃J)⁻¹` at the converged pose — the
///    Cramér–Rao lower bound on pose uncertainty — instead of the previous ad-hoc MSE scaling.
#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn refine_pose_lm_weighted(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    initial_pose: Pose,
    corner_covariances: &[Matrix2<f64>; 4],
) -> (Pose, [[f64; 6]; 6]) {
    // Huber threshold in Mahalanobis units.
    // k = 1.345 gives 95% asymptotic efficiency under Gaussian noise (standard choice).
    const HUBER_K: f64 = 1.345;

    let mut pose = initial_pose;
    let s_tag = tag_size;
    // Modern OpenCV 4.6+ Convention: Origin at Top-Left, CW winding
    let obj_pts = [
        Vector3::new(0.0, 0.0, 0.0),     // 0: Top-Left
        Vector3::new(s_tag, 0.0, 0.0),   // 1: Top-Right
        Vector3::new(s_tag, s_tag, 0.0), // 2: Bottom-Right
        Vector3::new(0.0, s_tag, 0.0),   // 3: Bottom-Left
    ];

    // W_i = Σ_i^{-1}: precompute pure structure-tensor information matrices.
    let mut info_matrices = [Matrix2::<f64>::zeros(); 4];
    for i in 0..4 {
        info_matrices[i] = corner_covariances[i]
            .try_inverse()
            .unwrap_or(Matrix2::identity());
    }

    // Nielsen trust-region state.
    let mut lambda = 1e-3_f64;
    let mut nu = 2.0_f64;
    let mut current_cost = huber_mahalanobis_cost(
        intrinsics,
        corners,
        &obj_pts,
        &pose,
        &info_matrices,
        HUBER_K,
    );

    for _ in 0..20 {
        let (jtj, jtr) = build_normal_equations(
            intrinsics,
            corners,
            &obj_pts,
            &pose,
            &info_matrices,
            HUBER_K,
        );

        // Gate 1: gradient convergence — solver is at a stationary point.
        if jtr.amax() < 1e-8 {
            break;
        }

        // Marquardt diagonal scaling: D = diag(J^T W̃ J).
        let d_diag = Vector6::new(
            jtj[(0, 0)].max(1e-8),
            jtj[(1, 1)].max(1e-8),
            jtj[(2, 2)].max(1e-8),
            jtj[(3, 3)].max(1e-8),
            jtj[(4, 4)].max(1e-8),
            jtj[(5, 5)].max(1e-8),
        );

        // Solve (J^T W̃ J + λD) δξ = J^T W̃ r
        let mut jtj_damped = jtj;
        for k in 0..6 {
            jtj_damped[(k, k)] += lambda * d_diag[k];
        }

        let delta = if let Some(chol) = jtj_damped.cholesky() {
            chol.solve(&jtr)
        } else {
            // Ill-conditioned; increase damping and retry.
            lambda *= 10.0;
            nu = 2.0;
            continue;
        };

        // Nielsen gain ratio: ρ = actual / predicted cost reduction.
        // Predicted: L(0) − L(δ) = ½ δᵀ(λD·δ + J^T W̃ r)
        let predicted_reduction = 0.5 * delta.dot(&(lambda * d_diag.component_mul(&delta) + jtr));

        // SE(3) exponential-map update.
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
        let actual_reduction = current_cost - new_cost;

        let rho = if predicted_reduction > 1e-12 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        if rho > 0.0 {
            // Accept: shrink lambda toward Gauss-Newton regime.
            pose = new_pose;
            current_cost = new_cost;
            lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            nu = 2.0;
            // Gate 2: step size convergence.
            if delta.norm() < 1e-7 {
                break;
            }
        } else {
            // Reject: grow lambda with doubling backoff.
            lambda *= nu;
            nu *= 2.0;
        }
    }

    // Covariance: Σ_pose = (J^T W̃ J)^{-1} at the converged pose.
    // This is the Cramér–Rao lower bound on pose uncertainty given the Huber-weighted
    // information matrices. No ad-hoc MSE scaling — W_i already encodes the noise model.
    let (jtj_final, _) = build_normal_equations(
        intrinsics,
        corners,
        &obj_pts,
        &pose,
        &info_matrices,
        HUBER_K,
    );
    let covariance = jtj_final.try_inverse().unwrap_or(Matrix6::identity());

    (pose, covariance.into())
}
