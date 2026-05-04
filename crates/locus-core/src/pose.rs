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

// ---------------------------------------------------------------------------
// Distortion model storage (embedded in CameraIntrinsics)
// ---------------------------------------------------------------------------

/// Lens distortion coefficients stored alongside the intrinsic parameters.
///
/// Variants correspond to the two supported distortion models plus the ideal
/// pinhole (no distortion) case.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DistortionCoeffs {
    /// No distortion — ideal pinhole / pre-rectified image.
    None,
    /// Brown-Conrady polynomial radial + tangential distortion (OpenCV convention).
    ///
    /// Coefficient order: `k1, k2, p1, p2, k3`.
    #[cfg(feature = "non_rectified")]
    BrownConrady {
        /// Radial coefficient k1.
        k1: f64,
        /// Radial coefficient k2.
        k2: f64,
        /// Tangential coefficient p1.
        p1: f64,
        /// Tangential coefficient p2.
        p2: f64,
        /// Radial coefficient k3.
        k3: f64,
    },
    /// Kannala-Brandt equidistant fisheye model.
    ///
    /// Coefficient order: `k1, k2, k3, k4`.
    #[cfg(feature = "non_rectified")]
    KannalaBrandt {
        /// Fisheye coefficient k1.
        k1: f64,
        /// Fisheye coefficient k2.
        k2: f64,
        /// Fisheye coefficient k3.
        k3: f64,
        /// Fisheye coefficient k4.
        k4: f64,
    },
}

impl DistortionCoeffs {
    /// Returns `true` for any non-pinhole distortion model.
    #[must_use]
    #[inline]
    pub const fn is_distorted(&self) -> bool {
        !matches!(self, Self::None)
    }
}

// ---------------------------------------------------------------------------
// CameraIntrinsics
// ---------------------------------------------------------------------------

/// Camera intrinsics parameters with optional lens distortion.
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
    /// Lens distortion model and coefficients. Defaults to [`DistortionCoeffs::None`].
    pub distortion: DistortionCoeffs,
}

impl CameraIntrinsics {
    /// Create new intrinsics with no distortion (ideal pinhole).
    ///
    /// Existing call sites remain unchanged.
    #[must_use]
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            distortion: DistortionCoeffs::None,
        }
    }

    /// Create new intrinsics with Brown-Conrady distortion.
    #[cfg(feature = "non_rectified")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn with_brown_conrady(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            distortion: DistortionCoeffs::BrownConrady { k1, k2, p1, p2, k3 },
        }
    }

    /// Create new intrinsics with Kannala-Brandt fisheye distortion.
    #[cfg(feature = "non_rectified")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn with_kannala_brandt(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            distortion: DistortionCoeffs::KannalaBrandt { k1, k2, k3, k4 },
        }
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

    /// Map a pixel coordinate `(px, py)` in the distorted image to an ideal
    /// (undistorted) pixel coordinate.
    ///
    /// For [`DistortionCoeffs::None`] this is an identity operation.
    #[must_use]
    pub fn undistort_pixel(&self, px: f64, py: f64) -> [f64; 2] {
        match self.distortion {
            DistortionCoeffs::None => [px, py],
            #[cfg(feature = "non_rectified")]
            DistortionCoeffs::BrownConrady { k1, k2, p1, p2, k3 } => {
                let m = crate::camera::BrownConradyModel { k1, k2, p1, p2, k3 };
                let xn = (px - self.cx) / self.fx;
                let yn = (py - self.cy) / self.fy;
                let [xu, yu] = crate::camera::CameraModel::undistort(&m, xn, yn);
                [xu * self.fx + self.cx, yu * self.fy + self.cy]
            },
            #[cfg(feature = "non_rectified")]
            DistortionCoeffs::KannalaBrandt { k1, k2, k3, k4 } => {
                let m = crate::camera::KannalaBrandtModel { k1, k2, k3, k4 };
                let xn = (px - self.cx) / self.fx;
                let yn = (py - self.cy) / self.fy;
                let [xu, yu] = crate::camera::CameraModel::undistort(&m, xn, yn);
                [xu * self.fx + self.cx, yu * self.fy + self.cy]
            },
        }
    }

    /// Apply distortion to a normalized ideal point `(xn, yn)` and return
    /// the distorted pixel coordinates `(px, py)`.
    ///
    /// For [`DistortionCoeffs::None`] this projects without distortion.
    #[must_use]
    pub fn distort_normalized(&self, xn: f64, yn: f64) -> [f64; 2] {
        let [xd, yd] = match self.distortion {
            DistortionCoeffs::None => [xn, yn],
            #[cfg(feature = "non_rectified")]
            DistortionCoeffs::BrownConrady { k1, k2, p1, p2, k3 } => {
                let m = crate::camera::BrownConradyModel { k1, k2, p1, p2, k3 };
                crate::camera::CameraModel::distort(&m, xn, yn)
            },
            #[cfg(feature = "non_rectified")]
            DistortionCoeffs::KannalaBrandt { k1, k2, k3, k4 } => {
                let m = crate::camera::KannalaBrandtModel { k1, k2, k3, k4 };
                crate::camera::CameraModel::distort(&m, xn, yn)
            },
        };
        [xd * self.fx + self.cx, yd * self.fy + self.cy]
    }

    /// Compute the 2×2 Jacobian of the distortion map at normalized point `(xn, yn)`.
    ///
    /// Returns `[[∂xd/∂xn, ∂xd/∂yn], [∂yd/∂xn, ∂yd/∂yn]]`.
    #[must_use]
    #[cfg_attr(not(feature = "non_rectified"), allow(unused_variables))]
    pub(crate) fn distortion_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2] {
        match self.distortion {
            DistortionCoeffs::None => [[1.0, 0.0], [0.0, 1.0]],
            #[cfg(feature = "non_rectified")]
            DistortionCoeffs::BrownConrady { k1, k2, p1, p2, k3 } => {
                let m = crate::camera::BrownConradyModel { k1, k2, p1, p2, k3 };
                crate::camera::CameraModel::distort_jacobian(&m, xn, yn)
            },
            #[cfg(feature = "non_rectified")]
            DistortionCoeffs::KannalaBrandt { k1, k2, k3, k4 } => {
                let m = crate::camera::KannalaBrandtModel { k1, k2, k3, k4 };
                crate::camera::CameraModel::distort_jacobian(&m, xn, yn)
            },
        }
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
    let thresholds = ConsistencyThresholds::from_fpr(
        config.pose_consistency_fpr,
        config.pose_consistency_min_decisive_ratio,
    );
    let (pose, cov, _diag) = estimate_tag_pose_with_diagnostics(
        intrinsics,
        corners,
        tag_size,
        img,
        mode,
        config,
        external_covariances,
        thresholds,
    );
    (pose, cov)
}

/// Diagnostics emitted by `estimate_tag_pose_with_diagnostics` alongside
/// the refined pose. NaN / `u8::MAX` sentinels mark signals that were not
/// computed (IPPE failure, gate-disabled in non-`bench-internals` builds).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "bench-internals"), allow(dead_code))]
pub(crate) struct PoseDiagnostics {
    pub aggregate_d2: f32,
    pub max_corner_d2: f32,
    pub branch_d2_ratio: f32,
    /// Index (0 or 1) of the IPPE candidate the solver actually refined and
    /// returned. Sentinel `u8::MAX` ⇒ branch selection did not run (IPPE
    /// failure or gate disabled). Read by the bench-internals harness.
    #[cfg_attr(feature = "bench-internals", allow(dead_code))]
    pub branch_chosen: u8,
}

impl PoseDiagnostics {
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self {
            aggregate_d2: f32::NAN,
            max_corner_d2: f32::NAN,
            branch_d2_ratio: f32::NAN,
            branch_chosen: u8::MAX,
        }
    }
}

/// Precomputed χ² critical thresholds for one frame's worth of consistency
/// checks. `None` ⇒ gate disabled (zero overhead path).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConsistencyThresholds {
    /// `χ²(2; fpr)` — aggregate gate (8 obs − 6 fitted = 2 DOF).
    pub aggregate: f64,
    /// `χ²(1; fpr)` — per-corner gate.
    pub per_corner: f64,
    /// Branch-ratio escape clause. When the IPPE branch selector
    /// produced `alternate_d2 / primary_d2 ≥ min_decisive_ratio`, the
    /// chosen branch is decisive and the χ² gate is bypassed even if
    /// the post-LM aggregate / per-corner d² exceeds the threshold.
    ///
    /// Rationale: the gate's job is to catch IPPE branch ambiguity. A
    /// genuine ambiguity has both candidates at similar d² (ratio ~ 1).
    /// A scene-specific noise outlier (PSF artefact, lighting gradient)
    /// has a clear branch winner but high absolute residual — and
    /// nulling its pose is lossy. See PR for the rotation-tail analysis.
    pub min_decisive_ratio: f64,
}

impl ConsistencyThresholds {
    /// Build from a false-positive rate and branch-ratio escape clause.
    /// Returns `None` for `fpr ∉ (0, 1)`, which the rest of the gate
    /// interprets as "disabled".
    #[must_use]
    pub(crate) fn from_fpr(fpr: f64, min_decisive_ratio: f64) -> Option<Self> {
        if fpr > 0.0 && fpr < 1.0 {
            Some(Self {
                aggregate: chi2_critical(fpr, 2),
                per_corner: chi2_critical(fpr, 1),
                min_decisive_ratio,
            })
        } else {
            None
        }
    }
}

/// Variant of [`estimate_tag_pose_with_config`] that additionally returns
/// pose-consistency diagnostics for telemetry. `thresholds = None` means
/// the gate is disabled — the consistency check short-circuits to "accept"
/// and the disabled-path diagnostic d² compute is elided entirely on
/// non-`bench-internals` builds (zero-overhead legacy path).
#[must_use]
#[allow(clippy::missing_panics_doc, clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "pipeline::estimate_tag_pose_diag")]
pub(crate) fn estimate_tag_pose_with_diagnostics(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    img: Option<&ImageView>,
    mode: PoseEstimationMode,
    config: &crate::config::DetectorConfig,
    external_covariances: Option<&[Matrix2<f64>; 4]>,
    thresholds: Option<ConsistencyThresholds>,
) -> (Option<Pose>, Option<[[f64; 6]; 6]>, PoseDiagnostics) {
    // For distorted cameras, IPPE runs in ideal space; LM residuals run in
    // observed (distorted) space. Keep both forms.
    let ideal_corners: [[f64; 2]; 4] = if intrinsics.distortion == DistortionCoeffs::None {
        *corners
    } else {
        core::array::from_fn(|i| intrinsics.undistort_pixel(corners[i][0], corners[i][1]))
    };

    let Some(h_poly) = crate::decoder::Homography::square_to_quad(&ideal_corners) else {
        return (None, None, PoseDiagnostics::empty());
    };
    let h_pixel = h_poly.h;
    let h_norm = intrinsics.inv_matrix() * h_pixel;
    let scaler = 2.0 / tag_size;
    let mut h_metric = h_norm;
    h_metric.column_mut(0).scale_mut(scaler);
    h_metric.column_mut(1).scale_mut(scaler);

    let Some(candidates) = solve_ippe_square(&h_metric) else {
        return (None, None, PoseDiagnostics::empty());
    };

    // LM weighting (Some when Accurate-mode + GWLF/structure-tensor input).
    let covariances = build_lm_covariances(
        config,
        mode,
        img,
        &ideal_corners,
        &h_poly,
        external_covariances,
    );

    // Isotropic Σ⁻¹ = (1/σ²)·I for the χ² gate and the branch selector's
    // fallback path — see `DetectorConfig::pose_consistency_gate_sigma_px`
    // for the σ-decoupling rationale.
    let gate_info_matrices = isotropic_info_matrices(config.pose_consistency_gate_sigma_px);
    let selector_info = pick_selector_info(covariances.as_ref(), &gate_info_matrices);

    let (best_pose, branch_diag) = if let Some(t) = thresholds {
        select_ippe_branch(
            intrinsics,
            corners,
            &selector_info,
            tag_size,
            &candidates,
            t,
        )
    } else {
        let pose = find_best_pose(intrinsics, &ideal_corners, tag_size, &candidates);
        let branch_diag =
            disabled_branch_diagnostics(intrinsics, corners, &selector_info, tag_size, &candidates);
        (pose, branch_diag)
    };

    let (refined_pose, covariance) =
        if let (PoseEstimationMode::Accurate, _, Some(covs)) = (mode, img, covariances) {
            let (p, c) = crate::pose_weighted::refine_pose_lm_weighted(
                intrinsics,
                corners,
                tag_size,
                best_pose,
                &covs,
                config.corner_d2_gate_threshold,
            );
            (p, Some(c))
        } else {
            let p = refine_pose_lm(
                intrinsics,
                corners,
                tag_size,
                best_pose,
                config.huber_delta_px,
            );
            (p, None)
        };

    let verdict = pose_consistency_check(
        intrinsics,
        corners,
        &gate_info_matrices,
        tag_size,
        &refined_pose,
        branch_diag.ratio(),
        thresholds,
    );

    let diag = PoseDiagnostics {
        aggregate_d2: verdict.aggregate_d2 as f32,
        max_corner_d2: verdict.max_corner_d2 as f32,
        branch_d2_ratio: branch_diag.ratio() as f32,
        branch_chosen: branch_diag.chosen_idx,
    };

    if verdict.accepted {
        (Some(refined_pose), covariance, diag)
    } else {
        (None, None, diag)
    }
}

/// Per-corner covariances for the weighted LM solver.
///
/// Priority: GWLF external (Accurate + ext) > Structure Tensor (Accurate +
/// img) > `None` (Fast mode or no image — LM falls back to unweighted Huber).
/// Returns covariances directly; the χ² gate and IPPE branch selector use
/// their own isotropic info matrices ([`isotropic_info_matrices`]) so we no
/// longer materialize an inverted info-matrix array on the hot path.
#[inline]
fn build_lm_covariances(
    config: &crate::config::DetectorConfig,
    mode: PoseEstimationMode,
    img: Option<&ImageView>,
    ideal_corners: &[[f64; 2]; 4],
    h_poly: &crate::decoder::Homography,
    external_covariances: Option<&[Matrix2<f64>; 4]>,
) -> Option<[Matrix2<f64>; 4]> {
    if mode != PoseEstimationMode::Accurate {
        return None;
    }
    if let Some(ext) = external_covariances {
        return Some(*ext);
    }
    let image = img?;
    Some(crate::pose_weighted::compute_framework_uncertainty(
        image,
        ideal_corners,
        h_poly,
        config.tikhonov_alpha_max,
        config.sigma_n_sq,
        config.structure_tensor_radius,
    ))
}

/// Build a per-corner isotropic info matrix `Σ⁻¹ = (1/σ²)·I`. Used by the
/// χ² consistency gate and the IPPE branch selector, which need a noise
/// model that's independent of the LM's per-corner weighting so the
/// gate's χ²(2) calibration stays meaningful.
#[inline]
fn isotropic_info_matrices(sigma_px: f64) -> [Matrix2<f64>; 4] {
    let inv_var = 1.0 / (sigma_px * sigma_px).max(1e-6);
    [Matrix2::new(inv_var, 0.0, 0.0, inv_var); 4]
}

/// Maximum acceptable per-corner condition number `λ_max / λ_min` for the
/// Mahalanobis-primary branch selector. Above this, the info matrix is
/// pathologically anisotropic and the Mahalanobis ranking becomes unreliable
/// (residuals along strong edges get vanishing penalty), so the selector
/// falls back to its isotropic Euclidean cousin.
///
/// `100.0` accepts moderately-anisotropic structure-tensor info matrices
/// (typical for tag corners with one dominant edge) while rejecting the
/// pathological κ > 1e3 cases that EdLines GN-residual covariances produce.
const SELECTOR_MAX_CONDITION_NUMBER: f64 = 100.0;

/// True iff every per-corner info matrix is positive-definite and the
/// largest condition number `λ_max / λ_min` stays below
/// [`SELECTOR_MAX_CONDITION_NUMBER`]. The `lmin > 1e-12` check covers
/// degenerate / non-PD cases (`det <= 0` ⇒ `λ_min <= 0`).
#[inline]
fn info_matrices_well_conditioned(info: &[Matrix2<f64>; 4]) -> bool {
    for m in info {
        let trace = m[(0, 0)] + m[(1, 1)];
        let det = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
        let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let lmax = 0.5 * (trace + disc);
        let lmin = 0.5 * (trace - disc);
        if lmin <= 1e-12 || lmax / lmin > SELECTOR_MAX_CONDITION_NUMBER {
            return false;
        }
    }
    true
}

/// Choose the info matrices the IPPE branch selector should use to rank
/// candidates: LM info (Mahalanobis-primary) when covariances are available
/// *and* invert to a well-conditioned info matrix; isotropic σ_gate
/// fallback otherwise.
///
/// LM info gives the selector a calibrated weighting — branches whose
/// residuals lie in low-information directions are correctly preferred.
/// Anisotropic-info pitfalls (residuals "hidden" along degenerate edges)
/// are guarded by the condition-number check.
#[inline]
fn pick_selector_info(
    covariances: Option<&[Matrix2<f64>; 4]>,
    fallback: &[Matrix2<f64>; 4],
) -> [Matrix2<f64>; 4] {
    let Some(covs) = covariances else {
        return *fallback;
    };
    let info: [Matrix2<f64>; 4] =
        core::array::from_fn(|i| covs[i].try_inverse().unwrap_or_else(Matrix2::identity));
    if info_matrices_well_conditioned(&info) {
        info
    } else {
        *fallback
    }
}

/// Diagnostic d² compute for the disabled-gate branch. Only contributes to
/// the `bench-internals` telemetry SoA columns; on production builds it is
/// a no-op that returns sentinel NaNs (avoids 8 distortion projections per
/// tag on the legacy code path).
#[inline]
fn disabled_branch_diagnostics(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    info_matrices: &[Matrix2<f64>; 4],
    tag_size: f64,
    candidates: &[Pose; 2],
) -> BranchDiagnostics {
    #[cfg(feature = "bench-internals")]
    {
        let obj_pts = centered_tag_corners(tag_size);
        let (_, agg0) = per_corner_d2(intrinsics, corners, &obj_pts, info_matrices, &candidates[0]);
        let (_, agg1) = per_corner_d2(intrinsics, corners, &obj_pts, info_matrices, &candidates[1]);
        // The disabled-gate path uses lowest-reprojection-error selection
        // (`find_best_pose`), which on this code branch is the same call site
        // that picks `candidates[chosen_idx]`. Mirror that here so the
        // diagnostic `chosen_idx` reflects what actually shipped.
        let chosen_idx = u8::from(
            reprojection_error(intrinsics, corners, &obj_pts, &candidates[1])
                < reprojection_error(intrinsics, corners, &obj_pts, &candidates[0]),
        );
        BranchDiagnostics {
            primary_d2: agg0.min(agg1),
            alternate_d2: agg0.max(agg1),
            chosen_idx,
        }
    }
    #[cfg(not(feature = "bench-internals"))]
    {
        let _ = (intrinsics, corners, info_matrices, tag_size, candidates);
        BranchDiagnostics::empty()
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

/// Project a 3D camera-space point through the intrinsics (with distortion if present).
///
/// Returns `[u, v]` in pixel coordinates.
#[inline]
fn project_with_distortion(p_cam: &Vector3<f64>, intrinsics: &CameraIntrinsics) -> [f64; 2] {
    let z = p_cam.z.max(1e-4);
    let xn = p_cam.x / z;
    let yn = p_cam.y / z;
    intrinsics.distort_normalized(xn, yn)
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
///
/// 4. **Distortion-Aware Jacobian:** When `intrinsics` carries a non-trivial distortion model,
///    the projection `P_cam → pixel` routes through the distortion map, and the analytic
///    chain-rule Jacobian `∂[u,v]/∂P_cam = diag(fx,fy) · J_dist · J_normalize` is used,
///    ensuring the solver converges to the correct distortion-compensated pose.
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
            let z = p_cam.z.max(1e-4);
            let z_inv = 1.0 / z;
            let xn = p_cam.x / z;
            let yn = p_cam.y / z;

            // Apply distortion and compute pixel residual.
            // For DistortionCoeffs::None, distort_normalized is an identity pass-through.
            let [u_est, v_est] = intrinsics.distort_normalized(xn, yn);

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

            // Distortion Jacobian J_dist = [[∂xd/∂xn, ∂xd/∂yn], [∂yd/∂xn, ∂yd/∂yn]].
            // For DistortionCoeffs::None this is the 2×2 identity, which recovers the
            // original pinhole Jacobian exactly.
            let jd = intrinsics.distortion_jacobian(xn, yn);

            // Full Jacobian of pixel coordinates wrt camera-space point P_cam:
            //
            //   ∂u/∂P_cam = fx · [J_dist[0][0]/z, J_dist[0][1]/z, -(J_dist[0][0]·xn + J_dist[0][1]·yn)/z]
            //   ∂v/∂P_cam = fy · [J_dist[1][0]/z, J_dist[1][1]/z, -(J_dist[1][0]·xn + J_dist[1][1]·yn)/z]
            //
            // Derivation: u = fx·xd + cx, xd = distort_x(xn, yn), xn = Px/Pz, yn = Py/Pz
            //   ∂u/∂Px = fx·(∂xd/∂xn·1/z) = fx·jd[0][0]/z
            //   ∂u/∂Py = fx·(∂xd/∂yn·1/z) = fx·jd[0][1]/z
            //   ∂u/∂Pz = fx·(∂xd/∂xn·(-xn/z) + ∂xd/∂yn·(-yn/z))
            let du_dp = Vector3::new(
                intrinsics.fx * jd[0][0] * z_inv,
                intrinsics.fx * jd[0][1] * z_inv,
                -intrinsics.fx * (jd[0][0] * xn + jd[0][1] * yn) * z_inv,
            );
            let dv_dp = Vector3::new(
                intrinsics.fy * jd[1][0] * z_inv,
                intrinsics.fy * jd[1][1] * z_inv,
                -intrinsics.fy * (jd[1][0] * xn + jd[1][1] * yn) * z_inv,
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
/// Uses distortion-aware projection when `intrinsics` carries a distortion model.
fn huber_cost(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
    delta: f64,
) -> f64 {
    let mut cost = 0.0;
    for i in 0..4 {
        let p_cam = pose.rotation * obj_pts[i] + pose.translation;
        let p = project_with_distortion(&p_cam, intrinsics);
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
        let p_cam = pose.rotation * obj_pts[i] + pose.translation;
        let p = project_with_distortion(&p_cam, intrinsics);
        err_sq += (p[0] - corners[i][0]).powi(2) + (p[1] - corners[i][1]).powi(2);
    }
    err_sq
}

/// Squared Mahalanobis distance for a 2D residual under a 2×2 information
/// matrix `Σ⁻¹`. Lifted from the inner LM accumulator so the consistency
/// gate stays numerically identical to what the solver weights against.
#[inline]
pub(crate) fn mahalanobis_d2(residual: [f64; 2], info: &Matrix2<f64>) -> f64 {
    let r_u = residual[0];
    let r_v = residual[1];
    r_u * (info[(0, 0)] * r_u + info[(0, 1)] * r_v)
        + r_v * (info[(1, 0)] * r_u + info[(1, 1)] * r_v)
}

/// Per-corner and aggregate squared Mahalanobis distances for a candidate pose.
///
/// Returns `(per_corner, aggregate)` where each per-corner entry is `d²_i =
/// rᵢᵀ Σᵢ⁻¹ rᵢ` and `aggregate = Σ d²_i`. Residuals are taken in observed
/// (distorted) image space using `project_with_distortion` so the gate
/// matches the same noise model the LM solver fits.
#[inline]
pub(crate) fn per_corner_d2(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    info_matrices: &[Matrix2<f64>; 4],
    pose: &Pose,
) -> ([f64; 4], f64) {
    let mut per_corner = [0.0_f64; 4];
    let mut aggregate = 0.0_f64;
    for i in 0..4 {
        let p_cam = pose.rotation * obj_pts[i] + pose.translation;
        let p = project_with_distortion(&p_cam, intrinsics);
        let res = [corners[i][0] - p[0], corners[i][1] - p[1]];
        let d2 = mahalanobis_d2(res, &info_matrices[i]);
        per_corner[i] = d2;
        aggregate += d2;
    }
    (per_corner, aggregate)
}

/// Chi-squared upper-tail critical value `χ²_dof(1 - fpr)`. Only `dof ∈
/// {1, 2}` is supported (the only values used by the consistency gate);
/// `ConsistencyThresholds::from_fpr` already screens `fpr ∉ (0, 1)` so
/// this assumes a valid input.
#[inline]
#[must_use]
pub(crate) fn chi2_critical(fpr: f64, dof: u32) -> f64 {
    debug_assert!(
        0.0 < fpr && fpr < 1.0,
        "chi2_critical: fpr must lie in (0, 1)"
    );
    match dof {
        // χ²(1) ≡ Z²; Z = erfinv(1 - p) · √2.
        1 => {
            let z = std::f64::consts::SQRT_2 * erfinv(1.0 - fpr);
            z * z
        },
        // χ²(2) survival function is exp(-x/2), so the inverse is -2 ln(fpr).
        2 => -2.0 * fpr.ln(),
        _ => unreachable!("chi2_critical only supports dof ∈ {{1, 2}}"),
    }
}

/// Approximate inverse of the error function on `[-1, 1]`. Uses the
/// Winitzki rational expansion (max abs error ≈ 1.3e-3) which is easily
/// good enough to set χ² critical values — the gate's *threshold* is
/// itself only meaningful to within one decade of FPR.
#[inline]
fn erfinv(x: f64) -> f64 {
    let a = 0.147_f64;
    let ln1mx2 = (1.0 - x * x).ln();
    let term = 2.0 / (std::f64::consts::PI * a) + 0.5 * ln1mx2;
    let inner = (term * term - ln1mx2 / a).sqrt() - term;
    x.signum() * inner.sqrt()
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

/// Diagnostics from the Mahalanobis-aware IPPE branch selector. The
/// `primary` branch is the lower-d² candidate; `alternate` is the other.
/// Ratio `alternate / primary` < 1 means the primary branch was wrong.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BranchDiagnostics {
    pub primary_d2: f64,
    pub alternate_d2: f64,
    /// Index (0 or 1) of the IPPE candidate that was *actually* refined and
    /// returned to the caller — accounts for the alternate-branch swap rule.
    /// Sentinel `u8::MAX` ⇒ branch selection did not run.
    pub chosen_idx: u8,
}

impl BranchDiagnostics {
    /// Sentinel for paths that did not compute branch d² (IPPE failure or
    /// non-`bench-internals` builds with the gate disabled).
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "bench-internals", allow(dead_code))]
    pub(crate) fn empty() -> Self {
        Self {
            primary_d2: f64::NAN,
            alternate_d2: f64::NAN,
            chosen_idx: u8::MAX,
        }
    }

    /// `alternate_d2 / primary_d2`. Returns `1.0` when the primary is
    /// degenerately small (avoids amplifying noise) and `NaN` when no d²
    /// was computed.
    #[inline]
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.primary_d2 <= 1e-12 {
            return 1.0;
        }
        self.alternate_d2 / self.primary_d2
    }
}

/// IPPE branch selector with a Mahalanobis-aware safety net.
///
/// Ranking is by Mahalanobis aggregate d² under `info_matrices`. The caller
/// chooses what to pass: well-conditioned LM info matrices (anisotropic
/// Mahalanobis ranking — recovers branches whose residuals lie along
/// low-information directions) or the isotropic σ_gate fallback (which
/// reduces Mahalanobis to scaled Euclidean, matching the legacy
/// `find_best_pose` behavior). [`pick_selector_info`] makes that choice.
///
/// The flip rule still requires the alternate branch to be ≥ 2× better
/// AND below the χ²(2; fpr) threshold — both conditions guard against
/// flipping toward an alternate that is merely *less bad* than a
/// degenerate primary; that case is handled downstream by the consistency
/// gate rejecting both.
fn select_ippe_branch(
    intrinsics: &CameraIntrinsics,
    observed_corners: &[[f64; 2]; 4],
    info_matrices: &[Matrix2<f64>; 4],
    tag_size: f64,
    candidates: &[Pose; 2],
    thresholds: ConsistencyThresholds,
) -> (Pose, BranchDiagnostics) {
    let obj_pts = centered_tag_corners(tag_size);

    let (_, agg0) = per_corner_d2(
        intrinsics,
        observed_corners,
        &obj_pts,
        info_matrices,
        &candidates[0],
    );
    let (_, agg1) = per_corner_d2(
        intrinsics,
        observed_corners,
        &obj_pts,
        info_matrices,
        &candidates[1],
    );

    let (primary_idx, primary_d2, alternate_d2) = if agg0 <= agg1 {
        (0_usize, agg0, agg1)
    } else {
        (1_usize, agg1, agg0)
    };

    let chosen = if alternate_d2 < primary_d2 * 0.5 && alternate_d2 < thresholds.aggregate {
        1 - primary_idx
    } else {
        primary_idx
    };

    (
        candidates[chosen],
        BranchDiagnostics {
            primary_d2,
            alternate_d2,
            chosen_idx: chosen as u8,
        },
    )
}

/// Verdict produced by the pose-consistency gate.
///
/// `accepted` is the only field consulted on the hot-path; the d² values
/// are surfaced for the bench-internals telemetry columns and the ROC
/// harness. `aggregate_d2` is the χ²(2) statistic across all four corners;
/// `max_corner_d2` is the worst single-corner χ²(1) statistic.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConsistencyVerdict {
    pub accepted: bool,
    pub aggregate_d2: f64,
    pub max_corner_d2: f64,
}

/// Final reprojection-consistency check. Two-prong χ² gate with a
/// branch-ratio escape clause:
///
/// - **Aggregate** (DOF = 8 obs − 6 fitted = 2): rejects poses whose total
///   reprojection error is inconsistent with the noise model.
/// - **Per-corner** (DOF = 1): catches single-corner contamination that
///   the aggregate would average over.
/// - **Branch-ratio escape** (`alternate_d2 / primary_d2 ≥
///   min_decisive_ratio`): bypasses the χ² test when the IPPE branch
///   selector had overwhelming evidence. The gate's purpose is catching
///   IPPE branch ambiguity (both candidates similar d²); when one branch
///   is decisively better, a high post-LM residual is more likely scene-
///   specific noise than a wrong branch, and nulling the pose is lossy.
///
/// `branch_d2_ratio` is `BranchDiagnostics::ratio()` from the IPPE branch
/// selector. Pass `f64::NAN` or any value `< thresholds.min_decisive_ratio`
/// when no branch context is available (e.g. the ROC test harness) to keep
/// only the χ² test active.
///
/// When `thresholds = None` the gate is disabled — accepts unconditionally
/// and skips the d² compute on production builds (NaN sentinels), keeping
/// telemetry-only d² behind `bench-internals`.
fn pose_consistency_check(
    intrinsics: &CameraIntrinsics,
    observed_corners: &[[f64; 2]; 4],
    info_matrices: &[Matrix2<f64>; 4],
    tag_size: f64,
    refined_pose: &Pose,
    branch_d2_ratio: f64,
    thresholds: Option<ConsistencyThresholds>,
) -> ConsistencyVerdict {
    let Some(t) = thresholds else {
        // Disabled gate: skip 4 distortion projections in production
        // builds; keep them under bench-internals for telemetry.
        #[cfg(feature = "bench-internals")]
        {
            let obj_pts = centered_tag_corners(tag_size);
            let (per_corner, aggregate_d2) = per_corner_d2(
                intrinsics,
                observed_corners,
                &obj_pts,
                info_matrices,
                refined_pose,
            );
            return ConsistencyVerdict {
                accepted: true,
                aggregate_d2,
                max_corner_d2: per_corner.iter().copied().fold(0.0_f64, f64::max),
            };
        }
        #[cfg(not(feature = "bench-internals"))]
        {
            let _ = (
                intrinsics,
                observed_corners,
                info_matrices,
                tag_size,
                refined_pose,
            );
            return ConsistencyVerdict {
                accepted: true,
                aggregate_d2: f64::NAN,
                max_corner_d2: f64::NAN,
            };
        }
    };

    let obj_pts = centered_tag_corners(tag_size);
    let (per_corner, aggregate_d2) = per_corner_d2(
        intrinsics,
        observed_corners,
        &obj_pts,
        info_matrices,
        refined_pose,
    );
    let max_corner_d2 = per_corner.iter().copied().fold(0.0_f64, f64::max);
    let chi2_ok = aggregate_d2 <= t.aggregate && max_corner_d2 <= t.per_corner;
    // Branch-ratio escape: a finite ratio above the threshold bypasses the
    // χ² test. `is_finite()` is load-bearing: the ROC harness disables the
    // escape via `min_decisive_ratio = +∞`, and `+∞ >= +∞` is true — so we
    // need to filter +∞ explicitly. NaN propagates as `false` for free.
    let branch_decisive = branch_d2_ratio.is_finite() && branch_d2_ratio >= t.min_decisive_ratio;
    let accepted = chi2_ok || branch_decisive;

    ConsistencyVerdict {
        accepted,
        aggregate_d2,
        max_corner_d2,
    }
}

/// Bench/test hook around the private `pose_consistency_check`.
///
/// Returns `(accepted, aggregate_d2, max_corner_d2)`. Lets the ROC harness
/// exercise the gate with controlled noise without going through the full
/// pipeline, which is required for the realized-FPR acceptance assertion in
/// `regression_pose_consistency_roc`.
///
/// Disables the branch-ratio escape clause (`min_decisive_ratio = +∞`) so
/// the realized-FPR is purely a function of the χ²(2)/χ²(1) test — the
/// branch-ratio escape is meaningful only in the IPPE-driven pipeline and
/// would confound the ROC characterization the harness performs.
#[cfg(feature = "bench-internals")]
#[must_use]
pub fn bench_pose_consistency_d2(
    intrinsics: &CameraIntrinsics,
    observed_corners: &[[f64; 2]; 4],
    info_matrices: &[Matrix2<f64>; 4],
    tag_size: f64,
    refined_pose: &Pose,
    fpr: f64,
) -> (bool, f64, f64) {
    let v = pose_consistency_check(
        intrinsics,
        observed_corners,
        info_matrices,
        tag_size,
        refined_pose,
        f64::NAN,
        ConsistencyThresholds::from_fpr(fpr, f64::INFINITY),
    );
    (v.accepted, v.aggregate_d2, v.max_corner_d2)
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

    // Hoist χ² thresholds out of the per-tag inner loop — they depend only
    // on `pose_consistency_fpr` and `pose_consistency_min_decisive_ratio`
    // and are constant for the frame.
    let thresholds = ConsistencyThresholds::from_fpr(
        config.pose_consistency_fpr,
        config.pose_consistency_min_decisive_ratio,
    );

    // Process valid candidates in parallel.
    // We collect into a temporary Vec to avoid unsafe parallel writes to the batch.
    let results: Vec<_> = (0..v)
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

            // Only GWLF writes covariances calibrated as image-noise
            // variances; other extractors (EdLines, Erf) leave per-corner
            // GN-residual uncertainties in `batch.corner_covariances` that
            // are far tighter than σ_n and would mis-feed the LM solver.
            let ext_covs_opt: Option<[Matrix2<f64>; 4]> =
                if config.refinement_mode == crate::config::CornerRefinementMode::Gwlf {
                    let covs: [Matrix2<f64>; 4] = core::array::from_fn(|j| {
                        Matrix2::new(
                            f64::from(batch.corner_covariances[i][j * 4]),
                            f64::from(batch.corner_covariances[i][j * 4 + 1]),
                            f64::from(batch.corner_covariances[i][j * 4 + 2]),
                            f64::from(batch.corner_covariances[i][j * 4 + 3]),
                        )
                    });
                    covs.iter()
                        .any(|c| c.norm_squared() > 1e-12)
                        .then_some(covs)
                } else {
                    None
                };

            let (pose_opt, _, diag) = estimate_tag_pose_with_diagnostics(
                intrinsics,
                &corners,
                tag_size,
                img,
                mode,
                config,
                ext_covs_opt.as_ref(),
                thresholds,
            );

            let pose_data = if let Some(pose) = pose_opt {
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
            };

            (pose_data, diag)
        })
        .collect();

    for (i, (pose_data, diag)) in results.iter().enumerate() {
        if let Some(data) = pose_data {
            batch.poses[i] = Pose6D {
                data: *data,
                padding: 0.0,
            };
        } else {
            batch.poses[i] = Pose6D {
                data: [0.0; 7],
                padding: 0.0,
            };
        }
        #[cfg(feature = "bench-internals")]
        {
            batch.pose_consistency_d2[i] = diag.aggregate_d2;
            batch.pose_consistency_d2_max_corner[i] = diag.max_corner_d2;
            batch.ippe_branch_d2_ratio[i] = diag.branch_d2_ratio;
        }
        #[cfg(not(feature = "bench-internals"))]
        let _ = diag;
    }
}

// ---------------------------------------------------------------------------
// Phase 0 rotation-tail diagnostic harness: bench-internals-gated entry points.
//
// THROWAWAY: the dual-branch LM run, leave-one-out refit, and the
// `BenchPoseDiagnostics` surface here are exploratory instrumentation. Once
// Phase 0 lands its memo, revert the additions in this section and the
// matching block in `pose_weighted.rs`. The permanent foundation is the
// `bench-internals` feature plumbing in `locus-py` and the
// `compute_image_noise_floor` helper in `gradient.rs`.
// ---------------------------------------------------------------------------

/// Per-branch result of a dual-branch IPPE refinement run, used by the Phase 0
/// rotation-tail diagnostic to ask "would the *other* branch have given a
/// better rotation under the same observed corners?"
#[cfg(feature = "bench-internals")]
#[derive(Debug, Clone, Copy)]
pub struct BenchBranchResult {
    /// IPPE candidate index (0 or 1) before LM refinement.
    pub branch_idx: u8,
    /// IPPE pose used as the LM seed.
    pub initial_pose: Pose,
    /// Pose after LM convergence.
    pub refined_pose: Pose,
    /// `Σ d²_i` Mahalanobis aggregate at the refined pose.
    pub aggregate_d2: f64,
    /// Per-corner Mahalanobis d² at the refined pose.
    pub per_corner_d2: [f64; 4],
}

/// Bench-only: surface-level diagnostics returned alongside both branches.
/// `chosen_idx` matches what `estimate_tag_pose_with_diagnostics` would have
/// returned, so harness scripts can compare "what we shipped" vs "what the
/// alternate branch would have given."
#[cfg(feature = "bench-internals")]
#[derive(Debug, Clone, Copy)]
pub struct BenchPoseDiagnostics {
    /// `Σ d²_i` across all four corners (χ²(2) statistic).
    pub aggregate_d2: f32,
    /// Worst single-corner Mahalanobis d² (χ²(1) statistic).
    pub max_corner_d2: f32,
    /// `alternate_d2 / primary_d2` from the IPPE branch selector.
    pub branch_d2_ratio: f32,
    /// Index of the IPPE candidate that was actually refined (0 or 1).
    pub branch_chosen: u8,
}

#[cfg(feature = "bench-internals")]
impl From<PoseDiagnostics> for BenchPoseDiagnostics {
    fn from(d: PoseDiagnostics) -> Self {
        Self {
            aggregate_d2: d.aggregate_d2,
            max_corner_d2: d.max_corner_d2,
            branch_d2_ratio: d.branch_d2_ratio,
            branch_chosen: d.branch_chosen,
        }
    }
}

/// Bench-only: solve IPPE for a quad and run LM on **both** candidate branches,
/// returning each branch's refined pose plus its aggregate / per-corner
/// Mahalanobis d² under the supplied info matrices. This is the load-bearing
/// measurement for the `branch_flip` failure-mode classifier — it answers
/// directly whether the alternate branch was rotationally better than the one
/// the production solver picked.
///
/// THROWAWAY: revert with the rest of this section after Phase 0 ships.
#[cfg(feature = "bench-internals")]
#[must_use]
#[allow(clippy::missing_panics_doc, clippy::too_many_arguments)]
pub fn bench_estimate_both_branches(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    img: Option<&ImageView>,
    mode: crate::config::PoseEstimationMode,
    config: &crate::config::DetectorConfig,
    external_covariances: Option<&[Matrix2<f64>; 4]>,
) -> Option<[BenchBranchResult; 2]> {
    let ideal_corners: [[f64; 2]; 4] = if intrinsics.distortion == DistortionCoeffs::None {
        *corners
    } else {
        core::array::from_fn(|i| intrinsics.undistort_pixel(corners[i][0], corners[i][1]))
    };

    let h_poly = crate::decoder::Homography::square_to_quad(&ideal_corners)?;
    let h_pixel = h_poly.h;
    let h_norm = intrinsics.inv_matrix() * h_pixel;
    let scaler = 2.0 / tag_size;
    let mut h_metric = h_norm;
    h_metric.column_mut(0).scale_mut(scaler);
    h_metric.column_mut(1).scale_mut(scaler);

    let candidates = solve_ippe_square(&h_metric)?;
    let covariances = build_lm_covariances(
        config,
        mode,
        img,
        &ideal_corners,
        &h_poly,
        external_covariances,
    );
    let gate_info_matrices = isotropic_info_matrices(config.pose_consistency_gate_sigma_px);
    let info_matrices = pick_selector_info(covariances.as_ref(), &gate_info_matrices);

    let obj_pts = centered_tag_corners(tag_size);

    let refine = |seed: Pose| -> Pose {
        if let (crate::config::PoseEstimationMode::Accurate, _, Some(covs)) =
            (mode, img, covariances)
        {
            crate::pose_weighted::refine_pose_lm_weighted(
                intrinsics,
                corners,
                tag_size,
                seed,
                &covs,
                config.corner_d2_gate_threshold,
            )
            .0
        } else {
            refine_pose_lm(intrinsics, corners, tag_size, seed, config.huber_delta_px)
        }
    };

    let results: [BenchBranchResult; 2] = core::array::from_fn(|i| {
        let initial = candidates[i];
        let refined = refine(initial);
        let (per_corner, aggregate) =
            per_corner_d2(intrinsics, corners, &obj_pts, &info_matrices, &refined);
        BenchBranchResult {
            branch_idx: i as u8,
            initial_pose: initial,
            refined_pose: refined,
            aggregate_d2: aggregate,
            per_corner_d2: per_corner,
        }
    });

    Some(results)
}

/// Bench-only: leave-one-out pose refinement. Drops corner `drop_idx` and
/// re-runs LM with only 3 corners (under-determined but still
/// well-conditioned for a planar tag). Used by the `corner_outlier`
/// classifier to test whether removing a single high-IRLS-weight corner
/// substantially improves rotation.
///
/// Implementation note: with 3 corners the homography fit is exact (no
/// residual to minimize), so we feed all 4 corners to LM but inflate the
/// info-matrix entry of `drop_idx` to ~zero, effectively removing its
/// contribution to the normal equations.
///
/// THROWAWAY: revert with the rest of this section after Phase 0 ships.
#[cfg(feature = "bench-internals")]
#[must_use]
#[allow(clippy::missing_panics_doc, clippy::too_many_arguments)]
pub fn bench_refit_pose_drop_corner(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    initial_pose: Pose,
    drop_idx: usize,
    img: Option<&ImageView>,
    mode: crate::config::PoseEstimationMode,
    config: &crate::config::DetectorConfig,
) -> Pose {
    debug_assert!(drop_idx < 4);

    let ideal_corners: [[f64; 2]; 4] = if intrinsics.distortion == DistortionCoeffs::None {
        *corners
    } else {
        core::array::from_fn(|i| intrinsics.undistort_pixel(corners[i][0], corners[i][1]))
    };

    let Some(h_poly) = crate::decoder::Homography::square_to_quad(&ideal_corners) else {
        return initial_pose;
    };
    let covariances_opt = build_lm_covariances(config, mode, img, &ideal_corners, &h_poly, None);

    if let (crate::config::PoseEstimationMode::Accurate, Some(mut covs)) = (mode, covariances_opt) {
        // Inflate the dropped corner's covariance ⇒ its info matrix shrinks
        // toward zero ⇒ no contribution to LM normal equations.
        covs[drop_idx] = Matrix2::identity().scale(1.0e12);
        crate::pose_weighted::refine_pose_lm_weighted(
            intrinsics,
            corners,
            tag_size,
            initial_pose,
            &covs,
            config.corner_d2_gate_threshold,
        )
        .0
    } else {
        // Fast mode has no per-corner weighting; we approximate "drop corner"
        // by snapping the dropped pixel residual to the model prediction
        // before running LM, so its contribution to the Huber cost is zero.
        let p_cam = initial_pose.rotation * centered_tag_corners(tag_size)[drop_idx]
            + initial_pose.translation;
        let projected = project_with_distortion(&p_cam, intrinsics);
        let mut local_corners = *corners;
        local_corners[drop_idx] = projected;
        refine_pose_lm(
            intrinsics,
            &local_corners,
            tag_size,
            initial_pose,
            config.huber_delta_px,
        )
    }
}

/// Bench-only: run `estimate_tag_pose_with_diagnostics` and surface its
/// `PoseDiagnostics` (incl. `branch_chosen`) as a stable public type, so the
/// Python harness can pull diagnostics out of the same code path the
/// production detector uses.
///
/// THROWAWAY: revert after Phase 0; the bench harness won't outlive it.
#[cfg(feature = "bench-internals")]
#[must_use]
#[allow(clippy::missing_panics_doc, clippy::too_many_arguments)]
pub fn bench_estimate_tag_pose(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    img: Option<&ImageView>,
    mode: crate::config::PoseEstimationMode,
    config: &crate::config::DetectorConfig,
    external_covariances: Option<&[Matrix2<f64>; 4]>,
    fpr: Option<f64>,
) -> (Option<Pose>, Option<[[f64; 6]; 6]>, BenchPoseDiagnostics) {
    let thresholds = fpr.and_then(|f| {
        ConsistencyThresholds::from_fpr(f, config.pose_consistency_min_decisive_ratio)
    });
    let (pose, cov, diag) = estimate_tag_pose_with_diagnostics(
        intrinsics,
        corners,
        tag_size,
        img,
        mode,
        config,
        external_covariances,
        thresholds,
    );
    (pose, cov, BenchPoseDiagnostics::from(diag))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
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
