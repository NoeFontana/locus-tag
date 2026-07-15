//! 3D Pose Estimation (PnP) for fiducial markers.
//!
//! IPPE-Square seed → Levenberg-Marquardt refinement. The LM cost surface
//! (weighted Mahalanobis or unweighted Huber) is selected automatically by
//! per-corner covariance availability; see `estimate_tag_pose_with_diagnostics`.

#![allow(clippy::many_single_char_names, clippy::similar_names)]
use crate::batch::{DetectionBatch, Point2f, Pose6D};
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
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor mirrors the OpenCV Brown-Conrady intrinsic + distortion coefficient list; a struct would just be unpacked at each call site"
    )]
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
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor mirrors the OpenCV Kannala-Brandt intrinsic + fisheye coefficient list; a struct would just be unpacked at each call site"
    )]
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

    /// Apply an SE(3) **right** (body-frame) perturbation `pose · exp(δ)`, with
    /// `δ = [t | ω]` (translation then rotation-vector, matching the LM Jacobian
    /// column layout). To first order (`V(ω) ≈ I`): `R' = R · exp(ω)`,
    /// `t' = R · δt + t` — the increment is expressed in the tag's *own* frame and
    /// rotated into the camera by the current `R`.
    ///
    /// This is the single manifold update shared by **all** pose LMs (the corner
    /// unweighted/weighted solvers and `board.rs`). The body frame is used
    /// deliberately: because the object points are symmetric about the body origin,
    /// it decouples rotation from translation in the normal equations (`JᵀWJ`
    /// condition number ~3.4× better, rot↔trans coupling ~0.999→0.06), yielding
    /// faster convergence and a body-frame pose covariance — see
    /// `docs/explanation/coordinates.md` and the `right_perturbation` study.
    ///
    /// `R · exp(ω)` is a product of two exactly-orthonormal matrices, so the
    /// result stays on SO(3) to ~1e-16 per step; over the ≤20 LM iterations the
    /// accumulated drift (~1e-13) is far below the pose tolerances and needs no
    /// re-orthonormalization (the corner LMs have always used this raw-product
    /// update). Downstream quaternion extraction (`quat_from_so3`) tolerates it.
    #[inline]
    #[must_use]
    pub(crate) fn retract(&self, delta: &Vector6<f64>) -> Self {
        let exp_omega = Rotation3::new(Vector3::new(delta[3], delta[4], delta[5]))
            .matrix()
            .into_owned();
        let dt_body = Vector3::new(delta[0], delta[1], delta[2]);
        Self::new(
            self.rotation * exp_omega,
            self.rotation * dt_body + self.translation,
        )
    }

    /// SE(3) inverse: maps camera-frame points back into the tag body frame.
    /// `R' = Rᵀ`, `t' = -Rᵀ t`.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let rt = self.rotation.transpose();
        Self::new(rt, -(rt * self.translation))
    }

    /// The 6×6 SE(3) adjoint `Ad_T` for the twist ordering `[t | ω]`, i.e. the map
    /// taking a **body**-frame (right) tangent vector to the equivalent
    /// **camera**-frame (left) one: `δ_camera = Ad_T · δ_body` (from
    /// `exp(δ_camera)·T = T·exp(δ_body)`). Layout:
    /// `Ad_T = [[R, [t]_× R], [0, R]]`.
    ///
    /// Used to reframe a covariance between the camera and body tangent spaces:
    /// `Σ_camera = Ad_T · Σ_body · Ad_Tᵀ`, and inversely
    /// `Σ_body = Ad_{T⁻¹} · Σ_camera · Ad_{T⁻¹}ᵀ`.
    #[must_use]
    pub fn adjoint(&self) -> Matrix6<f64> {
        let r = self.rotation;
        let tx = self.translation.cross_matrix();
        let mut ad = Matrix6::zeros();
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&(tx * r));
        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        ad
    }

    /// Reframe a 6×6 pose covariance from the **camera** (left/spatial) tangent to
    /// the **body** (right) tangent at this pose:
    /// `Σ_body = Ad_{T⁻¹} · Σ_camera · Ad_{T⁻¹}ᵀ`.
    ///
    /// Since the LMs were migrated to right perturbation the emitted covariance is
    /// already body-frame; this helper exists for consumers that hold a legacy
    /// camera-frame covariance (or want to convert one they transformed elsewhere).
    /// NaN sentinels propagate unchanged.
    #[must_use]
    pub fn covariance_camera_to_body(&self, cov_camera: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
        let ad_inv = self.inverse().adjoint();
        let cov = Matrix6::from_fn(|i, j| cov_camera[i][j]);
        let out = ad_inv * cov * ad_inv.transpose();
        core::array::from_fn(|i| core::array::from_fn(|j| out[(i, j)]))
    }

    /// Inverse of [`covariance_camera_to_body`]: body → camera tangent,
    /// `Σ_camera = Ad_T · Σ_body · Ad_Tᵀ`.
    #[must_use]
    pub fn covariance_body_to_camera(&self, cov_body: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
        let ad = self.adjoint();
        let cov = Matrix6::from_fn(|i, j| cov_body[i][j]);
        let out = ad * cov * ad.transpose();
        core::array::from_fn(|i| core::array::from_fn(|j| out[(i, j)]))
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

/// One row of the SE(3) **right** (body-frame) perturbation Jacobian for a pixel
/// coordinate, given the body-frame projection gradient `dq = Rᵀ·∂proj/∂P_cam`
/// and the body point `pb`. Layout ξ = [t | ω]: translation columns are `dq`, the
/// rotation columns are the cross-product `dq·(−[pb]_×)`. This is the single
/// definition of the sign-sensitive rotation-row math, verified by
/// `corner_normal_equations_gradient_is_descent` and `test_jacobian_rotation_rows`.
#[inline]
pub(crate) fn body_frame_row(dq: &Vector3<f64>, pb: &Vector3<f64>) -> Vector6<f64> {
    Vector6::new(
        dq[0],
        dq[1],
        dq[2],
        -(dq[1] * pb.z - dq[2] * pb.y),
        -(dq[2] * pb.x - dq[0] * pb.z),
        -(dq[0] * pb.y - dq[1] * pb.x),
    )
}

/// Pinhole projection gradients `∂[u,v]/∂P_cam` at a camera-frame point (given
/// `z_inv = 1/z`, `x_z = x/z`, `y_z = y/z`). Shared by the board LMs and the
/// weighted corner LM, which are pinhole-only; the single-tag
/// [`corner_normal_equations`] uses the distortion-aware gradient instead.
#[inline]
#[must_use]
pub(crate) fn pinhole_projection_gradients(
    intrinsics: &CameraIntrinsics,
    z_inv: f64,
    x_z: f64,
    y_z: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(intrinsics.fx * z_inv, 0.0, -intrinsics.fx * x_z * z_inv),
        Vector3::new(0.0, intrinsics.fy * z_inv, -intrinsics.fy * y_z * z_inv),
    )
}

/// Shared body-frame (right-perturbation) SE(3) normal-equations accumulator used
/// by **every** pose LM (corner unweighted/weighted and the three `board.rs`
/// solvers). Construct once from the linearization pose (captures `Rᵀ`), `add`
/// each correspondence with its already-computed projection gradient — so the
/// distortion model is the caller's choice and this stays frame-canonical without
/// pinning pinhole — then `finish` for the symmetric `(JᵀWJ, JᵀWr)`.
///
/// Only the upper triangle of `JᵀWJ` is accumulated (exploiting its symmetry);
/// `finish` mirrors it. This retires the four hand-transcribed copies of the
/// Jacobian-row + normal-equations math the right-perturbation migration had
/// spread across the solvers.
pub(crate) struct BodyFrameNormalEquations {
    jtj: Matrix6<f64>,
    jtr: Vector6<f64>,
    rt: Matrix3<f64>,
}

impl BodyFrameNormalEquations {
    #[inline]
    pub(crate) fn new(pose: &Pose) -> Self {
        Self {
            jtj: Matrix6::zeros(),
            jtr: Vector6::zeros(),
            rt: pose.rotation.transpose(),
        }
    }

    /// Accumulate one correspondence. `pb` = body (object) point; `du_dp`/`dv_dp` =
    /// `∂[u,v]/∂P_cam` (pinhole or distortion-aware, computed by the caller);
    /// `[res_u, res_v]` = pixel residual; `w` = 2×2 information·robust weight
    /// (`Matrix2::identity()` for the unweighted GN steps).
    #[inline]
    pub(crate) fn add(
        &mut self,
        pb: &Vector3<f64>,
        du_dp: &Vector3<f64>,
        dv_dp: &Vector3<f64>,
        res_u: f64,
        res_v: f64,
        w: &Matrix2<f64>,
    ) {
        let row_u = body_frame_row(&(self.rt * du_dp), pb);
        let row_v = body_frame_row(&(self.rt * dv_dp), pb);
        let (w00, w01, w10, w11) = (w[(0, 0)], w[(0, 1)], w[(1, 0)], w[(1, 1)]);
        // JᵀWr = row_u·(W r)_u + row_v·(W r)_v
        self.jtr += row_u * (w00 * res_u + w01 * res_v) + row_v * (w10 * res_u + w11 * res_v);
        // JᵀWJ upper triangle: wᵢⱼ folded outer products, symmetrized in `finish`.
        for i in 0..6 {
            for j in i..6 {
                self.jtj[(i, j)] += w00 * row_u[i] * row_u[j]
                    + w11 * row_v[i] * row_v[j]
                    + w01 * row_u[i] * row_v[j]
                    + w10 * row_v[i] * row_u[j];
            }
        }
    }

    /// Symmetrize and return `(JᵀWJ, JᵀWr)`.
    #[inline]
    pub(crate) fn finish(mut self) -> (Matrix6<f64>, Vector6<f64>) {
        symmetrize_jtj6(&mut self.jtj);
        (self.jtj, self.jtr)
    }
}

/// Convert a near-orthogonal rotation matrix to a unit quaternion without the
/// Müller iterative algorithm (which can hang on degenerate input).
///
/// `UnitQuaternion::from_matrix` delegates to `Rotation3::from_matrix_eps` with
/// `max_iter = 0`, which nalgebra treats as `usize::MAX` ("loop until
/// convergence"). For degenerate inputs (e.g. near-singular IPPE outputs at
/// extreme tag angles), the Müller iterative algorithm never converges —
/// infinite loop.
///
/// This helper uses the closed-form Shepperd method instead: wrap the matrix
/// as a `Rotation3` (no orthogonalization) then extract the quaternion
/// analytically. All current callers feed SO(3)-by-construction matrices (LM
/// exp-map output, ground-truth `UnitQuaternion::to_rotation_matrix`), so the
/// hang is unreachable today — but this helper makes the safety property
/// explicit and immune to future refactors.
#[inline]
#[must_use]
pub fn quat_from_so3(r: Matrix3<f64>) -> UnitQuaternion<f64> {
    let rot = Rotation3::from_matrix_unchecked(r);
    UnitQuaternion::from_rotation_matrix(&rot)
}

/// Estimate a tag's 6-DoF pose from its four detected image corners.
///
/// Seeds with IPPE-Square (homography decomposition), then refines with
/// Levenberg-Marquardt. `tag_size` is the physical tag edge length in world
/// units.
///
/// The weighting branch is selected by data availability: when `img` is
/// supplied the solver can weight residuals by per-corner covariance
/// (Mahalanobis LM); otherwise it falls back to unweighted Huber LM unless
/// external covariances are provided. The returned covariance matrix is
/// populated only when the weighted branch actually ran.
///
/// # Panics
/// Panics if SVD orthogonalization fails (numerically near-impossible).
#[must_use]
#[tracing::instrument(skip_all, name = "pipeline::estimate_tag_pose")]
pub fn estimate_tag_pose(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    img: Option<&ImageView>,
) -> (Option<Pose>, Option<[[f64; 6]; 6]>) {
    estimate_tag_pose_with_config(
        intrinsics,
        corners,
        tag_size,
        img,
        &crate::config::DetectorConfig::default(),
        None,
    )
}

/// Estimate pose with explicit configuration for tuning parameters.
#[must_use]
#[tracing::instrument(skip_all, name = "pipeline::estimate_tag_pose")]
pub fn estimate_tag_pose_with_config(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    img: Option<&ImageView>,
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
    /// Index (0..=3) of the corner the outlier-aware LM masked and dropped
    /// in the 3-corner re-solve. Sentinel `u8::MAX` ⇒ no drop fired (trigger
    /// gate, dominance check, or aggregate-on-4 self-rejection kept the
    /// 4-corner pose). When `outlier_corner_idx != u8::MAX`, the stored
    /// pose covariance reflects 6 observations instead of 8.
    pub outlier_corner_idx: u8,
}

impl PoseDiagnostics {
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self {
            aggregate_d2: f32::NAN,
            max_corner_d2: f32::NAN,
            branch_d2_ratio: f32::NAN,
            branch_chosen: u8::MAX,
            outlier_corner_idx: u8::MAX,
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
#[tracing::instrument(skip_all, name = "pipeline::estimate_tag_pose_diag")]
pub(crate) fn estimate_tag_pose_with_diagnostics(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    img: Option<&ImageView>,
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

    // LM weighting (Some when GWLF/structure-tensor input is available).
    let covariances =
        build_lm_covariances(config, img, &ideal_corners, &h_poly, external_covariances);

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

    let (mut refined_pose, mut covariance) = if let Some(covs) = covariances {
        let (p, c) = crate::pose_weighted::refine_pose_lm_weighted(
            intrinsics, corners, tag_size, best_pose, &covs,
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

    // Outlier-aware corner drop: catastrophic per-corner outliers that
    // survive Huber attenuation bias the rotation tail. Gated on (a) the
    // weighted-LM path (we need Σ_c to compute per-corner d² and mask
    // one corner) and (b) the per-profile threshold opt-in.
    //
    // Note on the `Matrix2::identity` fallback below: the per-corner Σ_c
    // kernel (`finalize_corner_covariance` in `pose_weighted.rs`) is
    // Tikhonov-regularised with `+1.0` on both diagonals plus a flat-
    // window identity fallback, so `try_inverse() → None` is unreachable
    // on the production kernel. The 4-corner LM is intrinsically tight
    // (cannot drop one corner without losing pose rank), so a singular
    // Σ_c cannot be rejected by skipping the tag — we depend on the
    // Tikhonov invariant. The `debug_assert!` makes that contract
    // explicit; in release builds, the identity fallback is a defensive
    // last resort and is dead code under the invariant.
    let outlier_corner_idx = match (covariances.as_ref(), config.outlier_drop_d2_threshold) {
        (Some(covs), threshold) if threshold > 0.0 => {
            let info_matrices: [Matrix2<f64>; 4] = core::array::from_fn(|i| {
                debug_assert!(
                    covs[i].try_inverse().is_some(),
                    "Tikhonov invariant violated: per-corner Σ_c[{i}] is singular; \
                     `finalize_corner_covariance` is supposed to guarantee PD output",
                );
                covs[i].try_inverse().unwrap_or_else(Matrix2::identity)
            });
            match maybe_drop_outlier_corner(
                intrinsics,
                corners,
                tag_size,
                &info_matrices,
                refined_pose,
                threshold,
            ) {
                Some((idx, pose_3, cov_3)) => {
                    refined_pose = pose_3;
                    covariance = Some(cov_3);
                    idx
                },
                None => u8::MAX,
            }
        },
        _ => u8::MAX,
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
        outlier_corner_idx,
    };

    if verdict.accepted {
        (Some(refined_pose), covariance, diag)
    } else {
        (None, None, diag)
    }
}

/// Worst corner must dominate the second-worst by at least this factor
/// before the outlier-aware LM masks it. Suppresses arbitrary single-drop
/// picks when two corners are similarly noisy (the typical pattern when
/// the *quad* — not one corner — is the failure mode).
const OUTLIER_DROP_DOMINANCE_RATIO: f64 = 2.0;

/// Try to identify and drop a single catastrophically bad corner, re-running
/// the weighted LM and keeping the 3-corner pose iff its aggregate d² over
/// the **three kept corners** is strictly lower than the 4-corner pose's
/// aggregate d² over those same three corners.
///
/// `Some((idx, pose_3, cov_3))` ⇒ corner `idx` was dropped and the
/// returned pose / covariance reflect 6 observations.
/// `None` ⇒ no drop (trigger not exceeded, dominance failed, or self-
/// rejection kept the 4-corner pose).
///
/// The self-rejection metric — d² aggregate over the kept corners — is
/// what makes the mechanism safe. It asks "did dropping this corner
/// improve the fit on the others?" rather than "did dropping reduce raw
/// reprojection on all 4". The latter is provably violated under isotropic
/// Σ_c (the unmasked LM IS the 4-corner agg minimum by definition), but
/// the former is well-posed: if the 3-corner LM moved to a meaningfully
/// different pose, the kept corners' Huber-weighted residuals should
/// shrink — they were already small at pose_4 unless the LM was biased by
/// the outlier.
///
/// Masking is done by zeroing the dropped corner's info matrix, not by
/// inflating its covariance to a large finite value. Both are
/// mathematically equivalent on the LM cost, but zero info matches the
/// "skip this corner" semantics literally — and lets the LM share the
/// already-computed info matrices instead of inverting again.
fn maybe_drop_outlier_corner(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    tag_size: f64,
    info_matrices: &[Matrix2<f64>; 4],
    refined_pose: Pose,
    threshold: f64,
) -> Option<(u8, Pose, [[f64; 6]; 6])> {
    let obj_pts = centered_tag_corners(tag_size);
    let (per_corner_4, _) =
        per_corner_d2(intrinsics, corners, &obj_pts, info_matrices, &refined_pose);

    let (i_worst, d2_worst) =
        per_corner_4
            .iter()
            .enumerate()
            .fold((0_usize, 0.0_f64), |(idx, best), (i, &d2)| {
                if d2 > best { (i, d2) } else { (idx, best) }
            });
    if d2_worst <= threshold {
        return None;
    }

    let d2_second = per_corner_4
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != i_worst)
        .map(|(_, &d)| d)
        .fold(0.0_f64, f64::max);
    if d2_worst < OUTLIER_DROP_DOMINANCE_RATIO * d2_second {
        return None;
    }

    let mut masked_info = *info_matrices;
    masked_info[i_worst] = Matrix2::zeros();
    let (pose_3, cov_3) = crate::pose_weighted::refine_pose_lm_weighted_with_info(
        intrinsics,
        corners,
        tag_size,
        refined_pose,
        &masked_info,
    );

    // Self-rejection over the 3 kept corners (using the original,
    // un-masked info matrices so the comparison metric matches the
    // trigger's).
    let (per_corner_3, _) = per_corner_d2(intrinsics, corners, &obj_pts, info_matrices, &pose_3);
    let kept_sum = |per_corner: &[f64; 4]| -> f64 {
        per_corner
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (i != i_worst).then_some(d))
            .sum()
    };
    if kept_sum(&per_corner_3) < kept_sum(&per_corner_4) {
        Some((i_worst as u8, pose_3, cov_3))
    } else {
        None
    }
}

/// Per-corner covariances for the weighted LM solver. Priority: external
/// (GWLF) → image-derived structure tensor → `None`. The χ² gate and IPPE
/// branch selector use their own isotropic info matrices
/// ([`isotropic_info_matrices`]) so we don't materialize an inverted
/// info-matrix array on the hot path.
#[inline]
fn build_lm_covariances(
    config: &crate::config::DetectorConfig,
    img: Option<&ImageView>,
    ideal_corners: &[[f64; 2]; 4],
    h_poly: &crate::decoder::Homography,
    external_covariances: Option<&[Matrix2<f64>; 4]>,
) -> Option<[Matrix2<f64>; 4]> {
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
    // Per the Tikhonov invariant in `finalize_corner_covariance`, the
    // kernel-level Σ_c is guaranteed PD; `try_inverse → None` is
    // unreachable on production kernel output. The `info_matrices_well_
    // conditioned` check below provides an additional condition-number
    // safety net even when invertibility is guaranteed.
    let info: [Matrix2<f64>; 4] = core::array::from_fn(|i| {
        debug_assert!(
            covs[i].try_inverse().is_some(),
            "Tikhonov invariant violated: per-corner Σ_c[{i}] is singular",
        );
        covs[i].try_inverse().unwrap_or_else(Matrix2::identity)
    });
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
///
/// **Input convention**: `h` is a *metric* homography that maps the canonical unit-square
/// source `(-1, -1), (1, -1), (1, 1), (-1, 1)` (scaled by `tag_size / 2` to physical
/// metres centred at the tag origin) to normalised image coordinates `[x_n, y_n, 1]^T`
/// (i.e. `K^{-1} · H_pixel · diag(2/L, 2/L, 1)` where `L = tag_size`).
///
/// **Output convention**: both returned [`Pose`]s are in the *camera-from-tag* frame
/// (tag plane at z=0, centred at origin, +x right / +y down for typical AprilTag /
/// ArUco corner orderings `[TL, TR, BR, BL]`).
pub(crate) fn solve_ippe_square(h: &Matrix3<f64>) -> Option<[Pose; 2]> {
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
    // Discriminant sqrt(tr² − 4·det). Algebraically tr² − 4·det = (a−b)² + 4c², a
    // manifestly non-negative, cancellation-free form. The naive `tr² − 4·det`
    // suffers catastrophic cancellation exactly near frontal (a≈b, c≈0 ⇒ tr²≈4·det),
    // degrading the precision of s1,s2 — and thus the seed rotation direction
    // v = [s1²−b, c] — in precisely the regime where rotation accuracy is hardest.
    // Compute the equivalent sum-of-squares form directly (cannot be negative).
    let delta = ((a - b).powi(2) + 4.0 * c * c).sqrt();

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
        let r1_norm = r1.norm();
        // Guard the normalization: a degenerate homography with ‖h1‖→0 would make
        // `1/‖r1‖` non-finite. Bail (caller falls back / rejects) rather than emit
        // a NaN pose. Latent on validated input (square_to_quad rejects non-finite
        // reprojection upstream); this is defense-in-depth.
        if r1_norm < 1e-12 {
            return None;
        }
        let scale = 1.0 / r1_norm;
        r1 *= scale;

        // Orthogonalize r2 w.r.t r1. If h2 ∥ r1 the residual collapses to ~0, so
        // normalize via try_normalize and bail instead of dividing by ‖r2‖→0.
        let r2 = (h2 - r1 * (h2.dot(&r1))).try_normalize(1e-12)?;

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

/// Assemble the Huber-weighted normal equations `(JᵀWJ, JᵀWr)` for the 4-corner
/// reprojection problem at `pose`, plus the Huber cost at `pose`, shared by
/// `refine_pose_lm` and its gradient tests. Layout: ξ = [tx, ty, tz, rx, ry, rz].
///
/// The returned cost is bit-identical to [`huber_cost`] evaluated at the same
/// pose (both route through the same clamped normalized projection), so the LM
/// can reuse it as the linearization-point cost without a redundant re-projection
/// — mirroring `pose_weighted::build_normal_equations`'s `(jtj, jtr, cost)` triple.
fn corner_normal_equations(
    intrinsics: &CameraIntrinsics,
    corners: &[[f64; 2]; 4],
    obj_pts: &[Vector3<f64>; 4],
    pose: &Pose,
    huber_delta: f64,
) -> (Matrix6<f64>, Vector6<f64>, f64) {
    let mut ne = BodyFrameNormalEquations::new(pose);
    let mut cost = 0.0;
    for i in 0..4 {
        let pb = obj_pts[i];
        let p_cam = pose.rotation * pb + pose.translation;
        // Behind-camera policy (deliberately differs from the board/weighted LMs,
        // which DROP such correspondences): a single tag has exactly four corners
        // we want to keep, so we clamp `z` and let the Huber weight softly reject
        // the resulting large residual rather than discard a corner and lose rank.
        // The board/weighted solvers see many points, so dropping a few invalid
        // projections there is both safe and cheaper.
        let z = p_cam.z.max(1e-4);
        let z_inv = 1.0 / z;
        let xn = p_cam.x / z;
        let yn = p_cam.y / z;

        // Distortion-aware projection (identity pass-through for DistortionCoeffs::None).
        let [u_est, v_est] = intrinsics.distort_normalized(xn, yn);
        let res_u = corners[i][0] - u_est;
        let res_v = corners[i][1] - v_est;

        // Huber IRLS weight on the 2-D geometric pixel distance.
        let r_norm = (res_u * res_u + res_v * res_v).sqrt();
        let w = if r_norm <= huber_delta {
            1.0
        } else {
            huber_delta / r_norm
        };

        // Huber cost at the linearization point (identical formula/order to
        // `huber_cost`, so the caller can skip a redundant re-projection).
        if r_norm <= huber_delta {
            cost += 0.5 * r_norm * r_norm;
        } else {
            cost += huber_delta * (r_norm - 0.5 * huber_delta);
        }

        // Distortion-aware projection gradient ∂[u,v]/∂P_cam (chain rule through
        // the distortion map); the body-frame Jacobian + accumulation live in the
        // shared `BodyFrameNormalEquations`. Scalar Huber weight → isotropic W.
        let jd = intrinsics.distortion_jacobian(xn, yn);
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
        ne.add(
            &pb,
            &du_dp,
            &dv_dp,
            res_u,
            res_v,
            &Matrix2::new(w, 0.0, 0.0, w),
        );
    }
    let (jtj, jtr) = ne.finish();
    (jtj, jtr, cost)
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
    // The normal-equations builder also returns the Huber cost at the linearization
    // point, so we only rebuild (and re-project) after an accepted step — a rejected
    // step reuses the cached Hessian and cost unchanged. `f64::MAX` forces the first
    // rebuild; the loop always populates these before use.
    let mut jtj = Matrix6::<f64>::zeros();
    let mut jtr = Vector6::<f64>::zeros();
    let mut current_cost = f64::MAX;
    let mut needs_rebuild = true;

    // Allow up to 20 iterations; typically exits in 3-6 via the convergence gates below.
    for _ in 0..20 {
        if needs_rebuild {
            // Huber-weighted normal equations + cost for the 4-corner reprojection problem.
            let (rjtj, rjtr, cost) =
                corner_normal_equations(intrinsics, corners, &obj_pts, &pose, huber_delta);
            jtj = rjtj;
            jtr = rjtr;
            current_cost = cost;
            needs_rebuild = false;
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
        let new_pose = pose.retract(&delta);

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
            // Pose moved, so the cached Hessian/cost are stale — rebuild next iteration.
            needs_rebuild = true;
            // Nielsen's update rule: lambda scales by max(1/3, 1 - (2*rho - 1)^3).
            lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            nu = 2.0;

            // Gate 2: Step size convergence — pose has stopped moving.
            if delta.norm() < 1e-7 {
                break;
            }
        } else {
            // Reject step: pose unchanged, so the cached Hessian/cost stay valid;
            // grow lambda with doubling backoff to stay within the trust region.
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
) {
    refine_poses_soa_with_config(
        batch,
        v,
        intrinsics,
        tag_size,
        img,
        &crate::config::DetectorConfig::default(),
    );
}

/// Refine poses for all valid candidates with explicit config for tuning parameters.
#[expect(
    clippy::too_many_lines,
    reason = "single SoA sweep over all candidates; the per-corner setup, solve, and covariance write-back belong together for cache locality"
)]
#[tracing::instrument(skip_all, name = "pipeline::pose_refinement")]
pub fn refine_poses_soa_with_config(
    batch: &mut DetectionBatch,
    v: usize,
    intrinsics: &CameraIntrinsics,
    tag_size: f64,
    img: Option<&ImageView>,
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

    // Per-candidate work; takes plain references and returns a Copy tuple
    // so the rayon closure can write the SoA cells directly without
    // materialising a per-frame `Vec<TupleN>` outside the workspace arena.
    // Marked `#[inline]` so the call through the `for_each` closure stays
    // a direct call after monomorphisation — a closure-binding form was
    // opaque to LLVM and added measurable per-candidate overhead in the
    // pose microbenches.
    #[expect(
        clippy::inline_always,
        reason = "the measured per-candidate overhead documented above is exactly what forcing the inline removes"
    )]
    #[inline(always)]
    fn compute_one(
        intrinsics: &CameraIntrinsics,
        tag_size: f64,
        img: Option<&ImageView>,
        config: &crate::config::DetectorConfig,
        thresholds: Option<ConsistencyThresholds>,
        corners_row: &[Point2f; 4],
        covs_row: &[f32; 16],
    ) -> (Option<[f32; 7]>, PoseDiagnostics) {
        let corners = [
            [f64::from(corners_row[0].x), f64::from(corners_row[0].y)],
            [f64::from(corners_row[1].x), f64::from(corners_row[1].y)],
            [f64::from(corners_row[2].x), f64::from(corners_row[2].y)],
            [f64::from(corners_row[3].x), f64::from(corners_row[3].y)],
        ];

        // Only GWLF writes covariances calibrated as image-noise
        // variances; other extractors (EdLines, Erf) leave per-corner
        // GN-residual uncertainties in `batch.corner_covariances` that
        // are far tighter than σ_n and would mis-feed the LM solver.
        let ext_covs_opt: Option<[Matrix2<f64>; 4]> =
            if config.refinement_mode == crate::config::CornerRefinementMode::Gwlf {
                let covs: [Matrix2<f64>; 4] = core::array::from_fn(|j| {
                    Matrix2::new(
                        f64::from(covs_row[j * 4]),
                        f64::from(covs_row[j * 4 + 1]),
                        f64::from(covs_row[j * 4 + 2]),
                        f64::from(covs_row[j * 4 + 3]),
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
            config,
            ext_covs_opt.as_ref(),
            thresholds,
        );

        let pose_data = if let Some(pose) = pose_opt {
            let q = quat_from_so3(pose.rotation);
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
    }

    // Split the batch into disjoint per-column slices so each rayon worker
    // can write its target SoA cells directly (rayon's `Zip` proves the
    // per-index disjointness statically), eliminating the previous
    // per-frame `Vec<TupleN>` heap allocation that was drained sequentially
    // after the parallel map. Read-only inputs (`corners`,
    // `corner_covariances`) and write-only outputs (`poses`, bench
    // diagnostics) live on disjoint fields of `DetectionBatch`, so the
    // simultaneous shared/mutable borrows are sound under Rust's
    // disjoint-field rule.
    let corners_in = &batch.corners[..v];
    let covs_in = &batch.corner_covariances[..v];
    let poses_out = &mut batch.poses[..v];
    #[cfg(feature = "bench-internals")]
    let d2_out = &mut batch.pose_consistency_d2[..v];
    #[cfg(feature = "bench-internals")]
    let d2_max_out = &mut batch.pose_consistency_d2_max_corner[..v];
    #[cfg(feature = "bench-internals")]
    let branch_ratio_out = &mut batch.ippe_branch_d2_ratio[..v];
    #[cfg(feature = "bench-internals")]
    let outlier_idx_out = &mut batch.outlier_corner_idx[..v];

    // Rayon's `Zip` on `IndexedParallelIterator`s silently truncates to the
    // shortest input. A future off-by-one fix on any of these slices would
    // drop the last candidate's pose with no panic, so guard the contract
    // explicitly. Cheap: a few `debug_assert_eq!`s, only in debug builds.
    debug_assert_eq!(poses_out.len(), v);
    debug_assert_eq!(corners_in.len(), v);
    debug_assert_eq!(covs_in.len(), v);
    #[cfg(feature = "bench-internals")]
    {
        debug_assert_eq!(d2_out.len(), v);
        debug_assert_eq!(d2_max_out.len(), v);
        debug_assert_eq!(branch_ratio_out.len(), v);
        debug_assert_eq!(outlier_idx_out.len(), v);
    }

    #[cfg(feature = "bench-internals")]
    {
        poses_out
            .par_iter_mut()
            .zip(d2_out.par_iter_mut())
            .zip(d2_max_out.par_iter_mut())
            .zip(branch_ratio_out.par_iter_mut())
            .zip(outlier_idx_out.par_iter_mut())
            .zip(corners_in.par_iter())
            .zip(covs_in.par_iter())
            .for_each(
                |(
                    (((((pose_slot, d2_slot), d2_max_slot), branch_slot), outlier_slot), c_row),
                    cov_row,
                )| {
                    let (pose_data, diag) = compute_one(
                        intrinsics, tag_size, img, config, thresholds, c_row, cov_row,
                    );
                    *pose_slot = if let Some(data) = pose_data {
                        Pose6D { data, padding: 0.0 }
                    } else {
                        Pose6D {
                            data: [0.0; 7],
                            padding: 0.0,
                        }
                    };
                    *d2_slot = diag.aggregate_d2;
                    *d2_max_slot = diag.max_corner_d2;
                    *branch_slot = diag.branch_d2_ratio;
                    *outlier_slot = diag.outlier_corner_idx;
                },
            );
    }
    #[cfg(not(feature = "bench-internals"))]
    {
        poses_out
            .par_iter_mut()
            .zip(corners_in.par_iter())
            .zip(covs_in.par_iter())
            .for_each(|((pose_slot, c_row), cov_row)| {
                let (pose_data, _diag) = compute_one(
                    intrinsics, tag_size, img, config, thresholds, c_row, cov_row,
                );
                *pose_slot = if let Some(data) = pose_data {
                    Pose6D { data, padding: 0.0 }
                } else {
                    Pose6D {
                        data: [0.0; 7],
                        padding: 0.0,
                    }
                };
            });
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // The corner-LM normal equations must satisfy `jtr = -∂cost/∂ξ` for the LM
    // step `δ = (JᵀWJ)⁻¹ jtr` to be a descent direction (identical convention to
    // the weighted LM's `test_weighted_jacobian_matches_finite_difference`). A
    // sign error in any row flips `jtr[k]` vs the true gradient → relative_err ≈ 2.
    #[test]
    fn corner_normal_equations_gradient_is_descent() {
        let intrinsics = CameraIntrinsics::new(600.0, 600.0, 320.0, 240.0);
        let rot = Rotation3::new(Vector3::new(0.2, -0.15, 0.1))
            .matrix()
            .into_owned();
        let pose = Pose::new(rot, Vector3::new(0.15, 0.12, 0.5));
        let obj_pts = centered_tag_corners(0.04);
        // Non-zero residuals so the gradient exposes a sign error.
        let mut corners: [[f64; 2]; 4] =
            core::array::from_fn(|i| pose.project(&obj_pts[i], &intrinsics));
        corners[0][0] += 3.0;
        corners[0][1] -= 2.0;
        corners[1][0] -= 1.5;
        corners[2][1] += 1.2;
        let huber = 1e9; // pure L2 so cost is smooth for the finite difference

        let (_jtj, jtr, cost) =
            corner_normal_equations(&intrinsics, &corners, &obj_pts, &pose, huber);
        // The returned cost must be bit-identical to a standalone `huber_cost` at
        // the same pose, since the LM reuses it as the linearization-point cost.
        // Compare raw bits: this is an exact-equality contract, not an approximation.
        assert_eq!(
            cost.to_bits(),
            huber_cost(&intrinsics, &corners, &obj_pts, &pose, huber).to_bits()
        );
        let eps = 1e-6;
        for dof in 0..6 {
            let mut dp = Vector6::zeros();
            dp[dof] = eps;
            let cf = huber_cost(&intrinsics, &corners, &obj_pts, &pose.retract(&dp), huber);
            dp[dof] = -eps;
            let cb = huber_cost(&intrinsics, &corners, &obj_pts, &pose.retract(&dp), huber);
            let num_grad = (cf - cb) / (2.0 * eps);
            let scale = jtr[dof].abs().max(num_grad.abs()).max(1.0);
            let rel = (jtr[dof] + num_grad).abs() / scale;
            assert!(
                rel < 1e-3,
                "DOF {dof}: jtr={:.4} -grad={:.4} rel={rel:.3}",
                jtr[dof],
                -num_grad
            );
        }
    }

    // Demonstrates the sign fix: with the pre-fix (negated) rotation rows the
    // trust region rejected every rotation step, so `refine_pose_lm` returned the
    // init rotation. With the corrected rows it converges to the true rotation
    // from a perturbed init. (Would fail before the fix; passes after.)
    #[test]
    fn refine_pose_lm_recovers_rotation() {
        let intrinsics = CameraIntrinsics::new(700.0, 700.0, 320.0, 240.0);
        let true_rot = Rotation3::new(Vector3::new(0.25, -0.18, 0.12))
            .matrix()
            .into_owned();
        let true_pose = Pose::new(true_rot, Vector3::new(0.03, -0.02, 0.6));
        let obj_pts = centered_tag_corners(0.05);
        let corners: [[f64; 2]; 4] =
            core::array::from_fn(|i| true_pose.project(&obj_pts[i], &intrinsics));

        // Perturb the init rotation by ~5° and translation slightly.
        let perturbed_rot = Rotation3::new(Vector3::new(0.06, 0.05, -0.04)).matrix() * true_rot;
        let init = Pose::new(perturbed_rot, Vector3::new(0.05, 0.0, 0.62));

        let refined = refine_pose_lm(&intrinsics, &corners, 0.05, init, 1.5);

        let ang = |a: &Matrix3<f64>, b: &Matrix3<f64>| {
            (((a.transpose() * b).trace() - 1.0) * 0.5)
                .clamp(-1.0, 1.0)
                .acos()
        };
        let init_err = ang(&init.rotation, &true_rot);
        let refined_err = ang(&refined.rotation, &true_rot);
        assert!(
            refined_err < 0.1 * init_err,
            "rotation not refined: init_err={init_err:.4} refined_err={refined_err:.4}"
        );
    }

    // The rotation-Jacobian fix makes the fallback LM actually move rotation, so a
    // rank-deficient corner configuration now drives the Hessian singular along the
    // rotation DOFs. The Cholesky-failure / λ-backoff path must keep the solver
    // bounded — it must never emit a NaN/∞ pose (which would poison downstream PnP
    // and any covariance extraction). Guards the now-active path against degenerate
    // inputs reachable through `estimate_tag_pose(corners, None)`.
    #[test]
    fn refine_pose_lm_stays_finite_on_degenerate_corners() {
        let intrinsics = CameraIntrinsics::new(600.0, 600.0, 320.0, 240.0);
        let init = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 0.5));

        let finite = |p: &Pose| {
            p.translation.iter().all(|v| v.is_finite()) && p.rotation.iter().all(|v| v.is_finite())
        };

        // Case 1: all four corners coincident (zero-area quad, fully rank-deficient).
        let coincident = [[300.0, 250.0]; 4];
        let r1 = refine_pose_lm(&intrinsics, &coincident, 0.05, init, 1.5);
        assert!(
            finite(&r1),
            "coincident-corner refine produced a non-finite pose"
        );

        // Case 2: four collinear corners (rank-1 spatial support → rotation DOFs
        // about the line are unobservable).
        let collinear = [
            [100.0, 250.0],
            [200.0, 250.0],
            [300.0, 250.0],
            [400.0, 250.0],
        ];
        let r2 = refine_pose_lm(&intrinsics, &collinear, 0.05, init, 1.5);
        assert!(
            finite(&r2),
            "collinear-corner refine produced a non-finite pose"
        );
    }

    // `Pose::adjoint` must satisfy the defining identity relating the body-frame
    // (right) and camera-frame (left) tangents: `exp(Ad_T·δ)·T = T·exp(δ)`. We
    // verify it by comparing the right retract `T·exp(δ)` against the explicit
    // left update `exp(Ad_T·δ)·T` for a small twist (agreement to O(δ²)).
    #[test]
    fn adjoint_relates_body_and_camera_tangents() {
        let r = Rotation3::new(Vector3::new(0.3, -0.2, 0.15))
            .matrix()
            .into_owned();
        let pose = Pose::new(r, Vector3::new(0.12, -0.05, 0.7));
        let delta_body = Vector6::new(2e-4, -1.5e-4, 3e-4, 1e-4, -2e-4, 1.5e-4);

        // Right: T·exp(δ_body).
        let right = pose.retract(&delta_body);

        // Left: exp(Ad_T·δ_body)·T, built explicitly (rotation on the left, its
        // translation part added, matching the SE(3) left perturbation to O(δ²)).
        let delta_cam = pose.adjoint() * delta_body;
        let rot_l = Rotation3::new(Vector3::new(delta_cam[3], delta_cam[4], delta_cam[5]))
            .matrix()
            .into_owned();
        let t_l = Vector3::new(delta_cam[0], delta_cam[1], delta_cam[2]);
        let left = Pose::new(rot_l * pose.rotation, rot_l * pose.translation + t_l);

        assert!(
            (right.rotation - left.rotation).abs().max() < 1e-7,
            "rotation mismatch: {}",
            (right.rotation - left.rotation).abs().max()
        );
        assert!(
            (right.translation - left.translation).norm() < 1e-7,
            "translation mismatch: {}",
            (right.translation - left.translation).norm()
        );
    }

    // Reframing a covariance camera→body→camera must be the identity (the two
    // adjoint maps are exact inverses), and a body-frame covariance built from a
    // real (body-frame) Hessian must reframe to a valid camera-frame one.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn covariance_reframe_roundtrips() {
        let r = Rotation3::new(Vector3::new(-0.25, 0.1, 0.4))
            .matrix()
            .into_owned();
        let pose = Pose::new(r, Vector3::new(-0.08, 0.2, 0.55));

        // A representative SPD covariance (body frame).
        let intrinsics = CameraIntrinsics::new(700.0, 700.0, 320.0, 240.0);
        let obj_pts = centered_tag_corners(0.05);
        let corners: [[f64; 2]; 4] =
            core::array::from_fn(|i| pose.project(&obj_pts[i], &intrinsics));
        let (jtj, _, _) = corner_normal_equations(&intrinsics, &corners, &obj_pts, &pose, 1e9);
        let cov_body_m = jtj.try_inverse().expect("SPD Hessian invertible");
        let cov_body: [[f64; 6]; 6] =
            core::array::from_fn(|i| core::array::from_fn(|j| cov_body_m[(i, j)]));

        let cov_cam = pose.covariance_body_to_camera(&cov_body);
        let back = pose.covariance_camera_to_body(&cov_cam);
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (back[i][j] - cov_body[i][j]).abs() < 1e-9,
                    "roundtrip mismatch at ({i},{j}): {} vs {}",
                    back[i][j],
                    cov_body[i][j]
                );
            }
        }

        // The camera-frame covariance is a different matrix (frames differ) but
        // stays symmetric.
        for i in 0..6 {
            for j in 0..6 {
                assert!((cov_cam[i][j] - cov_cam[j][i]).abs() < 1e-12);
            }
        }
    }

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
    fn quaternion_extraction_does_not_hang_on_near_singular_input() {
        // Regression for the latent `UnitQuaternion::from_matrix` Müller-iteration hang.
        // That function delegates to `Rotation3::from_matrix_eps` with `max_iter = 0`,
        // which nalgebra treats as `usize::MAX` ("loop until convergence"). On a
        // matrix that is one cosmic-ray bit off SO(3), the iteration can fail to
        // converge — wedging the thread. `quat_from_so3` uses the closed-form
        // analytic extraction instead and must return promptly with a unit-norm
        // quaternion on the same input.
        let mut r = Matrix3::<f64>::identity();
        // Perturb off-diagonal entries by 1e-12 — orders of magnitude below the
        // floating-point round-off floor for `R R^T - I` produced by any LM
        // exp-map step, yet representative of "near-orthogonal but not exact".
        r[(0, 1)] = 1e-12;
        r[(1, 0)] = -1e-12;
        r[(0, 2)] = 1e-12;
        r[(2, 0)] = -1e-12;

        let start = std::time::Instant::now();
        let q = quat_from_so3(r);
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_millis(1),
            "quat_from_so3 took {elapsed:?} — expected sub-millisecond closed-form extraction",
        );
        assert!(
            (q.norm() - 1.0).abs() < 1e-9,
            "quat_from_so3 result is not unit-norm: |q| = {}",
            q.norm(),
        );
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

        let (est_pose, _) = estimate_tag_pose(&intrinsics, &img_pts, tag_size, None);
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

            if let (Some(est_pose), _) = estimate_tag_pose(&intrinsics, &img_pts, tag_size, None) {
                // Check if recovered pose is reasonably close
                // Note: noise decreases accuracy, so we use a loose threshold
                let t_err = (est_pose.translation - translation).norm();
                prop_assert!(t_err < 0.1 + noise * 0.1, "Translation error {} too high for noise {}", t_err, noise);

                let r_err = (est_pose.rotation - rotation).norm();
                prop_assert!(r_err < 0.1 + noise * 0.1, "Rotation error {} too high for noise {}", r_err, noise);
            }
        }
    }

    // ---------- Outlier-aware corner-drop policy ----------

    /// Project the centred tag corners under a known pose with no noise.
    /// Tests then perturb individual corners to drive the outlier-drop policy.
    fn synthetic_scene() -> (CameraIntrinsics, Pose, f64, [[f64; 2]; 4]) {
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.5));
        let tag_size = 0.16;
        let obj_pts = centered_tag_corners(tag_size);
        let mut corners = [[0.0_f64; 2]; 4];
        for i in 0..4 {
            corners[i] = pose.project(&obj_pts[i], &intrinsics);
        }
        (intrinsics, pose, tag_size, corners)
    }

    /// `Σ_c⁻¹ = σ⁻²·I` for all four corners — the info matrices the
    /// outlier helper consumes directly.
    fn isotropic_info(sigma_sq: f64) -> [Matrix2<f64>; 4] {
        let inv = 1.0 / sigma_sq;
        let m = Matrix2::new(inv, 0.0, 0.0, inv);
        [m, m, m, m]
    }

    /// Converge the 4-corner weighted LM seeded at GT pose. Shared by all
    /// outlier-drop tests; the helper then runs on this output.
    fn seed_with_lm(
        intrinsics: &CameraIntrinsics,
        corners: &[[f64; 2]; 4],
        tag_size: f64,
        gt_pose: Pose,
        info: &[Matrix2<f64>; 4],
    ) -> Pose {
        let (pose, _) = crate::pose_weighted::refine_pose_lm_weighted_with_info(
            intrinsics, corners, tag_size, gt_pose, info,
        );
        pose
    }

    #[test]
    fn outlier_drop_fires_on_single_dominant_outlier() {
        let (intrinsics, gt_pose, tag_size, mut corners) = synthetic_scene();
        // +25 px outlier on corner 0. The Huber kernel attenuates the LM
        // pull (weight ≈ k/s = 1.345/12.5 ≈ 0.11), so post-LM corner 0's
        // residual stays large enough that d²_0 ≫ 25 with σ²=4.
        corners[0][0] += 25.0;
        let info = isotropic_info(4.0);
        let pose_4 = seed_with_lm(&intrinsics, &corners, tag_size, gt_pose, &info);

        let dropped =
            maybe_drop_outlier_corner(&intrinsics, &corners, tag_size, &info, pose_4, 25.0);

        let (idx, pose_3, _cov_3) = dropped.expect("drop should fire on +25 px outlier");
        assert_eq!(idx, 0, "expected to drop corner 0");
        let err_dropped = (pose_3.translation - gt_pose.translation).norm();
        let err_full = (pose_4.translation - gt_pose.translation).norm();
        assert!(
            err_dropped < err_full,
            "drop should improve translation error: dropped={err_dropped} full={err_full}",
        );
    }

    #[test]
    fn outlier_drop_skips_clean_corners() {
        let (intrinsics, gt_pose, tag_size, mut corners) = synthetic_scene();
        // Sub-pixel noise on all four — no corner should trigger 5σ² = 25.
        let noise = [(0.3, -0.2), (-0.15, 0.25), (0.1, -0.05), (-0.2, 0.18)];
        for i in 0..4 {
            corners[i][0] += noise[i].0;
            corners[i][1] += noise[i].1;
        }
        let info = isotropic_info(4.0);
        let pose_4 = seed_with_lm(&intrinsics, &corners, tag_size, gt_pose, &info);

        let dropped =
            maybe_drop_outlier_corner(&intrinsics, &corners, tag_size, &info, pose_4, 25.0);

        assert!(
            dropped.is_none(),
            "no outlier should be dropped on sub-pixel noise"
        );
    }

    #[test]
    fn outlier_drop_dominance_check_rejects_two_outliers() {
        let (intrinsics, gt_pose, tag_size, mut corners) = synthetic_scene();
        // Two comparable +10 px outliers — neither dominates the other by 2×.
        corners[0][0] += 10.0;
        corners[1][0] += 10.0;
        let info = isotropic_info(4.0);
        let pose_4 = seed_with_lm(&intrinsics, &corners, tag_size, gt_pose, &info);

        let dropped =
            maybe_drop_outlier_corner(&intrinsics, &corners, tag_size, &info, pose_4, 25.0);

        assert!(
            dropped.is_none(),
            "two correlated outliers must fail dominance check"
        );
    }

    #[test]
    fn refine_pose_lm_with_info_handles_zero_info_on_one_corner() {
        // Locks the masking contract: a zero info matrix on one corner
        // makes its contribution to JᵀWJ, JᵀWr, and the Huber cost
        // identically zero, so the LM converges as if that corner were
        // absent — even when its residual would otherwise dominate.
        let (intrinsics, gt_pose, tag_size, mut corners) = synthetic_scene();
        corners[0][0] += 50.0;

        let mut info = isotropic_info(4.0);
        info[0] = Matrix2::zeros();

        let (pose, _) = crate::pose_weighted::refine_pose_lm_weighted_with_info(
            &intrinsics,
            &corners,
            tag_size,
            gt_pose,
            &info,
        );

        let err = (pose.translation - gt_pose.translation).norm();
        assert!(
            err < 1e-3,
            "masked LM should ignore corner 0 outlier; translation error {err}",
        );
    }
}
