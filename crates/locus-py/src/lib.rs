//! Python bindings for the Locus Tag library.
//!
//! ## NumPy allocation safety contract
//!
//! `PyArray::new(py, shape, false)` is `unsafe` because it returns an
//! uninitialized buffer. We always pair it with a full overwrite before the
//! array escapes the function. The contract relied on at every call site:
//!
//! - The returned `PyArrayN<T>` is freshly allocated, C-contiguous, and
//!   exclusively owned by the GIL-bound `py` token until we hand it back.
//! - No other Rust or Python reference observes the buffer while the
//!   `&mut [T]` borrow from `as_slice_mut` is live, so the slice cannot
//!   alias.
//! - Each call site fully writes the buffer before returning the array.
//!
//! Call-site SAFETY comments reference this note instead of restating it.
#![allow(
    unsafe_code,
    clippy::unused_self,
    missing_docs,
    clippy::trivially_copy_pass_by_ref
)]

use locus_core::ImageView;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{
    gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pyfunction, gen_stub_pymethods,
};

// ============================================================================
// Enums
// ============================================================================

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TagFamily {
    AprilTag16h5 = 0,
    AprilTag36h11 = 1,
    ArUco4x4_50 = 2,
    ArUco4x4_100 = 3,
    ArUco6x6_250 = 4,
}

impl From<TagFamily> for locus_core::TagFamily {
    fn from(f: TagFamily) -> Self {
        match f {
            TagFamily::AprilTag16h5 => locus_core::TagFamily::AprilTag16h5,
            TagFamily::AprilTag36h11 => locus_core::TagFamily::AprilTag36h11,
            TagFamily::ArUco4x4_50 => locus_core::TagFamily::ArUco4x4_50,
            TagFamily::ArUco4x4_100 => locus_core::TagFamily::ArUco4x4_100,
            TagFamily::ArUco6x6_250 => locus_core::TagFamily::ArUco6x6_250,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SegmentationConnectivity {
    Four = 0,
    Eight = 1,
}

impl From<SegmentationConnectivity> for locus_core::config::SegmentationConnectivity {
    fn from(c: SegmentationConnectivity) -> Self {
        match c {
            SegmentationConnectivity::Four => locus_core::config::SegmentationConnectivity::Four,
            SegmentationConnectivity::Eight => locus_core::config::SegmentationConnectivity::Eight,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CornerRefinementMode {
    None = 0,
    Erf = 1,
    Gwlf = 2,
}

impl From<CornerRefinementMode> for locus_core::config::CornerRefinementMode {
    fn from(m: CornerRefinementMode) -> Self {
        match m {
            CornerRefinementMode::None => locus_core::config::CornerRefinementMode::None,
            CornerRefinementMode::Erf => locus_core::config::CornerRefinementMode::Erf,
            CornerRefinementMode::Gwlf => locus_core::config::CornerRefinementMode::Gwlf,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum QuadExtractionMode {
    ContourRdp = 0,
    EdLines = 1,
}

impl From<QuadExtractionMode> for locus_core::config::QuadExtractionMode {
    fn from(m: QuadExtractionMode) -> Self {
        match m {
            QuadExtractionMode::ContourRdp => locus_core::config::QuadExtractionMode::ContourRdp,
            QuadExtractionMode::EdLines => locus_core::config::QuadExtractionMode::EdLines,
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum EdLinesImbalanceGatePolicy {
    Disabled = 0,
    Enabled = 1,
}

impl From<EdLinesImbalanceGatePolicy> for locus_core::config::EdLinesImbalanceGatePolicy {
    fn from(m: EdLinesImbalanceGatePolicy) -> Self {
        match m {
            EdLinesImbalanceGatePolicy::Disabled => {
                locus_core::config::EdLinesImbalanceGatePolicy::Disabled
            },
            EdLinesImbalanceGatePolicy::Enabled => {
                locus_core::config::EdLinesImbalanceGatePolicy::Enabled
            },
        }
    }
}

impl From<locus_core::config::EdLinesImbalanceGatePolicy> for EdLinesImbalanceGatePolicy {
    fn from(m: locus_core::config::EdLinesImbalanceGatePolicy) -> Self {
        match m {
            locus_core::config::EdLinesImbalanceGatePolicy::Disabled => Self::Disabled,
            locus_core::config::EdLinesImbalanceGatePolicy::Enabled => Self::Enabled,
        }
    }
}

// ============================================================================
// Structs
// ============================================================================

/// Identifies which lens distortion model the [`CameraIntrinsics`] coefficients apply to.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass_enum)]
#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum DistortionModel {
    /// Ideal pinhole camera — no distortion (pre-rectified image). Default.
    #[default]
    Pinhole = 0,
    /// Brown-Conrady polynomial radial + tangential distortion (OpenCV convention).
    ///
    /// `dist_coeffs` must contain exactly **5** values: `[k1, k2, p1, p2, k3]`.
    #[cfg(feature = "non_rectified")]
    BrownConrady = 1,
    /// Kannala-Brandt equidistant fisheye model.
    ///
    /// `dist_coeffs` must contain exactly **4** values: `[k1, k2, k3, k4]`.
    #[cfg(feature = "non_rectified")]
    KannalaBrandt = 2,
}

/// Camera intrinsic parameters with optional lens distortion.
///
/// Construct with just `(fx, fy, cx, cy)` for a rectified camera, or
/// pass `distortion_model` and `dist_coeffs` for a distorted camera:
///
/// ```python
/// # Pinhole (no distortion)
/// k = CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0)
///
/// # Brown-Conrady
/// k = CameraIntrinsics(
///     fx=800.0, fy=800.0, cx=400.0, cy=300.0,
///     distortion_model=DistortionModel.BrownConrady,
///     dist_coeffs=[-0.3, 0.1, 0.001, -0.002, 0.0],
/// )
///
/// # Kannala-Brandt fisheye
/// k = CameraIntrinsics(
///     fx=380.0, fy=380.0, cx=320.0, cy=240.0,
///     distortion_model=DistortionModel.KannalaBrandt,
///     dist_coeffs=[0.1, -0.01, 0.001, 0.0],
/// )
/// ```
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct CameraIntrinsics {
    #[pyo3(get, set)]
    pub fx: f64,
    #[pyo3(get, set)]
    pub fy: f64,
    #[pyo3(get, set)]
    pub cx: f64,
    #[pyo3(get, set)]
    pub cy: f64,
    /// The distortion model to use. Defaults to [`DistortionModel::Pinhole`].
    #[pyo3(get, set)]
    pub distortion_model: DistortionModel,
    /// Distortion coefficients corresponding to the chosen model.
    ///
    /// - `BrownConrady`: `[k1, k2, p1, p2, k3]` (5 values)
    /// - `KannalaBrandt`: `[k1, k2, k3, k4]` (4 values)
    /// - `Pinhole`: ignored (may be empty)
    #[pyo3(get, set)]
    pub dist_coeffs: Vec<f64>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl CameraIntrinsics {
    #[new]
    #[pyo3(signature = (fx, fy, cx, cy, distortion_model=DistortionModel::Pinhole, dist_coeffs=None))]
    #[cfg_attr(not(feature = "non_rectified"), allow(clippy::unnecessary_wraps))]
    fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        distortion_model: DistortionModel,
        dist_coeffs: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let dist_coeffs = dist_coeffs.unwrap_or_default();

        for (name, value, require_positive) in [
            ("fx", fx, true),
            ("fy", fy, true),
            ("cx", cx, false),
            ("cy", cy, false),
        ] {
            if !value.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "CameraIntrinsics.{name} must be finite, got {value}"
                )));
            }
            if require_positive && value <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "CameraIntrinsics.{name} must be > 0, got {value}"
                )));
            }
        }

        // Validate coefficient count for the chosen model.
        match distortion_model {
            #[cfg(feature = "non_rectified")]
            DistortionModel::BrownConrady => {
                if dist_coeffs.len() != 5 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "DistortionModel.BrownConrady requires exactly 5 dist_coeffs \
                         [k1, k2, p1, p2, k3], got {}",
                        dist_coeffs.len()
                    )));
                }
            },
            #[cfg(feature = "non_rectified")]
            DistortionModel::KannalaBrandt => {
                if dist_coeffs.len() != 4 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "DistortionModel.KannalaBrandt requires exactly 4 dist_coeffs \
                         [k1, k2, k3, k4], got {}",
                        dist_coeffs.len()
                    )));
                }
            },
            DistortionModel::Pinhole => {
                // Any (or no) coefficients are accepted for Pinhole; they are ignored.
            },
        }

        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            distortion_model,
            dist_coeffs,
        })
    }
}

impl From<CameraIntrinsics> for locus_core::CameraIntrinsics {
    fn from(c: CameraIntrinsics) -> Self {
        match c.distortion_model {
            DistortionModel::Pinhole => Self::new(c.fx, c.fy, c.cx, c.cy),
            #[cfg(feature = "non_rectified")]
            DistortionModel::BrownConrady => {
                // Validated in CameraIntrinsics::new — length is guaranteed to be 5.
                let coeffs = &c.dist_coeffs;
                Self::with_brown_conrady(
                    c.fx, c.fy, c.cx, c.cy, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4],
                )
            },
            #[cfg(feature = "non_rectified")]
            DistortionModel::KannalaBrandt => {
                // Validated in CameraIntrinsics::new — length is guaranteed to be 4.
                let coeffs = &c.dist_coeffs;
                Self::with_kannala_brandt(
                    c.fx, c.fy, c.cx, c.cy, coeffs[0], coeffs[1], coeffs[2], coeffs[3],
                )
            },
        }
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct PyPose {
    #[pyo3(get)]
    pub quaternion: [f64; 4], // x, y, z, w
    #[pyo3(get)]
    pub translation: [f64; 3],
}

// ============================================================================
// Result types
// ============================================================================

/// Intermediate pipeline artifacts emitted when `debug_telemetry=True`.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(get_all, frozen)]
pub struct PipelineTelemetryResult {
    pub binarized: Py<PyArray2<u8>>,
    pub threshold_map: Py<PyArray2<u8>>,
    pub subpixel_jitter: Option<Py<PyArray3<f32>>>,
    pub reprojection_errors: Option<Py<PyArray1<f32>>>,
    pub gwlf_fallback_count: usize,
    pub gwlf_avg_delta: f32,
    /// Per-candidate adaptive-router route labels (0 = low, 1 = high,
    /// 255 = Static/not-routed). One entry per Phase-A candidate, populated
    /// only when `debug_telemetry=True` and `QuadExtractionPolicy::AdaptivePpb`
    /// is active for at least one candidate.
    pub routed_to: Option<Py<PyArray1<u8>>>,
    /// Per-candidate pixels-per-bit estimates (`bbox.short_side / min_outer_dim`).
    /// One entry per Phase-A candidate, populated only when `debug_telemetry=True`.
    pub ppb_estimate: Option<Py<PyArray1<f32>>>,
}

/// Typed result returned by `Detector.detect` (Python).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(get_all, frozen)]
pub struct DetectionResult {
    pub ids: Py<PyArray1<i32>>,
    pub corners: Py<PyArray3<f32>>,
    pub error_rates: Py<PyArray1<f32>>,
    pub poses: Option<Py<PyArray2<f32>>>,
    pub rejected_corners: Py<PyArray3<f32>>,
    pub rejected_error_rates: Py<PyArray1<f32>>,
    /// Per-rejected-quad funnel status code (matches `locus.FunnelStatus`).
    /// Shape `(M,)` — same M as `rejected_corners`.
    pub rejected_funnel_status: Py<PyArray1<u8>>,
    pub telemetry: Option<Py<PipelineTelemetryResult>>,
}

/// Telemetry from `CharucoRefiner.estimate` (Python), populated when `debug_telemetry=True`.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(get_all, frozen)]
pub struct CharucoTelemetryResult {
    pub rejected_saddles: Py<PyArray2<f32>>,
    /// Structure-tensor determinant for the low-det rejection lane.
    /// NaN for entries from the singular-covariance lane — always gate
    /// on [`Self::rejected_reasons`] before computing aggregates.
    pub rejected_determinants: Py<PyArray1<f32>>,
    /// Reason code for each rejected saddle (u8 repr of
    /// [`locus_core::charuco::RejectionReason`]: 0 =
    /// `LowStructureTensorDet`, 1 = `SingularCovariance`).
    pub rejected_reasons: Py<PyArray1<u8>>,
}

/// Typed result returned by `CharucoRefiner.estimate` (Python).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(get_all, frozen)]
pub struct CharucoEstimateResult {
    pub ids: Py<PyArray1<i32>>,
    pub corners: Py<PyArray3<f32>>,
    pub saddle_ids: Py<PyArray1<i32>>,
    pub saddle_pts: Py<PyArray2<f32>>,
    pub saddle_obj: Py<PyArray2<f64>>,
    pub board_pose: Option<Py<PyArray1<f64>>>,
    pub board_cov: Option<Py<PyArray2<f64>>>,
    pub telemetry: Option<Py<CharucoTelemetryResult>>,
}

// ============================================================================
// Board topology types
// ============================================================================

/// Configuration for a ChAruco board.
///
/// Markers occupy the checkerboard squares where `(row + col)` is even.
/// The interior checkerboard corners (saddle points) are used for sub-pixel
/// pose estimation.
///
/// The `family` parameter is checked against the board marker count at
/// construction time: if the board needs more tag IDs than the dictionary
/// provides, a `ValueError` is raised.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
pub struct CharucoBoard {
    pub(crate) inner: std::sync::Arc<locus_core::board::CharucoTopology>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl CharucoBoard {
    /// Create a ChAruco board configuration.
    ///
    /// - `rows` / `cols`: number of checkerboard squares in each dimension.
    /// - `square_length`: physical side length of one square (metres).
    /// - `marker_length`: physical side length of one ArUco marker (metres).
    /// - `family`: tag family used on this board (determines max valid ID).
    #[new]
    fn new(
        rows: usize,
        cols: usize,
        square_length: f64,
        marker_length: f64,
        family: TagFamily,
    ) -> PyResult<Self> {
        let max_id = locus_core::TagFamily::from(family).max_id_count();
        locus_core::board::CharucoTopology::new(rows, cols, square_length, marker_length, max_id)
            .map(|t| Self {
                inner: std::sync::Arc::new(t),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Number of square rows.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    /// Number of square columns.
    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }
}

/// Configuration for an AprilGrid board.
///
/// Every grid cell contains one marker; tag IDs are assigned in row-major order.
///
/// The `family` parameter is checked against the board marker count at
/// construction time: if the board needs more tag IDs than the dictionary
/// provides, a `ValueError` is raised.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
pub struct AprilGrid {
    pub(crate) inner: std::sync::Arc<locus_core::board::AprilGridTopology>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl AprilGrid {
    /// Create an AprilGrid board configuration.
    ///
    /// - `rows` / `cols`: number of markers in each dimension.
    /// - `spacing`: gap between adjacent markers (metres).
    /// - `marker_length`: physical side length of one marker (metres).
    /// - `family`: tag family used on this board (determines max valid ID).
    #[new]
    fn new(
        rows: usize,
        cols: usize,
        spacing: f64,
        marker_length: f64,
        family: TagFamily,
    ) -> PyResult<Self> {
        let max_id = locus_core::TagFamily::from(family).max_id_count();
        locus_core::board::AprilGridTopology::new(rows, cols, spacing, marker_length, max_id)
            .map(|t| Self {
                inner: std::sync::Arc::new(t),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Number of marker rows.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    /// Number of marker columns.
    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }
}

// ============================================================================
// BoardEstimator
// ============================================================================

/// Estimator for multi-tag board poses (AprilGrid).
///
/// Uses the decoded tags as geometric priors to estimate the camera-to-board
/// transform via LO-RANSAC + Anisotropic Weighted Levenberg–Marquardt.
///
/// Reuse a single `BoardEstimator` across frames to amortise the one-time
/// scratch-buffer allocation.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(unsendable)]
pub struct BoardEstimator {
    inner: locus_core::board::BoardEstimator,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl BoardEstimator {
    /// Create a `BoardEstimator` for the given board.
    ///
    /// Reuse the same estimator across frames to avoid re-allocating scratch buffers.
    #[new]
    fn new(board: &AprilGrid) -> Self {
        Self {
            inner: locus_core::board::BoardEstimator::new(std::sync::Arc::clone(&board.inner)),
        }
    }

    /// Create a `BoardEstimator` for a ChAruco board.
    ///
    /// This uses the ArUco tags for board pose estimation without refining saddle points.
    #[classmethod]
    fn from_charuco(_cls: &Bound<'_, pyo3::types::PyType>, board: &CharucoBoard) -> Self {
        let topo = &board.inner;
        let april_grid_topo = locus_core::board::AprilGridTopology::from_obj_points(
            topo.rows,
            topo.cols,
            topo.marker_length,
            topo.obj_points.clone(),
        );
        Self {
            inner: locus_core::board::BoardEstimator::new(std::sync::Arc::new(april_grid_topo)),
        }
    }

    /// Run the board pose estimation pipeline on a single frame.
    ///
    /// Calls the standard tag detector internally, then solves for the board
    /// pose using all detected markers.
    ///
    /// Returns a [`BoardEstimateResult`] with fields:
    /// - `ids`:         `(N,) int32`  — decoded ArUco IDs
    /// - `corners`:     `(N, 4, 2) float32`  — tag corners
    /// - `board_pose`:  `(7,) float64` `[tx, ty, tz, qx, qy, qz, qw]` or `None`
    /// - `board_cov`:   `(6, 6) float64` **body-frame** se(3) pose covariance
    ///   (ordered `[t, ω]`; see `docs/explanation/coordinates.md §5.1`), or `None` if no
    ///   board was estimated **or** the LM Hessian was singular (the
    ///   pose still ships as a best-effort estimate; only the calibrated
    ///   uncertainty is unavailable).
    #[pyo3(signature = (detector, img, intrinsics))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "pyo3 #[pymethods] extraction requires owned PyReadonlyArray2/CameraIntrinsics arguments"
    )]
    fn estimate(
        &mut self,
        py: Python<'_>,
        detector: &mut Detector,
        img: PyReadonlyArray2<'_, u8>,
        intrinsics: CameraIntrinsics,
    ) -> PyResult<BoardEstimateResult> {
        let buffer = prepare_image_view(&img)?;
        let view = buffer.view();
        let core_intr = locus_core::CameraIntrinsics::from(intrinsics);
        let tag_size = self.inner.config.marker_length;
        // Read the detector's outlier-drop opt-in BEFORE the detect call —
        // `detect` borrows `detector` mutably until `batch_view` is dropped.
        let outlier_drop_d2_threshold = detector.inner.config().outlier_drop_d2_threshold;

        // 1. Run tag detection (releases GIL; populates detector's internal batch).
        // Board estimation requires per-tag poses as RANSAC seeds.
        let batch_view = py
            .detach(|| {
                detector
                    .inner
                    .detect(&view, Some(&core_intr), Some(tag_size), false)
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let n = batch_view.len();

        // 2. Run board pose estimation, applying the detector's per-profile
        // outlier-drop policy to the joint LM.
        let board_pose_raw = py.detach(|| {
            self.inner
                .estimate(&batch_view, &core_intr, outlier_drop_d2_threshold)
        });

        // 3. Package detections (ids + corners).
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let ids_arr = unsafe { PyArray1::<i32>::new(py, [n], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let corners_arr = unsafe { PyArray3::<f32>::new(py, [n, 4, 2], false) };

        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let ids_slice = unsafe { ids_arr.as_slice_mut() }
            .map_err(|e| PyRuntimeError::new_err(format!("ids array layout error: {e}")))?;

        // SAFETY: ids are u32; reinterpreting as i32 is well-defined (same width,
        // two's-complement). ArUco IDs fit within i32::MAX by convention.
        ids_slice.copy_from_slice(unsafe {
            std::slice::from_raw_parts(batch_view.ids.as_ptr().cast::<i32>(), n)
        });

        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let corners_slice = unsafe { corners_arr.as_slice_mut() }
            .map_err(|e| PyRuntimeError::new_err(format!("corners array layout error: {e}")))?;

        // SAFETY: Point2f is repr(C) with two f32 fields; flat reinterpretation as
        // &[f32] is sound.
        corners_slice.copy_from_slice(unsafe {
            std::slice::from_raw_parts(batch_view.corners.as_ptr().cast::<f32>(), n * 8)
        });

        // 4. Board pose and covariance.
        //
        // Singular `JᵀWJ` produces a NaN-filled covariance sentinel
        // (`refine_aw_lm` → `Matrix6::from_element(NAN)`). Surface this to
        // Python as `board_cov = None` so the typed `NDArray | None`
        // contract is honored: consumers branch on `cov is None`, not
        // `np.isnan(cov).any()`. The pose itself remains a best-effort
        // estimate.
        let (board_pose, board_cov) = if let Some(bp) = board_pose_raw {
            let q = locus_core::pose::quat_from_so3(bp.pose.rotation);
            let t = bp.pose.translation;

            // SAFETY: see "NumPy allocation safety contract" in module docs.
            let pose_arr = unsafe { PyArray1::<f64>::new(py, [7], false) };
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            let ps = unsafe { pose_arr.as_slice_mut() }
                .map_err(|e| PyRuntimeError::new_err(format!("pose array layout error: {e}")))?;
            ps[0] = t.x;
            ps[1] = t.y;
            ps[2] = t.z;
            ps[3] = q.i;
            ps[4] = q.j;
            ps[5] = q.k;
            ps[6] = q.w;

            let cov_opt = if bp.covariance.iter().all(|v| v.is_finite()) {
                // SAFETY: see "NumPy allocation safety contract" in module docs.
                let cov_arr = unsafe { PyArray2::<f64>::new(py, [6, 6], false) };
                // SAFETY: see "NumPy allocation safety contract" in module docs.
                let cs = unsafe { cov_arr.as_slice_mut() }
                    .map_err(|e| PyRuntimeError::new_err(format!("cov array layout error: {e}")))?;
                for row in 0..6 {
                    for col in 0..6 {
                        cs[row * 6 + col] = bp.covariance[(row, col)];
                    }
                }
                Some(cov_arr.unbind())
            } else {
                None
            };
            (Some(pose_arr.unbind()), cov_opt)
        } else {
            (None, None)
        };

        Ok(BoardEstimateResult {
            ids: ids_arr.unbind(),
            corners: corners_arr.unbind(),
            board_pose,
            board_cov,
        })
    }
}

/// Typed result returned by `BoardEstimator.estimate` (Python).
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(get_all, frozen)]
pub struct BoardEstimateResult {
    pub ids: Py<PyArray1<i32>>,
    pub corners: Py<PyArray3<f32>>,
    pub board_pose: Option<Py<PyArray1<f64>>>,
    pub board_cov: Option<Py<PyArray2<f64>>>,
}

// ============================================================================
// CharucoRefiner
// ============================================================================

/// Extracts ChAruco saddle points from decoded ArUco detections and estimates
/// the board pose via LO-RANSAC + Anisotropic Weighted Levenberg–Marquardt.
///
/// Reuse a single `CharucoRefiner` across frames to amortise the one-time
/// scratch-buffer allocation.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(unsendable)]
pub struct CharucoRefiner {
    inner: locus_core::charuco::CharucoRefiner,
    /// Production output buffer — no telemetry overhead.
    batch: locus_core::charuco::CharucoBatch,
    /// Debug output buffer — telemetry pre-allocated once.
    telem_batch: locus_core::charuco::CharucoBatch,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl CharucoRefiner {
    /// Create a `CharucoRefiner` for the given board.
    ///
    /// Reuse the same refiner across frames to avoid re-allocating scratch buffers.
    #[new]
    fn new(board: &CharucoBoard) -> Self {
        let inner = locus_core::charuco::CharucoRefiner::from_arc(
            std::sync::Arc::clone(&board.inner),
            locus_core::board::LoRansacConfig::default(),
        );
        let batch = inner.new_batch();
        let telem_batch = inner.new_batch_with_telemetry();
        Self {
            inner,
            batch,
            telem_batch,
        }
    }

    /// Run the full ChAruco pipeline on a single frame.
    ///
    /// Calls the standard ArUco tag detector internally, then extracts and
    /// refines saddle points, and estimates the board pose.
    ///
    /// Returns a dict with keys:
    /// - `ids`:         `(N,) int32`  — decoded ArUco IDs
    /// - `corners`:     `(N, 4, 2) float32`  — tag corners
    /// - `saddle_ids`:  `(S,) int32`  — accepted saddle-point indices
    /// - `saddle_pts`:  `(S, 2) float32`  — refined image coordinates
    /// - `saddle_obj`:  `(S, 3) float64`  — board-frame 3D coordinates
    /// - `board_pose`:  `(7,) float64` `[tx, ty, tz, qx, qy, qz, qw]` or `None`
    /// - `board_cov`:   `(6, 6) float64` **body-frame** se(3) pose covariance
    ///   (ordered `[t, ω]`; see `docs/explanation/coordinates.md §5.1`), or `None` if no
    ///   board was estimated **or** the LM Hessian was singular (the
    ///   pose still ships as a best-effort estimate; only the calibrated
    ///   uncertainty is unavailable).
    /// - `telemetry`:   dict with `"rejected_saddles"` `(R,2) float32`,
    ///                  `"rejected_determinants"` `(R,) float32`, and
    ///                  `"rejected_reasons"` `(R,) uint8` (0 =
    ///                  `LowStructureTensorDet`, 1 = `SingularCovariance`)
    ///                  or `None` (only populated when
    ///                  `debug_telemetry=True`). Always gate
    ///                  `rejected_determinants` aggregations on
    ///                  `rejected_reasons` — the `SingularCovariance`
    ///                  lane writes NaN into the determinants buffer.
    #[pyo3(signature = (detector, img, intrinsics, debug_telemetry = false))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "pyo3 #[pymethods] extraction requires owned PyReadonlyArray2/CameraIntrinsics arguments"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one cohesive board-estimate FFI routine (detect -> estimate -> marshal results/telemetry into NumPy); splitting would fragment the buffer marshalling"
    )]
    fn estimate(
        &mut self,
        py: Python<'_>,
        detector: &mut Detector,
        img: PyReadonlyArray2<'_, u8>,
        intrinsics: CameraIntrinsics,
        debug_telemetry: bool,
    ) -> PyResult<CharucoEstimateResult> {
        let buffer = prepare_image_view(&img)?;
        let view = buffer.view();
        let core_intr = locus_core::CameraIntrinsics::from(intrinsics);
        let tag_size = self.inner.config.marker_length;
        // Read the detector's outlier-drop opt-in BEFORE the detect call —
        // `detect` borrows `detector` mutably until `batch_view` is dropped.
        let outlier_drop_d2_threshold = detector.inner.config().outlier_drop_d2_threshold;

        // 1. Run ArUco detection (releases GIL; populates detector's internal batch).
        // Board estimation requires per-tag poses as RANSAC seeds.
        let batch_view = py
            .detach(|| {
                detector
                    .inner
                    .detect(&view, Some(&core_intr), Some(tag_size), false)
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let n = batch_view.len();

        // 2. Run ChAruco saddle extraction + board pose estimation, applying
        // the detector's per-profile outlier-drop policy to the joint LM.
        if debug_telemetry {
            py.detach(|| {
                self.inner.estimate_with_telemetry(
                    &batch_view,
                    &view,
                    &core_intr,
                    &mut self.telem_batch,
                    outlier_drop_d2_threshold,
                );
            });
        } else {
            py.detach(|| {
                self.inner.estimate(
                    &batch_view,
                    &view,
                    &core_intr,
                    &mut self.batch,
                    outlier_drop_d2_threshold,
                );
            });
        }
        let active_batch = if debug_telemetry {
            &self.telem_batch
        } else {
            &self.batch
        };
        let s = active_batch.count;

        // 3. Package ArUco detections (ids + corners).
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let ids_arr = unsafe { PyArray1::<i32>::new(py, [n], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let corners_arr = unsafe { PyArray3::<f32>::new(py, [n, 4, 2], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        unsafe {
            let ids_slice = ids_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("ids slice: {e}")))?;
            ids_slice.copy_from_slice(std::slice::from_raw_parts(
                batch_view.ids.as_ptr().cast::<i32>(),
                n,
            ));
            let corners_slice = corners_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("corners slice: {e}")))?;
            corners_slice.copy_from_slice(std::slice::from_raw_parts(
                batch_view.corners.as_ptr().cast::<f32>(),
                n * 8,
            ));
        }

        // 4. Package saddle-point detections.
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let saddle_ids_arr = unsafe { PyArray1::<i32>::new(py, [s], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let saddle_pts_arr = unsafe { PyArray2::<f32>::new(py, [s, 2], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let saddle_obj_arr = unsafe { PyArray2::<f64>::new(py, [s, 3], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        unsafe {
            let sid_slice = saddle_ids_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("saddle_ids slice: {e}")))?;
            for (dst, &src) in sid_slice.iter_mut().zip(active_batch.saddle_ids()) {
                // saddle IDs are bounded by board saddle count (≤ (rows-1)*(cols-1) ≤ ~400).
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "saddle IDs are bounded by the board saddle count (<= ~400), well within i32"
                )]
                {
                    *dst = src as i32;
                }
            }
            // SAFETY: Point2f is repr(C) with two f32 fields; reinterpreting as &[f32] is
            // sound for a packed, contiguous slice.  [f64; 3] has the same element type as
            // the target NumPy slice.
            let spts_slice = saddle_pts_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("saddle_pts slice: {e}")))?;
            spts_slice.copy_from_slice(std::slice::from_raw_parts(
                active_batch.saddle_image_pts().as_ptr().cast::<f32>(),
                s * 2,
            ));
            let sobj_slice = saddle_obj_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("saddle_obj slice: {e}")))?;
            sobj_slice.copy_from_slice(std::slice::from_raw_parts(
                active_batch.saddle_obj_pts().as_ptr().cast::<f64>(),
                s * 3,
            ));
        }

        // 5. Board pose and covariance (None if insufficient saddles or RANSAC failed).
        // board_pose.take() requires &mut, so we can't go through active_batch.
        let board_pose_raw = if debug_telemetry {
            self.telem_batch.board_pose.take()
        } else {
            self.batch.board_pose.take()
        };
        // Singular `JᵀWJ` produces a NaN-filled covariance sentinel
        // (`refine_pose_lm_weighted_with_info` → `Matrix6::from_element(NAN)`).
        // Surface to Python as `board_cov = None` so the typed
        // `NDArray | None` contract is honored.
        let (board_pose, board_cov) = if let Some(bp) = board_pose_raw {
            let q = locus_core::pose::quat_from_so3(bp.pose.rotation);
            let t = bp.pose.translation;
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            let pose_arr = unsafe { PyArray1::<f64>::new(py, [7], false) };
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            unsafe {
                let ps = pose_arr
                    .as_slice_mut()
                    .map_err(|e| PyRuntimeError::new_err(format!("pose slice: {e}")))?;
                ps[0] = t.x;
                ps[1] = t.y;
                ps[2] = t.z;
                ps[3] = q.i;
                ps[4] = q.j;
                ps[5] = q.k;
                ps[6] = q.w;
            }
            let cov_opt = if bp.covariance.iter().all(|v| v.is_finite()) {
                // SAFETY: see "NumPy allocation safety contract" in module docs.
                let cov_arr = unsafe { PyArray2::<f64>::new(py, [6, 6], false) };
                // SAFETY: see "NumPy allocation safety contract" in module docs.
                unsafe {
                    let cs = cov_arr
                        .as_slice_mut()
                        .map_err(|e| PyRuntimeError::new_err(format!("cov slice: {e}")))?;
                    for row in 0..6 {
                        for col in 0..6 {
                            cs[row * 6 + col] = bp.covariance[(row, col)];
                        }
                    }
                }
                Some(cov_arr.unbind())
            } else {
                None
            };
            (Some(pose_arr.unbind()), cov_opt)
        } else {
            (None, None)
        };

        // 6. Telemetry (populated only when debug_telemetry=True and batch has telemetry).
        let telemetry = if debug_telemetry && let Some(t) = self.telem_batch.telemetry.as_ref() {
            let r = t.count;
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            let rej_pts_arr = unsafe { PyArray2::<f32>::new(py, [r, 2], false) };
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            let rej_det_arr = unsafe { PyArray1::<f32>::new(py, [r], false) };
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            let rej_reason_arr = unsafe { PyArray1::<u8>::new(py, [r], false) };
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            unsafe {
                // SAFETY: Point2f is repr(C) [f32; 2]; flat reinterpretation is sound.
                let rpts = rej_pts_arr
                    .as_slice_mut()
                    .map_err(|e| PyRuntimeError::new_err(format!("rej_pts slice: {e}")))?;
                rpts.copy_from_slice(std::slice::from_raw_parts(
                    t.rejected_predictions.as_ptr().cast::<f32>(),
                    r * 2,
                ));
                let rdet = rej_det_arr
                    .as_slice_mut()
                    .map_err(|e| PyRuntimeError::new_err(format!("rej_det slice: {e}")))?;
                rdet.copy_from_slice(&t.rejected_determinants[..r]);
                let rreason = rej_reason_arr
                    .as_slice_mut()
                    .map_err(|e| PyRuntimeError::new_err(format!("rej_reason slice: {e}")))?;
                rreason.copy_from_slice(&t.rejected_reasons[..r]);
            }
            Some(Py::new(
                py,
                CharucoTelemetryResult {
                    rejected_saddles: rej_pts_arr.unbind(),
                    rejected_determinants: rej_det_arr.unbind(),
                    rejected_reasons: rej_reason_arr.unbind(),
                },
            )?)
        } else {
            None
        };

        Ok(CharucoEstimateResult {
            ids: ids_arr.unbind(),
            corners: corners_arr.unbind(),
            saddle_ids: saddle_ids_arr.unbind(),
            saddle_pts: saddle_pts_arr.unbind(),
            saddle_obj: saddle_obj_arr.unbind(),
            board_pose,
            board_cov,
            telemetry,
        })
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Copy a (possibly row-padded) image buffer into a contiguous `dst` slice.
///
/// # Safety
/// `src` must be valid for `height * stride` bytes and must not alias `dst`.
unsafe fn copy_strided_image(
    src: *const u8,
    dst: &mut [u8],
    height: usize,
    width: usize,
    stride: usize,
) {
    if stride == width {
        // SAFETY: caller guarantees src is valid for height*stride == dst.len() bytes.
        unsafe { std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len()) };
    } else {
        for y in 0..height {
            // SAFETY: y*stride < height*stride (valid by caller); y*width < dst.len().
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.add(y * stride),
                    dst.as_mut_ptr().add(y * width),
                    width,
                );
            }
        }
    }
}

/// Copies `n` elements from `ptr` into a freshly allocated NumPy 1-D array.
///
/// Returns `None` when the pointer is null or the count is zero — the common
/// shape for optional debug-telemetry columns. `label` appears in the error
/// message if NumPy refuses a mutable slice of the allocation.
///
/// # Safety
///
/// Caller must guarantee `ptr` is valid for `n * size_of::<T>()` bytes for the
/// duration of this function (see "NumPy allocation safety contract" in the
/// module docs).
unsafe fn optional_py_array1<T: numpy::Element + Copy>(
    py: Python<'_>,
    ptr: *const T,
    n: usize,
    label: &str,
) -> PyResult<Option<Py<PyArray1<T>>>> {
    if ptr.is_null() || n == 0 {
        return Ok(None);
    }
    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let arr = unsafe { PyArray1::<T>::new(py, [n], false) };
    // SAFETY: see caller-obligation note above.
    unsafe {
        arr.as_slice_mut()
            .map_err(|e| PyRuntimeError::new_err(format!("{label} slice: {e}")))?
            .copy_from_slice(std::slice::from_raw_parts(ptr, n));
    }
    Ok(Some(arr.unbind()))
}

fn build_pipeline_telemetry(
    py: Python<'_>,
    telem: &locus_core::TelemetryPayload,
) -> PyResult<PipelineTelemetryResult> {
    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let binarized_arr = unsafe { PyArray2::<u8>::new(py, [telem.height, telem.width], false) };
    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let threshold_arr = unsafe { PyArray2::<u8>::new(py, [telem.height, telem.width], false) };
    // SAFETY: Both pointers are valid for `height * stride` bytes (guaranteed by the Rust
    // arena that owns them for the lifetime of this `detect()` call).  The destination slices
    // are freshly allocated by PyArray2::new and exclusively owned here.  When stride > width
    // the source rows are padded, so we copy each row independently.
    unsafe {
        copy_strided_image(
            telem.binarized_ptr,
            binarized_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("binarized slice: {e}")))?,
            telem.height,
            telem.width,
            telem.stride,
        );
        copy_strided_image(
            telem.threshold_map_ptr,
            threshold_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("threshold slice: {e}")))?,
            telem.height,
            telem.width,
            telem.stride,
        );
    }

    let subpixel_jitter = if !telem.subpixel_jitter_ptr.is_null() && telem.num_jitter > 0 {
        let nj = telem.num_jitter;
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let arr = unsafe { PyArray3::<f32>::new(py, [nj, 4, 2], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        unsafe {
            arr.as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("jitter slice: {e}")))?
                .copy_from_slice(std::slice::from_raw_parts(
                    telem.subpixel_jitter_ptr,
                    nj * 8,
                ));
        }
        Some(arr.unbind())
    } else {
        None
    };

    // SAFETY: telemetry pointers and counts are populated by the Rust arena
    // that remains live for the duration of this call (see module docs).
    let reprojection_errors = unsafe {
        optional_py_array1(
            py,
            telem.reprojection_errors_ptr,
            telem.num_reprojection,
            "repro",
        )?
    };
    // SAFETY: routed_to_ptr is valid for `num_routed` u8 slots (DetectionBatch SoA contract).
    let routed_to =
        unsafe { optional_py_array1(py, telem.routed_to_ptr, telem.num_routed, "routed_to")? };
    // SAFETY: ppb_estimate_ptr is valid for `num_routed` f32 slots; same lifetime as routed_to.
    let ppb_estimate = unsafe {
        optional_py_array1(py, telem.ppb_estimate_ptr, telem.num_routed, "ppb_estimate")?
    };

    Ok(PipelineTelemetryResult {
        binarized: binarized_arr.unbind(),
        threshold_map: threshold_arr.unbind(),
        subpixel_jitter,
        reprojection_errors,
        gwlf_fallback_count: telem.gwlf_fallback_count,
        gwlf_avg_delta: telem.gwlf_avg_delta,
        routed_to,
        ppb_estimate,
    })
}

// Detector class
// ============================================================================

/// Wraps a raw `usize` so the address can cross the `py.detach()` `Send` boundary.
///
/// # Safety
///
/// The caller must ensure no concurrent access to the pointed-to data occurs
/// while this wrapper is alive outside the originating thread.
struct SendPtr(usize);

// SAFETY: `py.detach()` runs its closure synchronously on the *same OS thread*
// that holds the GIL. The pointer remains exclusively owned by that thread.
unsafe impl Send for SendPtr {}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(unsendable)]
pub struct Detector {
    inner: Box<locus_core::Detector>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl Detector {
    /// Detect tags and return a typed [`DetectionResult`] containing NumPy arrays.
    #[pyo3(signature = (img, intrinsics=None, tag_size=None, debug_telemetry=false))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "pyo3 #[pymethods] extraction requires owned PyReadonlyArray2/CameraIntrinsics arguments"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one cohesive detect FFI routine (run pipeline -> marshal detections/telemetry into NumPy arrays); splitting would fragment the buffer marshalling"
    )]
    fn detect(
        &mut self,
        py: Python<'_>,
        img: PyReadonlyArray2<'_, u8>,
        intrinsics: Option<CameraIntrinsics>,
        tag_size: Option<f64>,
        debug_telemetry: bool,
    ) -> PyResult<DetectionResult> {
        let buffer = prepare_image_view(&img)?;
        let view = buffer.view();

        if let Some(ref i) = intrinsics {
            validate_principal_point(i, view.width, view.height)?;
        }

        let has_intrinsics = intrinsics.is_some();
        let core_intrinsics = intrinsics.map(locus_core::CameraIntrinsics::from);

        // 1. Run core pipeline
        let detections = py
            .detach(|| {
                self.inner
                    .detect(&view, core_intrinsics.as_ref(), tag_size, debug_telemetry)
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let n = detections.len();

        // Allocate NumPy arrays
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let ids_arr = unsafe { PyArray1::<i32>::new(py, [n], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let corners_arr = unsafe { PyArray3::<f32>::new(py, [n, 4, 2], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let error_rates_arr = unsafe { PyArray1::<f32>::new(py, [n], false) };

        // Perform memory mapping
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        unsafe {
            let ids_slice = ids_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("ids slice: {e}")))?;
            let corners_slice = corners_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("corners slice: {e}")))?;
            let error_rates_slice = error_rates_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("error_rates slice: {e}")))?;

            // Direct memory block transfer (Zero-copy layout alignment)
            // LLVM will vectorize these into SIMD load/store instructions.
            ids_slice.copy_from_slice(std::slice::from_raw_parts(
                detections.ids.as_ptr().cast::<i32>(),
                n,
            ));
            corners_slice.copy_from_slice(std::slice::from_raw_parts(
                detections.corners.as_ptr().cast::<f32>(),
                n * 8,
            ));
            error_rates_slice.copy_from_slice(std::slice::from_raw_parts(
                detections.error_rates.as_ptr(),
                n,
            ));
        }

        // Rejected Quads: (M, 4, 2)
        let m = detections.rejected_corners.len();
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let rejected_arr = unsafe { PyArray3::<f32>::new(py, [m, 4, 2], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        unsafe {
            let rejected_slice = rejected_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("rejected slice: {e}")))?;
            rejected_slice.copy_from_slice(std::slice::from_raw_parts(
                detections.rejected_corners.as_ptr().cast::<f32>(),
                m * 8,
            ));
        }

        // Rejected Error Rates: (M,)
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let rejected_error_rates_arr = unsafe { PyArray1::<f32>::new(py, [m], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        unsafe {
            let rejected_error_rates_slice = rejected_error_rates_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("rejected error rates slice: {e}")))?;
            rejected_error_rates_slice.copy_from_slice(std::slice::from_raw_parts(
                detections.rejected_error_rates.as_ptr(),
                m,
            ));
        }

        // Rejected Funnel Status: (M,) — `FunnelStatus` is `#[repr(u8)]`,
        // so a pointer-cast reinterpret is sound and zero-copy.
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let rejected_funnel_status_arr = unsafe { PyArray1::<u8>::new(py, [m], false) };
        // SAFETY: `FunnelStatus` has `#[repr(u8)]` (see `locus_core::batch::FunnelStatus`),
        // so reading its bytes as `u8` is well-defined. The slice length matches `m` by
        // construction in `view_with_telemetry`.
        unsafe {
            let rejected_funnel_status_slice =
                rejected_funnel_status_arr.as_slice_mut().map_err(|e| {
                    PyRuntimeError::new_err(format!("rejected funnel status slice: {e}"))
                })?;
            rejected_funnel_status_slice.copy_from_slice(std::slice::from_raw_parts(
                detections.rejected_funnel_status.as_ptr().cast::<u8>(),
                m,
            ));
        }

        // Poses: Vectorized (N, 7) layout: [tx, ty, tz, qx, qy, qz, qw]
        let poses = if has_intrinsics && tag_size.is_some() {
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            let poses_arr = unsafe { PyArray2::<f32>::new(py, [n, 7], false) };
            // SAFETY: see "NumPy allocation safety contract" in module docs.
            unsafe {
                let poses_slice = poses_arr
                    .as_slice_mut()
                    .map_err(|e| PyRuntimeError::new_err(format!("poses slice: {e}")))?;

                // Optimised block copy for Pose6D (ignoring f32 padding)
                for (i, pose) in detections.poses.iter().enumerate() {
                    poses_slice[i * 7..(i + 1) * 7].copy_from_slice(&pose.data);
                }
            }
            Some(poses_arr.unbind())
        } else {
            None
        };

        // Telemetry (zero-copy intermediate images)
        let telemetry = if let Some(telem) = detections.telemetry {
            let telem_result = build_pipeline_telemetry(py, &telem)?;
            Some(Py::new(py, telem_result)?)
        } else {
            None
        };

        Ok(DetectionResult {
            ids: ids_arr.unbind(),
            corners: corners_arr.unbind(),
            error_rates: error_rates_arr.unbind(),
            poses,
            rejected_corners: rejected_arr.unbind(),
            rejected_error_rates: rejected_error_rates_arr.unbind(),
            rejected_funnel_status: rejected_funnel_status_arr.unbind(),
            telemetry,
        })
    }

    /// Detect tags in multiple frames concurrently using Rayon.
    ///
    /// Releases the Python GIL for the entire parallel section. The detector
    /// leases [`FrameContext`] objects from its internal pool; the pool size
    /// was set via `max_concurrent_frames` at construction time.
    ///
    /// Telemetry and rejected-corner data are not available via this method.
    #[pyo3(signature = (frames, intrinsics=None, tag_size=None))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "pyo3 #[pymethods] extraction requires owned Vec<PyReadonlyArray2>/CameraIntrinsics arguments"
    )]
    fn detect_concurrent(
        &self,
        py: Python<'_>,
        frames: Vec<PyReadonlyArray2<'_, u8>>,
        intrinsics: Option<CameraIntrinsics>,
        tag_size: Option<f64>,
    ) -> PyResult<Vec<DetectionResult>> {
        // Prepare image buffers while GIL is held (PyReadonlyArray2 is GIL-bound).
        // Each `FfiImageBuffer` is either a zero-copy borrow of the NumPy
        // buffer (fast path, when stride_y > width + 3) or an owned padded
        // copy (fallback for tightly-packed inputs — see `prepare_image_view`).
        let buffers: Vec<FfiImageBuffer<'_>> = frames
            .iter()
            .map(prepare_image_view)
            .collect::<PyResult<_>>()?;
        // `views` borrows from `buffers`; `buffers` must outlive the detach
        // call below. The compiler already enforces this — leaving the
        // comment as a reminder for the next reader.
        let views: Vec<ImageView<'_>> = buffers.iter().map(FfiImageBuffer::view).collect();

        if let Some(ref i) = intrinsics {
            for view in &views {
                validate_principal_point(i, view.width, view.height)?;
            }
        }

        let has_pose = intrinsics.is_some() && tag_size.is_some();
        let core_intrinsics = intrinsics.map(locus_core::CameraIntrinsics::from);

        // Extract a raw pointer to the engine (which is Send+Sync) before releasing
        // the GIL. `self.inner` is pinned on the heap (Box) and kept alive by `self`.
        //
        // SAFETY: `Detector` is `#[pyclass(unsendable)]` — Python guarantees no
        // concurrent Python-level access. `py.detach()` runs synchronously on the
        // same OS thread. The pointer remains valid for the duration of this call.
        // SAFETY: `addr_of!` avoids the "reference as raw pointer" lint by not
        // creating an intermediate `&T` before casting to `*const T`.
        let ptr = SendPtr(std::ptr::addr_of!(*self.inner.engine()) as usize);

        let results: Vec<Result<Vec<locus_core::Detection>, locus_core::DetectorError>> = py
            .detach(move || {
                // SAFETY: LocusEngine is thread-safe (Send+Sync) and kept alive by the Detector.
                let engine = unsafe { &*(ptr.0 as *const locus_core::LocusEngine) };
                engine.detect_concurrent(&views, core_intrinsics.as_ref(), tag_size)
            });

        results
            .into_iter()
            .map(|r| {
                let dets = r.map_err(|e| PyValueError::new_err(e.to_string()))?;
                build_detection_result_from_owned(py, &dets, has_pose)
            })
            .collect()
    }

    /// Returns the current detector configuration as a profile-JSON string.
    ///
    /// Python re-parses this into the Pydantic `DetectorConfig`, so the readback
    /// is total over every profile field (the former field-by-field FFI struct
    /// silently dropped knobs it did not re-copy).
    fn config(&self) -> PyResult<String> {
        self.inner
            .config()
            .to_profile_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Update the tag families to be detected.
    fn set_families(&mut self, families: Vec<i32>) -> PyResult<()> {
        let core_families = families
            .into_iter()
            .map(tag_family_from_i32)
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.set_families(&core_families);
        Ok(())
    }
}

fn tag_family_from_i32(f: i32) -> PyResult<locus_core::TagFamily> {
    match f {
        0 => Ok(locus_core::TagFamily::AprilTag16h5),
        1 => Ok(locus_core::TagFamily::AprilTag36h11),
        2 => Ok(locus_core::TagFamily::ArUco4x4_50),
        3 => Ok(locus_core::TagFamily::ArUco4x4_100),
        4 => Ok(locus_core::TagFamily::ArUco6x6_250),
        _ => Err(PyValueError::new_err(format!(
            "Invalid TagFamily value: {f}"
        ))),
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (config_json, decimation=None, threads=None, families=vec![]))]
fn _create_detector_from_config(
    config_json: &str,
    decimation: Option<usize>,
    threads: Option<usize>,
    families: Vec<i32>,
) -> PyResult<Detector> {
    // The Pydantic `DetectorConfig` crosses the FFI as its `model_dump_json()`
    // string — the same profile format Rust already reads — so the shipped JSON
    // profile is the single source of truth on both sides of the boundary.
    let cfg = locus_core::config::DetectorConfig::from_profile_json(config_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut builder = locus_core::DetectorBuilder::new().with_config(cfg);

    if let Some(d) = decimation {
        builder = builder.with_decimation(d);
    }
    if let Some(t) = threads {
        builder = builder.with_threads(t);
    }
    for f in families {
        builder = builder.with_family(tag_family_from_i32(f)?);
    }

    let detector = builder
        .validated_build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Detector {
        inner: Box::new(detector),
    })
}

// ============================================================================
// DetectorBuilder
// ============================================================================

/// Fluent builder for constructing a [`Detector`].
///
/// Methods return `self` so they can be chained in Python:
/// ```python
/// detector = (
///     locus.DetectorBuilder()
///         .with_decimation(2)
///         .with_family(locus.TagFamily.AprilTag36h11)
///         .with_corner_refinement(locus.CornerRefinementMode.Gwlf)
///         .build()
/// )
/// ```
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
pub struct DetectorBuilder {
    inner: Option<locus_core::DetectorBuilder>,
}

impl DetectorBuilder {
    fn take_inner(slf: &Py<Self>, py: Python<'_>) -> PyResult<locus_core::DetectorBuilder> {
        slf.borrow_mut(py).inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("DetectorBuilder has already been consumed by build()")
        })
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl DetectorBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(locus_core::DetectorBuilder::new()),
        }
    }

    fn with_decimation(slf: Py<Self>, py: Python<'_>, decimation: usize) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_decimation(decimation);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_threads(slf: Py<Self>, py: Python<'_>, threads: usize) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_threads(threads);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_family(slf: Py<Self>, py: Python<'_>, family: TagFamily) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_family(family.into());
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_upscale_factor(slf: Py<Self>, py: Python<'_>, factor: usize) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_upscale_factor(factor);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_corner_refinement(
        slf: Py<Self>,
        py: Python<'_>,
        mode: CornerRefinementMode,
    ) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_corner_refinement(mode.into());
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_connectivity(
        slf: Py<Self>,
        py: Python<'_>,
        connectivity: SegmentationConnectivity,
    ) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_connectivity(connectivity.into());
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_threshold_tile_size(slf: Py<Self>, py: Python<'_>, size: usize) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_threshold_tile_size(size);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_threshold_min_range(slf: Py<Self>, py: Python<'_>, range: u8) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_threshold_min_range(range);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_adaptive_threshold_constant(
        slf: Py<Self>,
        py: Python<'_>,
        c: i16,
    ) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_adaptive_threshold_constant(c);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_quad_min_area(slf: Py<Self>, py: Python<'_>, area: u32) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_quad_min_area(area);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_quad_min_fill_ratio(slf: Py<Self>, py: Python<'_>, ratio: f32) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_quad_min_fill_ratio(ratio);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_quad_min_edge_score(slf: Py<Self>, py: Python<'_>, score: f64) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_quad_min_edge_score(score);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_max_hamming_error(slf: Py<Self>, py: Python<'_>, errors: u32) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_max_hamming_error(errors);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_decoder_min_contrast(
        slf: Py<Self>,
        py: Python<'_>,
        contrast: f64,
    ) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_decoder_min_contrast(contrast);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_gwlf_transversal_alpha(
        slf: Py<Self>,
        py: Python<'_>,
        alpha: f64,
    ) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_gwlf_transversal_alpha(alpha);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_quad_max_elongation(
        slf: Py<Self>,
        py: Python<'_>,
        elongation: f64,
    ) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_quad_max_elongation(elongation);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_quad_min_density(slf: Py<Self>, py: Python<'_>, density: f64) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_quad_min_density(density);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_quad_extraction_mode(
        slf: Py<Self>,
        py: Python<'_>,
        mode: QuadExtractionMode,
    ) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_quad_extraction_mode(mode.into());
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_sharpening(slf: Py<Self>, py: Python<'_>, enable: bool) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_sharpening(enable);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    fn with_max_concurrent_frames(slf: Py<Self>, py: Python<'_>, n: usize) -> PyResult<Py<Self>> {
        let b = Self::take_inner(&slf, py)?.with_max_concurrent_frames(n);
        slf.borrow_mut(py).inner = Some(b);
        Ok(slf)
    }

    /// Consume the builder and return a ready-to-use [`Detector`].
    fn build(&mut self) -> PyResult<Detector> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("DetectorBuilder has already been consumed"))?;
        let detector = inner
            .validated_build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Detector {
            inner: Box::new(detector),
        })
    }
}

fn validate_principal_point(
    intrinsics: &CameraIntrinsics,
    width: usize,
    height: usize,
) -> PyResult<()> {
    if intrinsics.cx < 0.0 || intrinsics.cx >= width as f64 {
        return Err(PyValueError::new_err(format!(
            "CameraIntrinsics.cx ({cx}) must lie in [0, {width}) for a {width}x{height} image",
            cx = intrinsics.cx,
        )));
    }
    if intrinsics.cy < 0.0 || intrinsics.cy >= height as f64 {
        return Err(PyValueError::new_err(format!(
            "CameraIntrinsics.cy ({cy}) must lie in [0, {height}) for a {width}x{height} image",
            cy = intrinsics.cy,
        )));
    }
    Ok(())
}

/// FFI-layer image buffer that hides whether `detect()` borrows zero-copy
/// from the NumPy buffer or has materialised a padded copy.
///
/// Why this exists: AVX2 `sample_bilinear_v8` (and its NEON fallback) issues
/// 32-bit gathers on 8-bit data and may read up to 3 bytes past the last
/// logical pixel — see the SAFETY block at `crates/locus-core/src/decoder.rs`
/// and the runtime gate in `crates/locus-core/src/simd/sampler.rs`. A
/// tightly-packed `np.zeros((H, W), dtype=np.uint8)` NumPy buffer carries no
/// such trailing slack, so the FFI either:
///   * takes the **fast path** (`Borrowed`) when the input already exposes
///     ≥3 trailing bytes (e.g. a column-prefix view from a wider parent
///     allocation), or
///   * takes the **fallback path** (`Padded`) by copying `H × W` bytes into
///     a tightly-packed scratch `Vec<u8>` over-allocated by 3 trailing
///     guard bytes.
///
/// The fallback violates the otherwise-strict "no copies at the FFI" rule
/// from `docs/engineering/constraints.md` §2, but it is a one-shot
/// per-`detect()` cost that preserves the pre-PR-#287 public API (every
/// `np.zeros((H, W))` call site continues to work transparently). PRs that
/// touched this fallback should reconsider the trade-off if frame copy
/// becomes a measurable cost.
enum FfiImageBuffer<'py> {
    /// Zero-copy view over the NumPy buffer. Used when the input has
    /// stride_y > width (≥3 trailing pad bytes per row).
    Borrowed {
        data: &'py [u8],
        width: usize,
        height: usize,
        stride: usize,
    },
    /// Owned padded scratch. The `storage` is a tightly-packed
    /// `width × height + 3` buffer; the leading `width × height` bytes hold
    /// the row-by-row copy of the input, the trailing 3 bytes are guard
    /// padding for SIMD gather.
    Padded {
        storage: Vec<u8>,
        width: usize,
        height: usize,
    },
}

impl FfiImageBuffer<'_> {
    /// Build an `ImageView` that borrows from `self`. The lifetime of the
    /// returned view is tied to `&self`, so callers must keep the buffer
    /// alive for as long as the view is used.
    fn view(&self) -> ImageView<'_> {
        match self {
            FfiImageBuffer::Borrowed {
                data,
                width,
                height,
                stride,
            } => {
                // SAFETY-equivalent: `data` was validated by `prepare_image_view`
                // (stride >= width, length >= height * stride). `ImageView::new`
                // re-validates the size invariants; on the borrowed path we
                // could even call `new_unchecked`, but we keep this honest.
                ImageView {
                    data,
                    width: *width,
                    height: *height,
                    stride: *stride,
                }
            },
            FfiImageBuffer::Padded {
                storage,
                width,
                height,
            } => {
                // For the copy path the layout is tightly packed with
                // `stride == width`, plus 3 trailing guard bytes in `storage`
                // that satisfy `ImageView::has_simd_padding()`.
                ImageView {
                    data: storage,
                    width: *width,
                    height: *height,
                    stride: *width,
                }
            },
        }
    }
}

fn prepare_image_view<'py>(img: &'py PyReadonlyArray2<'_, u8>) -> PyResult<FfiImageBuffer<'py>> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let strides = img.strides();
    let stride_x = strides[1];

    if stride_x != 1 {
        return Err(PyValueError::new_err(
            "Array must be C-contiguous. Call np.ascontiguousarray(image) first.",
        ));
    }

    // F2 (kept): reject negative or otherwise-invalid row strides before the
    // unsigned cast. NumPy yields `strides[0] < 0` for reverse-axis views
    // (e.g. `arr[::-1, :]`); `.cast_unsigned()` would wrap to a huge value
    // and the subsequent `from_raw_parts` would build a giant slice → UB.
    // `stride < width` is independently caught by `ImageView::new`, but we
    // surface a typed `PyValueError` at the FFI boundary instead of a
    // wrapped `PyRuntimeError`.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "image width is bounded by the NumPy array dimensions, well within isize"
    )]
    let width_isize = width as isize;
    if strides[0] < width_isize {
        return Err(PyValueError::new_err(format!(
            "Array row stride ({stride}) must be >= width ({width}); negative \
             strides (e.g. from `arr[::-1, :]`) are not C-contiguous. Call \
             np.ascontiguousarray(image) first.",
            stride = strides[0],
        )));
    }

    let stride_y = strides[0].cast_unsigned();

    // Fast-path detection. Cover the full row pitch `(H * stride_y)` so that
    // when `stride_y > width` (e.g. a `parent[:, :W]` slice from a wider
    // parent allocation), the trailing `(stride_y - width)` bytes of pad per
    // row are exposed and `ImageView::has_simd_padding()` can pass without
    // forcing a copy.
    let slice_len = if height > 0 && width > 0 {
        height * stride_y
    } else {
        0
    };
    // SAFETY: The returned `FfiImageBuffer<'py>::Borrowed` (and its
    // `view()`) borrow from `img` via the explicit `'py` lifetime tie on
    // the signature, so the slice cannot outlive the NumPy buffer. We
    // expose at most `height * stride_y` bytes from `img.data()`:
    //   * `stride_y` is validated above to be `>= width` (and was originally
    //     a non-negative `isize` from NumPy, so the unsigned cast is exact).
    //   * For C-contiguous arrays (`stride_y == width`), the slice length
    //     equals `(H-1)*stride_y + W = H*W`, which is the full allocation.
    //   * For strided views from a wider parent (the common
    //     `np.pad(...)[:, :W]` pattern), `arr.data() == parent.data()` and
    //     the parent has at least `H * stride_y` bytes — the slice stays
    //     within the parent allocation.
    //   * A user creating a column-offset view (`parent[:, c:c+W]` with
    //     `c > 0`) would have `arr.data() == parent.data() + c`, making the
    //     last `c` bytes of the slice unsafe. The FFI contract documented in
    //     `docs/engineering/ffi_contracts.md` §1 forbids such views — callers
    //     should pad with `np.pad(img, ((0, 0), (0, 3)))[:, :W]` instead.
    let data = unsafe { std::slice::from_raw_parts(img.data(), slice_len) };
    let view = ImageView::new(data, width, height, stride_y)
        .map_err(|e| PyRuntimeError::new_err(e.clone()))?;

    // F1 (redesigned): SIMD kernels (e.g. AVX2 `sample_bilinear_v8` in
    // `decoder.rs`) issue 32-bit gathers on 8-bit data and may read up to
    // 3 bytes past the last logical pixel. If the input has ≥3 trailing
    // bytes of slack (`has_simd_padding()`), return the zero-copy borrow.
    // Otherwise, transparently copy into a padded scratch — this preserves
    // the pre-PR public API for the natural NumPy idiom
    // `np.zeros((H, W), dtype=np.uint8)`, at the cost of one `H * W`-byte
    // allocation + copy per `detect()` call.
    //
    // Design rationale: PR #287's initial design rejected under-padded
    // buffers with a `PyValueError`. The project owner chose the copy
    // fallback over the reject after the xfail-comment menu at
    // `tests/test_ffi_contracts.py:115-122` was reconsidered. See the
    // CHANGELOG entry and `docs/engineering/ffi_contracts.md` §1 for the
    // full record.
    if view.has_simd_padding() {
        return Ok(FfiImageBuffer::Borrowed {
            data,
            width,
            height,
            stride: stride_y,
        });
    }

    // Copy path. Repack into a tight (stride == width) buffer with 3
    // trailing guard bytes that satisfy `has_simd_padding()`. We do not
    // try to preserve the source `stride_y` because the tight layout makes
    // downstream cache access more predictable and the source stride
    // information is irrelevant once we own the bytes.
    let logical = width.checked_mul(height).ok_or_else(|| {
        PyValueError::new_err(format!("Image dimensions {width}x{height} overflow usize",))
    })?;
    let storage_len = logical.checked_add(3).ok_or_else(|| {
        PyValueError::new_err(format!("Image dimensions {width}x{height} overflow usize",))
    })?;
    let mut storage = vec![0u8; storage_len];
    // Copy row by row to handle source `stride_y >= width` correctly. The
    // `Borrowed` fast path already covered `stride_y >= width + 3`; this
    // branch only fires for `stride_y ∈ {width, width + 1, width + 2}`.
    for y in 0..height {
        let src_row = view.get_row(y);
        let dst_off = y * width;
        storage[dst_off..dst_off + width].copy_from_slice(src_row);
    }
    // The trailing 3 bytes of `storage` are already zeroed by `vec![0u8; …]`,
    // which is what `has_simd_padding()` checks.

    Ok(FfiImageBuffer::Padded {
        storage,
        width,
        height,
    })
}

/// Convert an owned [`Vec<locus_core::Detection>`] to a [`DetectionResult`] with NumPy arrays.
///
/// Used by [`Detector::detect_concurrent`] where the detection pipeline runs
/// in a GIL-free closure and returns owned data.
///
/// Rejected corners are not available via this path (only valid detections are returned).
#[allow(clippy::unnecessary_wraps)] // PyResult is required by callers that chain with `?`
fn build_detection_result_from_owned(
    py: Python<'_>,
    detections: &[locus_core::Detection],
    has_pose: bool,
) -> PyResult<DetectionResult> {
    let n = detections.len();

    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let ids_arr = unsafe { PyArray1::<i32>::new(py, [n], false) };
    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let corners_arr = unsafe { PyArray3::<f32>::new(py, [n, 4, 2], false) };
    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let error_rates_arr = unsafe { PyArray1::<f32>::new(py, [n], false) };

    // SAFETY: see "NumPy allocation safety contract" in module docs.
    unsafe {
        let ids_sl = ids_arr
            .as_slice_mut()
            .map_err(|e| PyRuntimeError::new_err(format!("ids: {e}")))?;
        let corners_sl = corners_arr
            .as_slice_mut()
            .map_err(|e| PyRuntimeError::new_err(format!("corners: {e}")))?;
        let error_sl = error_rates_arr
            .as_slice_mut()
            .map_err(|e| PyRuntimeError::new_err(format!("error_rates: {e}")))?;

        for (i, det) in detections.iter().enumerate() {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "decoded tag IDs index a finite family codebook (thousands of entries), far below i32::MAX"
            )]
            {
                ids_sl[i] = det.id as i32;
            }
            error_sl[i] = det.hamming as f32;
            for (j, corner) in det.corners.iter().enumerate() {
                corners_sl[i * 8 + j * 2] = corner[0] as f32;
                corners_sl[i * 8 + j * 2 + 1] = corner[1] as f32;
            }
        }
    }

    let poses = if has_pose {
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        let poses_arr = unsafe { PyArray2::<f32>::new(py, [n, 7], false) };
        // SAFETY: see "NumPy allocation safety contract" in module docs.
        unsafe {
            let poses_sl = poses_arr
                .as_slice_mut()
                .map_err(|e| PyRuntimeError::new_err(format!("poses: {e}")))?;
            for (i, det) in detections.iter().enumerate() {
                if let Some(ref pose) = det.pose {
                    // Reconstruct [tx, ty, tz, qx, qy, qz, qw] from Pose struct.
                    let r = nalgebra::Rotation3::from_matrix_unchecked(pose.rotation);
                    let q = nalgebra::UnitQuaternion::from_rotation_matrix(&r);
                    poses_sl[i * 7] = pose.translation[0] as f32;
                    poses_sl[i * 7 + 1] = pose.translation[1] as f32;
                    poses_sl[i * 7 + 2] = pose.translation[2] as f32;
                    poses_sl[i * 7 + 3] = q.i as f32;
                    poses_sl[i * 7 + 4] = q.j as f32;
                    poses_sl[i * 7 + 5] = q.k as f32;
                    poses_sl[i * 7 + 6] = q.w as f32;
                }
            }
        }
        Some(poses_arr.unbind())
    } else {
        None
    };

    // Rejected corners are not available on the owned path.
    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let rejected_arr = unsafe { PyArray3::<f32>::new(py, [0, 4, 2], false) };
    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let rejected_error_rates_arr = unsafe { PyArray1::<f32>::new(py, [0], false) };
    // SAFETY: see "NumPy allocation safety contract" in module docs.
    let rejected_funnel_status_arr = unsafe { PyArray1::<u8>::new(py, [0], false) };

    Ok(DetectionResult {
        ids: ids_arr.unbind(),
        corners: corners_arr.unbind(),
        error_rates: error_rates_arr.unbind(),
        poses,
        rejected_corners: rejected_arr.unbind(),
        rejected_error_rates: rejected_error_rates_arr.unbind(),
        rejected_funnel_status: rejected_funnel_status_arr.unbind(),
        telemetry: None,
    })
}

// ============================================================================
// Profiling
// ============================================================================

#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
fn init_tracy() {
    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt;
        // SAFETY: tracy initialization is thread-safe.
        unsafe {
            std::env::set_var("TRACY_NO_INVARIANT_CHECK", "1");
        }
        let subscriber = tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default());
        tracing::subscriber::set_global_default(subscriber).ok();
    }
}

#[cfg(feature = "profiles")]
#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction)]
#[pyfunction]
fn _shipped_profile_json(name: &str) -> PyResult<&'static str> {
    locus_core::config::shipped_profile_json(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unknown shipped profile {name:?}; expected one of \
             ['standard', 'grid', 'high_accuracy']"
        ))
    })
}

#[pymodule]
fn locus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Enums
    m.add_class::<TagFamily>()?;
    m.add_class::<SegmentationConnectivity>()?;
    m.add_class::<CornerRefinementMode>()?;
    m.add_class::<QuadExtractionMode>()?;
    m.add_class::<EdLinesImbalanceGatePolicy>()?;
    // Config / misc structs
    m.add_class::<DistortionModel>()?;
    m.add_class::<CameraIntrinsics>()?;
    m.add_class::<PyPose>()?;
    // Result types
    m.add_class::<PipelineTelemetryResult>()?;
    m.add_class::<DetectionResult>()?;
    m.add_class::<CharucoTelemetryResult>()?;
    m.add_class::<CharucoEstimateResult>()?;
    m.add_class::<BoardEstimateResult>()?;
    // Board topology
    m.add_class::<CharucoBoard>()?;
    m.add_class::<AprilGrid>()?;
    m.add_class::<BoardEstimator>()?;
    m.add_class::<CharucoRefiner>()?;
    // Detector
    m.add_class::<Detector>()?;
    m.add_class::<DetectorBuilder>()?;

    m.add_function(wrap_pyfunction!(_create_detector_from_config, m)?)?;
    m.add_function(wrap_pyfunction!(init_tracy, m)?)?;
    #[cfg(feature = "profiles")]
    m.add_function(wrap_pyfunction!(_shipped_profile_json, m)?)?;
    Ok(())
}

/// `&mut Detector` appears as a `#[pymethods]` argument (board estimation
/// mutably borrows the detector to run `detect`). pyo3-stub-gen only implements
/// `PyStubType` for `&T`, so teach it that `&mut Detector` types the same as
/// `Detector` in a stub. `&mut` is a fundamental type, so this local impl is
/// coherent. Lets the arguments generate without a per-parameter override
/// (which does not compose with the `stub-gen` feature gate via `cfg_attr`).
#[cfg(feature = "stub-gen")]
impl pyo3_stub_gen::PyStubType for &mut Detector {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        <Detector as pyo3_stub_gen::PyStubType>::type_output()
    }
}

#[cfg(feature = "stub-gen")]
/// Gather stub info from the workspace-root `pyproject.toml`. This crate ships
/// no `pyproject.toml` of its own (maturin config lives at the repo root), so we
/// cannot use `define_stub_info_gatherer!` (it hardcodes the crate-dir path).
/// Must live in the library crate, not the `stub_gen` binary — `inventory`
/// collection only sees items linked into the same crate.
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    pyo3_stub_gen::StubInfo::from_pyproject_toml(manifest_dir.join("../../pyproject.toml"))
}
