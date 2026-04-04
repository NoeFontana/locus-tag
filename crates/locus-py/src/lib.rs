//! Python bindings for the Locus Tag library.
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
use pyo3::types::PyDict;

// ============================================================================
// Enums
// ============================================================================

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

#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CornerRefinementMode {
    None = 0,
    Edge = 1,
    GridFit = 2,
    Erf = 3,
    Gwlf = 4,
}

impl From<CornerRefinementMode> for locus_core::config::CornerRefinementMode {
    fn from(m: CornerRefinementMode) -> Self {
        match m {
            CornerRefinementMode::None => locus_core::config::CornerRefinementMode::None,
            CornerRefinementMode::Edge => locus_core::config::CornerRefinementMode::Edge,
            CornerRefinementMode::GridFit => locus_core::config::CornerRefinementMode::GridFit,
            CornerRefinementMode::Erf => locus_core::config::CornerRefinementMode::Erf,
            CornerRefinementMode::Gwlf => locus_core::config::CornerRefinementMode::Gwlf,
        }
    }
}

#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DecodeMode {
    Hard = 0,
    Soft = 1,
}

impl From<DecodeMode> for locus_core::config::DecodeMode {
    fn from(m: DecodeMode) -> Self {
        match m {
            DecodeMode::Hard => locus_core::config::DecodeMode::Hard,
            DecodeMode::Soft => locus_core::config::DecodeMode::Soft,
        }
    }
}

#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PoseEstimationMode {
    Fast = 0,
    Accurate = 1,
}

impl From<PoseEstimationMode> for locus_core::config::PoseEstimationMode {
    fn from(m: PoseEstimationMode) -> Self {
        match m {
            PoseEstimationMode::Fast => locus_core::config::PoseEstimationMode::Fast,
            PoseEstimationMode::Accurate => locus_core::config::PoseEstimationMode::Accurate,
        }
    }
}

// ============================================================================
// Structs
// ============================================================================

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
pub struct CameraIntrinsics {
    #[pyo3(get, set)]
    pub fx: f64,
    #[pyo3(get, set)]
    pub fy: f64,
    #[pyo3(get, set)]
    pub cx: f64,
    #[pyo3(get, set)]
    pub cy: f64,
}

#[pymethods]
impl CameraIntrinsics {
    #[new]
    fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }
}

impl From<CameraIntrinsics> for locus_core::CameraIntrinsics {
    fn from(c: CameraIntrinsics) -> Self {
        Self::new(c.fx, c.fy, c.cx, c.cy)
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct PyPose {
    #[pyo3(get)]
    pub quaternion: [f64; 4], // x, y, z, w
    #[pyo3(get)]
    pub translation: [f64; 3],
}

// ============================================================================
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyDetectorConfig {
    pub threshold_tile_size: usize,
    pub threshold_min_range: u8,
    pub enable_bilateral: bool,
    pub bilateral_sigma_space: f32,
    pub bilateral_sigma_color: f32,
    pub enable_sharpening: bool,
    pub enable_adaptive_window: bool,
    pub threshold_min_radius: usize,
    pub threshold_max_radius: usize,
    pub adaptive_threshold_constant: i16,
    pub adaptive_threshold_gradient_threshold: u8,
    pub quad_min_area: u32,
    pub quad_max_aspect_ratio: f32,
    pub quad_min_fill_ratio: f32,
    pub quad_max_fill_ratio: f32,
    pub quad_min_edge_length: f64,
    pub quad_min_edge_score: f64,
    pub subpixel_refinement_sigma: f64,
    pub segmentation_margin: i16,
    pub segmentation_connectivity: SegmentationConnectivity,
    pub upscale_factor: usize,
    pub decoder_min_contrast: f64,
    pub refinement_mode: CornerRefinementMode,
    pub decode_mode: DecodeMode,
    pub max_hamming_error: u32,
    pub gwlf_transversal_alpha: f64,
    pub quad_max_elongation: f64,
    pub quad_min_density: f64,
    pub quad_extraction_mode: i32,
}

impl From<locus_core::config::DetectorConfig> for PyDetectorConfig {
    fn from(c: locus_core::config::DetectorConfig) -> Self {
        Self {
            threshold_tile_size: c.threshold_tile_size,
            threshold_min_range: c.threshold_min_range,
            enable_bilateral: c.enable_bilateral,
            bilateral_sigma_space: c.bilateral_sigma_space,
            bilateral_sigma_color: c.bilateral_sigma_color,
            enable_sharpening: c.enable_sharpening,
            enable_adaptive_window: c.enable_adaptive_window,
            threshold_min_radius: c.threshold_min_radius,
            threshold_max_radius: c.threshold_max_radius,
            adaptive_threshold_constant: c.adaptive_threshold_constant,
            adaptive_threshold_gradient_threshold: c.adaptive_threshold_gradient_threshold,
            quad_min_area: c.quad_min_area,
            quad_max_aspect_ratio: c.quad_max_aspect_ratio,
            quad_min_fill_ratio: c.quad_min_fill_ratio,
            quad_max_fill_ratio: c.quad_max_fill_ratio,
            quad_min_edge_length: c.quad_min_edge_length,
            quad_min_edge_score: c.quad_min_edge_score,
            subpixel_refinement_sigma: c.subpixel_refinement_sigma,
            segmentation_margin: c.segmentation_margin,
            segmentation_connectivity: match c.segmentation_connectivity {
                locus_core::config::SegmentationConnectivity::Four => {
                    SegmentationConnectivity::Four
                },
                locus_core::config::SegmentationConnectivity::Eight => {
                    SegmentationConnectivity::Eight
                },
            },
            upscale_factor: c.upscale_factor,
            decoder_min_contrast: c.decoder_min_contrast,
            refinement_mode: match c.refinement_mode {
                locus_core::config::CornerRefinementMode::None => CornerRefinementMode::None,
                locus_core::config::CornerRefinementMode::Edge => CornerRefinementMode::Edge,
                locus_core::config::CornerRefinementMode::GridFit => CornerRefinementMode::GridFit,
                locus_core::config::CornerRefinementMode::Erf => CornerRefinementMode::Erf,
                locus_core::config::CornerRefinementMode::Gwlf => CornerRefinementMode::Gwlf,
            },
            decode_mode: match c.decode_mode {
                locus_core::config::DecodeMode::Hard => DecodeMode::Hard,
                locus_core::config::DecodeMode::Soft => DecodeMode::Soft,
            },
            max_hamming_error: c.max_hamming_error,
            gwlf_transversal_alpha: c.gwlf_transversal_alpha,
            quad_max_elongation: c.quad_max_elongation,
            quad_min_density: c.quad_min_density,
            quad_extraction_mode: match c.quad_extraction_mode {
                locus_core::config::QuadExtractionMode::ContourRdp => 0,
                locus_core::config::QuadExtractionMode::EdLines => 1,
            },
        }
    }
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
#[pyclass]
pub struct CharucoBoard {
    pub(crate) inner: std::sync::Arc<locus_core::board::CharucoTopology>,
}

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
#[pyclass]
pub struct AprilGrid {
    pub(crate) inner: std::sync::Arc<locus_core::board::AprilGridTopology>,
}

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
// CharucoRefiner
// ============================================================================

/// Extracts ChAruco saddle points from decoded ArUco detections and estimates
/// the board pose via LO-RANSAC + Anisotropic Weighted Levenberg–Marquardt.
///
/// Reuse a single `CharucoRefiner` across frames to amortise the one-time
/// scratch-buffer allocation.
#[pyclass(unsendable)]
pub struct CharucoRefiner {
    inner: locus_core::charuco::CharucoRefiner,
    /// Production output buffer — no telemetry overhead.
    batch: locus_core::charuco::CharucoBatch,
    /// Debug output buffer — telemetry pre-allocated once.
    telem_batch: locus_core::charuco::CharucoBatch,
}

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
    /// - `board_cov`:   `(6, 6) float64` pose covariance or `None`
    /// - `telemetry`:   dict with `"rejected_saddles"` `(R,2) float32` and
    ///                  `"rejected_determinants"` `(R,) float32`, or `None`
    ///                  (only populated when `debug_telemetry=True`)
    #[pyo3(signature = (detector, img, intrinsics, debug_telemetry = false))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    fn estimate<'py>(
        &mut self,
        py: Python<'py>,
        detector: &mut Detector,
        img: PyReadonlyArray2<'_, u8>,
        intrinsics: CameraIntrinsics,
        debug_telemetry: bool,
    ) -> Result<Bound<'py, PyDict>, PyErr> {
        let view = prepare_image_view(&img)?;
        let core_intr = locus_core::CameraIntrinsics::from(intrinsics);

        // 1. Run ArUco detection (releases GIL; populates detector's internal batch).
        let batch_view = py
            .detach(|| {
                detector.inner.detect(
                    &view,
                    Some(&core_intr),
                    None,
                    locus_core::config::PoseEstimationMode::Fast,
                    false,
                )
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let n = batch_view.len();

        // 2. Run ChAruco saddle extraction + board pose estimation.
        //    Dispatch to the telemetry or production monomorphisation based on the flag.
        if debug_telemetry {
            py.detach(|| {
                self.inner.estimate_with_telemetry(
                    &batch_view,
                    &view,
                    &core_intr,
                    &mut self.telem_batch,
                );
            });
        } else {
            py.detach(|| {
                self.inner
                    .estimate(&batch_view, &view, &core_intr, &mut self.batch);
            });
        }
        let active_batch = if debug_telemetry {
            &self.telem_batch
        } else {
            &self.batch
        };
        let s = active_batch.count;

        // 3. Package ArUco detections (ids + corners).
        let ids_arr = unsafe { PyArray1::<i32>::new(py, [n], false) };
        let corners_arr = unsafe { PyArray3::<f32>::new(py, [n, 4, 2], false) };
        unsafe {
            let ids_slice = ids_arr.as_slice_mut().expect("ids slice");
            ids_slice.copy_from_slice(std::slice::from_raw_parts(
                batch_view.ids.as_ptr().cast::<i32>(),
                n,
            ));
            let corners_slice = corners_arr.as_slice_mut().expect("corners slice");
            corners_slice.copy_from_slice(std::slice::from_raw_parts(
                batch_view.corners.as_ptr().cast::<f32>(),
                n * 8,
            ));
        }

        // 4. Package saddle-point detections.
        let saddle_ids_arr = unsafe { PyArray1::<i32>::new(py, [s], false) };
        let saddle_pts_arr = unsafe { PyArray2::<f32>::new(py, [s, 2], false) };
        let saddle_obj_arr = unsafe { PyArray2::<f64>::new(py, [s, 3], false) };
        unsafe {
            let sid_slice = saddle_ids_arr.as_slice_mut().expect("saddle_ids slice");
            for (dst, &src) in sid_slice.iter_mut().zip(active_batch.saddle_ids()) {
                // saddle IDs are bounded by board saddle count (≤ (rows-1)*(cols-1) ≤ ~400).
                #[allow(clippy::cast_possible_wrap)]
                {
                    *dst = src as i32;
                }
            }
            // SAFETY: Point2f is repr(C) with two f32 fields; reinterpreting as &[f32] is
            // sound for a packed, contiguous slice.  [f64; 3] has the same element type as
            // the target NumPy slice.
            let spts_slice = saddle_pts_arr.as_slice_mut().expect("saddle_pts slice");
            spts_slice.copy_from_slice(std::slice::from_raw_parts(
                active_batch.saddle_image_pts().as_ptr().cast::<f32>(),
                s * 2,
            ));
            let sobj_slice = saddle_obj_arr.as_slice_mut().expect("saddle_obj slice");
            sobj_slice.copy_from_slice(std::slice::from_raw_parts(
                active_batch.saddle_obj_pts().as_ptr().cast::<f64>(),
                s * 3,
            ));
        }

        let dict = PyDict::new(py);
        dict.set_item("ids", ids_arr)?;
        dict.set_item("corners", corners_arr)?;
        dict.set_item("saddle_ids", saddle_ids_arr)?;
        dict.set_item("saddle_pts", saddle_pts_arr)?;
        dict.set_item("saddle_obj", saddle_obj_arr)?;

        // 5. Board pose and covariance (None if insufficient saddles or RANSAC failed).
        // We need a mutable reference here — reborrow from the appropriate batch.
        let board_pose = if debug_telemetry {
            self.telem_batch.board_pose.take()
        } else {
            self.batch.board_pose.take()
        };
        if let Some(board_pose) = board_pose {
            let q = nalgebra::UnitQuaternion::from_matrix(&board_pose.pose.rotation);
            let t = board_pose.pose.translation;
            let pose_arr = unsafe { PyArray1::<f64>::new(py, [7], false) };
            unsafe {
                let ps = pose_arr.as_slice_mut().expect("pose slice");
                ps[0] = t.x;
                ps[1] = t.y;
                ps[2] = t.z;
                ps[3] = q.i;
                ps[4] = q.j;
                ps[5] = q.k;
                ps[6] = q.w;
            }
            let cov_arr = unsafe { PyArray2::<f64>::new(py, [6, 6], false) };
            unsafe {
                let cs = cov_arr.as_slice_mut().expect("cov slice");
                for row in 0..6 {
                    for col in 0..6 {
                        cs[row * 6 + col] = board_pose.covariance[(row, col)];
                    }
                }
            }
            dict.set_item("board_pose", pose_arr)?;
            dict.set_item("board_cov", cov_arr)?;
        } else {
            dict.set_item("board_pose", py.None())?;
            dict.set_item("board_cov", py.None())?;
        }

        // 6. Telemetry (None unless debug_telemetry=True).
        if debug_telemetry {
            if let Some(t) = self.telem_batch.telemetry.as_ref() {
                let r = t.count;
                let rej_pts_arr = unsafe { PyArray2::<f32>::new(py, [r, 2], false) };
                let rej_det_arr = unsafe { PyArray1::<f32>::new(py, [r], false) };
                unsafe {
                    // SAFETY: Point2f is repr(C) [f32; 2]; flat reinterpretation is sound.
                    let rpts = rej_pts_arr.as_slice_mut().expect("rej_pts slice");
                    rpts.copy_from_slice(std::slice::from_raw_parts(
                        t.rejected_predictions.as_ptr().cast::<f32>(),
                        r * 2,
                    ));
                    let rdet = rej_det_arr.as_slice_mut().expect("rej_det slice");
                    rdet.copy_from_slice(&t.rejected_determinants[..r]);
                }
                let telem_dict = PyDict::new(py);
                telem_dict.set_item("rejected_saddles", rej_pts_arr)?;
                telem_dict.set_item("rejected_determinants", rej_det_arr)?;
                dict.set_item("telemetry", telem_dict)?;
            } else {
                dict.set_item("telemetry", py.None())?;
            }
        } else {
            dict.set_item("telemetry", py.None())?;
        }

        Ok(dict)
    }
}

// Detector class
// ============================================================================

#[pyclass(unsendable)]
pub struct Detector {
    inner: Box<locus_core::Detector>,
}

#[pymethods]
impl Detector {
    /// Detect tags and return a dictionary of NumPy arrays (SoA layout).
    #[pyo3(signature = (img, intrinsics=None, tag_size=None, pose_estimation_mode=PoseEstimationMode::Fast, debug_telemetry=false))]
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::too_many_lines)]
    fn detect<'py>(
        &mut self,
        py: Python<'py>,
        img: PyReadonlyArray2<'_, u8>,
        intrinsics: Option<CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_estimation_mode: PoseEstimationMode,
        debug_telemetry: bool,
    ) -> Result<Bound<'py, PyDict>, PyErr> {
        let view = prepare_image_view(&img)?;

        let core_intrinsics = intrinsics.map(locus_core::CameraIntrinsics::from);
        let core_pose_mode = locus_core::config::PoseEstimationMode::from(pose_estimation_mode);

        // 1. Run core pipeline
        let detections = py
            .detach(|| {
                self.inner.detect(
                    &view,
                    core_intrinsics.as_ref(),
                    tag_size,
                    core_pose_mode,
                    debug_telemetry,
                )
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let n = detections.len();

        // Allocate NumPy arrays (Unsafe because memory is uninitialized)
        let ids_arr = unsafe { PyArray1::<i32>::new(py, [n], false) };
        let corners_arr = unsafe { PyArray3::<f32>::new(py, [n, 4, 2], false) };
        let error_rates_arr = unsafe { PyArray1::<f32>::new(py, [n], false) };

        // Perform memory mapping
        unsafe {
            let ids_slice = ids_arr
                .as_slice_mut()
                .expect("failed to get mutable slice for ids");
            let corners_slice = corners_arr
                .as_slice_mut()
                .expect("failed to get mutable slice for corners");
            let error_rates_slice = error_rates_arr
                .as_slice_mut()
                .expect("failed to get mutable slice for error_rates");

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

        let dict = PyDict::new(py);
        dict.set_item("ids", ids_arr)?;
        dict.set_item("corners", corners_arr)?;
        dict.set_item("error_rates", error_rates_arr)?;

        // Rejected Quads: (M, 4, 2)
        let m = detections.rejected_corners.len();
        let rejected_arr = unsafe { PyArray3::<f32>::new(py, [m, 4, 2], false) };
        unsafe {
            let rejected_slice = rejected_arr
                .as_slice_mut()
                .expect("failed to get mutable slice for rejected_corners");
            rejected_slice.copy_from_slice(std::slice::from_raw_parts(
                detections.rejected_corners.as_ptr().cast::<f32>(),
                m * 8,
            ));
        }
        dict.set_item("rejected_corners", rejected_arr)?;

        // Rejected Error Rates: (M,)
        let rejected_error_rates_arr = unsafe { PyArray1::<f32>::new(py, [m], false) };
        unsafe {
            let rejected_error_rates_slice = rejected_error_rates_arr
                .as_slice_mut()
                .expect("failed to get mutable slice for rejected_error_rates");
            rejected_error_rates_slice.copy_from_slice(std::slice::from_raw_parts(
                detections.rejected_error_rates.as_ptr(),
                m,
            ));
        }
        dict.set_item("rejected_error_rates", rejected_error_rates_arr)?;

        // Poses: Vectorized (N, 7) layout: [tx, ty, tz, qx, qy, qz, qw]
        if intrinsics.is_some() && tag_size.is_some() {
            let poses_arr = unsafe { PyArray2::<f32>::new(py, [n, 7], false) };
            unsafe {
                let poses_slice = poses_arr.as_slice_mut().expect("failed to get poses slice");

                // Optmized block copy for Pose6D (ignoring f32 padding)
                for (i, pose) in detections.poses.iter().enumerate() {
                    poses_slice[i * 7..(i + 1) * 7].copy_from_slice(&pose.data);
                }
            }
            dict.set_item("poses", poses_arr)?;
        } else {
            dict.set_item("poses", py.None())?;
        }

        // 3. Telemetry (Zero-copy intermediate images)
        if let Some(telemetry) = detections.telemetry {
            let tel_dict = PyDict::new(py);
            unsafe {
                let binarized_arr =
                    PyArray2::<u8>::new(py, [telemetry.height, telemetry.width], false);
                let dest_slice = binarized_arr
                    .as_slice_mut()
                    .expect("Failed to get PyArray slice");
                let src_slice = std::slice::from_raw_parts(
                    telemetry.binarized_ptr,
                    telemetry.height * telemetry.stride,
                );

                if telemetry.stride == telemetry.width {
                    // Contiguous memory layout
                    std::ptr::copy_nonoverlapping(
                        src_slice.as_ptr(),
                        dest_slice.as_mut_ptr(),
                        dest_slice.len(),
                    );
                } else {
                    // Strided memory layout
                    for y in 0..telemetry.height {
                        let src_offset = y * telemetry.stride;
                        let dest_offset = y * telemetry.width;
                        std::ptr::copy_nonoverlapping(
                            src_slice.as_ptr().add(src_offset),
                            dest_slice.as_mut_ptr().add(dest_offset),
                            telemetry.width,
                        );
                    }
                }

                let threshold_arr =
                    PyArray2::<u8>::new(py, [telemetry.height, telemetry.width], false);
                let dest_slice = threshold_arr
                    .as_slice_mut()
                    .expect("Failed to get PyArray slice");
                let src_slice = std::slice::from_raw_parts(
                    telemetry.threshold_map_ptr,
                    telemetry.height * telemetry.stride,
                );

                if telemetry.stride == telemetry.width {
                    // Contiguous memory layout
                    std::ptr::copy_nonoverlapping(
                        src_slice.as_ptr(),
                        dest_slice.as_mut_ptr(),
                        dest_slice.len(),
                    );
                } else {
                    // Strided memory layout
                    for y in 0..telemetry.height {
                        let src_offset = y * telemetry.stride;
                        let dest_offset = y * telemetry.width;
                        std::ptr::copy_nonoverlapping(
                            src_slice.as_ptr().add(src_offset),
                            dest_slice.as_mut_ptr().add(dest_offset),
                            telemetry.width,
                        );
                    }
                }

                tel_dict.set_item("binarized", &binarized_arr)?;
                tel_dict.set_item("threshold_map", &threshold_arr)?;
                tel_dict.set_item("gwlf_fallback_count", telemetry.gwlf_fallback_count)?;
                tel_dict.set_item("gwlf_avg_delta", telemetry.gwlf_avg_delta)?;

                // Subpixel Jitter
                if !telemetry.subpixel_jitter_ptr.is_null() && telemetry.num_jitter > 0 {
                    let nj = telemetry.num_jitter;
                    // Jitter is [nj, 4, 2]
                    let jitter_arr = PyArray3::<f32>::new(py, [nj, 4, 2], false);
                    let jitter_slice = jitter_arr
                        .as_slice_mut()
                        .expect("Failed to get jitter slice");
                    let src_jitter =
                        std::slice::from_raw_parts(telemetry.subpixel_jitter_ptr, nj * 8);
                    jitter_slice.copy_from_slice(src_jitter);
                    tel_dict.set_item("subpixel_jitter", &jitter_arr)?;
                }

                // Reprojection Errors
                if !telemetry.reprojection_errors_ptr.is_null() && telemetry.num_reprojection > 0 {
                    let nr = telemetry.num_reprojection;
                    let repro_arr = PyArray1::<f32>::new(py, [nr], false);
                    let repro_slice = repro_arr.as_slice_mut().expect("Failed to get repro slice");
                    let src_repro =
                        std::slice::from_raw_parts(telemetry.reprojection_errors_ptr, nr);
                    repro_slice.copy_from_slice(src_repro);
                    tel_dict.set_item("reprojection_errors", &repro_arr)?;
                }
            }
            dict.set_item("telemetry", tel_dict)?;
        } else {
            dict.set_item("telemetry", py.None())?;
        }

        Ok(dict)
    }

    /// Returns the current detector configuration.
    fn config(&self) -> PyDetectorConfig {
        PyDetectorConfig::from(self.inner.config())
    }

    /// Update the tag families to be detected.
    fn set_families(&mut self, families: Vec<i32>) -> PyResult<()> {
        let mut core_families = Vec::new();
        for f in families {
            let family = match f {
                0 => locus_core::TagFamily::AprilTag16h5,
                1 => locus_core::TagFamily::AprilTag36h11,
                2 => locus_core::TagFamily::ArUco4x4_50,
                3 => locus_core::TagFamily::ArUco4x4_100,
                4 => locus_core::TagFamily::ArUco6x6_250,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid TagFamily value: {f}"
                    )));
                },
            };
            core_families.push(family);
        }
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

#[pyfunction]
#[pyo3(signature = (decimation=1, threads=0, families=vec![], **kwargs))]
fn create_detector(
    decimation: usize,
    threads: usize,
    families: Vec<i32>,
    kwargs: Option<Bound<'_, PyDict>>,
) -> PyResult<Detector> {
    let mut builder = locus_core::DetectorBuilder::new()
        .with_decimation(decimation)
        .with_threads(threads);

    for f in families {
        builder = builder.with_family(tag_family_from_i32(f)?);
    }

    if let Some(args) = kwargs {
        if let Some(val) = args.get_item("enable_sharpening")? {
            builder = builder.with_sharpening(val.extract()?);
        }
        if let Some(val) = args.get_item("upscale_factor")? {
            builder = builder.with_upscale_factor(val.extract()?);
        }
        if let Some(val) = args.get_item("threshold_tile_size")? {
            builder = builder.with_threshold_tile_size(val.extract()?);
        }
        if let Some(val) = args.get_item("threshold_min_range")? {
            builder = builder.with_threshold_min_range(val.extract()?);
        }
        if let Some(val) = args.get_item("adaptive_threshold_constant")? {
            builder = builder.with_adaptive_threshold_constant(val.extract()?);
        }
        if let Some(val) = args.get_item("quad_min_area")? {
            builder = builder.with_quad_min_area(val.extract()?);
        }
        if let Some(val) = args.get_item("quad_min_fill_ratio")? {
            builder = builder.with_quad_min_fill_ratio(val.extract()?);
        }
        if let Some(val) = args.get_item("quad_min_edge_score")? {
            builder = builder.with_quad_min_edge_score(val.extract()?);
        }
        if let Some(val) = args.get_item("decoder_min_contrast")? {
            builder = builder.with_decoder_min_contrast(val.extract()?);
        }
        if let Some(val) = args.get_item("max_hamming_error")? {
            builder = builder.with_max_hamming_error(val.extract()?);
        }
        if let Some(val) = args.get_item("gwlf_transversal_alpha")? {
            builder = builder.with_gwlf_transversal_alpha(val.extract()?);
        }
        if let Some(val) = args.get_item("quad_max_elongation")? {
            builder = builder.with_quad_max_elongation(val.extract()?);
        }
        if let Some(val) = args.get_item("quad_min_density")? {
            builder = builder.with_quad_min_density(val.extract()?);
        }
        if let Some(val) = args.get_item("quad_extraction_mode")? {
            let i: i32 = val.extract()?;
            let mode = match i {
                0 => locus_core::config::QuadExtractionMode::ContourRdp,
                1 => locus_core::config::QuadExtractionMode::EdLines,
                _ => return Err(PyValueError::new_err("Invalid quad_extraction_mode")),
            };
            builder = builder.with_quad_extraction_mode(mode);
        }
        if let Some(val) = args.get_item("refinement_mode")? {
            let i: i32 = val.extract()?;
            let mode = match i {
                0 => locus_core::config::CornerRefinementMode::None,
                1 => locus_core::config::CornerRefinementMode::Edge,
                2 => locus_core::config::CornerRefinementMode::GridFit,
                3 => locus_core::config::CornerRefinementMode::Erf,
                4 => locus_core::config::CornerRefinementMode::Gwlf,
                _ => return Err(PyValueError::new_err("Invalid refinement_mode")),
            };
            builder = builder.with_corner_refinement(mode);
        }
        if let Some(val) = args.get_item("decode_mode")? {
            let i: i32 = val.extract()?;
            let mode = match i {
                0 => locus_core::config::DecodeMode::Hard,
                1 => locus_core::config::DecodeMode::Soft,
                _ => return Err(PyValueError::new_err("Invalid decode_mode")),
            };
            builder = builder.with_decode_mode(mode);
        }
        if let Some(val) = args.get_item("segmentation_connectivity")? {
            let i: i32 = val.extract()?;
            let conn = match i {
                0 => locus_core::config::SegmentationConnectivity::Four,
                1 => locus_core::config::SegmentationConnectivity::Eight,
                _ => return Err(PyValueError::new_err("Invalid connectivity")),
            };
            builder = builder.with_connectivity(conn);
        }
    }

    Ok(Detector {
        inner: Box::new(builder.build()),
    })
}

fn prepare_image_view<'a>(img: &PyReadonlyArray2<'_, u8>) -> PyResult<ImageView<'a>> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let strides = img.strides();
    let stride_y = strides[0].cast_unsigned();
    let stride_x = strides[1];

    if stride_x == 1 {
        let required_size = if height > 0 && width > 0 {
            (height - 1) * stride_y + width
        } else {
            0
        };
        let data = unsafe { std::slice::from_raw_parts(img.data(), required_size) };
        ImageView::new(data, width, height, stride_y)
            .map_err(|e| PyRuntimeError::new_err(e.clone()))
    } else {
        Err(PyValueError::new_err(
            "Array must be C-contiguous. Call np.ascontiguousarray(image) first.",
        ))
    }
}

// ============================================================================
// Profiling
// ============================================================================

#[pyfunction]
fn init_tracy() {
    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt;
        unsafe {
            std::env::set_var("TRACY_NO_INVARIANT_CHECK", "1");
        }
        let subscriber = tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default());
        tracing::subscriber::set_global_default(subscriber).ok();
    }
}

#[pyfunction]
fn production_config() -> Detector {
    Detector {
        inner: Box::new(locus_core::Detector::with_config(
            locus_core::DetectorConfig::production_default(),
        )),
    }
}

#[pyfunction]
fn fast_config() -> Detector {
    Detector {
        inner: Box::new(locus_core::Detector::with_config(
            locus_core::DetectorConfig::fast_default(),
        )),
    }
}

#[pymodule]
fn locus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Detector>()?;
    m.add_class::<PyDetectorConfig>()?;
    m.add_class::<TagFamily>()?;
    m.add_class::<SegmentationConnectivity>()?;
    m.add_class::<CornerRefinementMode>()?;
    m.add_class::<DecodeMode>()?;
    m.add_class::<PoseEstimationMode>()?;
    m.add_class::<CameraIntrinsics>()?;
    m.add_class::<PyPose>()?;
    m.add_class::<CharucoBoard>()?;
    m.add_class::<AprilGrid>()?;
    m.add_class::<CharucoRefiner>()?;

    m.add_function(wrap_pyfunction!(create_detector, m)?)?;
    m.add_function(wrap_pyfunction!(production_config, m)?)?;
    m.add_function(wrap_pyfunction!(fast_config, m)?)?;
    m.add_function(wrap_pyfunction!(init_tracy, m)?)?;
    Ok(())
}
