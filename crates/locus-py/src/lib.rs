//! Python bindings for the Locus Tag library.
#![allow(
    unsafe_code,
    clippy::unused_self,
    missing_docs,
    clippy::trivially_copy_pass_by_ref
)]

use locus_core::image::ImageView;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ============================================================================
// Detection and Stats (Python-compatible wrappers)
// ============================================================================

/// Python-compatible detection result.
#[pyclass]
#[derive(Clone)]
pub struct Detection {
    /// Tag ID.
    #[pyo3(get)]
    pub id: u32,
    /// Center coordinates (x, y).
    #[pyo3(get)]
    pub center: [f64; 2],
    /// Corner coordinates (4x2).
    #[pyo3(get)]
    pub corners: [[f64; 2]; 4],
    /// Hamming distance.
    #[pyo3(get)]
    pub hamming: u32,
    /// Decision margin.
    #[pyo3(get)]
    pub decision_margin: f64,
    /// Extracted bits from the tag.
    #[pyo3(get)]
    pub bits: u64,
}

/// Python-compatible pipeline statistics.
#[pyclass]
#[derive(Clone, Default)]
pub struct PipelineStats {
    /// Adaptive thresholding time (ms).
    #[pyo3(get)]
    pub threshold_ms: f64,
    /// Segmentation time (ms).
    #[pyo3(get)]
    pub segmentation_ms: f64,
    /// Quad extraction time (ms).
    #[pyo3(get)]
    pub quad_extraction_ms: f64,
    /// Decoding time (ms).
    #[pyo3(get)]
    pub decoding_ms: f64,
    /// Total time (ms).
    #[pyo3(get)]
    pub total_ms: f64,
    #[pyo3(get)]
    pub num_candidates: usize,
    /// Number of final detections.
    #[pyo3(get)]
    pub num_detections: usize,
    /// Number of candidates rejected due to low contrast.
    #[pyo3(get)]
    pub num_rejected_by_contrast: usize,
    /// Number of candidates that failed decoding.
    #[pyo3(get)]
    pub num_rejected_by_hamming: usize,
}

#[pyclass]
#[derive(Clone)]
pub struct FullDetectionResult {
    #[pyo3(get)]
    pub detections: Vec<Detection>,
    #[pyo3(get)]
    pub candidates: Vec<Detection>,
    pub binarized: Option<Vec<u8>>,
    pub labels: Option<Vec<u32>>,
    #[pyo3(get)]
    pub stats: PipelineStats,
    pub width: usize,
    pub height: usize,
}

#[pymethods]
impl FullDetectionResult {
    /// Get the binarized image as a numpy array.
    fn get_binarized(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if let Some(data) = &self.binarized {
            let array = Array2::from_shape_vec((self.height, self.width), data.clone())
                .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {e}")))?;
            Ok(Some(array.into_pyarray(py).into_any().unbind()))
        } else {
            Ok(None)
        }
    }

    /// Get the labels image as a numpy array.
    fn get_labels(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if let Some(data) = &self.labels {
            let array = Array2::from_shape_vec((self.height, self.width), data.clone())
                .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {e}")))?;
            Ok(Some(array.into_pyarray(py).into_any().unbind()))
        } else {
            Ok(None)
        }
    }
}

impl From<(locus_core::FullDetectionResult, usize, usize)> for FullDetectionResult {
    fn from(tuple: (locus_core::FullDetectionResult, usize, usize)) -> Self {
        let (res, width, height) = tuple;
        Self {
            detections: res.detections.into_iter().map(Detection::from).collect(),
            candidates: res.candidates.into_iter().map(Detection::from).collect(),
            binarized: res.binarized,
            labels: res.labels,
            stats: PipelineStats::from(res.stats),
            width,
            height,
        }
    }
}

impl From<locus_core::PipelineStats> for PipelineStats {
    fn from(s: locus_core::PipelineStats) -> Self {
        Self {
            threshold_ms: s.threshold_ms,
            segmentation_ms: s.segmentation_ms,
            quad_extraction_ms: s.quad_extraction_ms,
            decoding_ms: s.decoding_ms,
            total_ms: s.total_ms,
            num_candidates: s.num_candidates,
            num_detections: s.num_detections,
            num_rejected_by_contrast: s.num_rejected_by_contrast,
            num_rejected_by_hamming: s.num_rejected_by_hamming,
        }
    }
}

impl From<locus_core::Detection> for Detection {
    fn from(d: locus_core::Detection) -> Self {
        Self {
            id: d.id,
            center: d.center,
            corners: d.corners,
            hamming: d.hamming,
            decision_margin: d.decision_margin,
            bits: d.bits,
        }
    }
}

// ============================================================================
// TagFamily enum for per-call decoder selection
// ============================================================================

/// Tag family enum for selecting which decoders to use.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TagFamily {
    /// AprilTag 36h11 family (587 codes).
    AprilTag36h11 = 0,
    /// AprilTag 16h5 family (30 codes).
    AprilTag16h5 = 1,
    /// ArUco 4x4_50 dictionary.
    ArUco4x4_50 = 2,
    /// ArUco 4x4_100 dictionary.
    ArUco4x4_100 = 3,
}

#[pymethods]
impl TagFamily {
    fn __reduce__(&self) -> (PyObject, (u8,)) {
        Python::with_gil(|py| {
            let cls = py.get_type::<Self>();
            (cls.into_any().unbind(), (*self as u8,))
        })
    }
}

impl From<TagFamily> for locus_core::config::TagFamily {
    fn from(f: TagFamily) -> Self {
        match f {
            TagFamily::AprilTag36h11 => locus_core::config::TagFamily::AprilTag36h11,
            TagFamily::AprilTag16h5 => locus_core::config::TagFamily::AprilTag16h5,
            TagFamily::ArUco4x4_50 => locus_core::config::TagFamily::ArUco4x4_50,
            TagFamily::ArUco4x4_100 => locus_core::config::TagFamily::ArUco4x4_100,
        }
    }
}

// ============================================================================
// SegmentationConnectivity enum
// ============================================================================

/// Segmentation connectivity mode.
#[pyclass(eq, eq_int, module = "locus")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentationConnectivity {
    /// 4-connectivity: Pixels connect horizontally and vertically only.
    Four = 0,
    /// 8-connectivity: Pixels connect horizontally, vertically, and diagonally.
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

#[pymethods]
impl SegmentationConnectivity {
    fn __reduce__(&self) -> (PyObject, (u8,)) {
        Python::with_gil(|py| {
            let cls = py.get_type::<Self>();
            (cls.into_any().unbind(), (*self as u8,))
        })
    }
}

// ============================================================================
// CornerRefinementMode enum
// ============================================================================

/// Mode for subpixel corner refinement.
#[pyclass(eq, eq_int, module = "locus")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum CornerRefinementMode {
    /// No subpixel refinement (integer pixel precision).
    None = 0,
    /// Edge-based refinement using gradient maxima (Default).
    Edge = 1,
    /// GridFit: Optimizes corners by maximizing code contrast.
    GridFit = 2,
    /// Erf: Fits a Gaussian to the gradient profile for sub-pixel edge alignment.
    Erf = 3,
}

impl From<CornerRefinementMode> for locus_core::config::CornerRefinementMode {
    fn from(m: CornerRefinementMode) -> Self {
        match m {
            CornerRefinementMode::None => locus_core::config::CornerRefinementMode::None,
            CornerRefinementMode::Edge => locus_core::config::CornerRefinementMode::Edge,
            CornerRefinementMode::GridFit => locus_core::config::CornerRefinementMode::GridFit,
            CornerRefinementMode::Erf => locus_core::config::CornerRefinementMode::Erf,
        }
    }
}

#[pymethods]
impl CornerRefinementMode {
    fn __reduce__(&self) -> (PyObject, (u8,)) {
        Python::with_gil(|py| {
            let cls = py.get_type::<Self>();
            (cls.into_any().unbind(), (*self as u8,))
        })
    }
}

// ============================================================================
// DecodeMode enum
// ============================================================================

/// Decoding strategy mode.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeMode {
    /// Hard decision (Hamming distance) - Standard method.
    Hard = 0,
    /// Soft decision (LLR accumulation) - Better for low contrast/noise.
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

#[pymethods]
impl DecodeMode {
    fn __reduce__(&self) -> (PyObject, (u8,)) {
        Python::with_gil(|py| {
            let cls = py.get_type::<Self>();
            (cls.into_any().unbind(), (*self as u8,))
        })
    }
}

// ============================================================================
// Detector class with persistent state
// ============================================================================

/// The main detector class. Holds reusable detector state.
///
/// Use this for efficient repeated detection on multiple images.
///
/// Example:
///     detector = locus.Detector()
///     detections = detector.detect(image)
///
///     # With custom config
///     detector = locus.Detector(
///         threshold_tile_size=16,
///         quad_min_area=200,
///     )
#[pyclass(unsendable)]
pub struct Detector {
    inner: locus_core::Detector,
}

#[pymethods]
impl Detector {
    /// Create a new detector with optional configuration.
    ///
    /// Args:
    ///     threshold_tile_size: Tile size for adaptive thresholding (default: 4)
    ///     threshold_min_range: Min intensity range for valid tiles (default: 5)
    ///     enable_bilateral: Enable bilateral pre-filtering (default: false)
    ///     bilateral_sigma_space: Bilateral spatial sigma (default: 3.0)
    ///     bilateral_sigma_color: Bilateral color sigma (default: 30.0)
    ///     enable_adaptive_window: Enable adaptive window sizing (default: false)
    ///     threshold_min_radius: Min threshold window radius (default: 2)
    ///     threshold_max_radius: Max threshold window radius (default: 7)
    ///     quad_min_area: Minimum quad area in pixels (default: 64)
    ///     quad_max_aspect_ratio: Maximum bounding box aspect ratio (default: 3.0)
    ///     quad_min_fill_ratio: Minimum pixel fill ratio (default: 0.3)
    ///     quad_max_fill_ratio: Maximum pixel fill ratio (default: 0.95)
    ///     quad_min_edge_length: Minimum edge length in pixels (default: 4.0)
    ///     quad_min_edge_score: Minimum edge gradient score (default: 0.4)
    ///     segmentation_connectivity: Connectivity mode (default: Eight)
    ///     decode_mode: Decoding strategy (default: Hard)
    #[new]
    #[pyo3(signature = (
        threshold_tile_size = 8,
        threshold_min_range = 10,
        enable_bilateral = false,
        bilateral_sigma_space = 0.8,
        bilateral_sigma_color = 30.0,
        enable_sharpening = false,
        enable_adaptive_window = false,
        threshold_min_radius = 2,
        threshold_max_radius = 15,
        adaptive_threshold_constant = 0,
        adaptive_threshold_gradient_threshold = 10,
        quad_min_area = 16,
        quad_max_aspect_ratio = 10.0,
        quad_min_fill_ratio = 0.10,
        quad_max_fill_ratio = 0.98,
        quad_min_edge_length = 2.0,
        quad_min_edge_score = 0.1,
        subpixel_refinement_sigma = 0.6,
        segmentation_margin = 1,
        segmentation_connectivity = SegmentationConnectivity::Eight,
        upscale_factor = 1,
        decoder_min_contrast = 20.0,
        refinement_mode = CornerRefinementMode::Erf,
        decode_mode = DecodeMode::Hard,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        threshold_tile_size: usize,
        threshold_min_range: u8,
        enable_bilateral: bool,
        bilateral_sigma_space: f32,
        bilateral_sigma_color: f32,
        enable_sharpening: bool,
        enable_adaptive_window: bool,
        threshold_min_radius: usize,
        threshold_max_radius: usize,
        adaptive_threshold_constant: i16,
        adaptive_threshold_gradient_threshold: u8,
        quad_min_area: u32,
        quad_max_aspect_ratio: f32,
        quad_min_fill_ratio: f32,
        quad_max_fill_ratio: f32,
        quad_min_edge_length: f64,
        quad_min_edge_score: f64,
        subpixel_refinement_sigma: f64,
        segmentation_margin: i16,
        segmentation_connectivity: SegmentationConnectivity,
        upscale_factor: usize,
        decoder_min_contrast: f64,
        refinement_mode: CornerRefinementMode,
        decode_mode: DecodeMode,
    ) -> Self {
        let config = locus_core::DetectorConfig {
            threshold_tile_size,
            threshold_min_range,
            enable_bilateral,
            bilateral_sigma_space,
            bilateral_sigma_color,
            enable_sharpening,
            enable_adaptive_window,
            threshold_min_radius,
            threshold_max_radius,
            adaptive_threshold_constant,
            adaptive_threshold_gradient_threshold,
            quad_min_area,
            quad_max_aspect_ratio,
            quad_min_fill_ratio,
            quad_max_fill_ratio,
            quad_min_edge_length,
            quad_min_edge_score,
            subpixel_refinement_sigma,
            segmentation_margin,
            segmentation_connectivity: segmentation_connectivity.into(),
            upscale_factor,
            decoder_min_contrast,
            refinement_mode: refinement_mode.into(),
            decode_mode: decode_mode.into(),
        };
        Self {
            inner: locus_core::Detector::with_config(config),
        }
    }

    /// DEBUG: Get the current sharpening status.
    #[getter]
    fn enable_sharpening(&self) -> bool {
        self.inner.get_config().enable_sharpening
    }

    /// Detect tags in the image using default decoders.
    #[pyo3(signature = (img, decimation = 1))]
    #[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
    fn detect(&mut self, img: PyReadonlyArray2<u8>, decimation: usize) -> PyResult<Vec<Detection>> {
        let view = create_image_view(&img)?;
        let options = locus_core::DetectOptions::builder()
            .decimation(decimation)
            .build();
        let detections = self.inner.detect_with_options(&view, &options);
        Ok(detections.into_iter().map(Detection::from).collect())
    }

    /// Detect tags using specific tag families (for performance).
    ///
    /// Args:
    ///     img: Grayscale image as numpy array
    ///     families: List of TagFamily to decode
    #[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
    fn detect_with_options(
        &mut self,
        img: PyReadonlyArray2<u8>,
        families: Vec<TagFamily>,
    ) -> PyResult<Vec<Detection>> {
        let view = create_image_view(&img)?;
        let core_families: Vec<locus_core::config::TagFamily> =
            families.into_iter().map(Into::into).collect();
        let options = locus_core::DetectOptions::with_families(&core_families);
        let detections = self.inner.detect_with_options(&view, &options);
        Ok(detections.into_iter().map(Detection::from).collect())
    }

    /// Detect tags with timing statistics.
    #[pyo3(signature = (img, decimation = 1))]
    #[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
    fn detect_with_stats(
        &mut self,
        img: PyReadonlyArray2<u8>,
        decimation: usize,
    ) -> PyResult<(Vec<Detection>, PipelineStats)> {
        let view = create_image_view(&img)?;
        let options = locus_core::DetectOptions::builder()
            .decimation(decimation)
            .build();
        let (detections, stats) = self.inner.detect_with_stats_and_options(&view, &options);
        Ok((
            detections.into_iter().map(Detection::from).collect(),
            PipelineStats::from(stats),
        ))
    }

    /// Debugging: Extract quad candidates without decoding.
    ///
    /// Returns a list of all quad candidates found in the image, even those that
    /// fail decoding. Useful for debugging segmentation and quad fitting.
    #[pyo3(signature = (img, decimation = 1))]
    #[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
    fn extract_candidates(
        &mut self,
        img: PyReadonlyArray2<u8>,
        decimation: usize,
    ) -> PyResult<(Vec<Detection>, PipelineStats)> {
        let view = create_image_view(&img)?;
        let options = locus_core::DetectOptions::builder()
            .decimation(decimation)
            .build();
        let (candidates, stats) = self.inner.extract_candidates(&view, &options);
        Ok((
            candidates.into_iter().map(Detection::from).collect(),
            PipelineStats::from(stats),
        ))
    }

    /// Perform full detection and return all intermediate debug data.
    #[pyo3(signature = (img, decimation = 1))]
    #[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
    fn detect_full(
        &mut self,
        img: PyReadonlyArray2<u8>,
        decimation: usize,
    ) -> PyResult<FullDetectionResult> {
        let view = create_image_view(&img)?;
        let options = locus_core::DetectOptions::builder()
            .decimation(decimation)
            .build();
        let res = self.inner.detect_full(&view, &options);
        // If upscaling is enabled in config, the binarized/labeled images will be larger.
        let upscale = self.inner.get_config().upscale_factor;
        let width = view.width * upscale;
        let height = view.height * upscale;
        Ok(FullDetectionResult::from((res, width, height)))
    }

    /// Set the tag families to decode by default.
    fn set_families(&mut self, families: Vec<TagFamily>) {
        let core_families: Vec<locus_core::config::TagFamily> =
            families.into_iter().map(Into::into).collect();
        self.inner.set_families(&core_families);
    }

    /// Get the identification code (bits) for a specific tag ID and family.
    fn get_tag_code(&self, family: TagFamily, id: u32) -> Option<u64> {
        let core_family: locus_core::config::TagFamily = family.into();
        match core_family {
            locus_core::config::TagFamily::AprilTag36h11 => {
                if (id as usize) < locus_core::dictionaries::APRILTAG_36H11.codes.len() {
                    Some(locus_core::dictionaries::APRILTAG_36H11.codes[id as usize])
                } else {
                    None
                }
            },
            _ => None, // Not implemented for others yet in this quick debug helper
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Create an ImageView from a PyReadonlyArray2, enforcing C-contiguous layout.
#[allow(clippy::cast_sign_loss)]
fn create_image_view<'a>(img: &'a PyReadonlyArray2<'a, u8>) -> PyResult<ImageView<'a>> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let strides = img.strides();
    let stride_y = strides[0] as usize;
    let stride_x = strides[1];

    // Check for C-contiguous behavior (inner stride == 1)
    if stride_x == 1 {
        let required_size = if height > 0 && width > 0 {
            (height - 1) * stride_y + width
        } else {
            0
        };

        // If the slice is safe, we borrow directly
        // Note: PyReadonlyArray2::data() returns a raw pointer. We must ensure it's valid.
        // If it's contiguous with inner stride 1, we can treat it as a slice (with gaps if stride_y > width).
        // locus_core::ImageView handles stride > width correctly.

        let data = unsafe { std::slice::from_raw_parts(img.data(), required_size) };
        let view = ImageView::new(data, width, height, stride_y)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(view)
    } else {
        // Non-contiguous (e.g. sliced columns, generalized slicing)
        // We reject this to enforce zero-copy usage.
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input array must be C-contiguous (row-major). Found non-contiguous stride. \
             Use numpy.ascontiguousarray() if needed.",
        ));
    }
}

// ============================================================================
// Legacy function-based API (for backward compatibility)
// ============================================================================

/// A dummy detection function for Phase 0 verification.
#[pyfunction]
fn dummy_detect() -> String {
    format!("{} - Python Bindings Active", locus_core::core_info())
}

/// Detect tags in an image. Zero-copy ingestion of NumPy arrays.
#[pyfunction]
#[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
fn detect_tags(img: PyReadonlyArray2<u8>) -> PyResult<Vec<Detection>> {
    let view = create_image_view(&img)?;
    let mut detector = locus_core::Detector::new();
    let detections = detector.detect(&view);
    Ok(detections.into_iter().map(Detection::from).collect())
}

/// Detect tags and return timing stats.
#[pyfunction]
#[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
fn detect_tags_with_stats(img: PyReadonlyArray2<u8>) -> PyResult<(Vec<Detection>, PipelineStats)> {
    let view = create_image_view(&img)?;
    let mut detector = locus_core::Detector::new();
    let (detections, stats) = detector.detect_with_stats(&view);
    Ok((
        detections.into_iter().map(Detection::from).collect(),
        PipelineStats::from(stats),
    ))
}

/// For debugging: Apply thresholding and return the binarized image.
#[pyfunction]
#[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
fn debug_threshold(img: PyReadonlyArray2<u8>) -> PyResult<PyObject> {
    let view = create_image_view(&img)?;
    let width = view.width;
    let height = view.height;

    let mut output = vec![0u8; width * height];
    let arena = bumpalo::Bump::new();
    let engine = locus_core::threshold::ThresholdEngine::new();
    let stats = engine.compute_tile_stats(&arena, &view);
    engine.apply_threshold(&arena, &view, &stats, &mut output);

    Python::with_gil(|py| {
        let array = numpy::PyArray1::from_vec(py, output);
        let array2d = array.reshape([height, width]).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Failed to reshape NumPy array")
        })?;
        Ok(array2d.into_any().unbind())
    })
}

/// For debugging: Return the labeled connected components.
#[pyfunction]
#[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
fn debug_segmentation(img: PyReadonlyArray2<u8>) -> PyResult<PyObject> {
    let view = create_image_view(&img)?;
    let width = view.width;
    let height = view.height;

    let arena = bumpalo::Bump::new();
    let engine = locus_core::threshold::ThresholdEngine::new();
    let stats = engine.compute_tile_stats(&arena, &view);
    let mut binarized = vec![0u8; width * height];
    engine.apply_threshold(&arena, &view, &stats, &mut binarized);

    let labels =
        locus_core::segmentation::label_components(&arena, &binarized, width, height, false);
    let labels_vec = labels.to_vec();

    Python::with_gil(|py| {
        let array = numpy::PyArray1::from_vec(py, labels_vec);
        let array2d = array.reshape([height, width]).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Failed to reshape NumPy array")
        })?;
        Ok(array2d.into_any().unbind())
    })
}

// ============================================================================
// Module registration
// ============================================================================

/// The locus Python module.
#[pymodule]
fn locus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<Detection>()?;
    m.add_class::<FullDetectionResult>()?;
    m.add_class::<PipelineStats>()?;
    m.add_class::<TagFamily>()?;
    m.add_class::<SegmentationConnectivity>()?;
    m.add_class::<CornerRefinementMode>()?;
    m.add_class::<DecodeMode>()?;
    m.add_class::<Detector>()?;

    // Legacy functions (for backward compatibility)
    m.add_function(wrap_pyfunction!(dummy_detect, m)?)?;
    m.add_function(wrap_pyfunction!(detect_tags, m)?)?;
    m.add_function(wrap_pyfunction!(detect_tags_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(debug_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(debug_segmentation, m)?)?;
    Ok(())
}
