//! Python bindings for the Locus Tag library.
#![allow(unsafe_code)]

use locus_core::image::ImageView;
use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
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
    /// Number of quad candidates.
    #[pyo3(get)]
    pub num_candidates: usize,
    /// Number of final detections.
    #[pyo3(get)]
    pub num_detections: usize,
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
    ///     quad_min_edge_score: Minimum edge gradient score (default: 2.0)
    #[new]
    #[pyo3(signature = (
        threshold_tile_size = 4,
        threshold_min_range = 5,
        enable_bilateral = true,
        bilateral_sigma_space = 0.8,
        bilateral_sigma_color = 30.0,
        enable_adaptive_window = true,
        threshold_min_radius = 2,
        threshold_max_radius = 7,
        quad_min_area = 32,
        quad_max_aspect_ratio = 3.0,
        quad_min_fill_ratio = 0.3,
        quad_max_fill_ratio = 0.95,
        quad_min_edge_length = 4.0,
        quad_min_edge_score = 1.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        threshold_tile_size: usize,
        threshold_min_range: u8,
        enable_bilateral: bool,
        bilateral_sigma_space: f32,
        bilateral_sigma_color: f32,
        enable_adaptive_window: bool,
        threshold_min_radius: usize,
        threshold_max_radius: usize,
        quad_min_area: u32,
        quad_max_aspect_ratio: f32,
        quad_min_fill_ratio: f32,
        quad_max_fill_ratio: f32,
        quad_min_edge_length: f64,
        quad_min_edge_score: f64,
    ) -> Self {
        let config = locus_core::DetectorConfig {
            threshold_tile_size,
            threshold_min_range,
            enable_bilateral,
            bilateral_sigma_space,
            bilateral_sigma_color,
            enable_adaptive_window,
            threshold_min_radius,
            threshold_max_radius,
            quad_min_area,
            quad_max_aspect_ratio,
            quad_min_fill_ratio,
            quad_max_fill_ratio,
            quad_min_edge_length,
            quad_min_edge_score,
        };
        Self {
            inner: locus_core::Detector::with_config(config),
        }
    }

    /// Detect tags in the image using default decoders.
    #[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
    fn detect(&mut self, img: PyReadonlyArray2<u8>) -> PyResult<Vec<Detection>> {
        let view = create_image_view(&img)?;
        let detections = self.inner.detect(&view);
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
    #[allow(clippy::cast_sign_loss, clippy::needless_pass_by_value)]
    fn detect_with_stats(
        &mut self,
        img: PyReadonlyArray2<u8>,
    ) -> PyResult<(Vec<Detection>, PipelineStats)> {
        let view = create_image_view(&img)?;
        let (detections, stats) = self.inner.detect_with_stats(&view);
        Ok((
            detections.into_iter().map(Detection::from).collect(),
            PipelineStats::from(stats),
        ))
    }

    /// Set the tag families to decode by default.
    fn set_families(&mut self, families: Vec<TagFamily>) {
        let core_families: Vec<locus_core::config::TagFamily> =
            families.into_iter().map(Into::into).collect();
        self.inner.set_families(&core_families);
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Create an ImageView from a PyReadonlyArray2, validating strides.
#[allow(clippy::cast_sign_loss)]
fn create_image_view<'a>(img: &'a PyReadonlyArray2<'a, u8>) -> PyResult<ImageView<'a>> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let strides = img.strides();
    let stride = strides[0] as usize;

    if strides[1] != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Image must have C-contiguous rows (inner stride must be 1)",
        ));
    }

    let required_size = if height > 0 && width > 0 {
        (height - 1) * stride + width
    } else {
        0
    };

    let data = unsafe { std::slice::from_raw_parts(img.data(), required_size) };
    ImageView::new(data, width, height, stride).map_err(pyo3::exceptions::PyRuntimeError::new_err)
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
    let engine = locus_core::threshold::ThresholdEngine::new();
    let stats = engine.compute_tile_stats(&view);
    engine.apply_threshold(&view, &stats, &mut output);

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

    let engine = locus_core::threshold::ThresholdEngine::new();
    let stats = engine.compute_tile_stats(&view);
    let mut binarized = vec![0u8; width * height];
    engine.apply_threshold(&view, &stats, &mut binarized);

    let arena = bumpalo::Bump::new();
    let labels = locus_core::segmentation::label_components(&arena, &binarized, width, height);
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
    m.add_class::<PipelineStats>()?;
    m.add_class::<TagFamily>()?;
    m.add_class::<Detector>()?;

    // Legacy functions (for backward compatibility)
    m.add_function(wrap_pyfunction!(dummy_detect, m)?)?;
    m.add_function(wrap_pyfunction!(detect_tags, m)?)?;
    m.add_function(wrap_pyfunction!(detect_tags_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(debug_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(debug_segmentation, m)?)?;
    Ok(())
}
