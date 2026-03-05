//! Python bindings for the Locus Tag library.
#![allow(
    unsafe_code,
    clippy::unused_self,
    missing_docs,
    clippy::trivially_copy_pass_by_ref
)]

use locus_core::ImageView;
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ============================================================================
// Enums
// ============================================================================

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TagFamily {
    AprilTag36h11 = 0,
    AprilTag41h12 = 1,
    ArUco4x4_50 = 2,
    ArUco4x4_100 = 3,
}

impl From<TagFamily> for locus_core::TagFamily {
    fn from(f: TagFamily) -> Self {
        match f {
            TagFamily::AprilTag36h11 => locus_core::TagFamily::AprilTag36h11,
            TagFamily::AprilTag41h12 => locus_core::TagFamily::AprilTag41h12,
            TagFamily::ArUco4x4_50 => locus_core::TagFamily::ArUco4x4_50,
            TagFamily::ArUco4x4_100 => locus_core::TagFamily::ArUco4x4_100,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentationConnectivity {
    Four = 0,
    Eight = 1,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum CornerRefinementMode {
    None = 0,
    Edge = 1,
    GridFit = 2,
    Erf = 3,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeMode {
    Hard = 0,
    Soft = 1,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoseEstimationMode {
    Fast = 0,
    Accurate = 1,
}

// ============================================================================
// Structs
// ============================================================================

#[pyclass]
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

#[pyclass]
#[derive(Clone)]
pub struct PyPose {
    #[pyo3(get)]
    pub rotation: [[f64; 3]; 3],
    #[pyo3(get)]
    pub translation: [f64; 3],
}

// ============================================================================
// Detector class
// ============================================================================

#[pyclass(unsendable)]
pub struct Detector {
    inner: locus_core::Detector,
}

#[pymethods]
impl Detector {
    /// Detect tags and return a dictionary of NumPy arrays (SoA layout).
    fn detect(&mut self, py: Python<'_>, img: PyReadonlyArray2<u8>) -> PyResult<PyObject> {
        let view = prepare_image_view(&img)?;
        let detections = self.inner.detect(&view);

        let n = detections.len();
        let mut ids = Vec::with_capacity(n);
        let mut centers = Vec::with_capacity(n * 2);
        let mut corners = Vec::with_capacity(n * 8);
        let mut hamming = Vec::with_capacity(n);
        let mut decision_margin = Vec::with_capacity(n);

        for det in detections {
            ids.push(det.id);
            centers.push(det.center[0]);
            centers.push(det.center[1]);
            for c in &det.corners {
                corners.push(c[0]);
                corners.push(c[1]);
            }
            hamming.push(det.hamming);
            decision_margin.push(det.decision_margin);
        }

        let dict = PyDict::new(py);
        dict.set_item("ids", ids.into_pyarray(py))?;
        
        let centers_arr = Array2::from_shape_vec((n, 2), centers)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("centers", centers_arr.into_pyarray(py))?;

        let corners_arr = Array3::from_shape_vec((n, 4, 2), corners)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("corners", corners_arr.into_pyarray(py))?;

        dict.set_item("hamming", hamming.into_pyarray(py))?;
        dict.set_item("decision_margin", decision_margin.into_pyarray(py))?;

        Ok(dict.into())
    }
}

#[pyfunction]
#[pyo3(signature = (decimation=1, threads=0, families=vec![], **kwargs))]
fn create_detector(
    decimation: usize,
    threads: usize,
    families: Vec<i32>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Detector> {
    let mut builder = locus_core::DetectorBuilder::new()
        .with_decimation(decimation)
        .with_threads(threads);

    for f in families {
        let family = match f {
            0 => locus_core::TagFamily::AprilTag36h11,
            1 => locus_core::TagFamily::AprilTag41h12,
            2 => locus_core::TagFamily::ArUco4x4_50,
            3 => locus_core::TagFamily::ArUco4x4_100,
            _ => return Err(PyValueError::new_err(format!("Invalid TagFamily value: {}", f))),
        };
        builder = builder.with_family(family);
    }

    if let Some(args) = kwargs {
        if let Some(val) = args.get_item("upscale_factor")? {
            builder = builder.with_upscale_factor(val.extract()?);
        }
        if let Some(val) = args.get_item("refinement_mode")? {
            let i: i32 = val.extract()?;
            let mode = match i {
                0 => locus_core::config::CornerRefinementMode::None,
                1 => locus_core::config::CornerRefinementMode::Edge,
                2 => locus_core::config::CornerRefinementMode::GridFit,
                3 => locus_core::config::CornerRefinementMode::Erf,
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
        inner: builder.build(),
    })
}

fn prepare_image_view<'a>(img: &'a PyReadonlyArray2<'a, u8>) -> PyResult<ImageView<'a>> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let strides = img.strides();
    let stride_y = strides[0] as usize;
    let stride_x = strides[1];

    if stride_x == 1 {
        let required_size = if height > 0 && width > 0 {
            (height - 1) * stride_y + width
        } else {
            0
        };
        let data = unsafe { std::slice::from_raw_parts(img.data(), required_size) };
        ImageView::new(data, width, height, stride_y).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        Err(PyValueError::new_err("Array must be C-contiguous"))
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
        let subscriber = tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default());
        tracing::subscriber::set_global_default(subscriber).ok();
    }
}

#[pymodule]
fn locus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Detector>()?;
    m.add_class::<TagFamily>()?;
    m.add_class::<SegmentationConnectivity>()?;
    m.add_class::<CornerRefinementMode>()?;
    m.add_class::<DecodeMode>()?;
    m.add_class::<PoseEstimationMode>()?;
    m.add_class::<CameraIntrinsics>()?;
    m.add_class::<PyPose>()?;

    m.add_function(wrap_pyfunction!(create_detector, m)?)?;
    m.add_function(wrap_pyfunction!(init_tracy, m)?)?;
    Ok(())
}
