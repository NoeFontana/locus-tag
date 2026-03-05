//! Python bindings for the Locus Tag library.
#![allow(
    unsafe_code,
    clippy::unused_self,
    missing_docs,
    clippy::trivially_copy_pass_by_ref
)]

use locus_core::ImageView;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
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

impl From<SegmentationConnectivity> for locus_core::config::SegmentationConnectivity {
    fn from(c: SegmentationConnectivity) -> Self {
        match c {
            SegmentationConnectivity::Four => locus_core::config::SegmentationConnectivity::Four,
            SegmentationConnectivity::Eight => locus_core::config::SegmentationConnectivity::Eight,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum CornerRefinementMode {
    None = 0,
    Edge = 1,
    GridFit = 2,
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

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

impl From<CameraIntrinsics> for locus_core::CameraIntrinsics {
    fn from(c: CameraIntrinsics) -> Self {
        Self::new(c.fx, c.fy, c.cx, c.cy)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPose {
    #[pyo3(get)]
    pub quaternion: [f64; 4], // x, y, z, w
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
    #[pyo3(signature = (img, intrinsics=None, tag_size=None, pose_estimation_mode=PoseEstimationMode::Fast))]
    fn detect(
        &mut self,
        py: Python<'_>,
        img: PyReadonlyArray2<u8>,
        intrinsics: Option<CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_estimation_mode: PoseEstimationMode,
    ) -> PyResult<PyObject> {
        let view = prepare_image_view(&img)?;
        
        let core_intrinsics = intrinsics.map(locus_core::CameraIntrinsics::from);
        let core_pose_mode = locus_core::config::PoseEstimationMode::from(pose_estimation_mode);

        // 1. Run core pipeline
        let detections = py.allow_threads(|| {
            self.inner.detect(
                &view,
                core_intrinsics.as_ref(),
                tag_size,
                core_pose_mode,
            )
        });

        let n = detections.len();

        // Allocate NumPy arrays
        let ids_arr = PyArray1::<i32>::zeros(py, [n], false);
        let corners_arr = PyArray3::<f32>::zeros(py, [n, 4, 2], false);
        let error_rates_arr = PyArray1::<f32>::zeros(py, [n], false);

        // Perform memory mapping
        unsafe {
            let ids_slice = ids_arr.as_slice_mut().unwrap();
            let corners_slice = corners_arr.as_slice_mut().unwrap();
            let err_slice = error_rates_arr.as_slice_mut().unwrap();

            for (i, det) in detections.iter().enumerate() {
                ids_slice[i] = det.id as i32;
                for j in 0..4 {
                    corners_slice[i * 8 + j * 2] = det.corners[j][0] as f32;
                    corners_slice[i * 8 + j * 2 + 1] = det.corners[j][1] as f32;
                }
                err_slice[i] = det.hamming as f32;
            }
        }

        let dict = PyDict::new(py);
        dict.set_item("ids", ids_arr)?;
        dict.set_item("corners", corners_arr)?;
        dict.set_item("error_rates", error_rates_arr)?;

        // Poses: Vectorized (N, 7) layout: [tx, ty, tz, qx, qy, qz, qw]
        if intrinsics.is_some() && tag_size.is_some() {
            let poses_arr = PyArray2::<f32>::zeros(py, [n, 7], false);
            unsafe {
                let poses_slice = poses_arr.as_slice_mut().unwrap();
                for (i, det) in detections.iter().enumerate() {
                    if let Some(pose) = &det.pose {
                        let q = nalgebra::UnitQuaternion::from_matrix(&pose.rotation);
                        let t = pose.translation;
                        
                        let offset = i * 7;
                        poses_slice[offset] = t.x as f32;
                        poses_slice[offset + 1] = t.y as f32;
                        poses_slice[offset + 2] = t.z as f32;
                        poses_slice[offset + 3] = q.coords.x as f32;
                        poses_slice[offset + 4] = q.coords.y as f32;
                        poses_slice[offset + 5] = q.coords.z as f32;
                        poses_slice[offset + 6] = q.coords.w as f32;
                    }
                }
            }
            dict.set_item("poses", poses_arr)?;
        } else {
            dict.set_item("poses", py.None())?;
        }

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
        Err(PyValueError::new_err("Array must be C-contiguous. Call np.ascontiguousarray(image) first."))
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
