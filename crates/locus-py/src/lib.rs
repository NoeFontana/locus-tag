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

#[pyclass(eq, eq_int, skip_from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TagFamily {
    AprilTag16h5 = 0,
    AprilTag25h9 = 1,
    AprilTag36h10 = 2,
    AprilTag36h11 = 3,
    ArUco4x4_50 = 4,
    ArUco4x4_100 = 5,
}

impl From<TagFamily> for locus_core::TagFamily {
    fn from(f: TagFamily) -> Self {
        match f {
            TagFamily::AprilTag16h5 => locus_core::TagFamily::AprilTag16h5,
            TagFamily::AprilTag25h9 => locus_core::TagFamily::AprilTag25h9,
            TagFamily::AprilTag36h10 => locus_core::TagFamily::AprilTag36h10,
            TagFamily::AprilTag36h11 => locus_core::TagFamily::AprilTag36h11,
            TagFamily::ArUco4x4_50 => locus_core::TagFamily::ArUco4x4_50,
            TagFamily::ArUco4x4_100 => locus_core::TagFamily::ArUco4x4_100,
        }
    }
}

#[pyclass(eq, eq_int, from_py_object)]
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

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

#[pyclass(eq, eq_int, from_py_object)]
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

#[pyclass(eq, eq_int, from_py_object)]
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
            },
            decode_mode: match c.decode_mode {
                locus_core::config::DecodeMode::Hard => DecodeMode::Hard,
                locus_core::config::DecodeMode::Soft => DecodeMode::Soft,
            },
            max_hamming_error: c.max_hamming_error,
        }
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
        let detections = py.detach(|| {
            self.inner.detect(
                &view,
                core_intrinsics.as_ref(),
                tag_size,
                core_pose_mode,
                debug_telemetry,
            )
        });

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
            }
            dict.set_item("telemetry", tel_dict)?;
        } else {
            dict.set_item("telemetry", py.None())?;
        }

        Ok(dict)
    }

    /// Returns the current detector configuration.
    fn config(&self) -> PyDetectorConfig {
        PyDetectorConfig::from(*self.inner.config())
    }

    /// Update the tag families to be detected.
    fn set_families(&mut self, families: Vec<i32>) -> PyResult<()> {
        let mut core_families = Vec::new();
        for f in families {
            let family = match f {
                0 => locus_core::TagFamily::AprilTag16h5,
                1 => locus_core::TagFamily::AprilTag25h9,
                2 => locus_core::TagFamily::AprilTag36h10,
                3 => locus_core::TagFamily::AprilTag36h11,
                4 => locus_core::TagFamily::ArUco4x4_50,
                5 => locus_core::TagFamily::ArUco4x4_100,
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
        let family = match f {
            0 => locus_core::TagFamily::AprilTag16h5,
            1 => locus_core::TagFamily::AprilTag25h9,
            2 => locus_core::TagFamily::AprilTag36h10,
            3 => locus_core::TagFamily::AprilTag36h11,
            4 => locus_core::TagFamily::ArUco4x4_50,
            5 => locus_core::TagFamily::ArUco4x4_100,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid TagFamily value: {f}"
                )));
            },
        };
        builder = builder.with_family(family);
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

    m.add_function(wrap_pyfunction!(create_detector, m)?)?;
    m.add_function(wrap_pyfunction!(production_config, m)?)?;
    m.add_function(wrap_pyfunction!(fast_config, m)?)?;
    m.add_function(wrap_pyfunction!(init_tracy, m)?)?;
    Ok(())
}
