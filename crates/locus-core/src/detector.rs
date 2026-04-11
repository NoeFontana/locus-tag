use crate::batch::{DetectionBatch, DetectionBatchView};
use crate::config::DetectorConfig;
use crate::decoder::{TagDecoder, family_to_decoder};
use crate::error::DetectorError;
use crate::image::ImageView;
use bumpalo::Bump;

/// Per-thread mutable state for the detection pipeline.
///
/// Owns the arena allocator and the fixed-capacity SoA batch. Construct one
/// per thread and reuse across frames to preserve the zero-allocation hot-path.
pub struct FrameContext {
    /// Memory pool for ephemeral per-frame allocations.
    pub arena: Bump,
    /// Vectorized storage for quad candidates and results.
    pub batch: DetectionBatch,
    /// Reusable buffer for upscaling.
    pub upscale_buf: Vec<u8>,
}

impl FrameContext {
    /// Create a new frame context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
            batch: DetectionBatch::new(),
            upscale_buf: Vec::new(),
        }
    }

    /// Reset the state for a new frame.
    pub fn reset(&mut self) {
        self.arena.reset();
    }
}

impl Default for FrameContext {
    fn default() -> Self {
        Self::new()
    }
}

/// The primary entry point for the Locus perception library.
///
/// `Detector` encapsulates the entire detection pipeline. For concurrent
/// multi-frame detection, construct with a `max_concurrent_frames > 1` via
/// [`DetectorBuilder::with_max_concurrent_frames`] and call
/// [`Detector::detect_concurrent`].
pub struct Detector {
    engine: LocusEngine,
    ctx: FrameContext,
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector {
    /// Create a new detector with the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a builder to configure a new detector.
    #[must_use]
    pub fn builder() -> DetectorBuilder {
        DetectorBuilder::new()
    }

    /// Create a detector with custom pipeline configuration.
    #[must_use]
    pub fn with_config(config: DetectorConfig) -> Self {
        Self::builder().with_config(config).build()
    }

    /// Access the internal frame context (for advanced inspection or FFI).
    #[must_use]
    pub fn state(&self) -> &FrameContext {
        &self.ctx
    }

    /// Access the shared engine (for advanced FFI use only).
    #[must_use]
    pub fn engine(&self) -> &LocusEngine {
        &self.engine
    }

    /// Detect tags in the provided image.
    ///
    /// This method is the main execution pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`DetectorError`] if the input image cannot be decimated, upscaled,
    /// or if an intermediate image view cannot be constructed.
    pub fn detect(
        &mut self,
        img: &ImageView,
        intrinsics: Option<&crate::pose::CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_mode: crate::config::PoseEstimationMode,
        debug_telemetry: bool,
    ) -> Result<DetectionBatchView<'_>, DetectorError> {
        self.engine.detect_with_context(
            img,
            &mut self.ctx,
            intrinsics,
            tag_size,
            pose_mode,
            debug_telemetry,
        )
    }

    /// Detect tags in multiple frames concurrently using Rayon.
    ///
    /// Delegates to the internal [`LocusEngine`] pool. The pool is sized to
    /// `max_concurrent_frames` at construction time (see
    /// [`DetectorBuilder::with_max_concurrent_frames`]). Telemetry is not
    /// available via this method.
    pub fn detect_concurrent(
        &self,
        frames: &[ImageView<'_>],
        intrinsics: Option<&crate::pose::CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_mode: crate::config::PoseEstimationMode,
    ) -> Vec<Result<Vec<crate::Detection>, DetectorError>> {
        self.engine
            .detect_concurrent(frames, intrinsics, tag_size, pose_mode)
    }

    /// Clear all decoders and set new ones based on tag families.
    pub fn set_families(&mut self, families: &[crate::config::TagFamily]) {
        self.engine.set_families(families);
    }

    /// Get the current detector configuration.
    #[must_use]
    pub fn config(&self) -> DetectorConfig {
        self.engine.config()
    }

    /// Returns a cloned copy of the internal detection batch.
    /// Exclusively for benchmarking.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn bench_api_get_batch_cloned(&self) -> DetectionBatch {
        let mut new_batch = DetectionBatch::new();
        new_batch.corners.copy_from_slice(&self.ctx.batch.corners);
        new_batch
            .homographies
            .copy_from_slice(&self.ctx.batch.homographies);
        new_batch.ids.copy_from_slice(&self.ctx.batch.ids);
        new_batch.payloads.copy_from_slice(&self.ctx.batch.payloads);
        new_batch
            .error_rates
            .copy_from_slice(&self.ctx.batch.error_rates);
        new_batch.poses.copy_from_slice(&self.ctx.batch.poses);
        new_batch
            .status_mask
            .copy_from_slice(&self.ctx.batch.status_mask);
        new_batch
            .funnel_status
            .copy_from_slice(&self.ctx.batch.funnel_status);
        new_batch
            .corner_covariances
            .copy_from_slice(&self.ctx.batch.corner_covariances);
        new_batch
    }
}

/// Core detection pipeline — shared by [`Detector`] and [`LocusEngine`].
///
/// Runs the full pipeline (thresholding → segmentation → quad extraction →
/// homography → decoding → pose) using the provided mutable [`FrameContext`].
/// Returns a [`DetectionBatchView`] whose lifetime is tied to `ctx`.
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn run_detection_pipeline<'ctx>(
    config: &DetectorConfig,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    img: &ImageView,
    ctx: &'ctx mut FrameContext,
    intrinsics: Option<&crate::pose::CameraIntrinsics>,
    tag_size: Option<f64>,
    pose_mode: crate::config::PoseEstimationMode,
    debug_telemetry: bool,
) -> Result<DetectionBatchView<'ctx>, DetectorError> {
    ctx.reset();
    let state = ctx;

    let (detection_img, _effective_scale, refinement_img) = if config.decimation > 1 {
        let new_w = img.width / config.decimation;
        let new_h = img.height / config.decimation;
        let decimated_data = state.arena.alloc_slice_fill_copy(new_w * new_h, 0u8);
        let decimated_img = img
            .decimate_to(config.decimation, decimated_data)
            .map_err(DetectorError::Preprocessing)?;
        (decimated_img, 1.0 / config.decimation as f64, *img)
    } else if config.upscale_factor > 1 {
        let new_w = img.width * config.upscale_factor;
        let new_h = img.height * config.upscale_factor;
        state.upscale_buf.resize(new_w * new_h, 0);

        let upscaled_img = img
            .upscale_to(config.upscale_factor, &mut state.upscale_buf)
            .map_err(DetectorError::Preprocessing)?;
        (upscaled_img, config.upscale_factor as f64, upscaled_img)
    } else {
        (*img, 1.0, *img)
    };

    let img = &detection_img;

    // 1b. Optional Laplacian sharpening
    let sharpened_img = if config.enable_sharpening {
        let sharpened = state
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);
        crate::filter::laplacian_sharpen(img, sharpened);

        ImageView::new(sharpened, img.width, img.height, img.width)
            .map_err(DetectorError::InvalidImage)?
    } else {
        *img
    };

    let binarized = state
        .arena
        .alloc_slice_fill_copy(img.width * img.height, 0u8);
    let threshold_map = state
        .arena
        .alloc_slice_fill_copy(img.width * img.height, 0u8);

    // 1. Thresholding & 2. Segmentation & 3. Quad Extraction
    let (n, unrefined) = {
        let engine = crate::threshold::ThresholdEngine::from_config(config);
        let tile_stats = engine.compute_tile_stats(&state.arena, &sharpened_img);
        engine.apply_threshold_with_map(
            &state.arena,
            &sharpened_img,
            &tile_stats,
            binarized,
            threshold_map,
        );

        // 2. Segmentation (SIMD Fused RLE + LSL)
        let label_result = crate::simd_ccl_fusion::label_components_lsl(
            &state.arena,
            &sharpened_img,
            threshold_map,
            config.segmentation_connectivity == crate::config::SegmentationConnectivity::Eight,
            config.quad_min_area,
        );

        // 3. Quad Extraction (SoA)
        let (n, unrefined) = crate::quad::extract_quads_soa(
            &mut state.batch,
            &sharpened_img,
            &label_result,
            config,
            config.decimation,
            &refinement_img,
            debug_telemetry,
        );

        // 3.5 Fast-Path Funnel Gate
        // Rejects candidates early based on boundary contrast
        crate::funnel::apply_funnel_gate(
            &mut state.batch,
            n,
            &sharpened_img,
            &tile_stats,
            config.threshold_tile_size,
            config.decoder_min_contrast,
            1.0 / config.decimation as f64,
        );

        (n, unrefined)
    };

    // Compute subpixel jitter if requested
    let mut jitter_ptr = std::ptr::null();
    let mut num_jitter = 0;
    if let (true, Some(unrefined_pts)) = (debug_telemetry, unrefined) {
        num_jitter = unrefined_pts.len();
        // Store 4 corners * 2 (dx, dy) per candidate = 8 floats
        let jitter = state.arena.alloc_slice_fill_copy(num_jitter * 8, 0.0f32);
        for (i, unrefined_corners) in unrefined_pts.iter().enumerate() {
            for (j, unrefined_corner) in unrefined_corners.iter().enumerate() {
                let dx = state.batch.corners[i][j].x - unrefined_corner.x as f32;
                let dy = state.batch.corners[i][j].y - unrefined_corner.y as f32;
                jitter[i * 8 + j * 2] = dx;
                jitter[i * 8 + j * 2 + 1] = dy;
            }
        }
        jitter_ptr = jitter.as_ptr();
    }

    // 4. Homography Pass (SoA)
    crate::decoder::compute_homographies_soa(
        &state.batch.corners[0..n],
        &state.batch.status_mask[0..n],
        &mut state.batch.homographies[0..n],
    );

    // Optional: GWLF Refinement
    let mut gwlf_fallback_count = 0;
    let mut gwlf_avg_delta = 0.0f32;
    if config.refinement_mode == crate::config::CornerRefinementMode::Gwlf {
        let mut total_delta = 0.0f32;
        let mut count = 0;
        for i in 0..n {
            let coarse = [
                [state.batch.corners[i][0].x, state.batch.corners[i][0].y],
                [state.batch.corners[i][1].x, state.batch.corners[i][1].y],
                [state.batch.corners[i][2].x, state.batch.corners[i][2].y],
                [state.batch.corners[i][3].x, state.batch.corners[i][3].y],
            ];
            if let Some((refined, covs)) = crate::gwlf::refine_quad_gwlf_with_cov(
                &refinement_img,
                &coarse,
                config.gwlf_transversal_alpha,
            ) {
                for j in 0..4 {
                    let dx = refined[j][0] - coarse[j][0];
                    let dy = refined[j][1] - coarse[j][1];
                    total_delta += (dx * dx + dy * dy).sqrt();
                    count += 1;

                    state.batch.corners[i][j].x = refined[j][0];
                    state.batch.corners[i][j].y = refined[j][1];

                    // Store 2x2 covariance (4 floats) for each corner
                    state.batch.corner_covariances[i][j * 4] = covs[j][(0, 0)] as f32;
                    state.batch.corner_covariances[i][j * 4 + 1] = covs[j][(0, 1)] as f32;
                    state.batch.corner_covariances[i][j * 4 + 2] = covs[j][(1, 0)] as f32;
                    state.batch.corner_covariances[i][j * 4 + 3] = covs[j][(1, 1)] as f32;
                }
            } else {
                gwlf_fallback_count += 1;
            }
        }
        if count > 0 {
            gwlf_avg_delta = total_delta / count as f32;
        }

        // Recompute homographies after refinement
        crate::decoder::compute_homographies_soa(
            &state.batch.corners[0..n],
            &state.batch.status_mask[0..n],
            &mut state.batch.homographies[0..n],
        );
    }

    // 5. Decoding Pass (SoA)
    crate::decoder::decode_batch_soa(&mut state.batch, n, &refinement_img, decoders, config);

    // Partition valid candidates to the front [0..v]
    let v = state.batch.partition(n);

    // 6. Pose Refinement (SoA)
    let (repro_errors_ptr, num_repro) = run_pose_refinement(
        &mut state.batch,
        &state.arena,
        v,
        intrinsics,
        tag_size,
        &refinement_img,
        pose_mode,
        config,
        debug_telemetry,
    );

    // Detectors return corners at pixel centers (indices + 0.5) following OpenCV conventions.
    // No additional adjustment needed as the internal pipeline is now unbiased.

    let telemetry = if debug_telemetry {
        Some(crate::batch::TelemetryPayload {
            binarized_ptr: binarized.as_ptr(),
            threshold_map_ptr: threshold_map.as_ptr(),
            subpixel_jitter_ptr: jitter_ptr,
            num_jitter,
            reprojection_errors_ptr: repro_errors_ptr,
            num_reprojection: num_repro,
            gwlf_fallback_count,
            gwlf_avg_delta,
            width: img.width,
            height: img.height,
            stride: img.width,
        })
    } else {
        None
    };

    Ok(state.batch.view_with_telemetry(v, n, telemetry))
}

/// Run pose refinement on valid candidates and optionally compute reprojection errors.
///
/// Returns `(reprojection_errors_ptr, num_reprojection)` for telemetry.
#[allow(clippy::too_many_arguments)]
fn run_pose_refinement(
    batch: &mut crate::batch::DetectionBatch,
    arena: &bumpalo::Bump,
    v: usize,
    intrinsics: Option<&crate::pose::CameraIntrinsics>,
    tag_size: Option<f64>,
    refinement_img: &ImageView,
    pose_mode: crate::config::PoseEstimationMode,
    config: &DetectorConfig,
    debug_telemetry: bool,
) -> (*const f32, usize) {
    let mut repro_errors_ptr = std::ptr::null();
    let mut num_repro = 0;

    if let (Some(intr), Some(size)) = (intrinsics, tag_size) {
        crate::pose::refine_poses_soa_with_config(
            batch,
            v,
            intr,
            size,
            Some(refinement_img),
            pose_mode,
            config,
        );

        if debug_telemetry {
            num_repro = v;
            let repro_errors = arena.alloc_slice_fill_copy(num_repro, 0.0f32);

            let model_pts = [
                nalgebra::Vector3::new(0.0, 0.0, 0.0),
                nalgebra::Vector3::new(size, 0.0, 0.0),
                nalgebra::Vector3::new(size, size, 0.0),
                nalgebra::Vector3::new(0.0, size, 0.0),
            ];

            for (i, repro_error) in repro_errors.iter_mut().enumerate().take(v) {
                let det_pose_data = batch.poses[i].data;
                let det_t = nalgebra::Vector3::new(
                    f64::from(det_pose_data[0]),
                    f64::from(det_pose_data[1]),
                    f64::from(det_pose_data[2]),
                );
                let det_q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    f64::from(det_pose_data[6]),
                    f64::from(det_pose_data[3]),
                    f64::from(det_pose_data[4]),
                    f64::from(det_pose_data[5]),
                ));

                let pose = crate::pose::Pose {
                    rotation: *det_q.to_rotation_matrix().matrix(),
                    translation: det_t,
                };

                let mut sum_sq_err = 0.0;
                for (j, model_pt) in model_pts.iter().enumerate() {
                    let proj = pose.project(model_pt, intr);
                    let dx = proj[0] - f64::from(batch.corners[i][j].x);
                    let dy = proj[1] - f64::from(batch.corners[i][j].y);
                    sum_sq_err += dx * dx + dy * dy;
                }
                *repro_error = (sum_sq_err / 4.0).sqrt() as f32;
            }
            repro_errors_ptr = repro_errors.as_ptr();
        }
    }

    (repro_errors_ptr, num_repro)
}

/// A builder for configuring and instantiating a [`Detector`].
pub struct DetectorBuilder {
    config: DetectorConfig,
    families: Vec<crate::config::TagFamily>,
    max_concurrent_frames: usize,
}

impl DetectorBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DetectorConfig::default(),
            families: Vec::new(),
            max_concurrent_frames: 1,
        }
    }

    /// Use an existing configuration.
    #[must_use]
    pub fn with_config(mut self, config: DetectorConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the decimation factor for the input image.
    #[must_use]
    pub fn with_decimation(mut self, decimation: usize) -> Self {
        self.config.decimation = decimation;
        self
    }

    /// Add a tag family to be detected.
    #[must_use]
    pub fn with_family(mut self, family: crate::config::TagFamily) -> Self {
        if !self.families.contains(&family) {
            self.families.push(family);
        }
        self
    }

    /// Set the thread count for parallel processing.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.config.nthreads = threads;
        self
    }

    /// Set the upscale factor for detecting small tags.
    #[must_use]
    pub fn with_upscale_factor(mut self, factor: usize) -> Self {
        self.config.upscale_factor = factor;
        self
    }

    /// Set the corner refinement mode.
    #[must_use]
    pub fn with_corner_refinement(mut self, mode: crate::config::CornerRefinementMode) -> Self {
        self.config.refinement_mode = mode;
        self
    }

    /// Set the decoding mode (Hard vs Soft).
    #[must_use]
    pub fn with_decode_mode(mut self, mode: crate::config::DecodeMode) -> Self {
        self.config.decode_mode = mode;
        self
    }

    /// Set the segmentation connectivity (4-way or 8-way).
    #[must_use]
    pub fn with_connectivity(
        mut self,
        connectivity: crate::config::SegmentationConnectivity,
    ) -> Self {
        self.config.segmentation_connectivity = connectivity;
        self
    }

    /// Set the tile size for adaptive thresholding.
    #[must_use]
    pub fn with_threshold_tile_size(mut self, size: usize) -> Self {
        self.config.threshold_tile_size = size;
        self
    }

    /// Set the minimum intensity range for valid tiles.
    #[must_use]
    pub fn with_threshold_min_range(mut self, range: u8) -> Self {
        self.config.threshold_min_range = range;
        self
    }

    /// Set the constant subtracted from local mean in adaptive thresholding.
    #[must_use]
    pub fn with_adaptive_threshold_constant(mut self, c: i16) -> Self {
        self.config.adaptive_threshold_constant = c;
        self
    }

    /// Set the minimum quad area.
    #[must_use]
    pub fn with_quad_min_area(mut self, area: u32) -> Self {
        self.config.quad_min_area = area;
        self
    }

    /// Set the minimum fill ratio.
    #[must_use]
    pub fn with_quad_min_fill_ratio(mut self, ratio: f32) -> Self {
        self.config.quad_min_fill_ratio = ratio;
        self
    }

    /// Set the minimum edge alignment score.
    #[must_use]
    pub fn with_quad_min_edge_score(mut self, score: f64) -> Self {
        self.config.quad_min_edge_score = score;
        self
    }

    /// Set the maximum number of Hamming errors allowed.
    #[must_use]
    pub fn with_max_hamming_error(mut self, errors: u32) -> Self {
        self.config.max_hamming_error = errors;
        self
    }

    /// Set the minimum contrast for decoder bit classification.
    #[must_use]
    pub fn with_decoder_min_contrast(mut self, contrast: f64) -> Self {
        self.config.decoder_min_contrast = contrast;
        self
    }

    /// Set the GWLF transversal alpha.
    #[must_use]
    pub fn with_gwlf_transversal_alpha(mut self, alpha: f64) -> Self {
        self.config.gwlf_transversal_alpha = alpha;
        self
    }

    /// Set the maximum elongation allowed for a component.
    #[must_use]
    pub fn with_quad_max_elongation(mut self, elongation: f64) -> Self {
        self.config.quad_max_elongation = elongation;
        self
    }

    /// Set the minimum pixel density required to pass the moments gate.
    #[must_use]
    pub fn with_quad_min_density(mut self, density: f64) -> Self {
        self.config.quad_min_density = density;
        self
    }

    /// Set the quad extraction mode.
    #[must_use]
    pub fn with_quad_extraction_mode(mut self, mode: crate::config::QuadExtractionMode) -> Self {
        self.config.quad_extraction_mode = mode;
        self
    }

    /// Enable or disable Laplacian sharpening.
    #[must_use]
    pub fn with_sharpening(mut self, enable: bool) -> Self {
        self.config.enable_sharpening = enable;
        self
    }

    /// Build the [`Detector`] instance.
    ///
    /// The internal pool is sized to `max_concurrent_frames` (default 1).
    /// For purely single-frame use (`detect`), the default is optimal.
    /// Increase via [`DetectorBuilder::with_max_concurrent_frames`] when
    /// calling [`Detector::detect_concurrent`] with large batches.
    #[must_use]
    pub fn build(self) -> Detector {
        let pool_size = self.max_concurrent_frames.max(1);
        let decoders = self.build_decoders();
        let engine = LocusEngine::new(self.config, decoders, pool_size);
        Detector {
            engine,
            ctx: FrameContext::new(),
        }
    }

    /// Build the detector, validating the configuration first.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if the configuration is invalid (e.g., incompatible
    /// `quad_extraction_mode` + `refinement_mode` or `decode_mode` combination).
    pub fn validated_build(self) -> Result<Detector, crate::error::ConfigError> {
        self.config.validate()?;
        Ok(self.build())
    }

    /// Set the number of frames that [`Detector::detect_concurrent`] can process
    /// simultaneously. This pre-allocates that many [`FrameContext`] objects in
    /// the internal pool.
    ///
    /// Defaults to `1` (sequential). For batch workloads set this to the
    /// expected batch size or `rayon::current_num_threads()`.
    #[must_use]
    pub fn with_max_concurrent_frames(mut self, n: usize) -> Self {
        self.max_concurrent_frames = n.max(1);
        self
    }

    /// Build a standalone [`LocusEngine`] with an explicit pool size.
    ///
    /// Intended for advanced Rust users who want to manage their own
    /// `FrameContext` lifecycle. Python users should use `build()` instead.
    ///
    /// `pool_size = 0` falls back to `rayon::current_num_threads()`.
    #[must_use]
    pub fn build_engine(self) -> LocusEngine {
        let pool_size = if self.max_concurrent_frames == 0 {
            rayon::current_num_threads()
        } else {
            self.max_concurrent_frames
        };
        let decoders = self.build_decoders();
        LocusEngine::new(self.config, decoders, pool_size)
    }

    fn build_decoders(&self) -> Vec<Box<dyn TagDecoder + Send + Sync>> {
        let families = if self.families.is_empty() {
            vec![crate::config::TagFamily::AprilTag36h11]
        } else {
            self.families.clone()
        };
        families.into_iter().map(family_to_decoder).collect()
    }
}

impl Default for DetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LocusEngine
// ============================================================================

/// A shared, thread-safe detection engine.
///
/// Separates the immutable pipeline configuration from mutable per-frame state,
/// enabling one engine to be shared across many threads. Each concurrent caller
/// either supplies its own [`FrameContext`] (explicit API) or leases one from the
/// engine's internal lock-free pool (implicit API via [`LocusEngine::detect_concurrent`]).
///
/// # Construction
///
/// ```ignore
/// let engine = locus_core::DetectorBuilder::new()
///     .with_family(TagFamily::AprilTag36h11)
///     .build_engine();
/// ```
pub struct LocusEngine {
    config: DetectorConfig,
    decoders: Vec<Box<dyn TagDecoder + Send + Sync>>,
    /// Lock-free pool of reusable frame contexts.
    pool: crossbeam_queue::ArrayQueue<Box<FrameContext>>,
}

impl LocusEngine {
    /// Create a new engine with a pre-populated context pool.
    ///
    /// `pool_size` must be ≥ 1. Callers should use [`DetectorBuilder::build_engine`]
    /// rather than calling this directly.
    #[must_use]
    pub fn new(
        config: DetectorConfig,
        decoders: Vec<Box<dyn TagDecoder + Send + Sync>>,
        pool_size: usize,
    ) -> Self {
        let capacity = pool_size.max(1);
        let pool = crossbeam_queue::ArrayQueue::new(capacity);
        for _ in 0..pool_size {
            // Box<FrameContext> to heap-allocate the ~200 KB DetectionBatch.
            let _ = pool.push(Box::new(FrameContext::new()));
        }
        Self {
            config,
            decoders,
            pool,
        }
    }

    /// Run the detection pipeline using an explicitly supplied context.
    ///
    /// The returned [`DetectionBatchView`] borrows from `ctx`; drop the view before
    /// calling this method again on the same context or returning it to the pool.
    ///
    /// # Errors
    ///
    /// Returns [`DetectorError`] if the input image is invalid.
    pub fn detect_with_context<'ctx>(
        &self,
        img: &ImageView,
        ctx: &'ctx mut FrameContext,
        intrinsics: Option<&crate::pose::CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_mode: crate::config::PoseEstimationMode,
        debug_telemetry: bool,
    ) -> Result<DetectionBatchView<'ctx>, DetectorError> {
        run_detection_pipeline(
            &self.config,
            &self.decoders,
            img,
            ctx,
            intrinsics,
            tag_size,
            pose_mode,
            debug_telemetry,
        )
    }

    /// Detect tags in multiple frames concurrently using Rayon.
    ///
    /// Pool contexts are leased to Rayon threads; each result is assembled into an
    /// owned `Vec<Detection>` before the context is returned. Telemetry is
    /// unavailable in this mode (debug overhead would outlive the arena).
    ///
    /// If the pool is exhausted (more concurrent callers than pool size), a
    /// temporary overflow context is allocated and discarded after use.
    pub fn detect_concurrent(
        &self,
        frames: &[ImageView<'_>],
        intrinsics: Option<&crate::pose::CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_mode: crate::config::PoseEstimationMode,
    ) -> Vec<Result<Vec<crate::Detection>, DetectorError>> {
        use rayon::prelude::*;
        frames
            .par_iter()
            .map(|img| {
                // Pop a context from the pool, or create a temporary overflow context.
                let (mut ctx, to_pool) = if let Some(c) = self.pool.pop() {
                    (c, true)
                } else {
                    (Box::new(FrameContext::new()), false)
                };

                let owned = run_detection_pipeline(
                    &self.config,
                    &self.decoders,
                    img,
                    &mut ctx,
                    intrinsics,
                    tag_size,
                    pose_mode,
                    false, // telemetry disabled: arena pointers would not survive pool return
                )
                .map(|view| {
                    // Extract owned data BEFORE releasing ctx back to the pool.
                    // `view` borrows from ctx.batch; reassemble_owned() copies into Vec.
                    // The view is implicitly dropped at the end of this closure.
                    view.reassemble_owned()
                });

                if to_pool {
                    let _ = self.pool.push(ctx);
                }
                owned
            })
            .collect()
    }

    /// Clear all decoders and replace them with the given tag families.
    pub fn set_families(&mut self, families: &[crate::config::TagFamily]) {
        self.decoders.clear();
        for &family in families {
            self.decoders
                .push(crate::decoder::family_to_decoder(family));
        }
    }

    /// Get the current detector configuration.
    #[must_use]
    pub fn config(&self) -> DetectorConfig {
        self.config
    }
}
