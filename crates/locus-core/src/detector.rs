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
    /// Reusable buffer for the ROI super-resolution rescue pass. Split per
    /// rescue into two sub-slices: one for the upscaled output (plus SIMD
    /// padding) and one for the Lanczos horizontal-pass scratch.
    pub rescue_buf: Vec<u8>,
}

impl FrameContext {
    /// Create a new frame context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
            batch: DetectionBatch::new(),
            upscale_buf: Vec::new(),
            rescue_buf: Vec::new(),
        }
    }

    /// Create a new frame context pre-sized for the given configuration.
    ///
    /// When `config.roi_rescue.enabled` is set, the arena is allocated with
    /// enough capacity to hold the expected peak working set, and
    /// `rescue_buf` is grown to fit `max_rescues_per_frame` ROIs at the
    /// configured `upscale_factor` / `max_roi_side_px`. Otherwise this is
    /// byte-equivalent to [`FrameContext::new`].
    #[must_use]
    pub fn new_for_config(config: &DetectorConfig) -> Self {
        let arena_hint = arena_capacity_hint(config);
        let mut ctx = Self {
            arena: if arena_hint > 0 {
                Bump::with_capacity(arena_hint)
            } else {
                Bump::new()
            },
            batch: DetectionBatch::new(),
            upscale_buf: Vec::new(),
            rescue_buf: Vec::new(),
        };
        ctx.ensure_rescue_capacity(config);
        ctx
    }

    /// Reset the state for a new frame.
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Grow `rescue_buf` to fit the configured policy. No-op when rescue is
    /// disabled or the buffer already has sufficient capacity.
    pub(crate) fn ensure_rescue_capacity(&mut self, config: &DetectorConfig) {
        let needed = rescue_buf_size(config);
        if needed > self.rescue_buf.len() {
            self.rescue_buf.resize(needed, 0);
        }
    }
}

/// Per-slot layout of `rescue_buf`: `(out_slot_bytes, scratch_slot_bytes)`.
/// Returns `None` when rescue is disabled. `out_slot_bytes` already includes
/// the 3-byte SIMD padding so a single slot can be split cleanly into the
/// `out` and `scratch` arguments of [`crate::decoder::decode_with_rescue`].
pub(crate) fn rescue_slot_layout(config: &DetectorConfig) -> Option<(usize, usize)> {
    if !config.roi_rescue.enabled {
        return None;
    }
    let side = usize::from(config.roi_rescue.max_roi_side_px);
    let factor = usize::from(config.roi_rescue.upscale_factor);
    let out_plane = (side * factor) * (side * factor);
    let scratch_plane = match config.roi_rescue.interpolation {
        crate::config::RescueInterpolation::Bilinear => 0,
        crate::config::RescueInterpolation::Lanczos3 => (side * factor) * side,
    };
    // +3 bytes on the out plane keeps the x86_64 AVX2 gather path active in
    // sample_bilinear_v8 (see upscale_roi_to_buf for the padding contract).
    Some((out_plane + 3, scratch_plane))
}

/// Bytes of `rescue_buf` needed for one full frame of rescue attempts:
/// `max_rescues_per_frame * (out_slot + scratch_slot)`. Returns `0` when
/// rescue is disabled.
pub(crate) fn rescue_buf_size(config: &DetectorConfig) -> usize {
    let Some((out_slot, scratch_slot)) = rescue_slot_layout(config) else {
        return 0;
    };
    (out_slot + scratch_slot) * usize::from(config.roi_rescue.max_rescues_per_frame)
}

/// Peak per-frame `Bump` working set. Used as the `with_capacity` seed so the
/// first frame does not trigger a chunk realloc on long-running services.
fn arena_capacity_hint(config: &DetectorConfig) -> usize {
    // Conservative pre-size only when rescue is enabled; otherwise keep the
    // historical `Bump::new()` behavior.
    if !config.roi_rescue.enabled {
        return 0;
    }
    // Leave room for the existing steady-state per-frame allocations
    // (binarized, threshold_map, sharpened, decimation/upscale buffer) plus a
    // headroom factor for quad contour scratch. Empirically sized at
    // 2.5 × max(1080p single-channel) = ~5 MB to cover typical HD scenes.
    5 * 1024 * 1024
}

impl Default for FrameContext {
    fn default() -> Self {
        Self::new()
    }
}

/// ROI super-resolution rescue pass. Runs on candidates marked
/// `FailedDecode` that passed the fast-path funnel, ordered by first-pass
/// Hamming distance (ascending), capped by `max_rescues_per_frame`.
///
/// Successful promotions flip `status_mask[i]` to `Valid` so the subsequent
/// `partition` groups rescued tags alongside first-pass valids. Telemetry
/// columns (`rescue_attempted`, `rescue_hamming`) are written only when
/// `debug_telemetry` is set.
#[allow(clippy::too_many_arguments)]
fn run_rescue_stage(
    batch: &mut DetectionBatch,
    arena: &Bump,
    rescue_buf: &mut [u8],
    n: usize,
    config: &DetectorConfig,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    refinement_img: &ImageView,
    debug_telemetry: bool,
) {
    use crate::batch::{CandidateState, FunnelStatus};

    if n == 0 {
        return;
    }
    let Some((out_slot_len, scratch_slot_len)) = rescue_slot_layout(config) else {
        return;
    };
    let slot_len = out_slot_len + scratch_slot_len;

    // Arena-allocated index list of funnel-survivors that failed to decode.
    let indices = arena.alloc_slice_fill_copy(n, 0u16);
    let mut cand_count = 0usize;
    for i in 0..n {
        if batch.status_mask[i] == CandidateState::FailedDecode
            && batch.funnel_status[i] == FunnelStatus::PassedContrast
        {
            indices[cand_count] = i as u16;
            cand_count += 1;
        }
    }
    if cand_count == 0 {
        return;
    }
    let indices = &mut indices[..cand_count];

    // Prioritize lowest first-pass Hamming — those are the closest to the
    // max-hamming-error cliff, where super-resolution has the highest win
    // rate vs false-positive cost. `error_rates` holds non-negative Hamming
    // distances, so `total_cmp` is safe and branch-free.
    indices.sort_unstable_by(|a, b| {
        batch.error_rates[usize::from(*a)].total_cmp(&batch.error_rates[usize::from(*b)])
    });

    // `rescue_buf` is pre-sized to `slot_len * max_rescues_per_frame` by
    // `ensure_rescue_capacity` at frame start, so the per-frame cap is the
    // only binding constraint.
    let take = indices
        .len()
        .min(usize::from(config.roi_rescue.max_rescues_per_frame));
    debug_assert!(rescue_buf.len() >= take * slot_len);

    for (&packed_idx, slot) in indices
        .iter()
        .zip(rescue_buf.chunks_exact_mut(slot_len))
        .take(take)
    {
        let idx = usize::from(packed_idx);
        let (out_slot, rest) = slot.split_at_mut(out_slot_len);
        let scratch_slot = &mut rest[..scratch_slot_len];

        let promoted = crate::decoder::decode_with_rescue(
            batch,
            idx,
            refinement_img,
            decoders,
            config,
            out_slot,
            scratch_slot,
        );
        if debug_telemetry {
            batch.rescue_attempted[idx] = 1;
            if promoted {
                // `decode_with_rescue` wrote `error_rates[idx] = hamming` on
                // success; clamp guarantees `u8` fit regardless of
                // `rescue_max_hamming` (which is itself a `u8`).
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let h = batch.error_rates[idx].clamp(0.0, 255.0) as u8;
                batch.rescue_hamming[idx] = h;
            }
        }
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
    min_outer_dim: u32,
    img: &ImageView,
    ctx: &'ctx mut FrameContext,
    intrinsics: Option<&crate::pose::CameraIntrinsics>,
    tag_size: Option<f64>,
    pose_mode: crate::config::PoseEstimationMode,
    debug_telemetry: bool,
) -> Result<DetectionBatchView<'ctx>, DetectorError> {
    let has_distortion = intrinsics.is_some_and(|k| k.distortion.is_distorted());
    if has_distortion && config.any_route_uses_edlines() {
        return Err(DetectorError::Config(
            crate::error::ConfigError::EdLinesUnsupportedWithDistortion,
        ));
    }

    ctx.reset();
    ctx.ensure_rescue_capacity(config);
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

        // Dual refinement views for the adaptive policy router.
        // Under `Static`, extract_single_quad only consults `refinement_low`
        // (byte-identical to today's single-view behavior). Under
        // `AdaptivePpb`, the low-PPB route reads `refinement_low` and the
        // high-PPB (EdLines-preferred) route reads `refinement_high`, which
        // aliases the un-sharpened source so EdLines operates on its
        // preferred input.
        let refinement_low = refinement_img;
        let refinement_high = refinement_img;

        // 3. Quad Extraction (SoA). Distorted cameras run RDP in straight
        // space via `_with_camera`; the `_` arm is the pinhole flow.
        #[cfg(feature = "non_rectified")]
        let (n, unrefined) = match intrinsics.map(|k| (k, k.distortion)) {
            Some((k, crate::pose::DistortionCoeffs::BrownConrady { k1, k2, p1, p2, k3 })) => {
                let model = crate::camera::BrownConradyModel { k1, k2, p1, p2, k3 };
                crate::quad::extract_quads_soa_with_camera(
                    &mut state.batch,
                    &sharpened_img,
                    &label_result,
                    config,
                    config.decimation,
                    &refinement_low,
                    &refinement_high,
                    min_outer_dim,
                    debug_telemetry,
                    &model,
                    k,
                )
            },
            Some((k, crate::pose::DistortionCoeffs::KannalaBrandt { k1, k2, k3, k4 })) => {
                let model = crate::camera::KannalaBrandtModel { k1, k2, k3, k4 };
                crate::quad::extract_quads_soa_with_camera(
                    &mut state.batch,
                    &sharpened_img,
                    &label_result,
                    config,
                    config.decimation,
                    &refinement_low,
                    &refinement_high,
                    min_outer_dim,
                    debug_telemetry,
                    &model,
                    k,
                )
            },
            _ => crate::quad::extract_quads_soa(
                &mut state.batch,
                &sharpened_img,
                &label_result,
                config,
                config.decimation,
                &refinement_low,
                &refinement_high,
                min_outer_dim,
                debug_telemetry,
            ),
        };
        #[cfg(not(feature = "non_rectified"))]
        let (n, unrefined) = crate::quad::extract_quads_soa(
            &mut state.batch,
            &sharpened_img,
            &label_result,
            config,
            config.decimation,
            &refinement_low,
            &refinement_high,
            min_outer_dim,
            debug_telemetry,
        );

        // 3.5 Fast-Path Funnel Gate — rejects candidates early based on
        // boundary contrast. The midpoint sampling assumes straight-line
        // edges, which is violated under distortion (midpoint falls inside
        // the tag and produces a spurious 0-contrast reject). Correctness
        // without the funnel is covered by `decode_batch_soa_with_camera`.
        if !has_distortion {
            crate::funnel::apply_funnel_gate(
                &mut state.batch,
                n,
                &sharpened_img,
                &tile_stats,
                config.threshold_tile_size,
                config.decoder_min_contrast,
                1.0 / config.decimation as f64,
            );
        }

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

    // 5. Decoding Pass (SoA) — dispatch on distortion model
    // For rectified cameras (PinholeModel or no intrinsics), the compiler eliminates
    // the distortion path entirely via monomorphization of CameraModel::IS_RECTIFIED.
    match intrinsics.map(|k| &k.distortion) {
        #[cfg(feature = "non_rectified")]
        Some(crate::pose::DistortionCoeffs::BrownConrady { k1, k2, p1, p2, k3 }) => {
            let model = crate::camera::BrownConradyModel {
                k1: *k1,
                k2: *k2,
                p1: *p1,
                p2: *p2,
                k3: *k3,
            };
            crate::decoder::decode_batch_soa_with_camera(
                &mut state.batch,
                n,
                &refinement_img,
                decoders,
                config,
                intrinsics,
                &model,
            );
        },
        #[cfg(feature = "non_rectified")]
        Some(crate::pose::DistortionCoeffs::KannalaBrandt { k1, k2, k3, k4 }) => {
            let model = crate::camera::KannalaBrandtModel {
                k1: *k1,
                k2: *k2,
                k3: *k3,
                k4: *k4,
            };
            crate::decoder::decode_batch_soa_with_camera(
                &mut state.batch,
                n,
                &refinement_img,
                decoders,
                config,
                intrinsics,
                &model,
            );
        },
        _ => {
            // No distortion or no intrinsics — use the existing SIMD path with PinholeModel.
            crate::decoder::decode_batch_soa(
                &mut state.batch,
                n,
                &refinement_img,
                decoders,
                config,
            );
        },
    }

    // 5.5 ROI Super-Resolution Rescue — opt-in second-chance pass for
    // candidates whose first-pass decode missed by a small Hamming delta.
    // Skipped under distortion: the rescue homography uses
    // `Homography::square_to_quad`, which is only valid for the pinhole
    // flow; distortion-aware sampling lives in `decode_batch_soa_with_camera`.
    if config.roi_rescue.enabled && !has_distortion {
        run_rescue_stage(
            &mut state.batch,
            &state.arena,
            &mut state.rescue_buf,
            n,
            config,
            decoders,
            &refinement_img,
            debug_telemetry,
        );
    }

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
            routed_to_ptr: state.batch.routed_to.as_ptr(),
            ppb_estimate_ptr: state.batch.ppb_estimate.as_ptr(),
            num_routed: n,
            rescue_attempted_ptr: if config.roi_rescue.enabled {
                state.batch.rescue_attempted.as_ptr()
            } else {
                std::ptr::null()
            },
            rescue_hamming_ptr: if config.roi_rescue.enabled {
                state.batch.rescue_hamming.as_ptr()
            } else {
                std::ptr::null()
            },
            num_rescued: if config.roi_rescue.enabled { n } else { 0 },
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

    /// Set the quad extraction policy (per-candidate dispatch).
    ///
    /// Under [`QuadExtractionPolicy::Static`] (the default) the detector honours
    /// `with_quad_extraction_mode` and `with_corner_refinement` for every
    /// candidate. Under [`QuadExtractionPolicy::AdaptivePpb`], those settings
    /// are overridden per-candidate by the nested low/high route configuration.
    ///
    /// Validation runs when the final [`DetectorConfig`] is consumed by the
    /// pipeline, so invalid combinations (e.g. EdLines paired with Erf on a
    /// route, degenerate routes) surface as `DetectorError::Config(...)` at
    /// `detect()` time.
    ///
    /// [`QuadExtractionPolicy`]: crate::config::QuadExtractionPolicy
    /// [`QuadExtractionPolicy::Static`]: crate::config::QuadExtractionPolicy::Static
    /// [`QuadExtractionPolicy::AdaptivePpb`]: crate::config::QuadExtractionPolicy::AdaptivePpb
    #[must_use]
    pub fn with_extraction_policy(mut self, policy: crate::config::QuadExtractionPolicy) -> Self {
        self.config.quad_extraction_policy = policy;
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
        let ctx = FrameContext::new_for_config(&engine.config);
        Detector { engine, ctx }
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
    /// Minimum outer tag dimension (data grid + 2-pixel border) across
    /// configured decoders. Drives the per-candidate PPB estimate under
    /// [`crate::config::QuadExtractionPolicy::AdaptivePpb`]. Cached to avoid
    /// re-walking the decoder list on every call.
    min_outer_dim: u32,
    /// Lock-free pool of reusable frame contexts.
    pool: crossbeam_queue::ArrayQueue<Box<FrameContext>>,
}

/// Compute the minimum outer tag dimension (grid + 2 border bits) across the
/// decoder list. Returns a floor of `6` when the list is empty so that a
/// mis-configured detector never divides by zero downstream.
fn compute_min_outer_dim(decoders: &[Box<dyn TagDecoder + Send + Sync>]) -> u32 {
    decoders
        .iter()
        .map(|d| d.dimension() as u32 + 2)
        .min()
        .unwrap_or(6)
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
            let _ = pool.push(Box::new(FrameContext::new_for_config(&config)));
        }
        let min_outer_dim = compute_min_outer_dim(&decoders);
        Self {
            config,
            decoders,
            min_outer_dim,
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
            self.min_outer_dim,
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
                    (Box::new(FrameContext::new_for_config(&self.config)), false)
                };

                let owned = run_detection_pipeline(
                    &self.config,
                    &self.decoders,
                    self.min_outer_dim,
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
        self.min_outer_dim = compute_min_outer_dim(&self.decoders);
    }

    /// Get the current detector configuration.
    #[must_use]
    pub fn config(&self) -> DetectorConfig {
        self.config
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod rescue_buf_tests {
    use super::*;

    #[test]
    fn rescue_buf_size_is_zero_when_disabled() {
        let cfg = DetectorConfig::default();
        assert!(!cfg.roi_rescue.enabled);
        assert_eq!(rescue_buf_size(&cfg), 0);
    }

    #[test]
    fn rescue_buf_size_bilinear_omits_scratch() {
        let mut cfg = DetectorConfig::default();
        cfg.roi_rescue.enabled = true;
        cfg.roi_rescue.interpolation = crate::config::RescueInterpolation::Bilinear;
        cfg.roi_rescue.upscale_factor = 2;
        cfg.roi_rescue.max_roi_side_px = 32;
        cfg.roi_rescue.max_rescues_per_frame = 4;
        // out_plane = (32*2)^2 = 4096; scratch = 0; +3 padding; × 4 rescues.
        let expected = (4096 + 3) * 4;
        assert_eq!(rescue_buf_size(&cfg), expected);
    }

    #[test]
    fn rescue_buf_size_lanczos_includes_scratch() {
        let mut cfg = DetectorConfig::default();
        cfg.roi_rescue.enabled = true;
        cfg.roi_rescue.interpolation = crate::config::RescueInterpolation::Lanczos3;
        cfg.roi_rescue.upscale_factor = 4;
        cfg.roi_rescue.max_roi_side_px = 32;
        cfg.roi_rescue.max_rescues_per_frame = 2;
        // out_plane = 128^2 = 16384; scratch = 128*32 = 4096; +3 padding; × 2.
        let expected = (16384 + 4096 + 3) * 2;
        assert_eq!(rescue_buf_size(&cfg), expected);
    }

    #[test]
    fn ensure_rescue_capacity_grows_then_stays() {
        let mut cfg = DetectorConfig::default();
        cfg.roi_rescue.enabled = true;
        cfg.roi_rescue.interpolation = crate::config::RescueInterpolation::Bilinear;
        cfg.roi_rescue.upscale_factor = 2;
        cfg.roi_rescue.max_roi_side_px = 16;
        cfg.roi_rescue.max_rescues_per_frame = 1;

        let mut ctx = FrameContext::new();
        assert_eq!(ctx.rescue_buf.len(), 0);
        ctx.ensure_rescue_capacity(&cfg);
        let first = ctx.rescue_buf.len();
        assert!(first >= rescue_buf_size(&cfg));
        ctx.ensure_rescue_capacity(&cfg);
        assert_eq!(ctx.rescue_buf.len(), first, "second call is a no-op");
    }

    struct RescueHarness {
        cfg: DetectorConfig,
        batch: crate::batch::DetectionBatch,
        arena: Bump,
        rescue_buf: Vec<u8>,
        pixels: Vec<u8>,
        decoders: Vec<Box<dyn crate::decoder::TagDecoder + Send + Sync>>,
    }

    impl RescueHarness {
        fn new() -> Self {
            let mut cfg = DetectorConfig::default();
            cfg.roi_rescue.enabled = true;
            cfg.roi_rescue.interpolation = crate::config::RescueInterpolation::Bilinear;
            cfg.roi_rescue.upscale_factor = 2;
            cfg.roi_rescue.max_roi_side_px = 32;
            cfg.roi_rescue.max_rescues_per_frame = 2;
            let rescue_buf = vec![0u8; rescue_buf_size(&cfg)];
            Self {
                cfg,
                batch: crate::batch::DetectionBatch::new(),
                arena: Bump::new(),
                rescue_buf,
                pixels: vec![0u8; 64 * 64],
                decoders: vec![],
            }
        }

        fn run(&mut self, n: usize) {
            let img = crate::image::ImageView::new(&self.pixels, 64, 64, 64).expect("valid view");
            run_rescue_stage(
                &mut self.batch,
                &self.arena,
                &mut self.rescue_buf,
                n,
                &self.cfg,
                &self.decoders,
                &img,
                true,
            );
        }
    }

    #[test]
    fn run_rescue_stage_no_op_when_no_candidates() {
        // No tags in the batch → rescue is a fast-exit no-op, and it must
        // not touch any telemetry slot.
        let mut harness = RescueHarness::new();
        harness.run(0);
        assert_eq!(harness.batch.rescue_attempted[0], 0);
    }

    #[test]
    fn run_rescue_stage_skips_when_no_funnel_survivors() {
        // All candidates are `FailedDecode` but none passed the funnel —
        // rescue must skip them because the funnel check would have caught
        // the true positives.
        use crate::batch::{CandidateState, FunnelStatus};

        let mut harness = RescueHarness::new();
        for i in 0..3 {
            harness.batch.status_mask[i] = CandidateState::FailedDecode;
            harness.batch.funnel_status[i] = FunnelStatus::RejectedContrast;
        }
        harness.run(3);
        for i in 0..3 {
            assert_eq!(harness.batch.rescue_attempted[i], 0);
        }
    }
}

#[cfg(all(test, feature = "non_rectified"))]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::{CornerRefinementMode, DecodeMode, PoseEstimationMode, QuadExtractionMode};
    use crate::error::ConfigError;
    use crate::pose::CameraIntrinsics;

    #[test]
    fn edlines_with_distortion_is_rejected_at_detect_time() {
        let config = DetectorConfig::builder()
            .quad_extraction_mode(QuadExtractionMode::EdLines)
            .refinement_mode(CornerRefinementMode::None)
            .decode_mode(DecodeMode::Hard)
            .build();

        let mut detector = Detector::with_config(config);
        let pixels = vec![0u8; 64 * 64];
        let img = ImageView::new(&pixels, 64, 64, 64).expect("valid view");
        let intrinsics = CameraIntrinsics::with_brown_conrady(
            800.0, 800.0, 32.0, 32.0, -0.3, 0.1, 0.001, -0.002, 0.0,
        );

        let err = detector
            .detect(
                &img,
                Some(&intrinsics),
                None,
                PoseEstimationMode::Fast,
                false,
            )
            .expect_err("distorted EdLines must fail");

        assert!(
            matches!(
                err,
                DetectorError::Config(ConfigError::EdLinesUnsupportedWithDistortion)
            ),
            "expected EdLinesUnsupportedWithDistortion, got {err:?}"
        );
    }

    #[test]
    fn edlines_with_pinhole_is_accepted() {
        let config = DetectorConfig::builder()
            .quad_extraction_mode(QuadExtractionMode::EdLines)
            .refinement_mode(CornerRefinementMode::None)
            .decode_mode(DecodeMode::Hard)
            .build();

        let mut detector = Detector::with_config(config);
        let pixels = vec![0u8; 64 * 64];
        let img = ImageView::new(&pixels, 64, 64, 64).expect("valid view");
        let intrinsics = CameraIntrinsics::new(800.0, 800.0, 32.0, 32.0);

        detector
            .detect(
                &img,
                Some(&intrinsics),
                None,
                PoseEstimationMode::Fast,
                false,
            )
            .expect("edlines with pinhole intrinsics must succeed");
    }
}
