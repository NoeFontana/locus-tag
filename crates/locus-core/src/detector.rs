use crate::batch::{DetectionBatch, DetectionBatchView};
use crate::config::DetectorConfig;
use crate::decoder::{TagDecoder, family_to_decoder};
use crate::error::DetectorError;
use crate::image::ImageView;
use bumpalo::Bump;

/// Internal state container for the detector.
///
/// Owns the memory pools and reusable buffers to ensure zero-allocation
/// in the detection hot-path.
pub struct DetectorState {
    /// Memory pool for ephemeral per-frame allocations.
    pub arena: Bump,
    /// Vectorized storage for quad candidates and results.
    pub batch: DetectionBatch,
    /// Reusable buffer for upscaling.
    pub upscale_buf: Vec<u8>,
}

impl DetectorState {
    /// Create a new internal state container.
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

impl Default for DetectorState {
    fn default() -> Self {
        Self::new()
    }
}

/// The primary entry point for the Locus perception library.
///
/// `Detector` encapsulates the entire detection pipeline.
pub struct Detector {
    config: DetectorConfig,
    decoders: Vec<Box<dyn TagDecoder + Send + Sync>>,
    state: DetectorState,
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

    /// Access the internal state (for advanced inspection or FFI).
    #[must_use]
    pub fn state(&self) -> &DetectorState {
        &self.state
    }

    /// Clear all decoders and set new ones based on tag families.
    pub fn set_families(&mut self, families: &[crate::config::TagFamily]) {
        self.decoders.clear();
        for &family in families {
            self.decoders
                .push(crate::decoder::family_to_decoder(family));
        }
    }

    /// Detect tags in the provided image.
    ///
    /// This method is the main execution pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`DetectorError`] if the input image cannot be decimated, upscaled,
    /// or if an intermediate image view cannot be constructed.
    #[allow(clippy::similar_names)]
    #[allow(clippy::too_many_lines)]
    pub fn detect(
        &mut self,
        img: &ImageView,
        intrinsics: Option<&crate::pose::CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_mode: crate::config::PoseEstimationMode,
        debug_telemetry: bool,
    ) -> Result<DetectionBatchView<'_>, DetectorError> {
        self.state.reset();
        let state = &mut self.state;

        let (detection_img, _effective_scale, refinement_img) = if self.config.decimation > 1 {
            let new_w = img.width / self.config.decimation;
            let new_h = img.height / self.config.decimation;
            let decimated_data = state.arena.alloc_slice_fill_copy(new_w * new_h, 0u8);
            let decimated_img = img
                .decimate_to(self.config.decimation, decimated_data)
                .map_err(DetectorError::Preprocessing)?;
            (decimated_img, 1.0 / self.config.decimation as f64, *img)
        } else if self.config.upscale_factor > 1 {
            let new_w = img.width * self.config.upscale_factor;
            let new_h = img.height * self.config.upscale_factor;
            state.upscale_buf.resize(new_w * new_h, 0);

            let upscaled_img = img
                .upscale_to(self.config.upscale_factor, &mut state.upscale_buf)
                .map_err(DetectorError::Preprocessing)?;
            (
                upscaled_img,
                self.config.upscale_factor as f64,
                upscaled_img,
            )
        } else {
            (*img, 1.0, *img)
        };

        let img = &detection_img;

        // 1a. Optional bilateral pre-filtering
        let filtered_img = if self.config.enable_bilateral {
            let filtered = state
                .arena
                .alloc_slice_fill_copy(img.width * img.height, 0u8);
            crate::filter::bilateral_filter(
                &state.arena,
                img,
                filtered,
                3, // spatial radius
                self.config.bilateral_sigma_space,
                self.config.bilateral_sigma_color,
            );
            ImageView::new(filtered, img.width, img.height, img.width)
                .map_err(DetectorError::InvalidImage)?
        } else {
            *img
        };

        // 1b. Optional Laplacian sharpening
        let sharpened_img = if self.config.enable_sharpening {
            let sharpened = state
                .arena
                .alloc_slice_fill_copy(filtered_img.width * filtered_img.height, 0u8);
            crate::filter::laplacian_sharpen(&filtered_img, sharpened);

            ImageView::new(
                sharpened,
                filtered_img.width,
                filtered_img.height,
                filtered_img.width,
            )
            .map_err(DetectorError::InvalidImage)?
        } else {
            filtered_img
        };

        let binarized = state
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);
        let threshold_map = state
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);

        // 1. Thresholding & 2. Segmentation & 3. Quad Extraction
        let (n, unrefined) = {
            let engine = crate::threshold::ThresholdEngine::from_config(&self.config);
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
                self.config.segmentation_connectivity
                    == crate::config::SegmentationConnectivity::Eight,
                self.config.quad_min_area,
            );

            // 3. Quad Extraction (SoA)
            let (n, unrefined) = crate::quad::extract_quads_soa(
                &mut state.batch,
                &sharpened_img,
                &label_result,
                &self.config,
                self.config.decimation,
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
                self.config.threshold_tile_size,
                self.config.decoder_min_contrast,
                1.0 / self.config.decimation as f64,
            );

            (n, unrefined)
        };

        // Compute subpixel jitter if requested
        let mut jitter_ptr = std::ptr::null();
        let mut num_jitter = 0;
        if let (true, Some(unrefined_pts)) = (debug_telemetry, unrefined) {
            // Number of candidates to store jitter for (all extracted ones)
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
        crate::homography::compute_homographies_soa(
            &state.batch.corners[0..n],
            &state.batch.status_mask[0..n],
            &mut state.batch.homographies[0..n],
        );

        // Optional: GWLF Refinement
        let mut gwlf_fallback_count = 0;
        let mut gwlf_avg_delta = 0.0f32;
        if self.config.refinement_mode == crate::config::CornerRefinementMode::Gwlf {
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
                    self.config.gwlf_transversal_alpha,
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
            crate::homography::compute_homographies_soa(
                &state.batch.corners[0..n],
                &state.batch.status_mask[0..n],
                &mut state.batch.homographies[0..n],
            );
        }

        // 5. Decoding Pass (SoA)
        crate::decoder::decode_batch_soa(
            &mut state.batch,
            n,
            &refinement_img,
            &self.decoders,
            &self.config,
        );

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
            &self.config,
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

        Ok(self.state.batch.view_with_telemetry(v, n, telemetry))
    }

    /// Get the current detector configuration.
    #[must_use]
    pub fn config(&self) -> DetectorConfig {
        self.config
    }

    /// Returns a cloned copy of the internal detection batch.
    /// Exclusively for benchmarking.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn bench_api_get_batch_cloned(&self) -> DetectionBatch {
        let mut new_batch = DetectionBatch::new();
        new_batch.corners.copy_from_slice(&self.state.batch.corners);
        new_batch
            .homographies
            .copy_from_slice(&self.state.batch.homographies);
        new_batch.ids.copy_from_slice(&self.state.batch.ids);
        new_batch
            .payloads
            .copy_from_slice(&self.state.batch.payloads);
        new_batch
            .error_rates
            .copy_from_slice(&self.state.batch.error_rates);
        new_batch.poses.copy_from_slice(&self.state.batch.poses);
        new_batch
            .status_mask
            .copy_from_slice(&self.state.batch.status_mask);
        new_batch
            .funnel_status
            .copy_from_slice(&self.state.batch.funnel_status);
        new_batch
            .corner_covariances
            .copy_from_slice(&self.state.batch.corner_covariances);
        new_batch
    }
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
}

impl DetectorBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DetectorConfig::default(),
            families: Vec::new(),
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

    /// Enable or disable Laplacian sharpening.
    #[must_use]
    pub fn with_sharpening(mut self, enable: bool) -> Self {
        self.config.enable_sharpening = enable;
        self
    }

    /// Build the [`Detector`] instance.
    #[must_use]
    pub fn build(self) -> Detector {
        let mut decoders = Vec::new();
        let families = if self.families.is_empty() {
            vec![crate::config::TagFamily::AprilTag36h11]
        } else {
            self.families
        };
        for family in families {
            decoders.push(family_to_decoder(family));
        }

        Detector {
            config: self.config,
            decoders,
            state: DetectorState::new(),
        }
    }
}

impl Default for DetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
