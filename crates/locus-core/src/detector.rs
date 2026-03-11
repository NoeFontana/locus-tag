use crate::batch::{DetectionBatch, DetectionBatchView};
use crate::config::DetectorConfig;
use crate::decoder::{TagDecoder, family_to_decoder};
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
    /// # Panics
    ///
    /// Panics if the input image cannot be decimated or upscaled according to the configuration.
    #[allow(clippy::similar_names)]
    #[allow(clippy::too_many_lines)]
    pub fn detect(
        &mut self,
        img: &ImageView,
        intrinsics: Option<&crate::pose::CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_mode: crate::config::PoseEstimationMode,
        debug_telemetry: bool,
    ) -> DetectionBatchView<'_> {
        self.state.reset();
        let state = &mut self.state;

        let (detection_img, _effective_scale, refinement_img) = if self.config.decimation > 1 {
            let new_w = img.width / self.config.decimation;
            let new_h = img.height / self.config.decimation;
            let decimated_data = state.arena.alloc_slice_fill_copy(new_w * new_h, 0u8);
            let decimated_img = img
                .decimate_to(self.config.decimation, decimated_data)
                .expect("decimation failed");
            (decimated_img, 1.0 / self.config.decimation as f64, *img)
        } else if self.config.upscale_factor > 1 {
            let new_w = img.width * self.config.upscale_factor;
            let new_h = img.height * self.config.upscale_factor;
            state.upscale_buf.resize(new_w * new_h, 0);

            let upscaled_img = img
                .upscale_to(self.config.upscale_factor, &mut state.upscale_buf)
                .expect("valid upscaled view");
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
            ImageView::new(filtered, img.width, img.height, img.width).expect("valid filtered view")
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
            .expect("valid sharpened view")
        } else {
            filtered_img
        };

        let binarized = state
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);
        let threshold_map = state
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);

        // 1. Thresholding
        {
            let engine = crate::threshold::ThresholdEngine::from_config(&self.config);
            let tile_stats = engine.compute_tile_stats(&state.arena, &sharpened_img);
            engine.apply_threshold_with_map(
                &state.arena,
                &sharpened_img,
                &tile_stats,
                binarized,
                threshold_map,
            );
        }

        // 2. Segmentation
        let label_result = crate::segmentation::label_components_threshold_model(
            &state.arena,
            sharpened_img.data,
            sharpened_img.stride,
            threshold_map,
            img.width,
            img.height,
            self.config.segmentation_connectivity == crate::config::SegmentationConnectivity::Eight,
            self.config.quad_min_area,
            self.config.segmentation_margin,
        );

        // 3. Quad Extraction (SoA)
        let n = crate::quad::extract_quads_soa(
            &mut state.batch,
            &sharpened_img,
            &label_result,
            &self.config,
            self.config.decimation,
            &refinement_img,
        );

        // 4. Homography Pass (SoA)
        crate::decoder::compute_homographies_soa(
            &state.batch.corners[0..n],
            &mut state.batch.homographies[0..n],
        );

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
        if let (Some(intr), Some(size)) = (intrinsics, tag_size) {
            crate::pose::refine_poses_soa(
                &mut state.batch,
                v,
                intr,
                size,
                Some(&refinement_img),
                pose_mode,
            );
        }

        let telemetry = if debug_telemetry {
            Some(crate::batch::TelemetryPayload {
                binarized_ptr: binarized.as_ptr(),
                threshold_map_ptr: threshold_map.as_ptr(),
                width: img.width,
                height: img.height,
                stride: img.width,
            })
        } else {
            None
        };

        self.state.batch.view_with_telemetry(v, telemetry)
    }

    /// Get the current detector configuration.
    #[must_use]
    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }
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
