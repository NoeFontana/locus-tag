use crate::batch::DetectionBatch;
use crate::config::DetectorConfig;
use crate::decoder::{TagDecoder, family_to_decoder};
use crate::image::ImageView;
use crate::Detection;
use bumpalo::Bump;

/// Internal state container for the detector.
/// 
/// Owns the memory pools and reusable buffers to ensure zero-allocation 
/// in the detection hot-path.
pub struct DetectorState {
    pub arena: Bump,
    pub batch: DetectionBatch,
    pub upscale_buf: Vec<u8>,
    pub results: Vec<Detection>,
}

impl DetectorState {
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
            batch: DetectionBatch::new(),
            upscale_buf: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.arena.reset();
        self.results.clear();
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

impl Detector {
    /// Create a new detector with the default configuration.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a builder to configure a new detector.
    pub fn builder() -> DetectorBuilder {
        DetectorBuilder::new()
    }

    /// Create a detector with custom pipeline configuration.
    pub fn with_config(config: DetectorConfig) -> Self {
        Self::builder().with_config(config).build()
    }

    /// Access the internal state (for advanced inspection or FFI).
    pub fn state(&self) -> &DetectorState {
        &self.state
    }

    /// Clear all decoders and set new ones based on tag families.
    pub fn set_families(&mut self, families: &[crate::config::TagFamily]) {
        self.decoders.clear();
        for &family in families {
            self.decoders.push(crate::decoder::family_to_decoder(family));
        }
    }

    /// Detect tags in the provided image.
    ///
    /// This method is the main execution pipeline.
    pub fn detect(
        &mut self,
        img: &ImageView,
        intrinsics: Option<&crate::pose::CameraIntrinsics>,
        tag_size: Option<f64>,
        pose_mode: crate::config::PoseEstimationMode,
    ) -> Vec<Detection> {
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
            (upscaled_img, self.config.upscale_factor as f64, upscaled_img)
        } else {
            (*img, 1.0, *img)
        };

        let img = &detection_img;

        // 1a. Optional bilateral pre-filtering
        let filtered_img = if self.config.enable_bilateral {
            let filtered = state.arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
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
            let sharpened = state.arena.alloc_slice_fill_copy(filtered_img.width * filtered_img.height, 0u8);
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

        let binarized = state.arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
        let threshold_map = state.arena.alloc_slice_fill_copy(img.width * img.height, 0u8);

        // 1. Thresholding
        {
            let engine = crate::threshold::ThresholdEngine::from_config(&self.config);
            let stats = engine.compute_tile_stats(&state.arena, &sharpened_img);
            engine.apply_threshold_with_map(
                &state.arena,
                &sharpened_img,
                &stats,
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
            &state.batch.corners[0..n * 4],
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
        state.results = state.batch.reassemble(v);

        // Final coordinate adjustment to align with pixel center convention (UMICH/OpenCV)
        for d in &mut state.results {
            for corner in &mut d.corners {
                corner[0] += 0.5;
                corner[1] += 0.5;
            }
            d.center[0] += 0.5;
            d.center[1] += 0.5;
        }

        state.results.clone()
    }


    /// Detect tags with specific options.
    pub fn detect_with_options(
        &mut self,
        img: &ImageView,
        options: &crate::config::DetectOptions,
    ) -> Vec<crate::Detection> {
        self.detect_with_stats_and_options(img, options).detections
    }

    /// Detect tags and return comprehensive results with statistics.
    pub fn detect_with_stats(&mut self, img: &ImageView) -> crate::FullDetectionResult {
        self.detect_with_stats_and_options(img, &crate::config::DetectOptions::default())
    }

    /// Detect tags and return comprehensive results with statistics.
    pub fn detect_with_stats_and_options(
        &mut self,
        img: &ImageView,
        options: &crate::config::DetectOptions,
    ) -> crate::FullDetectionResult {
        let start = std::time::Instant::now();
        
        // Temporarily override families if options are provided
        let old_decoders = if !options.families.is_empty() {
            let mut new_decoders = Vec::new();
            for &family in &options.families {
                new_decoders.push(crate::decoder::family_to_decoder(family));
            }
            Some(std::mem::replace(&mut self.decoders, new_decoders))
        } else {
            None
        };

        let detections = self.detect(
            img,
            options.intrinsics.as_ref(),
            options.tag_size,
            options.pose_estimation_mode,
        );
        let result = crate::FullDetectionResult {
            detections: detections.clone(),
            stats: crate::PipelineStats {
                total_ms: start.elapsed().as_secs_f64() * 1000.0,
                threshold_ms: 0.0,
                segmentation_ms: 0.0,
                quad_extraction_ms: 0.0,
                decoding_ms: 0.0,
                num_candidates: 0,
                num_detections: detections.len(),
            },
        };

        if let Some(old) = old_decoders {
            self.decoders = old;
        }

        result
    }

    /// Get the current detector configuration.
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
    pub fn new() -> Self {
        Self {
            config: DetectorConfig::default(),
            families: Vec::new(),
        }
    }

    /// Use an existing configuration.
    pub fn with_config(mut self, config: DetectorConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the decimation factor for the input image.
    pub fn with_decimation(mut self, decimation: usize) -> Self {
        self.config.decimation = decimation;
        self
    }

    /// Add a tag family to be detected.
    pub fn with_family(mut self, family: crate::config::TagFamily) -> Self {
        if !self.families.contains(&family) {
            self.families.push(family);
        }
        self
    }

    /// Set the thread count for parallel processing.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.config.nthreads = threads;
        self
    }

    /// Set the upscale factor for detecting small tags.
    pub fn with_upscale_factor(mut self, factor: usize) -> Self {
        self.config.upscale_factor = factor;
        self
    }

    /// Set the corner refinement mode.
    pub fn with_corner_refinement(mut self, mode: crate::config::CornerRefinementMode) -> Self {
        self.config.refinement_mode = mode;
        self
    }

    /// Set the decoding mode (Hard vs Soft).
    pub fn with_decode_mode(mut self, mode: crate::config::DecodeMode) -> Self {
        self.config.decode_mode = mode;
        self
    }

    /// Set the segmentation connectivity (4-way or 8-way).
    pub fn with_connectivity(mut self, connectivity: crate::config::SegmentationConnectivity) -> Self {
        self.config.segmentation_connectivity = connectivity;
        self
    }

    /// Set the tile size for adaptive thresholding.
    pub fn with_threshold_tile_size(mut self, size: usize) -> Self {
        self.config.threshold_tile_size = size;
        self
    }

    /// Set the minimum intensity range for valid tiles.
    pub fn with_threshold_min_range(mut self, range: u8) -> Self {
        self.config.threshold_min_range = range;
        self
    }

    /// Set the constant subtracted from local mean in adaptive thresholding.
    pub fn with_adaptive_threshold_constant(mut self, c: i16) -> Self {
        self.config.adaptive_threshold_constant = c;
        self
    }

    /// Set the minimum quad area.
    pub fn with_quad_min_area(mut self, area: u32) -> Self {
        self.config.quad_min_area = area;
        self
    }

    /// Set the minimum fill ratio.
    pub fn with_quad_min_fill_ratio(mut self, ratio: f32) -> Self {
        self.config.quad_min_fill_ratio = ratio;
        self
    }

    /// Set the minimum edge alignment score.
    pub fn with_quad_min_edge_score(mut self, score: f64) -> Self {
        self.config.quad_min_edge_score = score;
        self
    }

    /// Set the maximum number of Hamming errors allowed.
    pub fn with_max_hamming_error(mut self, errors: u32) -> Self {
        self.config.max_hamming_error = errors;
        self
    }

    /// Set the minimum contrast for decoder bit classification.
    pub fn with_decoder_min_contrast(mut self, contrast: f64) -> Self {
        self.config.decoder_min_contrast = contrast;
        self
    }

    /// Build the [`Detector`] instance.
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
