use crate::batch::DetectionBatch;
use crate::config::DetectorConfig;
use crate::decoder::{TagDecoder, family_to_decoder};
use crate::image::ImageView;
use crate::Detection;
use bumpalo::Bump;

/// The primary entry point for the Locus perception library.
///
/// `Detector` encapsulates the entire detection pipeline, managing its own
/// internal state and memory pools to provide a safe, high-performance API.
pub struct Detector {
    config: DetectorConfig,
    decoders: Vec<Box<dyn TagDecoder + Send + Sync>>,
    results: Vec<Detection>,
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
        Self {
            config,
            decoders: vec![crate::decoder::family_to_decoder(crate::config::TagFamily::AprilTag36h11)],
            results: Vec::new(),
        }
    }

    /// Detect tags in the provided image.
    ///
    /// This method is the main execution pipeline.
    pub fn detect(&mut self, img: &ImageView) -> &[Detection] {
        let arena = Bump::new();
        let mut batch = DetectionBatch::new();
        let mut upscale_buf = Vec::new();

        let (detection_img, effective_scale, refinement_img) = if self.config.decimation > 1 {
            let new_w = img.width / self.config.decimation;
            let new_h = img.height / self.config.decimation;
            let decimated_data = arena.alloc_slice_fill_copy(new_w * new_h, 0u8);
            let decimated_img = img
                .decimate_to(self.config.decimation, decimated_data)
                .expect("decimation failed");
            (decimated_img, 1.0 / self.config.decimation as f64, *img)
        } else if self.config.upscale_factor > 1 {
            let new_w = img.width * self.config.upscale_factor;
            let new_h = img.height * self.config.upscale_factor;
            upscale_buf.resize(new_w * new_h, 0);

            let upscaled_img = img
                .upscale_to(self.config.upscale_factor, &mut upscale_buf)
                .expect("valid upscaled view");
            (upscaled_img, self.config.upscale_factor as f64, upscaled_img)
        } else {
            (*img, 1.0, *img)
        };

        let img = &detection_img;

        // 1a. Optional bilateral pre-filtering
        let filtered_img = if self.config.enable_bilateral {
            let filtered = arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
            crate::filter::bilateral_filter(
                &arena,
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
            let sharpened = arena.alloc_slice_fill_copy(filtered_img.width * filtered_img.height, 0u8);
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

        let binarized = arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
        let threshold_map = arena.alloc_slice_fill_copy(img.width * img.height, 0u8);

        // 1. Thresholding
        {
            let engine = crate::threshold::ThresholdEngine::from_config(&self.config);
            let stats = engine.compute_tile_stats(&arena, &sharpened_img);
            engine.apply_threshold_with_map(
                &arena,
                &sharpened_img,
                &stats,
                binarized,
                threshold_map,
            );
        }

        // 2. Segmentation
        let label_result = crate::segmentation::label_components_threshold_model(
            &arena,
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
            &mut batch,
            &sharpened_img,
            &label_result,
            &self.config,
            self.config.decimation,
            &refinement_img,
        );

        // 4. Homography Pass (SoA)
        crate::decoder::compute_homographies_soa(
            &batch.corners[0..n * 4],
            &mut batch.homographies[0..n],
        );

        // 5. Decoding Pass (SoA)
        crate::decoder::decode_batch_soa(
            &mut batch,
            n,
            &refinement_img,
            &self.decoders,
            &self.config,
        );

        // Partition valid candidates to the front [0..v]
        let v = batch.partition(n);

        self.results = batch.reassemble(v);

        // Final coordinate adjustment and scaling
        let upscale_factor = self.config.upscale_factor.max(1);
        let inv_scale = if upscale_factor > 1 {
            1.0 / effective_scale
        } else {
            1.0
        };

        for d in &mut self.results {
            for corner in &mut d.corners {
                corner[0] = (corner[0] + 0.5) * inv_scale;
                corner[1] = (corner[1] + 0.5) * inv_scale;
            }
            d.center[0] = (d.center[0] + 0.5) * inv_scale;
            d.center[1] = (d.center[1] + 0.5) * inv_scale;
        }

        &self.results
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
            results: Vec::new(),
        }
    }
}

impl Default for DetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
