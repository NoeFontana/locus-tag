//! Core detection logic for the Locus Tag library.
//!
//! This crate implements the high-performance AprilTag detection pipeline,
//! including adaptive thresholding, segmentation, quad extraction, and decoding.
//!
//! # Configuration
//!
//! The detector supports two levels of configuration:
//! - [`config::DetectorConfig`]: Pipeline-level settings (immutable after construction)
//! - [`config::DetectOptions`]: Per-call detection options
//!
//! # Example
//!
//! ```ignore
//! use locus_core::{Detector, config::{DetectorConfig, DetectOptions}};
//!
//! // Create detector with custom config
//! let config = DetectorConfig::builder()
//!     .threshold_tile_size(16)
//!     .build();
//! let mut detector = Detector::with_config(config);
//!
//! // Detect with custom options
//! let options = DetectOptions::builder()
//!     .quad_min_area(200)
//!     .build();
//! let detections = detector.detect_with_options(&img, &options);
//! ```

/// Configuration types for the detector pipeline.
pub mod config;
/// Tag decoding traits and implementations.
pub mod decoder;
/// Tag family dictionaries (AprilTag, ArUco).
pub mod dictionaries;
/// Gradient computation for edge refinement.
pub mod gradient;
/// Image buffer abstractions.
pub mod image;
/// 3D Pose Estimation (PnP).
pub mod pose;
/// Quad extraction and geometric primitives.
pub mod quad;
/// Connected components labeling using Union-Find.
pub mod segmentation;
/// Utilities for testing and synthetic data generation.
pub mod test_utils;
/// Adaptive thresholding implementation.
pub mod threshold;

pub use config::{DetectOptions, DetectorConfig};

use crate::decoder::TagDecoder;
use crate::image::ImageView;
use crate::threshold::ThresholdEngine;
use bumpalo::Bump;

/// Result of a tag detection.
pub struct Detection {
    /// The decoded ID of the tag.
    pub id: u32,
    /// The center coordinates of the tag in image pixels (x, y).
    pub center: [f64; 2],
    /// The 4 corners of the tag in image pixels.
    pub corners: [[f64; 2]; 4],
    /// The number of hamming errors corrected during decoding.
    pub hamming: u32,
    /// The decision margin of the decoding (higher is more confident).
    pub decision_margin: f64,
    /// The 3D pose of the tag relative to the camera (if requested).
    pub pose: Option<crate::pose::Pose>,
}

/// Statistics for the detection pipeline stages.
#[derive(Default, Debug, Clone)]
pub struct PipelineStats {
    /// Time taken for adaptive thresholding in milliseconds.
    pub threshold_ms: f64,
    /// Time taken for connected components labeling in milliseconds.
    pub segmentation_ms: f64,
    /// Time taken for quad extraction and fitting in milliseconds.
    pub quad_extraction_ms: f64,
    /// Time taken for decoding in milliseconds.
    pub decoding_ms: f64,
    /// Total pipeline time in milliseconds.
    pub total_ms: f64,
}

/// The main entry point for detecting AprilTags.
///
/// The detector holds reusable state (arena allocator, threshold engine)
/// and can be configured at construction time via [`DetectorConfig`].
pub struct Detector {
    arena: Bump,
    config: DetectorConfig,
    threshold_engine: ThresholdEngine,
    decoders: Vec<Box<dyn TagDecoder + Send + Sync>>,
}

impl Detector {
    /// Create a new detector instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DetectorConfig::default())
    }

    /// Create a detector with custom pipeline configuration.
    #[must_use]
    pub fn with_config(config: DetectorConfig) -> Self {
        Self {
            arena: Bump::new(),
            config,
            threshold_engine: ThresholdEngine::from_config(&config),
            decoders: vec![Box::new(crate::decoder::AprilTag36h11)],
        }
    }

    /// Add a decoder to the pipeline.
    pub fn add_decoder(&mut self, decoder: Box<dyn TagDecoder + Send + Sync>) {
        self.decoders.push(decoder);
    }

    /// Clear all decoders and set new ones based on tag families.
    pub fn set_families(&mut self, families: &[config::TagFamily]) {
        self.decoders.clear();
        for family in families {
            self.decoders.push(family_to_decoder(*family));
        }
    }

    /// Primary detection entry point using detector's configured decoders.
    pub fn detect(&mut self, img: &ImageView) -> Vec<Detection> {
        self.detect_with_options(img, &DetectOptions::default())
    }

    /// Detection with custom per-call options (e.g., specific tag families).
    pub fn detect_with_options(
        &mut self,
        img: &ImageView,
        options: &DetectOptions,
    ) -> Vec<Detection> {
        self.detect_with_stats_and_options(img, options).0
    }

    /// Detection with detailed timing statistics.
    pub fn detect_with_stats(&mut self, img: &ImageView) -> (Vec<Detection>, PipelineStats) {
        self.detect_with_stats_and_options(img, &DetectOptions::default())
    }

    /// Detection with both custom options and timing statistics.
    pub fn detect_with_stats_and_options(
        &mut self,
        img: &ImageView,
        options: &DetectOptions,
    ) -> (Vec<Detection>, PipelineStats) {
        let mut stats = PipelineStats::default();
        let start_total = std::time::Instant::now();

        self.arena.reset();

        // 1. Thresholding
        let start_thresh = std::time::Instant::now();
        let tile_stats = self.threshold_engine.compute_tile_stats(img);
        let binarized = self
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);
        self.threshold_engine
            .apply_threshold(img, &tile_stats, binarized);
        stats.threshold_ms = start_thresh.elapsed().as_secs_f64() * 1000.0;

        // 2. Segmentation (Connected Components with stats)
        let start_seg = std::time::Instant::now();
        let label_result = crate::segmentation::label_components_with_stats(
            &self.arena,
            binarized,
            img.width,
            img.height,
        );
        stats.segmentation_ms = start_seg.elapsed().as_secs_f64() * 1000.0;

        // 3. Quad Fitting (Fast path with pre-filtering)
        let start_quad = std::time::Instant::now();
        let candidates =
            crate::quad::extract_quads_with_config(&self.arena, img, &label_result, &self.config);
        stats.quad_extraction_ms = start_quad.elapsed().as_secs_f64() * 1000.0;

        // 4. Decoding - select decoders based on options
        let start_decode = std::time::Instant::now();
        let src_points = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

        // Build decoder list: use options.families if specified, else use self.decoders
        let temp_decoders: Vec<Box<dyn TagDecoder + Send + Sync>>;
        let active_decoders: &[Box<dyn TagDecoder + Send + Sync>] = if options.families.is_empty() {
            &self.decoders
        } else {
            temp_decoders = options
                .families
                .iter()
                .map(|f| family_to_decoder(*f))
                .collect();
            &temp_decoders
        };

        let mut final_detections = Vec::new();
        for mut cand in candidates {
            if let Some(h) = crate::decoder::Homography::from_pairs(&src_points, &cand.corners) {
                for decoder in active_decoders {
                    if let Some(bits) = crate::decoder::sample_grid(img, &h, decoder.as_ref()) {
                        if let Some((id, hamming)) = decoder.decode(bits) {
                            cand.id = id;
                            cand.hamming = hamming;

                            // Compute 3D pose if requested
                            if let (Some(intrinsics), Some(tag_size)) =
                                (options.intrinsics, options.tag_size)
                            {
                                cand.pose = crate::pose::estimate_tag_pose(
                                    &intrinsics,
                                    &cand.corners,
                                    tag_size,
                                );
                            }

                            final_detections.push(cand);
                            break;
                        }
                    }
                }
            }
        }
        stats.decoding_ms = start_decode.elapsed().as_secs_f64() * 1000.0;
        stats.total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        (final_detections, stats)
    }

    /// Fast detection using decimation (2x downsampled).
    /// Detects tags using a gradient-based downsampling approach for speed.
    ///
    /// # Panics
    /// Panics if the downsampled image cannot be created (should not happen with valid dimensions).
    pub fn detect_gradient(&mut self, img: &ImageView) -> Vec<Detection> {
        // Decimate the image 2x
        let new_w = img.width / 2;
        let new_h = img.height / 2;

        // Simple 2x2 averaging downsample
        let mut downsampled = vec![0u8; new_w * new_h];
        for y in 0..new_h {
            for x in 0..new_w {
                let p00 = u16::from(img.get_pixel(x * 2, y * 2));
                let p10 = u16::from(img.get_pixel(x * 2 + 1, y * 2));
                let p01 = u16::from(img.get_pixel(x * 2, y * 2 + 1));
                let p11 = u16::from(img.get_pixel(x * 2 + 1, y * 2 + 1));
                downsampled[y * new_w + x] = ((p00 + p10 + p01 + p11) / 4) as u8;
            }
        }

        let view =
            ImageView::new(&downsampled, new_w, new_h, new_w).expect("Valid downsampled image");

        // Run detection on downsampled image
        let mut detections = self.detect(&view);

        // Scale corners back to original resolution
        for det in &mut detections {
            det.center[0] *= 2.0;
            det.center[1] *= 2.0;
            for corner in &mut det.corners {
                corner[0] *= 2.0;
                corner[1] *= 2.0;
            }
        }

        detections
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns version and build information for the core library.
#[must_use]
pub fn core_info() -> String {
    "Locus Core v0.1.0 Engine".to_string()
}

/// Convert a TagFamily enum to a boxed decoder instance.
fn family_to_decoder(family: config::TagFamily) -> Box<dyn TagDecoder + Send + Sync> {
    match family {
        config::TagFamily::AprilTag36h11 => Box::new(decoder::AprilTag36h11),
        config::TagFamily::AprilTag16h5 => Box::new(decoder::AprilTag16h5),
        config::TagFamily::ArUco4x4_50 => Box::new(decoder::ArUco4x4_50),
        config::TagFamily::ArUco4x4_100 => Box::new(decoder::ArUco4x4_100),
    }
}
