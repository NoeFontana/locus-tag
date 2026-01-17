//! Core detection logic for the Locus Tag library.
//!
//! Locus is a high-performance AprilTag and ArUco detector implemented in Rust.
//! This project serves as an experiment in LLM-assisted library development,
//! targeting 1-10ms latencies.
//!
//! The pipeline includes adaptive thresholding, segmentation, quad extraction, and decoding.
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
/// Edge-preserving filtering for small tag detection.
pub mod filter;
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
    /// Number of quad candidates passed to decoder.
    pub num_candidates: usize,
    /// Number of final detections.
    pub num_detections: usize,
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

        // 1. Thresholding with optional bilateral pre-filtering and adaptive window
        let start_thresh = std::time::Instant::now();
        
        // 1a. Optional bilateral pre-filtering for edge-preserving noise reduction
        let filtered_img = if self.config.enable_bilateral {
            let filtered = self.arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
            crate::filter::bilateral_filter(
                img,
                filtered,
                3, // spatial radius
                self.config.bilateral_sigma_space,
                self.config.bilateral_sigma_color,
            );
            ImageView::new(filtered, img.width, img.height, img.width)
                .expect("valid filtered view")
        } else {
            *img
        };

        let binarized = self.arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
        let threshold_map = self.arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
        
        {
            let _span = tracing::info_span!("threshold_adaptive").entered();
            // Compute integral image for O(1) local mean per pixel
            let integral = crate::threshold::compute_integral_image(&filtered_img);
            
            // 1b. Apply adaptive or fixed-window threshold
            if self.config.enable_adaptive_window {
                // Compute gradient map for adaptive window sizing
                let gradient = self.arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
                crate::filter::compute_gradient_map(&filtered_img, gradient);
                
                // Apply gradient-based adaptive window threshold
                crate::threshold::adaptive_threshold_gradient_window(
                    &filtered_img,
                    gradient,
                    &integral,
                    binarized,
                    self.config.threshold_min_radius,
                    self.config.threshold_max_radius,
                    40, // gradient threshold to distinguish edges from uniform regions
                    3,  // constant C
                );
            } else {
                // Fallback to fixed 13x13 window (radius=6)
                crate::threshold::adaptive_threshold_integral(
                    &filtered_img,
                    &integral,
                    binarized,
                    6,
                    3,
                );
            }
            
            // For threshold_map, store the local mean (for potential threshold-model segmentation)
            compute_threshold_map(&filtered_img, &integral, threshold_map, 6, 3);
        }
        stats.threshold_ms = start_thresh.elapsed().as_secs_f64() * 1000.0;

        // 2. Segmentation - Standard binary CCL
        // Use the binarized image directly for connected component labeling
        let start_seg = std::time::Instant::now();
        let label_result = {
            let _span = tracing::info_span!("segmentation").entered();
            crate::segmentation::label_components_threshold_model(
                &self.arena,
                img.data,
                threshold_map,
                img.width,
                img.height,
            )
        };
        stats.segmentation_ms = start_seg.elapsed().as_secs_f64() * 1000.0;

        // 3. Quad Fitting (Fast path with pre-filtering)
        // Gradients are computed lazily inside extract_quads for small components only
        let start_quad = std::time::Instant::now();
        let candidates = {
            let _span = tracing::info_span!("quad_extraction").entered();
            crate::quad::extract_quads_with_config(&self.arena, img, &label_result, &self.config)
        };
        stats.quad_extraction_ms = start_quad.elapsed().as_secs_f64() * 1000.0;
        stats.num_candidates = candidates.len();

        // 4. Decoding - select decoders based on options
        let start_decode = std::time::Instant::now();
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
            if let Some(h) = crate::decoder::Homography::square_to_quad(&cand.corners) {
                for decoder in active_decoders {
                    if let Some(bits) = crate::decoder::sample_grid(img, &h, decoder.as_ref()) {
                        if let Some((id, hamming)) = decoder.decode(bits) {
                            cand.id = id;
                            cand.hamming = hamming;
                            if let (Some(intrinsics), Some(tag_size)) =
                                (options.intrinsics, options.tag_size)
                            {
                                cand.pose = crate::pose::estimate_tag_pose(
                                    &intrinsics,
                                    &cand.corners,
                                    tag_size,
                                );
                            }
                            // Adjust coordinates for benchmark compliance (0.5 offset)
                            // Our internal system treats (0, 0) as pixel center. 
                            // Common benchmarks (ICRA, AprilTag) treat (0.5, 0.5) as pixel center.
                            for corner in &mut cand.corners {
                                corner[0] += 0.5;
                                corner[1] += 0.5;
                            }
                            cand.center[0] += 0.5;
                            cand.center[1] += 0.5;
                            
                            final_detections.push(cand);
                            break;
                        }
                    }
                }
            }
        }
        stats.decoding_ms = start_decode.elapsed().as_secs_f64() * 1000.0;
        stats.num_detections = final_detections.len();
        stats.total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        (final_detections, stats)
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

// Use family_to_decoder from the decoder module.
pub use decoder::family_to_decoder;

/// Compute per-pixel threshold map from integral image.
///
/// Stores local mean - C for each pixel, used by threshold-model segmentation.
fn compute_threshold_map(
    img: &image::ImageView,
    integral: &[u32],
    output: &mut [u8],
    radius: usize,
    c: i16,
) {
    let w = img.width;
    let h = img.height;
    let stride = w + 1;

    for y in 0..h {
        let dst_offset = y * w;

        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(h);
        let actual_height = (y1 - y0) as u32;

        for x in 0..w {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(w);
            let actual_width = (x1 - x0) as u32;
            let actual_area = actual_width * actual_height;

            let i00 = integral[y0 * stride + x0];
            let i01 = integral[y0 * stride + x1];
            let i10 = integral[y1 * stride + x0];
            let i11 = integral[y1 * stride + x1];

            // Prevent underflow: add first, then subtract
            let sum = (i11 + i00) - (i01 + i10);
            let mean = (sum / actual_area) as i16;
            let threshold = (mean - c).clamp(0, 255) as u8;
            output[dst_offset + x] = threshold;
        }
    }
}
