//! Core detection logic for the Locus Tag library.
//!
//! Locus is a high-performance AprilTag and ArUco detector implemented in Rust.
//! This project serves as an experiment in LLM-assisted library development,
//! targeting 1-10ms latencies.
//!
//! The pipeline includes adaptive thresholding, segmentation, quad extraction, and decoding.
//!
//! # Architecture Overview
//!
//! The Locus pipeline follows a Data-Oriented Design approach optimized for strict latency budgets:
//!
//! 1. **Preprocessing (Adaptive Thresholding)**:
//!    - Integral image computation for constant-time local window stats.
//!    - Multiversion SIMD kernels for min/max filtering.
//!    - Optional bilateral filtering and sharpening.
//!    - See the [Architecture Guide](../architecture/index.html) for memory layout details.
//!
//! 2. **Segmentation**:
//!    - Threshold-model aware connected components labeling (CCL).
//!    - Union-Find data structure with flat arrays for cache locality.
//!
//! 3. **Quad Extraction**:
//!    - Contour tracing and polygon approximation (Douglas-Peucker).
//!    - Sub-pixel corner refinement using intensity-based optimization.
//!    - Geometric heuristics filtering (area, aspect ratio).
//!
//! 4. **Decoding**:
//!    - Homography-based grid sampling.
//!    - Plugin architecture for multiple tag families (Apriltag 36h11, ArUco).
//!    - Hamming error correction.
//!
//! # Configuration
//!
//! The detector supports two levels of configuration:
//! - [`config::DetectorConfig`]: Pipeline-level settings (immutable after construction)
//! - [`config::DetectOptions`]: Per-call detection options
//!
//! # Example
//!
//! ```
//! # use locus_core::{Detector, config::{DetectorConfig, DetectOptions, TagFamily}};
//! # use locus_core::image::ImageView;
//! // Create detector with custom config
//! let config = DetectorConfig::builder()
//!     .threshold_tile_size(16)
//!     .quad_min_area(200)
//!     .build();
//! let mut detector = Detector::with_config(config);
//!
//! // Create a dummy image for demonstration
//! # let pixels = vec![128u8; 64 * 64];
//! # let img = ImageView::new(&pixels, 64, 64, 64).unwrap();
//!
//! // Detect with custom options (e.g., specific tag family)
//! let options = DetectOptions::builder()
//!     .families(&[TagFamily::AprilTag36h11])
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

pub use crate::config::{DetectOptions, DetectorConfig, TagFamily};
use crate::decoder::TagDecoder;
pub use crate::image::ImageView;
use bumpalo::Bump;
use rayon::prelude::*;

/// Result of a tag detection.
#[derive(Clone, Debug, Default)]
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
/// Pipeline-wide statistics for a single detection call.
#[derive(Clone, Copy, Debug, Default)]
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
    /// Number of candidates rejected due to low contrast.
    pub num_rejected_by_contrast: usize,
    /// Number of candidates that passed contrast check but failed to match any tag code (hamming distance too high).
    pub num_rejected_by_hamming: usize,
}

/// Full result of a detection including intermediate data for debugging.
pub struct FullDetectionResult {
    /// Final detections.
    pub detections: Vec<Detection>,
    /// All quad candidates found before decoding.
    pub candidates: Vec<Detection>,
    /// The binarized image (thresholded).
    pub binarized: Option<Vec<u8>>,
    /// Labeled connected components image.
    pub labels: Option<Vec<u32>>,
    /// Pipeline statistics.
    pub stats: PipelineStats,
}

/// The main entry point for detecting AprilTags.
///
/// The detector holds reusable state (arena allocator, threshold engine)
/// and can be configured at construction time via [`DetectorConfig`].
pub struct Detector {
    arena: Bump,
    config: DetectorConfig,
    decoders: Vec<Box<dyn TagDecoder + Send + Sync>>,
    upscale_buf: Vec<u8>,
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
            decoders: vec![Box::new(crate::decoder::AprilTag36h11)],
            upscale_buf: Vec::new(),
        }
    }

    /// Get the current detector configuration.
    pub fn get_config(&self) -> DetectorConfig {
        self.config
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
    ///
    /// # Panics
    /// Panics if the upscaled image buffer cannot be created or viewed.
    #[allow(clippy::too_many_lines, clippy::unwrap_used)]
    pub fn detect_with_stats_and_options(
        &mut self,
        img: &ImageView,
        options: &DetectOptions,
    ) -> (Vec<Detection>, PipelineStats) {
        let res = self.detect_internal(img, options, false);
        (res.detections, res.stats)
    }

    /// Debugging: Extract quad candidates without decoding.
    ///
    /// This runs the pipeline up to the quad extraction stage and returns all distinct
    /// quads found. Useful for visualizing what the detector "sees" before the
    /// decoder rejects invalid tags.
    #[allow(clippy::too_many_lines, clippy::unwrap_used)]
    pub fn extract_candidates(
        &mut self,
        img: &ImageView,
        options: &DetectOptions,
    ) -> (Vec<Detection>, PipelineStats) {
        // This is now just a shortcut to detect_internal with a flag if we want to keep it,
        // but it's better to just return the full result if needed.
        // For now, let's keep the API compatible.
        let res = self.detect_internal(img, options, true);
        (res.candidates, res.stats)
    }

    /// Perform full detection and return all intermediate debug data.
    pub fn detect_full(
        &mut self,
        img: &ImageView,
        options: &DetectOptions,
    ) -> FullDetectionResult {
        self.detect_internal(img, options, true)
    }

    /// Internal unified detection pipeline.
    #[allow(clippy::too_many_lines, clippy::unwrap_used)]
    fn detect_internal(
        &mut self,
        img: &ImageView,
        options: &DetectOptions,
        capture_debug: bool,
    ) -> FullDetectionResult {
        let mut stats = PipelineStats::default();
        let start_total = std::time::Instant::now();

        // 0. Decimation or Upscaling
        let decimation = options.decimation.max(1);
        let upscale_factor = self.config.upscale_factor.max(1);

        self.arena.reset();

        // We handle decimation and upscaling as mutually exclusive for simplicity in the hot path,
        // though upscaling is usually for tiny tags and decimation for high-res streams.
        let (detection_img, effective_scale, refinement_img) = if decimation > 1 {
            let new_w = img.width / decimation;
            let new_h = img.height / decimation;
            let decimated_data = self.arena.alloc_slice_fill_copy(new_w * new_h, 0u8);
            let decimated_img = img
                .decimate_to(decimation, decimated_data)
                .expect("decimation failed");
            (decimated_img, 1.0 / decimation as f64, *img)
        } else if upscale_factor > 1 {
            let new_w = img.width * upscale_factor;
            let new_h = img.height * upscale_factor;
            self.upscale_buf.resize(new_w * new_h, 0);

            let upscaled_img = img
                .upscale_to(upscale_factor, &mut self.upscale_buf)
                .expect("valid upscaled view");
            (upscaled_img, upscale_factor as f64, upscaled_img)
        } else {
            (*img, 1.0, *img)
        };

        let img = &detection_img;

        // Backup config for potential restoration
        let original_tile_size = self.config.threshold_tile_size;
        let original_max_radius = self.config.threshold_max_radius;
        let original_edge_score = self.config.quad_min_edge_score;
        let original_quad_min_area = self.config.quad_min_area;
        let original_quad_min_edge_length = self.config.quad_min_edge_length;

        if upscale_factor > 1 {
            // Auto-scale parameters for upscaling
            self.config.threshold_tile_size *= upscale_factor;
            self.config.threshold_max_radius *= upscale_factor;
            self.config.quad_min_edge_score /= upscale_factor as f64;
        } else if decimation > 1 {
            // Auto-scale parameters for decimation (downscaling)
            self.config.threshold_tile_size = (self.config.threshold_tile_size / decimation).max(1);
            self.config.threshold_max_radius =
                (self.config.threshold_max_radius / decimation).max(1);
            let factor_sq = (decimation * decimation) as f64;
            #[allow(clippy::cast_sign_loss)]
            {
                self.config.quad_min_area =
                    (f64::from(self.config.quad_min_area) / factor_sq).max(1.0) as u32;
            }
            self.config.quad_min_edge_length =
                (self.config.quad_min_edge_length / decimation as f64).max(1.0);
        }

        let start_thresh = std::time::Instant::now();

        // 1a. Optional bilateral pre-filtering
        let filtered_img = if self.config.enable_bilateral {
            let filtered = self
                .arena
                .alloc_slice_fill_copy(img.width * img.height, 0u8);
            crate::filter::bilateral_filter(
                &self.arena,
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
            let sharpened = self
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

        let binarized = self
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);
        let threshold_map = self
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);

        {
            let _span = tracing::info_span!("threshold_adaptive").entered();
            // Compute integral image for O(1) local mean per pixel
            let stride = sharpened_img.width + 1;
            let integral = self
                .arena
                .alloc_slice_fill_copy(stride * (sharpened_img.height + 1), 0u64);
            crate::threshold::compute_integral_image(&sharpened_img, integral);

            // 1c. Apply adaptive or fixed-window threshold
            if self.config.enable_adaptive_window {
                // Compute gradient map for adaptive window sizing
                let gradient = self
                    .arena
                    .alloc_slice_fill_copy(img.width * img.height, 0u8);
                crate::filter::compute_gradient_map(&sharpened_img, gradient);

                // Apply gradient-based adaptive window threshold
                crate::threshold::adaptive_threshold_gradient_window(
                    &sharpened_img,
                    gradient,
                    integral,
                    binarized,
                    self.config.threshold_min_radius,
                    self.config.threshold_max_radius,
                    self.config.adaptive_threshold_gradient_threshold,
                    self.config.adaptive_threshold_constant,
                );
            } else {
                // Fallback to fixed 13x13 window (radius=6)
                crate::threshold::adaptive_threshold_integral(
                    &sharpened_img,
                    integral,
                    binarized,
                    6,
                    self.config.adaptive_threshold_constant,
                );
            }

            // For threshold_map, store the local mean (for potential threshold-model segmentation)
            let map_radius = if self.config.enable_adaptive_window {
                self.config.threshold_max_radius
            } else {
                6
            };
            crate::threshold::compute_threshold_map(
                &sharpened_img,
                integral,
                threshold_map,
                map_radius,
                self.config.adaptive_threshold_constant,
            );
        }
        stats.threshold_ms = start_thresh.elapsed().as_secs_f64() * 1000.0;

        // 2. Segmentation - Standard binary CCL
        let start_seg = std::time::Instant::now();
        let label_result = {
            let _span = tracing::info_span!("segmentation").entered();
            crate::segmentation::label_components_threshold_model(
                &self.arena,
                sharpened_img.data,
                threshold_map,
                img.width,
                img.height,
                self.config.segmentation_connectivity == config::SegmentationConnectivity::Eight,
                self.config.quad_min_area, // Keep the noise suppression
                self.config.segmentation_margin,
            )
        };
        stats.segmentation_ms = start_seg.elapsed().as_secs_f64() * 1000.0;

        // 3. Quad Extraction
        let start_quad = std::time::Instant::now();
        let candidates = {
            let _span = tracing::info_span!("quad_extraction").entered();
            crate::quad::extract_quads_with_config(
                &self.arena,
                &sharpened_img,
                &label_result,
                &self.config,
                decimation,
                &refinement_img,
            )
        };
        stats.quad_extraction_ms = start_quad.elapsed().as_secs_f64() * 1000.0;
        stats.num_candidates = candidates.len();

        // 4. Decoding
        let start_decode = std::time::Instant::now();
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

        // We use a small closure to encapsulate decoding logic and avoid duplication.
        // It takes an owned Detection and tries to decode it.
        // Returns (Option<Detection>, bool_failed_contrast, bool_failed_hamming, u32_best_hamming)
        let decode_fn = |mut cand: Detection| -> (Option<Detection>, bool, bool, u32) {
            let h = crate::decoder::Homography::square_to_quad(&cand.corners);
            if h.is_none() {
                return (None, false, false, u32::MAX);
            }
            let h = h.expect("Already checked is_some");

            let mut passed_contrast = false;
            let mut min_hamming = u32::MAX;

            for decoder in active_decoders {
                if let Some(bits) = crate::decoder::sample_grid(
                    &refinement_img,
                    &h,
                    decoder.as_ref(),
                    self.config.decoder_min_contrast,
                ) {
                    passed_contrast = true;
                    // We call a lower-level decode that returns the best hamming even if > threshold.
                    // But for now let's just use the existing decode and maybe hack it.
                    // Actually, let's just use 255 as max_hamming to see what we get.
                    if let Some((id, hamming, rot)) = decoder.decode_full(bits, 255) {
                        if hamming < min_hamming {
                            min_hamming = hamming;
                        }

                        // If it's within the official threshold, it's a success
                        if hamming <= if decoder.name() == "36h11" { 4 } else { 1 } {
                            cand.id = id;
                            cand.hamming = hamming;

                            let mut reordered = [[0.0; 2]; 4];
                            for (i, item) in reordered.iter_mut().enumerate() {
                                let src_idx = (i + usize::from(rot)) % 4;
                                *item = cand.corners[src_idx];
                            }
                            cand.corners = reordered;

                            if let (Some(intrinsics), Some(tag_size)) = (options.intrinsics, options.tag_size)
                            {
                                cand.pose =
                                    crate::pose::estimate_tag_pose(&intrinsics, &cand.corners, tag_size);
                            }
                            return (Some(cand), false, false, hamming);
                        }
                    }
                }
            }
            (None, !passed_contrast, passed_contrast, min_hamming)
        };

        let mut final_detections: Vec<Detection> = Vec::new();
        let mut processed_candidates = Vec::new();
        
        use std::sync::atomic::{AtomicUsize, Ordering};
        let rejected_contrast = AtomicUsize::new(0);
        let rejected_hamming = AtomicUsize::new(0);


        let results: Vec<(Option<Detection>, bool, bool, u32, Detection)> = candidates
            .into_par_iter()
            .map(|cand| {
                let cand_copy = if capture_debug { Some(cand.clone()) } else { None };
                let (det, failed_contrast, failed_hamming, best_h) = decode_fn(cand);
                (det, failed_contrast, failed_hamming, best_h, cand_copy.unwrap_or(Detection::default()))
            })
            .collect();

        for (det, failed_contrast, failed_hamming, best_h, mut cand) in results {
            if failed_contrast {
                rejected_contrast.fetch_add(1, Ordering::Relaxed);
            }
            if failed_hamming {
                rejected_hamming.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(d) = det {
                final_detections.push(d);
            }
            if capture_debug {
                cand.hamming = best_h; 
                processed_candidates.push(cand);
            }
        }

        stats.num_rejected_by_contrast = rejected_contrast.load(Ordering::Relaxed);
        stats.num_rejected_by_hamming = rejected_hamming.load(Ordering::Relaxed);

        stats.decoding_ms = start_decode.elapsed().as_secs_f64() * 1000.0;
        stats.num_detections = final_detections.len();

        // Final coordinate adjustment and scaling
        let inv_scale = if upscale_factor > 1 {
            1.0 / effective_scale
        } else {
            1.0
        };

        for d in &mut final_detections {
            for corner in &mut d.corners {
                corner[0] = (corner[0] + 0.5) * inv_scale;
                corner[1] = (corner[1] + 0.5) * inv_scale;
            }
            d.center[0] = (d.center[0] + 0.5) * inv_scale;
            d.center[1] = (d.center[1] + 0.5) * inv_scale;

            if let (Some(intrinsics), Some(tag_size)) = (options.intrinsics, options.tag_size) {
                d.pose = crate::pose::estimate_tag_pose(&intrinsics, &d.corners, tag_size);
            }
        }

        if capture_debug {
            for d in &mut processed_candidates {
                for corner in &mut d.corners {
                    corner[0] = (corner[0] + 0.5) * inv_scale;
                    corner[1] = (corner[1] + 0.5) * inv_scale;
                }
                d.center[0] = (d.center[0] + 0.5) * inv_scale;
                d.center[1] = (d.center[1] + 0.5) * inv_scale;
            }
        }

        stats.total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        // Restore config
        self.config.threshold_tile_size = original_tile_size;
        self.config.threshold_max_radius = original_max_radius;
        self.config.quad_min_edge_score = original_edge_score;
        self.config.quad_min_area = original_quad_min_area;
        self.config.quad_min_edge_length = original_quad_min_edge_length;

        FullDetectionResult {
            detections: final_detections,
            candidates: processed_candidates,
            binarized: if capture_debug {
                Some(binarized.to_vec())
            } else {
                None
            },
            labels: if capture_debug {
                Some(label_result.labels.to_vec())
            } else {
                None
            },
            stats,
        }
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
