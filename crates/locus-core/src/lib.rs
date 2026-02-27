//! High-performance AprilTag and ArUco detection engine.
//!
//! Locus is a research-oriented, memory-safe fiducial marker detector targeting low
//! latency. It provides a performance-focused pipeline for robotics and computer vision,
//! with strict zero-heap allocation in the detection hot-path.
//!
//! # Key Features
//!
//! - **Performance**: SIMD-accelerated adaptive thresholding and connected components.
//! - **Accuracy**: Advanced sub-pixel refinement and probabilistic pose estimation.
//! - **Flexibility**: Pluggable tag families (AprilTag 36h11, 16h5, ArUco).
//! - **Memory Safety**: Arena-based memory management ([`bumpalo`]).
//!
//! # Architecture
//!
//! The pipeline is designed around Data-Oriented Design (DOD) principles:
//!
//! 1. **Preprocessing**: [`threshold::ThresholdEngine`] computes local tile statistics and performs adaptive binarization.
//! 2. **Segmentation**: [`segmentation::label_components_threshold_model`] identifies quad candidates using Union-Find.
//! 3. **Extraction**: [`quad::extract_quads_with_config`] traces contours and fits initial polygon candidates.
//! 4. **Decoding**: [`Detector`] samples bit grids and performs Hamming error correction via [`strategy::DecodingStrategy`].
//! 5. **Pose Estimation**: [`pose::estimate_tag_pose`] recovers 6-DOF transforms using IPPE or weighted LM.
//!
//! # Examples
//!
//! ```
//! use locus_core::{Detector, DetectorConfig, DetectOptions, TagFamily, ImageView};
//!
//! // Create a detector with default settings
//! let mut detector = Detector::new();
//!
//! // Create a view into your image data (zero-copy)
//! let pixels = vec![0u8; 640 * 480];
//! let img = ImageView::new(&pixels, 640, 480, 640).unwrap();
//!
//! // Detect tags (default family: AprilTag 36h11)
//! let detections = detector.detect(&img);
//!
//! for detection in detections {
//!     println!("Detected tag {} at {:?}", detection.id, detection.center);
//! }
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
/// Weighted pose estimation logic.
pub mod pose_weighted;
/// Quad extraction and geometric primitives.
pub mod quad;
/// Connected components labeling using Union-Find.
pub mod segmentation;
/// Decoding strategies (Hard vs Soft).
pub mod strategy;
/// Utilities for testing and synthetic data generation.
pub mod test_utils;
/// Adaptive thresholding implementation.
pub mod threshold;

pub use crate::config::{DetectOptions, DetectorConfig, TagFamily};
use crate::decoder::TagDecoder;
pub use crate::decoder::family_to_decoder;
pub use crate::image::ImageView;
use bumpalo::Bump;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    /// The extracted bits from the tag (for debugging).
    pub bits: u64,
    /// The 3D pose of the tag relative to the camera (if requested).
    pub pose: Option<crate::pose::Pose>,
    /// The covariance of the estimated 3D pose (6x6 matrix), if computed.
    /// Order: [tx, ty, tz, rx, ry, rz] (Lie Algebra se3).
    pub pose_covariance: Option<[[f64; 6]; 6]>,
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
    pub fn detect_full(&mut self, img: &ImageView, options: &DetectOptions) -> FullDetectionResult {
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

        let start_thresh = std::time::Instant::now();
        {
            let _span = tracing::info_span!("threshold_binarize").entered();
            if self.config.enable_adaptive_window {
                // Adaptive window is robust for tiny tags but slow (O(N) per pixel)
                let stride = sharpened_img.width + 1;
                let integral = self
                    .arena
                    .alloc_slice_fill_copy(stride * (sharpened_img.height + 1), 0u64);
                crate::threshold::compute_integral_image(&sharpened_img, integral);

                let gradient = self
                    .arena
                    .alloc_slice_fill_copy(img.width * img.height, 0u8);
                crate::filter::compute_gradient_map(&sharpened_img, gradient);

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
                threshold_map.fill(128); // Not implemented for adaptive
            } else {
                // Tile-based thresholding is fast (sub-2ms)
                let engine = crate::threshold::ThresholdEngine::from_config(&self.config);
                let stats = engine.compute_tile_stats(&self.arena, &sharpened_img);
                engine.apply_threshold_with_map(
                    &self.arena,
                    &sharpened_img,
                    &stats,
                    binarized,
                    threshold_map,
                );
            }
        }
        stats.threshold_ms = start_thresh.elapsed().as_secs_f64() * 1000.0;

        // 2. Segmentation - Standard binary CCL
        let start_seg = std::time::Instant::now();
        let label_result = {
            let _span = tracing::info_span!("segmentation").entered();
            crate::segmentation::label_components_threshold_model(
                &self.arena,
                sharpened_img.data,
                sharpened_img.stride,
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

        let (mut final_detections, mut processed_candidates, rejected_contrast, rejected_hamming) =
            match self.config.decode_mode {
                config::DecodeMode::Hard => {
                    Self::decode_candidates::<crate::strategy::HardStrategy>(
                        &self.config,
                        candidates,
                        &refinement_img,
                        active_decoders,
                        capture_debug,
                        options,
                    )
                },
                config::DecodeMode::Soft => {
                    Self::decode_candidates::<crate::strategy::SoftStrategy>(
                        &self.config,
                        candidates,
                        &refinement_img,
                        active_decoders,
                        capture_debug,
                        options,
                    )
                },
            };

        stats.num_rejected_by_contrast = rejected_contrast;
        stats.num_rejected_by_hamming = rejected_hamming;

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
                let (pose, covariance) = crate::pose::estimate_tag_pose(
                    &intrinsics,
                    &d.corners,
                    tag_size,
                    Some(img),
                    options.pose_estimation_mode,
                );
                d.pose = pose;
                d.pose_covariance = covariance;
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

impl Detector {
    fn decode_candidates<S: crate::strategy::DecodingStrategy>(
        config: &crate::config::DetectorConfig,
        candidates: Vec<Detection>,
        refinement_img: &ImageView,
        active_decoders: &[Box<dyn TagDecoder + Send + Sync>],
        capture_debug: bool,
        options: &DetectOptions,
    ) -> (Vec<Detection>, Vec<Detection>, usize, usize) {
        let rejected_contrast = AtomicUsize::new(0);
        let rejected_hamming = AtomicUsize::new(0);

        let results: Vec<_> = candidates
            .into_par_iter()
            .map(|cand| {
                let cand_copy = if capture_debug {
                    Some(cand.clone())
                } else {
                    None
                };
                let (det, failed_contrast, failed_hamming, best_h, bits) =
                    Self::decode_single_candidate::<S>(
                        config,
                        cand,
                        refinement_img,
                        active_decoders,
                        options,
                    );
                (
                    det,
                    failed_contrast,
                    failed_hamming,
                    best_h,
                    bits,
                    cand_copy.unwrap_or(Detection::default()),
                )
            })
            .collect();

        let mut final_detections = Vec::new();
        let mut processed_candidates = Vec::new();

        for (det, failed_contrast, failed_hamming, best_h, bits, mut cand) in results {
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
                cand.bits = bits;
                processed_candidates.push(cand);
            }
        }

        (
            final_detections,
            processed_candidates,
            rejected_contrast.load(Ordering::Relaxed),
            rejected_hamming.load(Ordering::Relaxed),
        )
    }

    #[allow(clippy::too_many_lines)]
    fn decode_single_candidate<S: crate::strategy::DecodingStrategy>(
        config: &crate::config::DetectorConfig,
        mut cand: Detection,
        refinement_img: &ImageView,
        active_decoders: &[Box<dyn TagDecoder + Send + Sync>],
        options: &DetectOptions,
    ) -> (Option<Detection>, bool, bool, u32, u64) {
        let scales = [1.0, 0.9, 1.1]; // Try most likely scales

        let mut best_overall_h = u32::MAX;
        let mut best_overall_code: Option<S::Code> = None;
        let mut passed_contrast = false;

        for scale in scales {
            let center = cand.center;
            let mut scaled_corners = cand.corners;
            if (scale - 1.0f64).abs() > 1e-4 {
                for c in &mut scaled_corners {
                    c[0] = center[0] + (c[0] - center[0]) * scale;
                    c[1] = center[1] + (c[1] - center[1]) * scale;
                }
            }

            let Some(h) = crate::decoder::Homography::square_to_quad(&scaled_corners) else {
                continue;
            };

            for decoder in active_decoders {
                if let Some(code) =
                    crate::decoder::sample_grid_generic::<S>(refinement_img, &h, decoder.as_ref())
                {
                    passed_contrast = true;
                    if let Some((id, hamming, rot)) = S::decode(&code, decoder.as_ref(), 255) {
                        if hamming < best_overall_h {
                            best_overall_h = hamming;
                            best_overall_code = Some(code.clone());
                        }

                        if hamming
                            <= match decoder.name() {
                                "36h11" | "Aruco36h11" => 4,
                                _ => 1,
                            }
                        {
                            cand.id = id;
                            cand.hamming = hamming;
                            cand.bits = S::to_debug_bits(&code);

                            // Correct rotation on ORIGINAL corners
                            let mut reordered = [[0.0; 2]; 4];
                            for (i, item) in reordered.iter_mut().enumerate() {
                                let src_idx = (i + usize::from(rot)) % 4;
                                *item = cand.corners[src_idx];
                            }
                            cand.corners = reordered;

                            // Always perform ERF refinement for finalists if requested
                            if config.refinement_mode == crate::config::CornerRefinementMode::Erf {
                                let decode_arena = bumpalo::Bump::new();
                                let refined_corners = crate::decoder::refine_corners_erf(
                                    &decode_arena,
                                    refinement_img,
                                    &cand.corners,
                                    config.subpixel_refinement_sigma,
                                );

                                // Verify that refined corners still yield a valid decode
                                if let Some(h_ref) =
                                    crate::decoder::Homography::square_to_quad(&refined_corners)
                                    && let Some(code_ref) = crate::decoder::sample_grid_generic::<S>(
                                        refinement_img,
                                        &h_ref,
                                        decoder.as_ref(),
                                    )
                                    && let Some((id_ref, hamming_ref, _)) =
                                        S::decode(&code_ref, decoder.as_ref(), 255)
                                {
                                    // Only keep if it's the same tag and hamming is not worse
                                    if id_ref == id && hamming_ref <= hamming {
                                        cand.corners = refined_corners;
                                        cand.hamming = hamming_ref;
                                        cand.bits = S::to_debug_bits(&code_ref);
                                    }
                                }
                            }

                            if let (Some(intrinsics), Some(tag_size)) =
                                (options.intrinsics, options.tag_size)
                            {
                                let (pose, covariance) = crate::pose::estimate_tag_pose(
                                    &intrinsics,
                                    &cand.corners,
                                    tag_size,
                                    Some(refinement_img),
                                    options.pose_estimation_mode,
                                );
                                cand.pose = pose;
                                cand.pose_covariance = covariance;
                            }
                            return (Some(cand), false, false, hamming, S::to_debug_bits(&code));
                        }
                    }
                }
            }

            // If 1.0 works perfectly, don't even try 0.9/1.1
            if best_overall_h == 0 {
                break;
            }
        }

        // Stage 2: Configurable Corner Refinement
        let max_h_for_refine = if active_decoders.iter().any(|d| d.name() == "36h11") {
            10
        } else {
            4
        };
        if best_overall_h
            > if active_decoders.iter().any(|d| d.name() == "36h11") {
                4
            } else {
                1
            }
            && best_overall_h <= max_h_for_refine
            && best_overall_code.is_some()
        {
            let mut current_corners = cand.corners;
            let best_code = best_overall_code
                .as_ref()
                .expect("best_overall_code is some")
                .clone();

            match config.refinement_mode {
                crate::config::CornerRefinementMode::None => {},
                crate::config::CornerRefinementMode::Edge => {
                    // Current edge-based subpixel refinement (Gradient Hill Climbing)
                    let nudge = 0.2;

                    for _pass in 0..2 {
                        let mut pass_improved = false;
                        for c_idx in 0..4 {
                            for (dx, dy) in
                                [(nudge, 0.0), (-nudge, 0.0), (0.0, nudge), (0.0, -nudge)]
                            {
                                let mut test_corners = current_corners;
                                test_corners[c_idx][0] += dx;
                                test_corners[c_idx][1] += dy;

                                if let Some(h) =
                                    crate::decoder::Homography::square_to_quad(&test_corners)
                                {
                                    for decoder in active_decoders {
                                        if let Some(code) = crate::decoder::sample_grid_generic::<S>(
                                            refinement_img,
                                            &h,
                                            decoder.as_ref(),
                                        ) && let Some((id, hamming, rot)) =
                                            S::decode(&code, decoder.as_ref(), 255)
                                            && hamming < best_overall_h
                                        {
                                            best_overall_h = hamming;
                                            best_overall_code = Some(code.clone());
                                            current_corners = test_corners;
                                            pass_improved = true;
                                            // Success check
                                            if hamming
                                                <= if decoder.name() == "36h11" { 4 } else { 1 }
                                            {
                                                cand.id = id;
                                                cand.hamming = hamming;
                                                cand.bits = S::to_debug_bits(&code);
                                                cand.corners = current_corners;

                                                // Fix rotation
                                                let mut reordered = [[0.0; 2]; 4];
                                                for (i, item) in reordered.iter_mut().enumerate() {
                                                    let src_idx = (i + usize::from(rot)) % 4;
                                                    *item = cand.corners[src_idx];
                                                }
                                                cand.corners = reordered;
                                                return (
                                                    Some(cand),
                                                    false,
                                                    false,
                                                    hamming,
                                                    S::to_debug_bits(&code),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if !pass_improved {
                            break;
                        }
                    }
                },
                crate::config::CornerRefinementMode::GridFit => {
                    // New GridFit optimization
                    for decoder in active_decoders {
                        // Recover ID and rotation from the best bits to construct the ideal target
                        if let Some((id, hamming, rot)) =
                            S::decode(&best_code, decoder.as_ref(), 255)
                        {
                            // Only proceed if this decoder explains the bits as well as our best guess
                            if hamming == best_overall_h {
                                // Fetch canonical code
                                if let Some(mut target_bits) = decoder.get_code(id as u16) {
                                    // Rotate to match observed orientation
                                    for _ in 0..rot {
                                        target_bits = crate::dictionaries::rotate90(
                                            target_bits,
                                            decoder.dimension(),
                                        );
                                    }

                                    // Run GridFit with the CLEAN target pattern
                                    // NOTE: GridFit uses contrast against binary target.
                                    let refined = crate::decoder::refine_corners_gridfit(
                                        refinement_img,
                                        &current_corners,
                                        decoder.as_ref(), // Use the decoder to get sample points
                                        target_bits, // Maximize contrast against GROUND TRUTH CODE
                                    );

                                    // Check if refined corners actually improve Hamming
                                    if let Some(h) =
                                        crate::decoder::Homography::square_to_quad(&refined)
                                        && let Some(code) = crate::decoder::sample_grid_generic::<S>(
                                            refinement_img,
                                            &h,
                                            decoder.as_ref(),
                                        )
                                        && let Some((id, hamming, rot)) =
                                            S::decode(&code, decoder.as_ref(), 255)
                                        && hamming < best_overall_h
                                    {
                                        best_overall_h = hamming;
                                        best_overall_code = Some(code.clone());
                                        current_corners = refined;
                                        // Success check
                                        if hamming <= if decoder.name() == "36h11" { 4 } else { 1 }
                                        {
                                            cand.id = id;
                                            cand.hamming = hamming;
                                            cand.bits = S::to_debug_bits(&code);
                                            cand.corners = current_corners;

                                            // Fix rotation
                                            let mut reordered = [[0.0; 2]; 4];
                                            for (i, item) in reordered.iter_mut().enumerate() {
                                                let src_idx = (i + usize::from(rot)) % 4;
                                                *item = cand.corners[src_idx];
                                            }
                                            cand.corners = reordered;

                                            return (
                                                Some(cand),
                                                false,
                                                false,
                                                hamming,
                                                S::to_debug_bits(&code),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                crate::config::CornerRefinementMode::Erf => {
                    let nudge = 0.2;
                    for _pass in 0..2 {
                        let mut pass_improved = false;
                        for c_idx in 0..4 {
                            for (dx, dy) in
                                [(nudge, 0.0), (-nudge, 0.0), (0.0, nudge), (0.0, -nudge)]
                            {
                                let mut test_corners = current_corners;
                                test_corners[c_idx][0] += dx;
                                test_corners[c_idx][1] += dy;

                                if let Some(h) =
                                    crate::decoder::Homography::square_to_quad(&test_corners)
                                {
                                    for decoder in active_decoders {
                                        if let Some(code) = crate::decoder::sample_grid_generic::<S>(
                                            refinement_img,
                                            &h,
                                            decoder.as_ref(),
                                        ) && let Some((id, hamming, rot)) =
                                            S::decode(&code, decoder.as_ref(), 255)
                                            && hamming < best_overall_h
                                        {
                                            best_overall_h = hamming;
                                            best_overall_code = Some(code.clone());
                                            current_corners = test_corners;
                                            pass_improved = true;

                                            if hamming
                                                <= if decoder.name() == "36h11" { 4 } else { 1 }
                                            {
                                                cand.id = id;
                                                cand.hamming = hamming;
                                                cand.bits = S::to_debug_bits(&code);
                                                cand.corners = current_corners;

                                                // Final ERF polish for maximum precision
                                                let decode_arena = bumpalo::Bump::new();
                                                let refined = crate::decoder::refine_corners_erf(
                                                    &decode_arena,
                                                    refinement_img,
                                                    &cand.corners,
                                                    config.subpixel_refinement_sigma,
                                                );

                                                // Verify refined corners
                                                if let Some(h_ref) =
                                                    crate::decoder::Homography::square_to_quad(
                                                        &refined,
                                                    )
                                                    && let Some(code_ref) =
                                                        crate::decoder::sample_grid_generic::<S>(
                                                            refinement_img,
                                                            &h_ref,
                                                            decoder.as_ref(),
                                                        )
                                                    && let Some((id_ref, hamming_ref, _)) =
                                                        S::decode(&code_ref, decoder.as_ref(), 255)
                                                    && id_ref == id
                                                    && hamming_ref <= hamming
                                                {
                                                    cand.corners = refined;
                                                    cand.hamming = hamming_ref;
                                                    cand.bits = S::to_debug_bits(&code_ref);
                                                }

                                                let mut reordered = [[0.0; 2]; 4];
                                                for (i, item) in reordered.iter_mut().enumerate() {
                                                    let src_idx = (i + usize::from(rot)) % 4;
                                                    *item = cand.corners[src_idx];
                                                }
                                                cand.corners = reordered;
                                                return (
                                                    Some(cand),
                                                    false,
                                                    false,
                                                    hamming,
                                                    S::to_debug_bits(&code),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if !pass_improved {
                            break;
                        }
                    }
                },
            }
        }

        (
            None,
            !passed_contrast,
            passed_contrast,
            best_overall_h,
            best_overall_code.map_or(0, |c| S::to_debug_bits(&c)),
        )
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
