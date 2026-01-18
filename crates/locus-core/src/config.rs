//! Configuration types for the detector pipeline.
//!
//! This module provides two configuration types:
//! - [`DetectorConfig`]: Pipeline-level configuration (immutable after construction)
//! - [`DetectOptions`]: Per-call options (e.g., which tag families to decode)

// ============================================================================
// DetectorConfig: Pipeline-level configuration
// ============================================================================

/// Segmentation connectivity mode.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SegmentationConnectivity {
    /// 4-connectivity: Pixels connect horizontally and vertically only.
    /// Required for separating checkerboard corners.
    Four,
    /// 8-connectivity: Pixels connect horizontally, vertically, and diagonally.
    /// Better for isolated tags with broken borders.
    Eight,
}

/// Pipeline-level configuration for the detector.
///
/// These settings affect the fundamental behavior of the detection pipeline
/// and are immutable after the `Detector` is constructed. Use the builder
/// pattern for ergonomic construction.
///
/// # Example
/// ```
/// use locus_core::config::DetectorConfig;
///
/// let config = DetectorConfig::builder()
///     .threshold_tile_size(16)
///     .quad_min_area(200)
///     .build();
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectorConfig {
    // Threshold parameters
    /// Tile size for adaptive thresholding (default: 8).
    /// Larger tiles are faster but less adaptive to local contrast.
    pub threshold_tile_size: usize,
    /// Minimum intensity range in a tile to be considered valid (default: 10).
    /// Tiles with lower range are treated as uniform (no edges).
    pub threshold_min_range: u8,

    // Adaptive filtering parameters
    /// Enable bilateral pre-filtering for edge-preserving noise reduction (default: true).
    pub enable_bilateral: bool,
    /// Bilateral spatial sigma for spatial smoothing (default: 3.0).
    pub bilateral_sigma_space: f32,
    /// Bilateral color sigma for edge preservation (default: 30.0).
    /// Higher values = more smoothing across edges.
    pub bilateral_sigma_color: f32,
    
    /// Enable Laplacian sharpening to enhance edges for small tags (default: false).
    pub enable_sharpening: bool,
    
    /// Enable adaptive threshold window sizing based on gradient (default: true).
    pub enable_adaptive_window: bool,
    /// Minimum threshold window radius for high-gradient regions (default: 2 = 5x5).
    pub threshold_min_radius: usize,
    /// Maximum threshold window radius for low-gradient regions (default: 7 = 15x15).
    pub threshold_max_radius: usize,

    // Quad filtering parameters
    /// Minimum quad area in pixels (default: 400).
    pub quad_min_area: u32,
    /// Maximum aspect ratio of bounding box (default: 3.0).
    pub quad_max_aspect_ratio: f32,
    /// Minimum fill ratio (pixel count / bbox area) (default: 0.3).
    pub quad_min_fill_ratio: f32,
    /// Maximum fill ratio (default: 0.95).
    pub quad_max_fill_ratio: f32,
    /// Minimum edge length in pixels (default: 4.0).
    pub quad_min_edge_length: f64,
    /// Minimum edge alignment score (0.0 to 1.0)
    pub quad_min_edge_score: f64,
    /// PSF blur factor for subpixel refinement (e.g., 0.6)
    pub subpixel_refinement_sigma: f64,
    /// Segmentation connectivity (4-way or 8-way).
    pub segmentation_connectivity: SegmentationConnectivity,
    /// Factor to upscale the image before detection (1 = no upscaling).
    /// Increasing this to 2 allows detecting smaller tags (e.g., < 15px)
    /// at the cost of processing speed (O(N^2)). Nearest-neighbor interpolation is used.
    pub upscale_factor: usize,
    
    // Decoder parameters
    /// Minimum contrast range for Otsu-based bit classification (default: 20.0).
    /// For checkerboard patterns with densely packed tags, lower values (e.g., 10.0)
    /// can improve recall on small/blurry tags.
    pub decoder_min_contrast: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            threshold_tile_size: 4, 
            threshold_min_range: 10, // Increased from 2 to reduce noise with tiled thresholding
            enable_bilateral: true,
            bilateral_sigma_space: 0.8,
            bilateral_sigma_color: 30.0,
            enable_sharpening: false,
            enable_adaptive_window: true,
            threshold_min_radius: 2,
            threshold_max_radius: 7,
            quad_min_area: 16,      // Lowered from 25 for small/distant tags
            quad_max_aspect_ratio: 8.0, // Increased from 3.0 to support extreme foreshortening
            quad_min_fill_ratio: 0.15, // Lowered from 0.3 for thin quads at tilt
            quad_max_fill_ratio: 0.98,
            quad_min_edge_length: 3.0, // Lowered from 4.0
            quad_min_edge_score: 0.4,  // Slightly lowered from 0.5
            subpixel_refinement_sigma: 0.6,
            segmentation_connectivity: SegmentationConnectivity::Eight,
            upscale_factor: 1,
            decoder_min_contrast: 20.0,
        }
    }
}

impl DetectorConfig {
    /// Create a new builder for `DetectorConfig`.
    #[must_use]
    pub fn builder() -> DetectorConfigBuilder {
        DetectorConfigBuilder::default()
    }
}

/// Builder for [`DetectorConfig`].
#[derive(Default)]
pub struct DetectorConfigBuilder {
    threshold_tile_size: Option<usize>,
    threshold_min_range: Option<u8>,
    enable_bilateral: Option<bool>,
    bilateral_sigma_space: Option<f32>,
    bilateral_sigma_color: Option<f32>,
    enable_sharpening: Option<bool>,
    enable_adaptive_window: Option<bool>,
    threshold_min_radius: Option<usize>,
    threshold_max_radius: Option<usize>,
    quad_min_area: Option<u32>,
    quad_max_aspect_ratio: Option<f32>,
    quad_min_fill_ratio: Option<f32>,
    quad_max_fill_ratio: Option<f32>,
    quad_min_edge_length: Option<f64>,
    /// Minimum gradient magnitude along edges (rejects weak candidates).
    pub quad_min_edge_score: Option<f64>,
    /// Sigma for Gaussian in subpixel refinement.
    pub subpixel_refinement_sigma: Option<f64>,
    /// Connectivity mode for segmentation (4 or 8).
    pub segmentation_connectivity: Option<SegmentationConnectivity>,
    /// Upscale factor for low-res images (1 = no upscale).
    pub upscale_factor: Option<usize>,
    /// Minimum contrast for decoder to accept a tag.
    pub decoder_min_contrast: Option<f64>,
}

impl DetectorConfigBuilder {
    /// Set the tile size for adaptive thresholding.
    #[must_use]
    pub fn threshold_tile_size(mut self, size: usize) -> Self {
        self.threshold_tile_size = Some(size);
        self
    }

    /// Set the minimum intensity range for valid tiles.
    #[must_use]
    pub fn threshold_min_range(mut self, range: u8) -> Self {
        self.threshold_min_range = Some(range);
        self
    }

    /// Set the minimum quad area.
    #[must_use]
    pub fn quad_min_area(mut self, area: u32) -> Self {
        self.quad_min_area = Some(area);
        self
    }

    /// Set the maximum aspect ratio.
    #[must_use]
    pub fn quad_max_aspect_ratio(mut self, ratio: f32) -> Self {
        self.quad_max_aspect_ratio = Some(ratio);
        self
    }

    /// Set the minimum fill ratio.
    #[must_use]
    pub fn quad_min_fill_ratio(mut self, ratio: f32) -> Self {
        self.quad_min_fill_ratio = Some(ratio);
        self
    }

    /// Set the maximum fill ratio.
    #[must_use]
    pub fn quad_max_fill_ratio(mut self, ratio: f32) -> Self {
        self.quad_max_fill_ratio = Some(ratio);
        self
    }

    /// Set the minimum edge length.
    #[must_use]
    pub fn quad_min_edge_length(mut self, length: f64) -> Self {
        self.quad_min_edge_length = Some(length);
        self
    }

    /// Set the minimum edge gradient score.
    #[must_use]
    pub fn quad_min_edge_score(mut self, score: f64) -> Self {
        self.quad_min_edge_score = Some(score);
        self
    }

    /// Enable or disable bilateral pre-filtering.
    #[must_use]
    pub fn enable_bilateral(mut self, enable: bool) -> Self {
        self.enable_bilateral = Some(enable);
        self
    }

    /// Set bilateral spatial sigma.
    #[must_use]
    pub fn bilateral_sigma_space(mut self, sigma: f32) -> Self {
        self.bilateral_sigma_space = Some(sigma);
        self
    }

    /// Set bilateral color sigma.
    #[must_use]
    pub fn bilateral_sigma_color(mut self, sigma: f32) -> Self {
        self.bilateral_sigma_color = Some(sigma);
        self
    }

    /// Enable or disable Laplacian sharpening.
    #[must_use]
    pub fn enable_sharpening(mut self, enable: bool) -> Self {
        self.enable_sharpening = Some(enable);
        self
    }

    /// Enable or disable adaptive threshold window sizing.
    #[must_use]
    pub fn enable_adaptive_window(mut self, enable: bool) -> Self {
        self.enable_adaptive_window = Some(enable);
        self
    }

    /// Set minimum threshold window radius.
    #[must_use]
    pub fn threshold_min_radius(mut self, radius: usize) -> Self {
        self.threshold_min_radius = Some(radius);
        self
    }

    /// Set maximum threshold window radius.
    #[must_use]
    pub fn threshold_max_radius(mut self, radius: usize) -> Self {
        self.threshold_max_radius = Some(radius);
        self
    }

    /// Build the configuration, using defaults for unset fields.
    #[must_use]
    pub fn build(self) -> DetectorConfig {
        let d = DetectorConfig::default();
        DetectorConfig {
            threshold_tile_size: self.threshold_tile_size.unwrap_or(d.threshold_tile_size),
            threshold_min_range: self.threshold_min_range.unwrap_or(d.threshold_min_range),
            enable_bilateral: self.enable_bilateral.unwrap_or(d.enable_bilateral),
            bilateral_sigma_space: self.bilateral_sigma_space.unwrap_or(d.bilateral_sigma_space),
            bilateral_sigma_color: self.bilateral_sigma_color.unwrap_or(d.bilateral_sigma_color),
            enable_sharpening: self.enable_sharpening.unwrap_or(d.enable_sharpening),
            enable_adaptive_window: self.enable_adaptive_window.unwrap_or(d.enable_adaptive_window),
            threshold_min_radius: self.threshold_min_radius.unwrap_or(d.threshold_min_radius),
            threshold_max_radius: self.threshold_max_radius.unwrap_or(d.threshold_max_radius),
            quad_min_area: self.quad_min_area.unwrap_or(d.quad_min_area),
            quad_max_aspect_ratio: self
                .quad_max_aspect_ratio
                .unwrap_or(d.quad_max_aspect_ratio),
            quad_min_fill_ratio: self.quad_min_fill_ratio.unwrap_or(d.quad_min_fill_ratio),
            quad_max_fill_ratio: self.quad_max_fill_ratio.unwrap_or(d.quad_max_fill_ratio),
            quad_min_edge_length: self.quad_min_edge_length.unwrap_or(d.quad_min_edge_length),
            quad_min_edge_score: self.quad_min_edge_score.unwrap_or(d.quad_min_edge_score),
            subpixel_refinement_sigma: self.subpixel_refinement_sigma.unwrap_or(d.subpixel_refinement_sigma),
            segmentation_connectivity: self.segmentation_connectivity.unwrap_or(d.segmentation_connectivity),
            upscale_factor: self.upscale_factor.unwrap_or(d.upscale_factor),
            decoder_min_contrast: self.decoder_min_contrast.unwrap_or(d.decoder_min_contrast),
        }
    }

    /// Set the segmentation connectivity.
    #[must_use]
    pub fn segmentation_connectivity(mut self, connectivity: SegmentationConnectivity) -> Self {
        self.segmentation_connectivity = Some(connectivity);
        self
    }

    /// Set the upscale factor (1 = no upscaling, 2 = 2x, etc.).
    #[must_use]
    pub fn upscale_factor(mut self, factor: usize) -> Self {
        self.upscale_factor = Some(factor);
        self
    }

    /// Set the minimum contrast for decoder bit classification.
    /// Lower values (e.g., 10.0) improve recall on small/blurry checkerboard tags.
    #[must_use]
    pub fn decoder_min_contrast(mut self, contrast: f64) -> Self {
        self.decoder_min_contrast = Some(contrast);
        self
    }
}

// ============================================================================
// DetectOptions: Per-call detection options
// ============================================================================

/// Tag family identifier for per-call decoder selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TagFamily {
    /// AprilTag 36h11 family (587 codes, 11-bit hamming distance).
    AprilTag36h11,
    /// AprilTag 16h5 family (30 codes, 5-bit hamming distance).
    AprilTag16h5,
    /// ArUco 4x4_50 dictionary.
    ArUco4x4_50,
    /// ArUco 4x4_100 dictionary.
    ArUco4x4_100,
}

impl TagFamily {
    /// Returns all available tag families.
    #[must_use]
    pub const fn all() -> &'static [TagFamily] {
        &[
            TagFamily::AprilTag36h11,
            TagFamily::AprilTag16h5,
            TagFamily::ArUco4x4_50,
            TagFamily::ArUco4x4_100,
        ]
    }
}

/// Per-call detection options.
///
/// These allow customizing which tag families to decode for a specific call,
/// enabling performance optimization when you know which tags to expect.
///
/// # Example
/// ```
/// use locus_core::config::{DetectOptions, TagFamily};
///
/// // Only search for AprilTag 36h11 tags (fastest)
/// let options = DetectOptions::with_families(&[TagFamily::AprilTag36h11]);
///
/// // Search for multiple families
/// let multi = DetectOptions::with_families(&[
///     TagFamily::AprilTag36h11,
///     TagFamily::ArUco4x4_50,
/// ]);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
pub struct DetectOptions {
    /// Tag families to attempt decoding. Empty means use detector defaults.
    pub families: Vec<TagFamily>,
    /// Camera intrinsics for 3D pose estimation. If None, pose is not computed.
    pub intrinsics: Option<crate::pose::CameraIntrinsics>,
    /// Physical size of the tag in world units (e.g. meters) for 3D pose estimation.
    pub tag_size: Option<f64>,
}

impl DetectOptions {
    /// Create a new builder for `DetectOptions`.
    #[must_use]
    pub fn builder() -> DetectOptionsBuilder {
        DetectOptionsBuilder::default()
    }
    /// Create options that decode only the specified tag families.
    #[must_use]
    pub fn with_families(families: &[TagFamily]) -> Self {
        Self {
            families: families.to_vec(),
            intrinsics: None,
            tag_size: None,
        }
    }

    /// Create options that decode all known tag families.
    #[must_use]
    pub fn all_families() -> Self {
        Self {
            families: TagFamily::all().to_vec(),
            intrinsics: None,
            tag_size: None,
        }
    }
}

/// Builder for [`DetectOptions`].
#[derive(Default)]
pub struct DetectOptionsBuilder {
    families: Vec<TagFamily>,
    intrinsics: Option<crate::pose::CameraIntrinsics>,
    tag_size: Option<f64>,
}

impl DetectOptionsBuilder {
    /// Set the tag families to decode.
    #[must_use]
    pub fn families(mut self, families: &[TagFamily]) -> Self {
        self.families = families.to_vec();
        self
    }

    /// Set camera intrinsics for pose estimation.
    #[must_use]
    pub fn intrinsics(mut self, fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        self.intrinsics = Some(crate::pose::CameraIntrinsics::new(fx, fy, cx, cy));
        self
    }

    /// Set physical tag size for pose estimation.
    #[must_use]
    pub fn tag_size(mut self, size: f64) -> Self {
        self.tag_size = Some(size);
        self
    }

    /// Build the options.
    #[must_use]
    pub fn build(self) -> DetectOptions {
        DetectOptions {
            families: self.families,
            intrinsics: self.intrinsics,
            tag_size: self.tag_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_config_builder() {
        let config = DetectorConfig::builder()
            .threshold_tile_size(16)
            .quad_min_area(1000)
            .build();
        assert_eq!(config.threshold_tile_size, 16);
        assert_eq!(config.quad_min_area, 1000);
        // Check defaults
        assert_eq!(config.threshold_min_range, 2);
    }

    #[test]
    fn test_detector_config_defaults() {
        let config = DetectorConfig::default();
        assert_eq!(config.threshold_tile_size, 4);
        assert_eq!(config.quad_min_area, 16);
    }

    #[test]
    fn test_detect_options_families() {
        let opt = DetectOptions::with_families(&[TagFamily::AprilTag36h11]);
        assert_eq!(opt.families.len(), 1);
        assert_eq!(opt.families[0], TagFamily::AprilTag36h11);
    }

    #[test]
    fn test_detect_options_default_empty() {
        let opt = DetectOptions::default();
        assert!(opt.families.is_empty());
    }

    #[test]
    fn test_all_families() {
        let opt = DetectOptions::all_families();
        assert_eq!(opt.families.len(), 4);
    }
}
