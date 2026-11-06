//! Error types for the detection pipeline.

use thiserror::Error;

/// Configuration validation errors.
#[derive(Debug, Clone, Error)]
pub enum ConfigError {
    /// Tile size below minimum of 2.
    #[error("threshold_tile_size must be >= 2, got {0}")]
    TileSizeTooSmall(usize),
    /// Radius outside `[1, 127]`.
    #[error("threshold_local_mean_radius must be in [1, 127], got {0}")]
    InvalidLocalMeanRadius(usize),
    /// Noise multiplier non-positive or non-finite.
    #[error("threshold_noise_k must be finite and > 0, got {0}")]
    InvalidNoiseK(f32),
    /// Border error budget outside `[0, 1]`.
    #[error("decoder_max_border_error_rate must be in [0, 1], got {0}")]
    InvalidBorderErrorRate(f32),
    /// Border bit count outside supported bounds.
    #[error("decoder_marker_border_bits must be in 1..={max}, got {0}", max = crate::config::MAX_MARKER_BORDER_BITS)]
    InvalidMarkerBorderBits(u8),
    /// Decimation factor below 1.
    #[error("decimation factor must be >= 1, got {0}")]
    InvalidDecimation(usize),
    /// Upscale factor below 1.
    #[error("upscale_factor must be >= 1, got {0}")]
    InvalidUpscaleFactor(usize),
    /// Fill ratio range invalid or inverted.
    #[error("fill ratio range invalid: min={min}, max={max} (must be 0.0..=1.0, min < max)")]
    InvalidFillRatio {
        /// Lower fill ratio bound.
        min: f32,
        /// Upper fill ratio bound.
        max: f32,
    },
    /// Minimum edge length non-positive.
    #[error("quad_min_edge_length must be positive, got {0}")]
    InvalidEdgeLength(f64),
    /// Structure tensor radius outside `[1, 8]`.
    #[error("structure_tensor_radius must be in [1, 8], got {0}")]
    InvalidStructureTensorRadius(u8),
    /// Pose consistency FPR outside `[0.0, 1.0)`.
    #[error("pose_consistency_fpr must be in [0.0, 1.0), got {0}")]
    InvalidPoseConsistencyFpr(f64),
    /// Outlier-drop d² threshold negative or non-finite.
    #[error("outlier_drop_d2_threshold must be finite and >= 0.0, got {0}")]
    InvalidOutlierDropD2Threshold(f64),
    /// EdLines joint Gauss-Newton solver is incompatible with 1D ERF corner refinement.
    #[error(
        "EdLines + Erf refinement are incompatible: use CornerRefinementMode::None with EdLines"
    )]
    EdLinesIncompatibleWithErf,
    /// EdLines line fitting assumes Euclidean pixel geometry and does not support lens distortion.
    #[error(
        "EdLines is geometrically incompatible with distorted cameras (BrownConrady / \
         KannalaBrandt): use QuadExtractionMode::ContourRdp"
    )]
    EdLinesUnsupportedWithDistortion,
    /// Straight-space extraction on an upscaled grid cannot apply unscaled lens intrinsics.
    #[error(
        "upscale_factor > 1 is not supported with a declared lens model (BrownConrady / \
         KannalaBrandt): straight-space extraction would apply the lens on the upscaled grid \
         using un-upscaled intrinsics"
    )]
    UpscaleUnsupportedWithDistortion,
    /// Degenerate adaptive policy where low and high modes are identical.
    #[error(
        "AdaptivePpb policy has identical low/high extraction modes; \
         use QuadExtractionPolicy::Static for single-mode operation"
    )]
    AdaptivePolicyDegenerate,
    /// Non-None static refinement passed alongside route-carrying adaptive policy.
    #[error(
        "AdaptivePpb policy requires refinement_mode = None (the routes carry their own \
         refinement modes), got {0:?}"
    )]
    AdaptivePolicyStaticRefinement(crate::config::CornerRefinementMode),
    /// Adaptive PPB threshold outside `(1.0, 5.0)`.
    #[error("AdaptivePpb threshold must be in (1.0, 5.0), got {0}")]
    AdaptivePolicyThresholdOutOfRange(f32),
    /// Profile JSON syntax error.
    #[error("profile JSON parse error: {0}")]
    ProfileParse(String),
}

/// Pipeline runtime errors.
#[derive(Debug, Clone, Error)]
pub enum DetectorError {
    /// Preprocessing failure.
    #[error("preprocessing failed: {0}")]
    Preprocessing(String),
    /// Invalid image dimensions or layout.
    #[error("invalid image: {0}")]
    InvalidImage(String),
    /// Configuration validation error.
    #[error("config error: {0}")]
    Config(#[from] ConfigError),
}
