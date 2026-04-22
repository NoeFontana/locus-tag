//! Error types for the detection pipeline.

use thiserror::Error;

/// Errors that can occur during detector configuration validation.
#[derive(Debug, Clone, Error)]
pub enum ConfigError {
    /// Threshold tile size must be >= 2.
    #[error("threshold_tile_size must be >= 2, got {0}")]
    TileSizeTooSmall(usize),
    /// Decimation factor must be >= 1.
    #[error("decimation factor must be >= 1, got {0}")]
    InvalidDecimation(usize),
    /// Upscale factor must be >= 1.
    #[error("upscale_factor must be >= 1, got {0}")]
    InvalidUpscaleFactor(usize),
    /// Minimum fill ratio must be in [0.0, 1.0].
    #[error("fill ratio range invalid: min={min}, max={max} (must be 0.0..=1.0, min < max)")]
    InvalidFillRatio {
        /// The minimum fill ratio that was set.
        min: f32,
        /// The maximum fill ratio that was set.
        max: f32,
    },
    /// Minimum edge length must be positive.
    #[error("quad_min_edge_length must be positive, got {0}")]
    InvalidEdgeLength(f64),
    /// Structure tensor radius must stay within the supported kernel bound.
    #[error("structure_tensor_radius must be <= 8, got {0}")]
    InvalidStructureTensorRadius(u8),
    /// Pose consistency FPR must be in `[0.0, 1.0)`.
    ///
    /// `0.0` disables the gate; positive values are interpreted as a target
    /// false-positive rate from which the chi-squared critical value is
    /// derived. Values >= 1.0 would accept everything (degenerate gate)
    /// and are explicitly rejected to surface configuration mistakes.
    #[error("pose_consistency_fpr must be in [0.0, 1.0), got {0}")]
    InvalidPoseConsistencyFpr(f64),
    /// EdLines is geometrically incompatible with Erf corner refinement.
    ///
    /// Erf performs an independent 1-D search per corner which destroys the
    /// planarity constraint established by the joint Gauss-Newton solver in
    /// the EdLines pipeline, degrading corner RMSE from ~0.17 px to ~0.59 px.
    #[error(
        "EdLines + Erf refinement are incompatible: use CornerRefinementMode::None or \
         CornerRefinementMode::Gwlf with EdLines"
    )]
    EdLinesIncompatibleWithErf,
    /// EdLines is statistically incompatible with Soft (LLR) decoding.
    ///
    /// EdLines produces a large number of background line-segment candidates.
    /// Soft decoding forces deep probabilistic evaluation of all of them,
    /// causing a 10–22% precision collapse compared to Hard decoding.
    #[error("EdLines + Soft decoding are incompatible: use DecodeMode::Hard with EdLines")]
    EdLinesIncompatibleWithSoftDecode,
    /// EdLines is geometrically incompatible with distorted cameras.
    ///
    /// EdLines' Huber IRLS line fit, micro-ray parabola, and GN refinement all
    /// assume Euclidean pixel geometry. A Brown-Conrady or Kannala-Brandt
    /// intrinsic bends marker edges into curves, invalidating that assumption.
    /// Use `QuadExtractionMode::ContourRdp` when passing distortion coefficients.
    #[error(
        "EdLines is geometrically incompatible with distorted cameras (BrownConrady / \
         KannalaBrandt): use QuadExtractionMode::ContourRdp"
    )]
    EdLinesUnsupportedWithDistortion,
    /// `AdaptivePpb` policy had `low_extraction == high_extraction`.
    ///
    /// A degenerate adaptive policy is a configuration mistake — both
    /// branches would produce identical behavior. Use `Static` instead.
    #[error(
        "AdaptivePpb policy has identical low/high extraction modes; \
         use QuadExtractionPolicy::Static for single-mode operation"
    )]
    AdaptivePolicyDegenerate,
    /// `AdaptivePpb` policy threshold fell outside the valid open interval.
    ///
    /// The PPB threshold must lie strictly inside `(1.0, 5.0)`. Values at or
    /// outside the endpoints are degenerate: `<=1.0` always routes to the
    /// high branch, `>=5.0` effectively never does.
    #[error("AdaptivePpb threshold must be in (1.0, 5.0), got {0}")]
    AdaptivePolicyThresholdOutOfRange(f32),
    /// Profile JSON failed to parse (malformed syntax or unknown fields).
    ///
    /// `serde_json::Error` is not `Clone`, so its `Display` form is captured
    /// at construction time. The wrapping `ConfigError` stays `Clone`.
    #[error("profile JSON parse error: {0}")]
    ProfileParse(String),
}

/// Errors that can occur during tag detection.
#[derive(Debug, Clone, Error)]
pub enum DetectorError {
    /// Image preprocessing (decimation or upscaling) failed.
    #[error("preprocessing failed: {0}")]
    Preprocessing(String),
    /// Image view construction failed (invalid dimensions/stride).
    #[error("invalid image: {0}")]
    InvalidImage(String),
    /// Configuration validation failed.
    #[error("config error: {0}")]
    Config(#[from] ConfigError),
}
