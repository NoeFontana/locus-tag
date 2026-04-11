//! Error types for the detection pipeline.

use std::fmt;

/// Errors that can occur during detector configuration validation.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Threshold tile size must be >= 2.
    TileSizeTooSmall(usize),
    /// Decimation factor must be >= 1.
    InvalidDecimation(usize),
    /// Upscale factor must be >= 1.
    InvalidUpscaleFactor(usize),
    /// Minimum fill ratio must be in [0.0, 1.0].
    InvalidFillRatio {
        /// The minimum fill ratio that was set.
        min: f32,
        /// The maximum fill ratio that was set.
        max: f32,
    },
    /// Minimum edge length must be positive.
    InvalidEdgeLength(f64),
    /// Structure tensor radius must stay within the supported kernel bound.
    InvalidStructureTensorRadius(u8),
    /// EdLines is geometrically incompatible with Erf corner refinement.
    ///
    /// Erf performs an independent 1-D search per corner which destroys the
    /// planarity constraint established by the joint Gauss-Newton solver in
    /// the EdLines pipeline, degrading corner RMSE from ~0.17 px to ~0.59 px.
    EdLinesIncompatibleWithErf,
    /// EdLines is statistically incompatible with Soft (LLR) decoding.
    ///
    /// EdLines produces a large number of background line-segment candidates.
    /// Soft decoding forces deep probabilistic evaluation of all of them,
    /// causing a 10–22% precision collapse compared to Hard decoding.
    EdLinesIncompatibleWithSoftDecode,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TileSizeTooSmall(size) => {
                write!(f, "threshold_tile_size must be >= 2, got {size}")
            },
            Self::InvalidDecimation(d) => {
                write!(f, "decimation factor must be >= 1, got {d}")
            },
            Self::InvalidUpscaleFactor(u) => {
                write!(f, "upscale_factor must be >= 1, got {u}")
            },
            Self::InvalidFillRatio { min, max } => {
                write!(
                    f,
                    "fill ratio range invalid: min={min}, max={max} (must be 0.0..=1.0, min < max)"
                )
            },
            Self::InvalidEdgeLength(l) => {
                write!(f, "quad_min_edge_length must be positive, got {l}")
            },
            Self::InvalidStructureTensorRadius(r) => {
                write!(f, "structure_tensor_radius must be <= 8, got {r}")
            },
            Self::EdLinesIncompatibleWithErf => write!(
                f,
                "EdLines + Erf refinement are incompatible: use CornerRefinementMode::None or \
                 CornerRefinementMode::Gwlf with EdLines"
            ),
            Self::EdLinesIncompatibleWithSoftDecode => write!(
                f,
                "EdLines + Soft decoding are incompatible: use DecodeMode::Hard with EdLines"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Errors that can occur during tag detection.
#[derive(Debug, Clone)]
pub enum DetectorError {
    /// Image preprocessing (decimation or upscaling) failed.
    Preprocessing(String),
    /// Image view construction failed (invalid dimensions/stride).
    InvalidImage(String),
    /// Configuration validation failed.
    Config(ConfigError),
}

impl fmt::Display for DetectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preprocessing(msg) => write!(f, "preprocessing failed: {msg}"),
            Self::InvalidImage(msg) => write!(f, "invalid image: {msg}"),
            Self::Config(e) => write!(f, "config error: {e}"),
        }
    }
}

impl std::error::Error for DetectorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ConfigError> for DetectorError {
    fn from(e: ConfigError) -> Self {
        Self::Config(e)
    }
}
