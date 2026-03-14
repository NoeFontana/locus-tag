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
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TileSizeTooSmall(size) => {
                write!(f, "threshold_tile_size must be >= 2, got {size}")
            }
            Self::InvalidDecimation(d) => {
                write!(f, "decimation factor must be >= 1, got {d}")
            }
            Self::InvalidUpscaleFactor(u) => {
                write!(f, "upscale_factor must be >= 1, got {u}")
            }
            Self::InvalidFillRatio { min, max } => {
                write!(
                    f,
                    "fill ratio range invalid: min={min}, max={max} (must be 0.0..=1.0, min < max)"
                )
            }
            Self::InvalidEdgeLength(l) => {
                write!(f, "quad_min_edge_length must be positive, got {l}")
            }
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
