//! High-performance AprilTag and ArUco detection engine.
//!
//! Locus is a research-oriented, memory-safe fiducial marker detector targeting low
//! latency. It provides a performance-focused pipeline for robotics and computer vision,
//! with strict zero-heap allocation in the detection hot-path.

/// Batched state container for Structure of Arrays (SoA) layout.
pub(crate) mod batch;
/// Board-level configuration and layout utilities.
pub mod board;
/// Zero-cost camera distortion model trait and implementations.
pub mod camera;
/// ChAruco board saddle-point extraction and pose estimation.
pub mod charuco;
/// Configuration types for the detector pipeline.
pub mod config;
/// Tag decoding traits and implementations.
pub(crate) mod decoder;
/// The primary public API for the detector.
pub mod detector;
/// Tag family dictionaries (AprilTag, ArUco).
pub(crate) mod dictionaries;
/// Unified ERF-based sub-pixel edge refinement shared by quad and decoder stages.
pub(crate) mod edge_refinement;
/// EDLines localized quad extraction.
pub(crate) mod edlines;
/// Error types for the detection pipeline.
pub mod error;
/// Edge-preserving filtering for small tag detection.
pub(crate) mod filter;
/// Fast-path decoding funnel.
pub(crate) mod funnel;
/// Gradient computation for edge refinement.
pub(crate) mod gradient;
/// Gradient-Weighted Line Fitting (GWLF).
pub(crate) mod gwlf;
/// Image buffer abstractions.
pub mod image;
/// 3D Pose Estimation (PnP).
pub mod pose;
/// Weighted pose estimation logic.
pub(crate) mod pose_weighted;
/// Quad extraction and geometric primitives.
pub(crate) mod quad;
/// Connected components labeling using Union-Find.
pub(crate) mod segmentation;
/// SIMD optimized mathematical kernels.
pub(crate) mod simd;
/// SIMD-Accelerated Connected Components Labeling (CCL).
pub(crate) mod simd_ccl_fusion;
/// Decoding strategies (Hard vs Soft).
pub(crate) mod strategy;
/// Utilities for testing and synthetic data generation.
#[cfg(any(test, feature = "bench-internals"))]
pub(crate) mod test_utils;
/// Adaptive thresholding implementation.
pub(crate) mod threshold;
/// Thread-local workspace arena for per-candidate allocations.
pub(crate) mod workspace;

// Re-exports for the public API
pub use crate::batch::TelemetryPayload;
pub use crate::board::{AprilGridTopology, BoardConfigError, CharucoTopology};
pub use crate::camera::{BrownConradyModel, CameraModel, KannalaBrandtModel, PinholeModel};
pub use crate::config::{
    CornerRefinementMode, DecodeMode, DetectOptions, DetectorConfig, PoseEstimationMode,
    QuadExtractionMode, TagFamily,
};
pub use crate::detector::{Detector, DetectorBuilder, FrameContext, LocusEngine};
pub use crate::error::{ConfigError, DetectorError};
pub use crate::image::ImageView;
pub use crate::pose::{CameraIntrinsics, DistortionCoeffs};

#[cfg(feature = "bench-internals")]
pub mod bench_api {
    //! Internal API exposed exclusively for benchmarking and integration testing.
    pub use crate::batch::*;
    pub use crate::camera::*;
    pub use crate::decoder::*;
    pub use crate::dictionaries::*;
    pub use crate::filter::*;
    pub use crate::funnel::*;
    pub use crate::gwlf::*;
    pub use crate::pose::*;
    pub use crate::pose_weighted::bench_compute_corner_covariance;
    pub use crate::quad::*;
    pub use crate::segmentation::*;
    pub use crate::simd::sampler::*;
    pub use crate::simd_ccl_fusion::*;
    pub use crate::strategy::*;
    pub use crate::test_utils::*;
    pub use crate::threshold::*;
}

/// A single tag detection result.
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
    /// The rotation of the tag relative to its canonical orientation (0-3).
    pub rotation: u8,
    /// The decision margin of the decoding (higher is more confident).
    pub decision_margin: f64,
    /// The extracted bits from the tag.
    pub bits: u64,
    /// The 3D pose of the tag relative to the camera (if requested).
    pub pose: Option<crate::pose::Pose>,
    /// The covariance of the estimated 3D pose (6x6 matrix), if computed.
    pub pose_covariance: Option<[[f64; 6]; 6]>,
}

impl Detection {
    /// Compute the axis-aligned bounding box (AABB) of the detection.
    ///
    /// Returns (min_x, min_y, max_x, max_y) in integer pixel coordinates.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn aabb(&self) -> (usize, usize, usize, usize) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let max_x = self
            .corners
            .iter()
            .fold(f64::NEG_INFINITY, |acc, p| acc.max(p[0]));
        let max_y = self
            .corners
            .iter()
            .fold(f64::NEG_INFINITY, |acc, p| acc.max(p[1]));
        for p in &self.corners {
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
        }

        (
            min_x.floor().max(0.0) as usize,
            min_y.floor().max(0.0) as usize,
            max_x.ceil().max(0.0) as usize,
            max_y.ceil().max(0.0) as usize,
        )
    }
}

/// A 2D point with f64 precision.
#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

/// A 3D pose (rotation + translation).
pub use crate::pose::Pose;

/// Returns version and build information for the core library.
#[must_use]
pub fn core_info() -> String {
    "Locus Core v0.1.0 Engine (Encapsulated)".to_string()
}
