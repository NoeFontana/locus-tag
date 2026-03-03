/// The maximum number of candidates in a single batch.
pub const MAX_CANDIDATES: usize = 256;

/// A 2D point with subpixel precision (f32).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Point2f {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

/// A 3x3 homography matrix (f32).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Matrix3x3 {
    /// The matrix elements in row-major or column-major format (internal use).
    pub data: [f32; 9],
}

/// A 6D pose representing translation and rotation (unit quaternion).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Pose6D {
    /// Translation (x, y, z) and Rotation as a unit quaternion (x, y, z, w).
    pub data: [f32; 7],
}

/// The lifecycle state of a candidate in the detection pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CandidateState {
    /// No candidate at this index.
    #[default]
    Empty = 0,
    /// Quad vertices have been extracted.
    Active = 1,
    /// Failed to decode the marker payload.
    FailedDecode = 2,
    /// Successfully decoded and mathematically verified.
    Valid = 3,
}

/// A batched state container for fiducial marker detections using a Structure of Arrays (SoA) layout.
/// This structure is designed for high-performance SIMD processing and zero heap allocations.
#[repr(C, align(32))]
pub struct DetectionBatch {
    /// Flattened array of sub-pixel quad vertices (4 corners per candidate).
    pub corners: [Point2f; MAX_CANDIDATES * 4],
    /// The 3x3 projection matrices.
    pub homographies: [Matrix3x3; MAX_CANDIDATES],
    /// The extracted bitstrings.
    pub payloads: [u64; MAX_CANDIDATES],
    /// The MSE or Log-Likelihood Ratio confidence scores.
    pub error_rates: [f32; MAX_CANDIDATES],
    /// Translation vectors and unit quaternions.
    pub poses: [Pose6D; MAX_CANDIDATES],
    /// A dense byte-array tracking the lifecycle of each candidate.
    pub status_mask: [CandidateState; MAX_CANDIDATES],
}

impl DetectionBatch {
    /// Creates a new DetectionBatch with all fields initialized to zero (Empty state).
    pub fn new() -> Self {
        Self {
            corners: [Point2f { x: 0.0, y: 0.0 }; MAX_CANDIDATES * 4],
            homographies: [Matrix3x3 { data: [0.0; 9] }; MAX_CANDIDATES],
            payloads: [0; MAX_CANDIDATES],
            error_rates: [0.0; MAX_CANDIDATES],
            poses: [Pose6D { data: [0.0; 7] }; MAX_CANDIDATES],
            status_mask: [CandidateState::Empty; MAX_CANDIDATES],
        }
    }

    /// Returns the maximum capacity of the batch.
    pub fn capacity(&self) -> usize {
        MAX_CANDIDATES
    }
}

impl Default for DetectionBatch {
    fn default() -> Self {
        Self::new()
    }
}
