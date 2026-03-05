/// The maximum number of candidates in a single batch.
pub(crate) const MAX_CANDIDATES: usize = 1024;

/// A 2D point with subpixel precision (f32).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Point2f {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

/// A 3x3 homography matrix (f32).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(32))]
pub(crate) struct Matrix3x3 {
    /// The matrix elements in row-major or column-major format (internal use).
    pub data: [f32; 9],
    /// Padding to ensure 64-byte size (cache line) and alignment for SIMD.
    pub padding: [f32; 7],
}

/// A 6D pose representing translation and rotation (unit quaternion).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(32))]
pub(crate) struct Pose6D {
    /// Translation (x, y, z) and Rotation as a unit quaternion (x, y, z, w).
    pub data: [f32; 7],
    /// Padding to 32-byte alignment.
    pub padding: f32,
}

/// The lifecycle state of a candidate in the detection pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum CandidateState {
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
    pub(crate) corners: [Point2f; MAX_CANDIDATES * 4],
    /// The 3x3 projection matrices.
    pub(crate) homographies: [Matrix3x3; MAX_CANDIDATES],
    /// The decoded IDs of the tags.
    pub(crate) ids: [u32; MAX_CANDIDATES],
    /// The extracted bitstrings.
    pub(crate) payloads: [u64; MAX_CANDIDATES],
    /// The MSE or Log-Likelihood Ratio confidence scores.
    pub(crate) error_rates: [f32; MAX_CANDIDATES],
    /// Translation vectors and unit quaternions.
    pub(crate) poses: [Pose6D; MAX_CANDIDATES],
    /// A dense byte-array tracking the lifecycle of each candidate.
    pub(crate) status_mask: [CandidateState; MAX_CANDIDATES],
}

impl DetectionBatch {
    /// Creates a new DetectionBatch with all fields initialized to zero (Empty state).
    #[must_use]
    #[allow(clippy::large_stack_arrays)]
    pub(crate) fn new() -> Self {
        *Box::new(Self {
            corners: [Point2f { x: 0.0, y: 0.0 }; MAX_CANDIDATES * 4],
            homographies: [Matrix3x3 {
                data: [0.0; 9],
                padding: [0.0; 7],
            }; MAX_CANDIDATES],
            ids: [0; MAX_CANDIDATES],
            payloads: [0; MAX_CANDIDATES],
            error_rates: [0.0; MAX_CANDIDATES],
            poses: [Pose6D {
                data: [0.0; 7],
                padding: 0.0,
            }; MAX_CANDIDATES],
            status_mask: [CandidateState::Empty; MAX_CANDIDATES],
        })
    }
    /// Returns the maximum capacity of the batch.
    #[must_use]
    pub(crate) fn capacity(&self) -> usize {
        MAX_CANDIDATES
    }

    /// Partitions the batch so that all `Valid` candidates are at the front `[0..V]`.
    /// Returns the number of valid candidates `V`.
    pub(crate) fn partition(&mut self, n: usize) -> usize {
        let mut v = 0;
        let n_clamped = n.min(MAX_CANDIDATES);
        for i in 0..n_clamped {
            if self.status_mask[i] == CandidateState::Valid {
                if i != v {
                    // Swap index i with index v across all parallel arrays.
                    for j in 0..4 {
                        self.corners.swap(i * 4 + j, v * 4 + j);
                    }
                    self.homographies.swap(i, v);
                    self.ids.swap(i, v);
                    self.payloads.swap(i, v);
                    self.error_rates.swap(i, v);
                    self.poses.swap(i, v);
                    self.status_mask.swap(i, v);
                }
                v += 1;
            }
        }
        v
    }

    /// Reassemble the batched SoA data into a list of discrete `Detection` objects.
    #[must_use]
    pub(crate) fn reassemble(&self, v: usize) -> Vec<crate::Detection> {
        let mut detections = Vec::with_capacity(v);
        for i in 0..v {
            let offset = i * 4;
            let corners = [
                [
                    f64::from(self.corners[offset].x),
                    f64::from(self.corners[offset].y),
                ],
                [
                    f64::from(self.corners[offset + 1].x),
                    f64::from(self.corners[offset + 1].y),
                ],
                [
                    f64::from(self.corners[offset + 2].x),
                    f64::from(self.corners[offset + 2].y),
                ],
                [
                    f64::from(self.corners[offset + 3].x),
                    f64::from(self.corners[offset + 3].y),
                ],
            ];

            let center = [
                (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4.0,
                (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4.0,
            ];

            // Reconstruct Pose if available (Z translation > 0)
            let pose = if self.poses[i].data[2] > 0.0 {
                let d = self.poses[i].data;
                // layout: [tx, ty, tz, qx, qy, qz, qw]
                let t = nalgebra::Vector3::new(f64::from(d[0]), f64::from(d[1]), f64::from(d[2]));
                let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    f64::from(d[6]), // w
                    f64::from(d[3]), // x
                    f64::from(d[4]), // y
                    f64::from(d[5]), // z
                ));
                Some(crate::pose::Pose {
                    rotation: q.to_rotation_matrix().into_inner(),
                    translation: t,
                })
            } else {
                None
            };

            detections.push(crate::Detection {
                id: self.ids[i],
                center,
                corners,
                hamming: self.error_rates[i] as u32,
                rotation: 0,
                decision_margin: 0.0,
                bits: self.payloads[i],
                pose,
                pose_covariance: None,
            });
        }
        detections
    }
}

/// Helper function to partition a batch, moving all valid candidates to the front.
pub(crate) fn partition_batch_soa(batch: &mut DetectionBatch, n: usize) -> usize {
    batch.partition(n)
}

impl Default for DetectionBatch {
    fn default() -> Self {
        Self::new()
    }
}
