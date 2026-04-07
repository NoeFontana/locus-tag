/// The maximum number of candidates in a single batch.
pub(crate) const MAX_CANDIDATES: usize = 1024;

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
#[repr(C, align(32))]
pub struct Matrix3x3 {
    /// The matrix elements in row-major or column-major format (internal use).
    pub data: [f32; 9],
    /// Padding to ensure 64-byte size (cache line) and alignment for SIMD.
    pub padding: [f32; 7],
}

/// A 6D pose representing translation and rotation (unit quaternion).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(32))]
pub struct Pose6D {
    /// Translation (x, y, z) and Rotation as a unit quaternion (x, y, z, w).
    pub data: [f32; 7],
    /// Padding to 32-byte alignment.
    pub padding: f32,
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

/// The status of a candidate in the fast-path decoding funnel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FunnelStatus {
    /// Candidate has not yet been processed by the funnel.
    #[default]
    None = 0,
    /// Candidate passed the O(1) contrast gate.
    PassedContrast = 1,
    /// Candidate rejected by the O(1) contrast gate.
    RejectedContrast = 2,
    /// Candidate rejected during homography DDA or SIMD sampling.
    RejectedSampling = 3,
}

/// A batched state container for fiducial marker detections using a Structure of Arrays (SoA) layout.
/// This structure is designed for high-performance SIMD processing and zero heap allocations.
#[repr(C, align(32))]
pub struct DetectionBatch {
    /// Flattened array of sub-pixel quad vertices (4 corners per candidate).
    pub corners: [[Point2f; 4]; MAX_CANDIDATES],
    /// The 3x3 projection matrices.
    pub homographies: [Matrix3x3; MAX_CANDIDATES],
    /// The decoded IDs of the tags.
    pub ids: [u32; MAX_CANDIDATES],
    /// The extracted bitstrings.
    pub payloads: [u64; MAX_CANDIDATES],
    /// The MSE or Log-Likelihood Ratio confidence scores.
    pub error_rates: [f32; MAX_CANDIDATES],
    /// Translation vectors and unit quaternions.
    pub poses: [Pose6D; MAX_CANDIDATES],
    /// A dense byte-array tracking the lifecycle of each candidate.
    pub status_mask: [CandidateState; MAX_CANDIDATES],
    /// Detailed status from the fast-path funnel.
    pub funnel_status: [FunnelStatus; MAX_CANDIDATES],
    /// Four 2x2 corner covariance matrices per quad (16 floats).
    /// Layout: [c0_xx, c0_xy, c0_yx, c0_yy, c1_xx, ...]
    pub corner_covariances: [[f32; 16]; MAX_CANDIDATES],
}

impl DetectionBatch {
    /// Creates a new DetectionBatch with all fields initialized to zero (Empty state).
    #[must_use]
    #[allow(clippy::large_stack_arrays)]
    pub fn new() -> Self {
        *Box::new(Self {
            corners: [[Point2f { x: 0.0, y: 0.0 }; 4]; MAX_CANDIDATES],
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
            funnel_status: [FunnelStatus::None; MAX_CANDIDATES],
            corner_covariances: [[0.0; 16]; MAX_CANDIDATES],
        })
    }
    /// Returns the maximum capacity of the batch.
    #[must_use]
    pub fn capacity(&self) -> usize {
        MAX_CANDIDATES
    }

    /// Partitions the batch so that all `Valid` candidates are at the front `[0..V]`.
    /// Returns the number of valid candidates `V`.
    pub fn partition(&mut self, n: usize) -> usize {
        let mut v = 0;
        let n_clamped = n.min(MAX_CANDIDATES);

        // Two-pointer partition:
        // Move Valid candidates to [0..v]
        // Move everything else to [v..n_clamped]
        for i in 0..n_clamped {
            if self.status_mask[i] == CandidateState::Valid {
                if i != v {
                    // Swap index i with index v across all parallel arrays.
                    self.corners.swap(i, v);
                    self.homographies.swap(i, v);
                    self.ids.swap(i, v);
                    self.payloads.swap(i, v);
                    self.error_rates.swap(i, v);
                    self.poses.swap(i, v);
                    self.status_mask.swap(i, v);
                    self.funnel_status.swap(i, v);
                    self.corner_covariances.swap(i, v);
                }
                v += 1;
            }
        }
        v
    }

    /// Reassemble the batched SoA data into a list of discrete `Detection` objects.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn reassemble(&self, v: usize) -> Vec<crate::Detection> {
        let mut detections = Vec::with_capacity(v);
        for i in 0..v {
            let corners = [
                [
                    f64::from(self.corners[i][0].x),
                    f64::from(self.corners[i][0].y),
                ],
                [
                    f64::from(self.corners[i][1].x),
                    f64::from(self.corners[i][1].y),
                ],
                [
                    f64::from(self.corners[i][2].x),
                    f64::from(self.corners[i][2].y),
                ],
                [
                    f64::from(self.corners[i][3].x),
                    f64::from(self.corners[i][3].y),
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
pub fn partition_batch_soa(batch: &mut DetectionBatch, n: usize) -> usize {
    batch.partition(n)
}

/// Metadata for zero-copy telemetry extraction.
#[derive(Debug, Clone, Copy)]
pub struct TelemetryPayload {
    /// Pointer to the binarized image buffer.
    pub binarized_ptr: *const u8,
    /// Pointer to the threshold map buffer.
    pub threshold_map_ptr: *const u8,
    /// Pointer to subpixel jitter data [4 corners * 2 (dx, dy)] per valid candidate.
    /// Allocated in the detector's arena.
    pub subpixel_jitter_ptr: *const f32,
    /// Number of valid candidates jitter data is available for.
    pub num_jitter: usize,
    /// Pointer to reprojection RMSE values per valid candidate.
    /// Allocated in the detector's arena.
    pub reprojection_errors_ptr: *const f32,
    /// Number of valid candidates reprojection data is available for.
    pub num_reprojection: usize,
    /// Number of quads that fell back to coarse corners during GWLF.
    pub gwlf_fallback_count: usize,
    /// Average Euclidean distance (delta) of GWLF refinement (pixels).
    pub gwlf_avg_delta: f32,
    /// Width of the buffers.
    pub width: usize,
    /// Height of the buffers.
    pub height: usize,
    /// Stride of the buffers.
    pub stride: usize,
}

// SAFETY: TelemetryPayload contains raw pointers to the detector's arena.
// These pointers are stable for the duration of the frame processing and
// are only accessed from the Python thread while the arena is still alive.
#[allow(unsafe_code)]
unsafe impl Send for TelemetryPayload {}
#[allow(unsafe_code)]
unsafe impl Sync for TelemetryPayload {}

/// A lightweight, borrowed view of the detection results.
///
/// This struct holds slices to the active elements in a [`DetectionBatch`].
/// It avoids heap allocations and provides efficient access to detection data.
#[derive(Debug, Clone, Copy)]
pub struct DetectionBatchView<'a> {
    /// Decoded IDs of the markers.
    pub ids: &'a [u32],
    /// Refined corners in image coordinates.
    pub corners: &'a [[Point2f; 4]],
    /// Computed homography matrices.
    pub homographies: &'a [Matrix3x3],
    /// Decoded bitstrings (paylods).
    pub payloads: &'a [u64],
    /// Confidence scores or error rates.
    pub error_rates: &'a [f32],
    /// 3D poses (rotation + translation).
    pub poses: &'a [Pose6D],
    /// Corner covariances (Fisher information priors).
    pub corner_covariances: &'a [[f32; 16]],
    /// Optional telemetry data for intermediate images.
    pub telemetry: Option<TelemetryPayload>,
    /// Corners of quads that were extracted but rejected during decoding or verification.
    pub rejected_corners: &'a [[Point2f; 4]],
    /// Error rates (e.g. Hamming distance) for rejected quads.
    pub rejected_error_rates: &'a [f32],
    /// Detailed status from the fast-path funnel for rejected quads.
    pub rejected_funnel_status: &'a [FunnelStatus],
}

impl DetectionBatchView<'_> {
    /// Returns the number of detections in the view.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns true if the view contains no detections.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Materialise an owned copy of the detection results from this view's slices.
    ///
    /// This is the correct escape hatch for pool-based concurrent detection: call
    /// `reassemble_owned()`, then drop the view, then return the `FrameContext` to
    /// the pool — the view borrows from the context, so extraction must happen first.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn reassemble_owned(&self) -> Vec<crate::Detection> {
        let v = self.ids.len();
        let mut detections = Vec::with_capacity(v);
        for i in 0..v {
            let corners = [
                [
                    f64::from(self.corners[i][0].x),
                    f64::from(self.corners[i][0].y),
                ],
                [
                    f64::from(self.corners[i][1].x),
                    f64::from(self.corners[i][1].y),
                ],
                [
                    f64::from(self.corners[i][2].x),
                    f64::from(self.corners[i][2].y),
                ],
                [
                    f64::from(self.corners[i][3].x),
                    f64::from(self.corners[i][3].y),
                ],
            ];
            let center = [
                (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4.0,
                (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4.0,
            ];
            let pose = if self.poses[i].data[2] > 0.0 {
                let d = self.poses[i].data;
                let t = nalgebra::Vector3::new(f64::from(d[0]), f64::from(d[1]), f64::from(d[2]));
                let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    f64::from(d[6]),
                    f64::from(d[3]),
                    f64::from(d[4]),
                    f64::from(d[5]),
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

impl DetectionBatch {
    /// Returns a borrowed view of the first `v` candidates in the batch.
    #[must_use]
    pub fn view(&self, v: usize) -> DetectionBatchView<'_> {
        let n = v.min(MAX_CANDIDATES);
        DetectionBatchView {
            ids: &self.ids[..n],
            corners: &self.corners[..n],
            homographies: &self.homographies[..n],
            payloads: &self.payloads[..n],
            error_rates: &self.error_rates[..n],
            poses: &self.poses[..n],
            corner_covariances: &self.corner_covariances[..n],
            telemetry: None,
            rejected_corners: &[],
            rejected_error_rates: &[],
            rejected_funnel_status: &[],
        }
    }

    /// Returns a borrowed view with telemetry data.
    #[must_use]
    pub fn view_with_telemetry(
        &self,
        v: usize,
        n: usize,
        telemetry: Option<TelemetryPayload>,
    ) -> DetectionBatchView<'_> {
        let v_clamped = v.min(MAX_CANDIDATES);
        let n_clamped = n.min(MAX_CANDIDATES);
        DetectionBatchView {
            ids: &self.ids[..v_clamped],
            corners: &self.corners[..v_clamped],
            homographies: &self.homographies[..v_clamped],
            payloads: &self.payloads[..v_clamped],
            error_rates: &self.error_rates[..v_clamped],
            poses: &self.poses[..v_clamped],
            corner_covariances: &self.corner_covariances[..v_clamped],
            telemetry,
            rejected_corners: &self.corners[v_clamped..n_clamped],
            rejected_error_rates: &self.error_rates[v_clamped..n_clamped],
            rejected_funnel_status: &self.funnel_status[v_clamped..n_clamped],
        }
    }
}

impl Default for DetectionBatch {
    fn default() -> Self {
        Self::new()
    }
}
