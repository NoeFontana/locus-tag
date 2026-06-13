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

// Soundness guard for `DetectionBatch::new_boxed`: the unsafe
// `Box::<Self>::new_zeroed().assume_init()` path treats `status_mask` and
// `funnel_status` slots as the default variants. If a future reorder or a
// dropped `= 0` literal moves either variant off discriminant 0, the build
// must fail here instead of silently producing invalid enum values at
// runtime. Pinning the SAFETY contract at compile time.
const _: () = {
    assert!(CandidateState::Empty as u8 == 0);
    assert!(FunnelStatus::None as u8 == 0);
};

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
    /// Aggregate Mahalanobis d² (χ²(2) statistic) from the pose-consistency
    /// gate. NaN means the gate did not run (e.g., `pose_consistency_fpr == 0`
    /// or pose estimation failed before the check). Only populated when the
    /// `bench-internals` feature is enabled.
    #[cfg(feature = "bench-internals")]
    pub pose_consistency_d2: [f32; MAX_CANDIDATES],
    /// Worst single-corner Mahalanobis d² (χ²(1) statistic) from the
    /// pose-consistency gate. NaN sentinel as above.
    #[cfg(feature = "bench-internals")]
    pub pose_consistency_d2_max_corner: [f32; MAX_CANDIDATES],
    /// IPPE branch ratio `alternate_d2 / primary_d2`. `1.0` ≈ ambiguous,
    /// `≫ 1` = primary clearly wins, `≪ 1` = primary was wrong (alternate
    /// fits much better). NaN sentinel as above.
    #[cfg(feature = "bench-internals")]
    pub ippe_branch_d2_ratio: [f32; MAX_CANDIDATES],
    /// Outlier-aware corner-drop telemetry. `u8::MAX` ⇒ no corner was
    /// dropped for this candidate (either the trigger didn't fire or the
    /// 3-corner self-rejection kept the 4-corner pose). Values `0..=3`
    /// identify which corner was masked when the 3-corner pose was kept;
    /// in that case the stored pose covariance reflects 6 observations
    /// instead of 8.
    ///
    /// Phase-D telemetry — inert in profiles where
    /// `pose.outlier_drop_d2_threshold = 0.0`.
    #[cfg(feature = "bench-internals")]
    pub outlier_corner_idx: [u8; MAX_CANDIDATES],
    /// Per-candidate `AdaptivePpb` route label. Debug-only telemetry.
    /// [`ROUTED_TO_LOW`] = low-PPB route, [`ROUTED_TO_HIGH`] = high-PPB
    /// route, [`ROUTED_TO_STATIC`] for `QuadExtractionPolicy::Static` /
    /// not-routed. Populated only when `debug_telemetry` is set.
    pub routed_to: [u8; MAX_CANDIDATES],
    /// Per-candidate pixels-per-bit estimate consumed by the adaptive router.
    /// Debug-only telemetry. Zero under `Static` (no routing performed).
    pub ppb_estimate: [f32; MAX_CANDIDATES],
}

/// Sentinel written into [`DetectionBatch::routed_to`] when the candidate
/// was not subjected to adaptive routing (`QuadExtractionPolicy::Static`).
pub const ROUTED_TO_STATIC: u8 = u8::MAX;

/// `AdaptivePpb` low-PPB route label written into [`DetectionBatch::routed_to`].
pub const ROUTED_TO_LOW: u8 = 0;

/// `AdaptivePpb` high-PPB route label written into [`DetectionBatch::routed_to`].
pub const ROUTED_TO_HIGH: u8 = 1;

impl DetectionBatch {
    /// Heap-allocate a default `DetectionBatch` without ever materializing it on the stack.
    ///
    /// The struct is ~215 KB (≈228 KB under `bench-internals`). Returning it by
    /// value — which a naive `new() -> Self` did — forces every caller's stack
    /// frame to hold a copy because debug-mode rustc does not perform RVO/NRVO.
    /// Chained construction (`Detector::with_config` → `DetectorBuilder::build`
    /// → `FrameContext::new` → `DetectionBatch::new`) accumulated several
    /// copies and overflowed the main-thread stack on Windows (1 MB default)
    /// and Linux worker threads (2 MB default).
    ///
    /// This version allocates a zeroed `Box<MaybeUninit<Self>>` directly on the
    /// heap and patches the few non-zero defaults in place via `slice::fill`.
    /// No `DetectionBatch`-sized value ever lives on the stack.
    #[must_use]
    #[allow(unsafe_code)]
    pub fn new_boxed() -> Box<Self> {
        // SAFETY: All-zero is a valid bit pattern for every field of
        // `DetectionBatch`:
        //  - `Point2f` / `Matrix3x3` / `Pose6D`: every component is `f32`
        //    (`0.0` is a valid float, matches the previous explicit default).
        //  - `[u32 / u64 / u8 / f32; N]` numeric arrays: zero matches the
        //    previous explicit `0` / `0.0` defaults.
        //  - `status_mask` / `funnel_status`: `#[repr(u8)]` enums whose
        //    default variants (`CandidateState::Empty` and
        //    `FunnelStatus::None`) both have discriminant `0`.
        //  - `corner_covariances`: nested `[f32; 16]` arrays — zero matches.
        // Fields whose default is non-zero (`routed_to`, and the four
        // `bench-internals` NaN / `u8::MAX` fields) are patched below before
        // the box is returned.
        let mut boxed: Box<Self> = unsafe { Box::<Self>::new_zeroed().assume_init() };
        // Patch non-zero defaults via in-place slice writes — no stack temp.
        boxed.routed_to.fill(ROUTED_TO_STATIC);
        #[cfg(feature = "bench-internals")]
        {
            boxed.pose_consistency_d2.fill(f32::NAN);
            boxed.pose_consistency_d2_max_corner.fill(f32::NAN);
            boxed.ippe_branch_d2_ratio.fill(f32::NAN);
            boxed.outlier_corner_idx.fill(u8::MAX);
        }
        boxed
    }
    /// Returns the maximum capacity of the batch.
    #[must_use]
    #[allow(clippy::unused_self)]
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
                    #[cfg(feature = "bench-internals")]
                    {
                        self.pose_consistency_d2.swap(i, v);
                        self.pose_consistency_d2_max_corner.swap(i, v);
                        self.ippe_branch_d2_ratio.swap(i, v);
                        self.outlier_corner_idx.swap(i, v);
                    }
                    self.routed_to.swap(i, v);
                    self.ppb_estimate.swap(i, v);
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
#[cfg(feature = "bench-internals")]
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
    /// Pointer to per-candidate adaptive-router route labels. Populated only
    /// when `debug_telemetry` is set; null otherwise.
    pub routed_to_ptr: *const u8,
    /// Pointer to per-candidate pixels-per-bit estimates. Populated only
    /// when `debug_telemetry` is set; null otherwise.
    pub ppb_estimate_ptr: *const f32,
    /// Number of candidates that the routing telemetry arrays cover (N from
    /// Phase A, before partition). `0` when routing telemetry is disabled.
    pub num_routed: usize,
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

// SAFETY: TelemetryPayload is read-only from the Python thread and does not
// contain any interior mutability.
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
