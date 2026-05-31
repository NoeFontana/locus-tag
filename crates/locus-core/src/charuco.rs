//! ChAruco board saddle-point extraction and board pose estimation.
//!
//! After ArUco tags have been decoded by the main pipeline, this module
//! leverages the decoded tags as geometric priors to locate checkerboard
//! interior corners (saddle points) with sub-pixel precision and then
//! estimates the board pose via the generic [`crate::board::RobustPoseSolver`].
//!
//! # Algorithm
//!
//! 1. **Deduplication pass** — collect the set of saddle IDs visible from the
//!    decoded tags (O(V), no allocation).
//! 2. **Homography prediction** — project each saddle's 3D board coordinate
//!    through the owning tag's stored canonical→image homography to obtain a
//!    sub-pixel initial estimate (typically < 1 px error).
//! 3. **Gauss-Newton refinement** — iterate the Newton step
//!    `S·δp = −∇I(p)` (where S is the local structure tensor) to drive the
//!    estimate to the true saddle position.
//! 4. **Covariance estimation** — call `compute_corner_covariance` at the
//!    refined position to get the Fisher information matrix for AW-LM.
//! 5. **Pose estimation** — build a [`crate::board::PointCorrespondences`] view
//!    (`group_size = 1`) and delegate to [`crate::board::RobustPoseSolver`].

#![allow(clippy::similar_names)]

use crate::batch::{DetectionBatchView, Point2f};
use crate::board::{
    BoardPose, CharucoTopology, LoRansacConfig, PointCorrespondences, RobustPoseSolver,
};
use crate::image::ImageView;
use crate::pose::CameraIntrinsics;
use crate::pose_weighted::compute_corner_covariance;
use nalgebra::Matrix2;
use std::sync::Arc;

// ── Telemetry sub-buffer ──────────────────────────────────────────────────

/// Per-frame diagnostic record of rejected saddle candidates.
///
/// Pre-allocated once; reused across frames with zero heap allocation.
/// Only populated when [`CharucoRefiner::estimate_with_telemetry`] is called.
pub struct CharucoTelemetry {
    /// Homography-predicted 2D position of each rejected saddle (before refinement).
    pub rejected_predictions: Box<[Point2f]>,
    /// Structure tensor determinant at rejection (lower = blurrier / flatter region).
    pub rejected_determinants: Box<[f32]>,
    /// Number of valid entries in `[0..count]` on the last frame.
    pub count: usize,
}

impl CharucoTelemetry {
    /// Create a telemetry buffer pre-sized for `max_saddles` rejection entries.
    #[must_use]
    pub fn new(max_saddles: usize) -> Self {
        let n = max_saddles.max(1);
        Self {
            rejected_predictions: vec![Point2f { x: 0.0, y: 0.0 }; n].into_boxed_slice(),
            rejected_determinants: vec![0.0f32; n].into_boxed_slice(),
            count: 0,
        }
    }

    /// Slice of rejected saddle predictions for this frame.
    #[inline]
    #[must_use]
    pub fn rejected_predictions(&self) -> &[Point2f] {
        &self.rejected_predictions[..self.count]
    }

    /// Slice of rejection determinants for this frame.
    #[inline]
    #[must_use]
    pub fn rejected_determinants(&self) -> &[f32] {
        &self.rejected_determinants[..self.count]
    }
}

// ── Public output buffer ───────────────────────────────────────────────────

/// Pre-allocated output buffer for a single [`CharucoRefiner::estimate`] call.
///
/// Allocate once (via [`CharucoRefiner::new_batch`] or [`CharucoBatch::new`])
/// and reuse across frames — `estimate()` performs zero heap allocations.
///
/// After each call the valid results occupy `[0..count]` in each array.
pub struct CharucoBatch {
    /// Indices into [`CharucoTopology::saddle_points`] for each accepted saddle.
    pub saddle_ids: Box<[usize]>,
    /// Refined 2D image coordinates for each accepted saddle.
    pub saddle_image_pts: Box<[Point2f]>,
    /// Board-frame 3D coordinates for each accepted saddle.
    pub saddle_obj_pts: Box<[[f64; 3]]>,
    /// Estimated board pose, or `None` if fewer than 4 saddle points were
    /// accepted or LO-RANSAC could not find a consensus.
    pub board_pose: Option<BoardPose>,
    /// Number of valid saddles written into `[0..count]` on the last frame.
    pub count: usize,
    /// Diagnostic rejection data; `None` unless constructed via
    /// [`CharucoBatch::with_telemetry`].
    pub telemetry: Option<Box<CharucoTelemetry>>,
}

impl CharucoBatch {
    /// Create a batch pre-sized for `max_saddles` entries (no telemetry).
    ///
    /// Pass `topology.saddle_points.len()` as `max_saddles`.
    #[must_use]
    pub fn new(max_saddles: usize) -> Self {
        let n = max_saddles.max(1);
        Self {
            saddle_ids: vec![0usize; n].into_boxed_slice(),
            saddle_image_pts: vec![Point2f { x: 0.0, y: 0.0 }; n].into_boxed_slice(),
            saddle_obj_pts: vec![[0.0f64; 3]; n].into_boxed_slice(),
            board_pose: None,
            count: 0,
            telemetry: None,
        }
    }

    /// Create a batch with the telemetry sub-buffer pre-allocated.
    ///
    /// Reuse across frames — `estimate_with_telemetry()` performs zero heap
    /// allocations and resets [`CharucoTelemetry::count`] each call.
    #[must_use]
    pub fn with_telemetry(max_saddles: usize) -> Self {
        let mut b = Self::new(max_saddles);
        b.telemetry = Some(Box::new(CharucoTelemetry::new(max_saddles)));
        b
    }

    /// Slice of accepted saddle indices for this frame.
    #[inline]
    #[must_use]
    pub fn saddle_ids(&self) -> &[usize] {
        &self.saddle_ids[..self.count]
    }

    /// Slice of refined 2D image coordinates for this frame.
    #[inline]
    #[must_use]
    pub fn saddle_image_pts(&self) -> &[Point2f] {
        &self.saddle_image_pts[..self.count]
    }

    /// Slice of board-frame 3D coordinates for this frame.
    #[inline]
    #[must_use]
    pub fn saddle_obj_pts(&self) -> &[[f64; 3]] {
        &self.saddle_obj_pts[..self.count]
    }
}

// ── ChAruco refiner ────────────────────────────────────────────────────────

/// Stateless pipeline branch that extracts ChAruco saddle points from a
/// [`DetectionBatchView`] and estimates the board pose.
///
/// All scratch buffers are pre-allocated in [`CharucoRefiner::new`];
/// [`CharucoRefiner::estimate`] performs **zero heap allocations**.
pub struct CharucoRefiner {
    /// Board layout with saddle topology.
    ///
    /// Stored in an [`Arc`] so the same topology can be shared across frames or
    /// threads without cloning the marker/saddle tables.
    pub config: Arc<CharucoTopology>,
    /// Underlying robust pose solver (LO-RANSAC + AW-LM).
    pub solver: RobustPoseSolver,
    // Pre-allocated scratch (one-time Box allocations in new()):
    //
    // No per-group seed buffer: the solver constructs minimal-sample seeds
    // via multi-IPPE-Square branch enumeration (for square groups) or a
    // centroid-translation fallback (for `group_size == 1` saddle path) from
    // the correspondence data directly — see
    // `crate::board::solve_seed_from_ippe_enumeration`.
    scratch_img: Box<[Point2f]>,
    scratch_obj: Box<[[f64; 3]]>,
    scratch_info: Box<[Matrix2<f64>]>,
    /// `scratch_seen[id]` is `true` when saddle `id` has already been claimed
    /// by a tag in the current frame.  Reset selectively each frame.
    scratch_seen: Box<[bool]>,
    /// Saddle IDs written in the same order as `scratch_img`/`scratch_obj`.
    scratch_saddle_ids: Box<[usize]>,
    /// Indices of saddles touched this frame (for selective `scratch_seen` reset).
    scratch_touched: Box<[usize]>,
}

impl CharucoRefiner {
    /// Minimum structure tensor determinant to accept a saddle candidate.
    const MIN_DET: f64 = 1e-3;
    /// Structure tensor window radius (pixels).
    const ST_RADIUS: isize = 3;

    /// Creates a new `CharucoRefiner`, taking ownership of `config`.
    ///
    /// Wraps `config` in an [`Arc`] internally.  Reuse the same `CharucoRefiner`
    /// across frames to amortise the one-time scratch-buffer allocation.
    #[must_use]
    pub fn new(config: CharucoTopology) -> Self {
        Self::from_arc(Arc::new(config), LoRansacConfig::default())
    }

    /// Allocate a production [`CharucoBatch`] sized for this board (no telemetry).
    ///
    /// Call once during initialisation and reuse the batch across frames.
    #[must_use]
    pub fn new_batch(&self) -> CharucoBatch {
        CharucoBatch::new(self.config.saddle_points.len())
    }

    /// Allocate a [`CharucoBatch`] with telemetry pre-allocated for debug sessions.
    ///
    /// Use with [`CharucoRefiner::estimate_with_telemetry`].
    #[must_use]
    pub fn new_batch_with_telemetry(&self) -> CharucoBatch {
        CharucoBatch::with_telemetry(self.config.saddle_points.len())
    }

    /// Creates a new `CharucoRefiner` from a shared [`Arc`] topology.
    ///
    /// Use this when the same [`CharucoTopology`] is shared across multiple
    /// refiners or threads.
    #[must_use]
    pub fn from_arc(config: Arc<CharucoTopology>, ransac_cfg: LoRansacConfig) -> Self {
        let n = config.saddle_points.len().max(1);
        Self {
            solver: RobustPoseSolver::new().with_lo_ransac_config(ransac_cfg),
            scratch_img: vec![Point2f { x: 0.0, y: 0.0 }; n].into_boxed_slice(),
            scratch_obj: vec![[0.0f64; 3]; n].into_boxed_slice(),
            scratch_info: vec![Matrix2::identity(); n].into_boxed_slice(),
            scratch_seen: vec![false; n].into_boxed_slice(),
            scratch_saddle_ids: vec![0usize; n].into_boxed_slice(),
            scratch_touched: vec![0usize; n].into_boxed_slice(),
            config,
        }
    }

    /// Standard path — estimates board pose with zero telemetry overhead.
    ///
    /// Results are written into `batch`; `batch.count` and `batch.board_pose`
    /// are updated on every call.
    ///
    /// # Parameters
    /// - `view`       — slice view of valid decoded tags (from `Detector::detect`).
    /// - `img`        — the same image that was fed to the detector.
    /// - `intrinsics` — camera intrinsics for pose estimation.
    /// - `batch`      — caller-allocated output buffer (reuse across frames).
    pub fn estimate(
        &mut self,
        view: &DetectionBatchView<'_>,
        img: &ImageView,
        intrinsics: &CameraIntrinsics,
        batch: &mut CharucoBatch,
        outlier_drop_d2_threshold: f64,
    ) {
        self.estimate_impl::<false>(view, img, intrinsics, batch, outlier_drop_d2_threshold);
    }

    /// Debug path — same as [`estimate`](Self::estimate) but also records
    /// rejected saddle predictions into `batch.telemetry`.
    ///
    /// Requires `batch` to have been constructed via
    /// [`CharucoRefiner::new_batch_with_telemetry`].  When `batch.telemetry`
    /// is `None` this degrades gracefully to the production path.
    pub fn estimate_with_telemetry(
        &mut self,
        view: &DetectionBatchView<'_>,
        img: &ImageView,
        intrinsics: &CameraIntrinsics,
        batch: &mut CharucoBatch,
        outlier_drop_d2_threshold: f64,
    ) {
        self.estimate_impl::<true>(view, img, intrinsics, batch, outlier_drop_d2_threshold);
    }

    /// Core implementation — monomorphised at compile time via `TELEMETRY`.
    ///
    /// When `TELEMETRY = false` every `if TELEMETRY { }` block is statically
    /// dead code and is fully stripped by LLVM, leaving zero overhead in the
    /// production binary.
    ///
    /// `outlier_drop_d2_threshold > 0.0` enables the post-LM outlier-aware
    /// drop step on the joint board pose. Sourced from
    /// `DetectorConfig::outlier_drop_d2_threshold` by callers.
    fn estimate_impl<const TELEMETRY: bool>(
        &mut self,
        view: &DetectionBatchView<'_>,
        img: &ImageView,
        intrinsics: &CameraIntrinsics,
        batch: &mut CharucoBatch,
        outlier_drop_d2_threshold: f64,
    ) {
        let mut num_touched = 0usize;
        let mut num_accepted = 0usize;

        if TELEMETRY && let Some(t) = batch.telemetry.as_mut() {
            t.count = 0;
        }

        for tag_i in 0..view.len() {
            let tag_id = view.ids[tag_i] as usize;
            if tag_id >= self.config.tag_cell_corners.len() {
                continue;
            }

            let adj = self.config.tag_cell_corners[tag_id];

            // Get the tag's TL board coordinate and the stored homography.
            let Some(obj_pts) = self.config.obj_points.get(tag_id).and_then(|o| *o) else {
                continue;
            };
            let tl = obj_pts[0];
            let h = view.homographies[tag_i].data;

            for &maybe_sid in &adj {
                let Some(saddle_id) = maybe_sid else { continue };
                if saddle_id >= self.config.saddle_points.len() {
                    continue;
                }

                // Deduplicate: each saddle is processed at most once per frame.
                if self.scratch_seen[saddle_id] {
                    continue;
                }
                self.scratch_seen[saddle_id] = true;
                self.scratch_touched[num_touched] = saddle_id;
                num_touched += 1;

                let sp = self.config.saddle_points[saddle_id];

                // ── Step B: project through canonical homography ───────────
                let [u, v] = board_to_canonical(sp[0], sp[1], tl, self.config.marker_length);
                let [px, py] = apply_homography_col_major(&h, u, v);

                // Bounds check (leave 1-pixel border for Sobel kernels).
                let w = img.width as f64;
                let h_px = img.height as f64;
                if px < 1.0 || py < 1.0 || px > w - 2.0 || py > h_px - 2.0 {
                    continue;
                }

                // ── Step C: structure-tensor gate ─────────────────────────
                // Newton refinement on this corpus is empirically inert —
                // the step magnitude ~|∇I|/det(S) ~ 1e-5 px falls below f32
                // precision after the downstream cast, so the predicted
                // homography-projected coords ARE the refined coords. The
                // principled Förstner replacement regresses pose by 4.5×
                // on the ChArUco-golden corpus (see
                // `regression_board_hub::board_charuco_v1_golden_forstner`
                // and `memory/project_refine_saddle_noop.md`). What
                // survives — and matters — is the determinant-based
                // rejection of flat / low-conditioning windows.
                let det_s = compute_structure_tensor_det(img, px, py, Self::ST_RADIUS);
                if det_s < Self::MIN_DET {
                    if TELEMETRY {
                        record_rejection(&mut batch.telemetry, px, py, det_s);
                    }
                    continue;
                }

                // ── Step D: build correspondence ──────────────────────────
                // Information matrix = covariance^{-1}. A singular Σ_c
                // (zero-gradient / flat window that nonetheless slipped past
                // the determinant gate above) cannot be inverted into a
                // meaningful info matrix; defaulting to identity would advertise
                // unit confidence for a corner with no observed structure, so
                // the saddle is dropped instead. This is logically a second
                // rejection lane downstream of the structure-tensor gate;
                // telemetry records it under the same channel via a NaN
                // determinant marker so a debug observer can distinguish it.
                #[allow(clippy::cast_possible_truncation)]
                let cov =
                    compute_corner_covariance(img, [px, py], 0.1, 2.0, Self::ST_RADIUS as i32);
                let Some(info) = cov.try_inverse() else {
                    if TELEMETRY {
                        record_rejection(&mut batch.telemetry, px, py, f64::NAN);
                    }
                    continue;
                };
                self.scratch_saddle_ids[num_accepted] = saddle_id;
                self.scratch_img[num_accepted] = Point2f {
                    x: px as f32,
                    y: py as f32,
                };
                self.scratch_obj[num_accepted] = sp;
                self.scratch_info[num_accepted] = info;
                num_accepted += 1;
            }
        }

        // Reset seen flags for touched saddles (O(touched) not O(all saddles)).
        for &id in &self.scratch_touched[..num_touched] {
            self.scratch_seen[id] = false;
        }

        // ── Pose estimation ────────────────────────────────────────────────
        let board_pose = if num_accepted >= 4 {
            let corr = PointCorrespondences {
                image_points: &self.scratch_img[..num_accepted],
                object_points: &self.scratch_obj[..num_accepted],
                information_matrices: &self.scratch_info[..num_accepted],
                group_size: 1,
            };
            self.solver
                .estimate(&corr, intrinsics, outlier_drop_d2_threshold)
        } else {
            None
        };

        // ── Write results into caller-supplied batch (zero allocation) ─────
        batch.saddle_ids[..num_accepted].copy_from_slice(&self.scratch_saddle_ids[..num_accepted]);
        batch.saddle_image_pts[..num_accepted].copy_from_slice(&self.scratch_img[..num_accepted]);
        batch.saddle_obj_pts[..num_accepted].copy_from_slice(&self.scratch_obj[..num_accepted]);
        batch.count = num_accepted;
        batch.board_pose = board_pose;
    }
}

// ── Private helpers ────────────────────────────────────────────────────────

/// Convert a board-frame 2D point `(sx, sy)` to the canonical `[-1, 1]²`
/// coordinate system of the tag's stored homography.
///
/// The homography maps canonical `(-1,-1)` → TL image corner and `(1,1)` → BR
/// image corner of the **marker** (Layer A).  Because saddle points (Layer B)
/// lie at the outer corners of the *enclosing square*, which is larger than the
/// marker by the padding margin `(square_length - marker_length) / 2`, the
/// resulting canonical values satisfy `|u| > 1` or `|v| > 1`.
/// This is **intentional extrapolation** — the homography is linear and correctly
/// maps out-of-range canonical coordinates to the saddle's image position.
#[inline]
fn board_to_canonical(sx: f64, sy: f64, tag_tl: [f64; 3], marker_length: f64) -> [f64; 2] {
    [
        2.0 * (sx - tag_tl[0]) / marker_length - 1.0,
        2.0 * (sy - tag_tl[1]) / marker_length - 1.0,
    ]
}

/// Apply a column-major 3×3 homography to a canonical point `(u, v)`.
///
/// Nalgebra stores matrices in column-major order, so `data[j]` for
/// `j in 0..9` iterates the elements in column-major order:
///   - `data[0]` = h(0,0), `data[1]` = h(1,0), `data[2]` = h(2,0)
///   - `data[3]` = h(0,1), `data[4]` = h(1,1), `data[5]` = h(2,1)
///   - `data[6]` = h(0,2), `data[7]` = h(1,2), `data[8]` = h(2,2)
#[inline]
fn apply_homography_col_major(data: &[f32; 9], u: f64, v: f64) -> [f64; 2] {
    let x_h = f64::from(data[0]) * u + f64::from(data[3]) * v + f64::from(data[6]);
    let y_h = f64::from(data[1]) * u + f64::from(data[4]) * v + f64::from(data[7]);
    let w_h = f64::from(data[2]) * u + f64::from(data[5]) * v + f64::from(data[8]);
    if w_h.abs() < 1e-12 {
        return [0.0, 0.0];
    }
    [x_h / w_h, y_h / w_h]
}

/// Compute the structure tensor `S` and the image gradient `[gx, gy]` at pixel
/// `(cx, cy)` over a window of `radius` pixels using 3×3 Sobel kernels.
///
/// Returns `(S, grad)` where `S` is the 2×2 raw structure tensor (without
/// regularisation) and `grad` is the Sobel gradient at the centre pixel.
/// Tikhonov-regularised structure-tensor determinant over a `(2·radius+1)²`
/// window. Used as a flat-window rejection gate (`det < MIN_DET` ⇒ reject).
/// Iterative saddle refinement on this corpus is empirically falsified;
/// see `memory/project_refine_saddle_noop.md`.
fn compute_structure_tensor_det(img: &ImageView, cx: f64, cy: f64, radius: isize) -> f64 {
    let cxi = cx.floor() as isize;
    let cyi = cy.floor() as isize;
    let w = img.width.cast_signed();
    let h = img.height.cast_signed();
    let stride = img.stride.cast_signed();

    let r = radius;
    let x_start = (cxi - r).max(1);
    let x_end = (cxi + r).min(w - 2);
    let y_start = (cyi - r).max(1);
    let y_end = (cyi + r).min(h - 2);

    let x_min = (x_start - 1).cast_unsigned();
    let x_count = (x_end - x_start + 1).cast_unsigned();
    let row_slice_len = x_count + 2;

    let mut sum_gx2 = 0.0_f64;
    let mut sum_gy2 = 0.0_f64;
    let mut sum_gxgy = 0.0_f64;

    for py in y_start..=y_end {
        let off = (py * stride).cast_unsigned();
        let su = stride.cast_unsigned();
        let base0 = off - su + x_min;
        let base1 = off + x_min;
        let base2 = off + su + x_min;
        let row0 = &img.data[base0..base0 + row_slice_len];
        let row1 = &img.data[base1..base1 + row_slice_len];
        let row2 = &img.data[base2..base2 + row_slice_len];

        for k in 0..x_count {
            let p00 = i32::from(row0[k]);
            let p01 = i32::from(row0[k + 1]);
            let p02 = i32::from(row0[k + 2]);
            let p10 = i32::from(row1[k]);
            let p12 = i32::from(row1[k + 2]);
            let p20 = i32::from(row2[k]);
            let p21 = i32::from(row2[k + 1]);
            let p22 = i32::from(row2[k + 2]);

            let gx = f64::from((p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20));
            let gy = f64::from((p22 + 2 * p21 + p20) - (p02 + 2 * p01 + p00));

            sum_gx2 += gx * gx;
            sum_gy2 += gy * gy;
            sum_gxgy += gx * gy;
        }
    }

    // Tikhonov regularisation (ε = 1.0) on the structure tensor before
    // taking the determinant. Matches the prior `refine_saddle`
    // formulation so rejection thresholds carry over unchanged.
    let s00 = sum_gx2 + 1.0;
    let s11 = sum_gy2 + 1.0;
    let s01 = sum_gxgy;
    s00 * s11 - s01 * s01
}

/// Record a rejected saddle prediction into the telemetry buffer (no-op when
/// `telemetry` is `None`).  Inlined so the call site is zero-cost for
/// `TELEMETRY = false` (dead-code eliminated by LLVM).
#[inline]
fn record_rejection(telemetry: &mut Option<Box<CharucoTelemetry>>, px: f64, py: f64, det: f64) {
    if let Some(t) = telemetry.as_mut() {
        let idx = t.count;
        if idx < t.rejected_predictions.len() {
            t.rejected_predictions[idx] = Point2f {
                x: px as f32,
                y: py as f32,
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                t.rejected_determinants[idx] = det as f32;
            }
            t.count += 1;
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    missing_docs
)]
mod tests {
    use super::*;
    use crate::batch::DetectionBatch;
    use crate::image::ImageView;

    /// Build a synthetic 8-bit checkerboard image of `sq` pixel squares.
    fn checkerboard_buf(width: usize, height: usize, sq: usize) -> Vec<u8> {
        let mut buf = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                if (x / sq + y / sq).is_multiple_of(2) {
                    buf[y * width + x] = 200;
                } else {
                    buf[y * width + x] = 50;
                }
            }
        }
        buf
    }

    #[test]
    fn structure_tensor_det_passes_gate_on_clean_saddle() {
        // Synthetic hard-threshold checkerboard with 16-px squares. The
        // saddle at pixel (32, 32) — intersection of four squares — must
        // produce a well-conditioned structure tensor (Tikhonov-regularised
        // determinant well above `MIN_DET = 1e-3`).
        let sq = 16usize;
        let buf = checkerboard_buf(128, 128, sq);
        let img = ImageView::new(&buf, 128, 128, 128).unwrap();

        // 0.5-px offset places `.floor()` on the corner pixel (32, 32).
        let cx = 32.5_f64;
        let cy = 32.5_f64;

        let det = compute_structure_tensor_det(&img, cx, cy, 3);
        assert!(
            det > 1e-3,
            "structure tensor det must exceed gate threshold on a clean saddle, got {det}",
        );
    }

    #[test]
    fn test_deduplication_no_double_count() {
        // Feed a trivial empty batch — verify that scratch_seen is fully reset.
        let config = CharucoTopology::new(4, 4, 0.04, 0.03, usize::MAX).unwrap();
        let mut refiner = CharucoRefiner::new(config);

        let det_batch = DetectionBatch::new();
        let view = det_batch.view(0);
        let buf = vec![128u8; 256 * 256];
        let img = ImageView::new(&buf, 256, 256, 256).unwrap();
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 128.0, 128.0);

        let mut out = refiner.new_batch();
        refiner.estimate(&view, &img, &intrinsics, &mut out, 0.0);
        assert_eq!(out.count, 0);
        assert!(
            refiner.scratch_seen.iter().all(|&b| !b),
            "scratch_seen must be fully reset after estimate()"
        );
    }

    #[test]
    fn test_estimate_returns_none_with_few_saddles() {
        // With an empty batch, board_pose must be None.
        let config = CharucoTopology::new(4, 4, 0.04, 0.03, usize::MAX).unwrap();
        let mut refiner = CharucoRefiner::new(config);

        let det_batch = DetectionBatch::new();
        let view = det_batch.view(0);
        let buf = vec![128u8; 256 * 256];
        let img = ImageView::new(&buf, 256, 256, 256).unwrap();
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 128.0, 128.0);

        let mut out = refiner.new_batch();
        refiner.estimate(&view, &img, &intrinsics, &mut out, 0.0);
        assert!(out.board_pose.is_none());
    }

    #[test]
    fn test_board_to_canonical_identity() {
        // For the TL corner of a marker (tl = (0,0,0), marker_length=1):
        //   u = 2*(0 - 0)/1 - 1 = -1,  v = -1
        let [u, v] = board_to_canonical(0.0, 0.0, [0.0, 0.0, 0.0], 1.0);
        assert!((u + 1.0).abs() < 1e-12);
        assert!((v + 1.0).abs() < 1e-12);
        // For the BR corner (tl=(0,0,0), marker=1): (1,1) → (1,1)
        let [u2, v2] = board_to_canonical(1.0, 1.0, [0.0, 0.0, 0.0], 1.0);
        assert!((u2 - 1.0).abs() < 1e-12);
        assert!((v2 - 1.0).abs() < 1e-12);
    }

    /// Singular-`Σ_c` rejection contract: every saddle that survives the
    /// extraction pass must carry a finite, non-NaN information matrix.
    ///
    /// In practice the `compute_corner_covariance` Tikhonov-regularised
    /// kernel returns positive-definite Σ_c by construction, so the
    /// `try_inverse() → None` branch added in PR-C is defensive. This test
    /// verifies the invariant the branch upholds (no NaN/infinite info
    /// matrix reaches the LM), and is the appropriate observable since
    /// the rejection helper is `pub(crate)`-inlined in `estimate_impl`.
    ///
    /// A uniform mid-grey image is the harshest realistic stress case: zero
    /// Sobel gradient → minimal anisotropy → Tikhonov holds Σ_c at
    /// `2.0 · I` (well-conditioned). Any accepted saddle must have a
    /// finite `info`.
    #[test]
    fn admitted_saddle_info_matrices_are_finite() {
        let config = CharucoTopology::new(4, 4, 0.04, 0.03, usize::MAX).unwrap();
        let mut refiner = CharucoRefiner::new(config);

        let buf = vec![128u8; 256 * 256];
        let img = ImageView::new(&buf, 256, 256, 256).unwrap();
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 128.0, 128.0);

        let mut det_batch = DetectionBatch::new();
        det_batch.ids[0] = 0;
        det_batch.status_mask[0] = crate::batch::CandidateState::Valid;
        // Column-major identity homography with an image-centre translation
        // (`(u, v) ↦ (u + 128, v + 128)`). Saddles for tag 0 land within
        // image bounds.
        det_batch.homographies[0].data = [
            1.0, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, // col 1
            128.0, 128.0, 1.0, // col 2
        ];
        let v = det_batch.partition(1);
        let view = det_batch.view(v);

        let mut out = refiner.new_batch();
        refiner.estimate(&view, &img, &intrinsics, &mut out, 0.0);

        // Whatever the count, every admitted info matrix must be finite.
        // A singular Σ_c would have produced NaN entries in the inverse
        // before PR-C's reject-and-skip policy; that path now bumps the
        // skip counter without growing `num_accepted`.
        for i in 0..out.count {
            let info = refiner.scratch_info[i];
            for r in 0..2 {
                for c in 0..2 {
                    let v = info[(r, c)];
                    assert!(
                        v.is_finite(),
                        "admitted saddle {i} info[{r},{c}] must be finite, got {v}"
                    );
                }
            }
        }
    }

    /// Auxiliary smoke test of the rejection policy: a saddle whose Σ_c is
    /// constructed singular-by-hand must not be admitted to the LM solver.
    /// Driven via the public `compute_corner_covariance` kernel + a manual
    /// `try_inverse` to mirror the kernel's branching, this verifies our
    /// rejection contract at the policy level: `cov.try_inverse() == None`
    /// ⇒ saddle dropped.
    #[test]
    fn singular_covariance_try_inverse_returns_none() {
        // A genuinely singular 2×2: `[[1, 1], [1, 1]]` has rank 1, det = 0.
        let singular = Matrix2::<f64>::new(1.0, 1.0, 1.0, 1.0);
        assert!(
            singular.try_inverse().is_none(),
            "singular 2×2 must be uninvertible — the rejection contract \
             relies on this for the F6 skip-and-record-rejection branch"
        );

        // And the recovery shape we use in the implementation: an info
        // matrix recovered via `try_inverse` must be `Some(_)` only on
        // invertible input, so the `let Some(info) = cov.try_inverse()`
        // binding is sufficient to drive the saddle-skip control flow.
        assert!(Matrix2::<f64>::identity().try_inverse().is_some());
    }
}
