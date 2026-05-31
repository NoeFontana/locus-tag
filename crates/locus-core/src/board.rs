//! Board-level configuration and layout utilities.

use crate::batch::{MAX_CANDIDATES, Point2f};
use crate::pose::{
    CameraIntrinsics, Pose, projection_jacobian, quat_from_so3, solve_ippe_square, symmetrize_jtj6,
};
use nalgebra::{Matrix2, Matrix3, Matrix6, UnitQuaternion, Vector3, Vector6};
use std::sync::Arc;

// ── Board configuration error ──────────────────────────────────────────────

/// Errors that can occur when constructing a board topology.
#[derive(Debug, Clone)]
pub enum BoardConfigError {
    /// The requested board requires more tag IDs than the chosen dictionary provides.
    DictionaryTooSmall {
        /// Number of markers the board needs.
        required: usize,
        /// Number of unique IDs the dictionary offers.
        available: usize,
    },
}

impl std::fmt::Display for BoardConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DictionaryTooSmall {
                required,
                available,
            } => write!(
                f,
                "board requires {required} tag IDs but the dictionary only has {available}"
            ),
        }
    }
}

impl std::error::Error for BoardConfigError {}

// ── LO-RANSAC Configuration ────────────────────────────────────────────────

/// Configuration for LO-RANSAC board pose estimation.
///
/// These parameters are tuned for sub-pixel, metrology-grade perception on
/// structured fiducial grids. See the accompanying architectural specification
/// for the mathematical justification behind each constant.
#[derive(Clone, Debug)]
pub struct LoRansacConfig {
    /// Minimum RANSAC iterations before dynamic stopping is permitted.
    /// Guarantees the sampler escapes spatially-correlated occlusion clusters
    /// (e.g., a cable over 4 adjacent tags) before the stopping rule fires.
    pub k_min: u32,
    /// Hard ceiling on RANSAC iterations (worst-case execution time bound).
    /// Caps runtime at ~20 µs/frame even on fundamentally degraded images.
    pub k_max: u32,
    /// Desired probability of finding at least one clean minimal sample.
    pub confidence: f64,
    /// Outer inlier threshold, **squared** (pixels²).
    ///
    /// Wide net for the initial consensus gate: a 4-point IPPE model on
    /// sub-pixel noisy data can tilt, so this threshold is intentionally
    /// generous enough to admit all geometrically consistent tags even under
    /// per-tag pose noise before the LO GN step refines things.
    pub tau_outer_sq: f64,
    /// Inner inlier threshold, **squared** (pixels²).
    ///
    /// Tight gate applied *within* the LO Gauss-Newton inner loop.
    /// Anything outside ~1 px after GN smoothing is a Hamming alias,
    /// chromatic aberration, or physical occlusion — exclude from further GN.
    pub tau_inner_sq: f64,
    /// AW-LM inlier threshold, **squared** (pixels²).
    ///
    /// Applied to the LO-verified pose to build the inlier set passed to the
    /// final Anisotropic Weighted LM step.  This should be **generous** (≥ the
    /// outer threshold) so that AW-LM can exploit the maximum number of
    /// geometrically consistent observations; its internal Huber weighting
    /// (k = 1.345) robustly downweights any residual outliers.
    pub tau_aw_lm_sq: f64,
    /// Maximum iterations for the LO inner loop.
    ///
    /// Gauss-Newton converges quadratically; if the basin is not reached within
    /// 5 steps the solver is thrashing — terminate early.
    pub lo_max_iterations: u32,
    /// Minimum outer inlier count needed to trigger the LO inner loop.
    pub min_inliers: usize,
}

impl Default for LoRansacConfig {
    fn default() -> Self {
        Self {
            k_min: 15,
            k_max: 50,
            confidence: 0.9999,
            tau_outer_sq: 100.0, // 10² px²   — raw IPPE seeds have 5-10px board error; match original
            tau_inner_sq: 1.0,   // 1.0² px²  — tight gate inside LO GN loop
            tau_aw_lm_sq: 100.0, // 10² px²   — generous AW-LM input; Huber handles outliers
            lo_max_iterations: 5,
            min_inliers: 4,
        }
    }
}

// ── Board topologies ───────────────────────────────────────────────────────

/// Typed topology for an AprilGrid board.
///
/// Every grid cell contains one marker; tag IDs are assigned in row-major order.
/// Use [`AprilGridTopology::new`] to construct with dictionary-bounds validation.
#[derive(Clone, Debug)]
pub struct AprilGridTopology {
    /// Number of rows in the marker grid.
    pub rows: usize,
    /// Number of columns in the marker grid.
    pub cols: usize,
    /// Physical side length of one marker (metres).
    pub marker_length: f64,
    /// Physical gap between adjacent markers (metres).
    pub spacing: f64,
    /// 3D object points for each tag ID (row-major, [TL, TR, BR, BL]).
    pub obj_points: Vec<Option<[[f64; 3]; 4]>>,
}

impl AprilGridTopology {
    /// Construct from pre-computed `obj_points` for adapters and tests.
    ///
    /// Bypasses the standard AprilGrid geometry calculation.  Callers are
    /// responsible for ensuring `obj_points` are consistent with `rows × cols`.
    /// This is primarily useful when adapting a [`CharucoTopology`]'s marker
    /// table for use with [`BoardEstimator`] (tag-only pose estimation without
    /// saddle refinement).
    #[must_use]
    pub fn from_obj_points(
        rows: usize,
        cols: usize,
        marker_length: f64,
        obj_points: Vec<Option<[[f64; 3]; 4]>>,
    ) -> Self {
        Self {
            rows,
            cols,
            marker_length,
            spacing: 0.0,
            obj_points,
        }
    }

    /// Build an AprilGrid topology, validating marker count against `max_tag_id`.
    ///
    /// `max_tag_id` is the number of unique IDs the target dictionary provides
    /// (obtain via [`crate::TagFamily::max_id_count`]).  Pass `usize::MAX` to skip the
    /// check (e.g. in tests).
    ///
    /// The origin `(0,0,0)` is at the geometric centre of the board.
    ///
    /// # Errors
    /// Returns [`BoardConfigError::DictionaryTooSmall`] when `rows × cols >
    /// max_tag_id`.
    pub fn new(
        rows: usize,
        cols: usize,
        spacing: f64,
        marker_length: f64,
        max_tag_id: usize,
    ) -> Result<Self, BoardConfigError> {
        let num_markers = rows * cols;
        if num_markers > max_tag_id {
            return Err(BoardConfigError::DictionaryTooSmall {
                required: num_markers,
                available: max_tag_id,
            });
        }

        let mut obj_points = vec![None; num_markers];
        let step = marker_length + spacing;
        let board_width = cols as f64 * marker_length + (cols - 1) as f64 * spacing;
        let board_height = rows as f64 * marker_length + (rows - 1) as f64 * spacing;
        let offset_x = -board_width / 2.0;
        let offset_y = -board_height / 2.0;

        for r in 0..rows {
            for c in 0..cols {
                let x = offset_x + c as f64 * step;
                let y = offset_y + r as f64 * step;
                let idx = r * cols + c;
                obj_points[idx] = Some([
                    [x, y, 0.0],
                    [x + marker_length, y, 0.0],
                    [x + marker_length, y + marker_length, 0.0],
                    [x, y + marker_length, 0.0],
                ]);
            }
        }

        Ok(Self {
            rows,
            cols,
            marker_length,
            spacing,
            obj_points,
        })
    }
}

/// Typed topology for a ChAruco board.
///
/// The board has two distinct layers of geometric primitives:
///
/// **Layer A — Tags**: ArUco markers that occupy the dark squares where `(row + col)` is even.
/// Each tag fills only the inner `marker_length × marker_length` area of its
/// `square_length × square_length` cell, leaving a white padding margin of
/// `(square_length - marker_length) / 2` on every side.
///
/// **Layer B — Saddles**: The `(rows-1) × (cols-1)` interior checkerboard corners.
/// These are the intersection points of the black squares' outer edges.
/// Crucially, saddle points lie *outside* the tag's physical boundary — they are at the
/// corners of the *enclosing square*, not the tag itself.
///
/// Use [`CharucoTopology::new`] to construct with dictionary-bounds validation.
#[derive(Clone, Debug)]
pub struct CharucoTopology {
    /// Number of square rows on the board.
    pub rows: usize,
    /// Number of square columns on the board.
    pub cols: usize,
    /// Physical side length of one ArUco marker (metres).
    pub marker_length: f64,
    /// Physical side length of one checkerboard square (metres).
    pub square_length: f64,
    /// 3D object points for each marker ID (indexed by detection ID, [TL, TR, BR, BL]).
    pub obj_points: Vec<Option<[[f64; 3]; 4]>>,
    /// 3D coordinates of interior checkerboard corners (saddle points).
    ///
    /// Saddle at interior-grid position `(sr, sc)` has ID `sr*(cols-1)+sc`.
    pub saddle_points: Vec<[f64; 3]>,
    /// For each marker index, the IDs of the 4 corners of its enclosing black square: `[TL, TR, BR, BL]`.
    ///
    /// These are **saddle-point IDs** (Layer B), not tag corner IDs (Layer A).
    /// Because markers occupy only the inner `marker_length × marker_length` area of their
    /// `square_length × square_length` cell, each saddle lies *outside* the physical tag
    /// boundary by a padding margin of `(square_length - marker_length) / 2`.
    /// `None` indicates the corner is on the board perimeter (no interior saddle there).
    pub tag_cell_corners: Vec<[Option<usize>; 4]>,
}

impl CharucoTopology {
    /// Build a ChAruco topology, validating marker count against `max_tag_id`.
    ///
    /// `max_tag_id` is the number of unique IDs the target dictionary provides
    /// (obtain via [`crate::TagFamily::max_id_count`]).  Pass `usize::MAX` to skip the
    /// check (e.g. in tests).
    ///
    /// The origin `(0,0,0)` is at the geometric centre of the board.
    ///
    /// # Errors
    /// Returns [`BoardConfigError::DictionaryTooSmall`] when the number of
    /// markers on the board exceeds `max_tag_id`.
    pub fn new(
        rows: usize,
        cols: usize,
        square_length: f64,
        marker_length: f64,
        max_tag_id: usize,
    ) -> Result<Self, BoardConfigError> {
        let num_markers = (rows * cols).div_ceil(2);
        if num_markers > max_tag_id {
            return Err(BoardConfigError::DictionaryTooSmall {
                required: num_markers,
                available: max_tag_id,
            });
        }

        let total_width = cols as f64 * square_length;
        let total_height = rows as f64 * square_length;
        let offset_x = -total_width / 2.0;
        let offset_y = -total_height / 2.0;
        let marker_padding = (square_length - marker_length) / 2.0;

        let mut obj_points = vec![None; num_markers];
        let mut marker_idx = 0;
        for r in 0..rows {
            for c in 0..cols {
                if (r + c) % 2 == 0 {
                    let x = offset_x + c as f64 * square_length + marker_padding;
                    let y = offset_y + r as f64 * square_length + marker_padding;
                    if marker_idx < obj_points.len() {
                        obj_points[marker_idx] = Some([
                            [x, y, 0.0],
                            [x + marker_length, y, 0.0],
                            [x + marker_length, y + marker_length, 0.0],
                            [x, y + marker_length, 0.0],
                        ]);
                        marker_idx += 1;
                    }
                }
            }
        }

        // ── Saddle points (interior checkerboard corners) ────────────────
        let saddle_cols = cols.saturating_sub(1);
        let num_saddles = rows.saturating_sub(1) * saddle_cols;
        let mut saddle_points = Vec::with_capacity(num_saddles);
        for sr in 0..rows.saturating_sub(1) {
            for sc in 0..saddle_cols {
                let x = offset_x + (sc + 1) as f64 * square_length;
                let y = offset_y + (sr + 1) as f64 * square_length;
                saddle_points.push([x, y, 0.0]);
            }
        }

        // ── Tag→square-corner adjacency (tag_cell_corners) ──────────────
        // Maps each marker to the 4 saddle IDs at the corners of its enclosing square.
        // The saddles lie outside the tag's physical boundary; see field-level docs.
        let mut tag_cell_corners = vec![[None; 4]; num_markers];
        let saddle_id = |sr: isize, sc: isize| -> Option<usize> {
            let saddle_rows_max: isize = rows.saturating_sub(1).cast_signed();
            let saddle_cols_max: isize = cols.saturating_sub(1).cast_signed();
            if sr < 0 || sc < 0 || sr >= saddle_rows_max || sc >= saddle_cols_max {
                None
            } else {
                Some(sr.cast_unsigned() * saddle_cols + sc.cast_unsigned())
            }
        };
        let mut midx = 0usize;
        for r in 0..rows {
            for c in 0..cols {
                if (r + c) % 2 == 0 {
                    let ri = r.cast_signed();
                    let ci = c.cast_signed();
                    tag_cell_corners[midx] = [
                        saddle_id(ri - 1, ci - 1), // TL corner of enclosing square
                        saddle_id(ri - 1, ci),     // TR corner of enclosing square
                        saddle_id(ri, ci),         // BR corner of enclosing square
                        saddle_id(ri, ci - 1),     // BL corner of enclosing square
                    ];
                    midx += 1;
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            marker_length,
            square_length,
            obj_points,
            saddle_points,
            tag_cell_corners,
        })
    }
}

// ── Result type ────────────────────────────────────────────────────────────

/// Result of a board pose estimation.
#[derive(Clone, Debug)]
pub struct BoardPose {
    /// The estimated 6-DOF pose.
    pub pose: Pose,
    /// The 6x6 pose covariance matrix in se(3) tangent space.
    /// Order: [tx, ty, tz, rx, ry, rz]
    pub covariance: Matrix6<f64>,
}

// ── Generic Correspondence Interface ──────────────────────────────────────

/// A flat, contiguous view of M 2D-3D point correspondences for pose estimation.
///
/// Points are organised in contiguous *groups* of [`group_size`](Self::group_size)
/// elements. The inlier bitmask operated on by [`RobustPoseSolver`] tracks one
/// bit *per group*, keeping the mask size bounded at 1 024 bits regardless of
/// `group_size`.
///
/// | Pipeline        | `group_size` | Bit semantics             |
/// |-----------------|--------------|---------------------------|
/// | AprilGrid       | 4            | 1 bit = 1 tag (4 corners) |
/// | ChAruco saddles | 1            | 1 bit = 1 saddle point    |
///
/// **Lifetime**: the slices are typically backed by pre-allocated scratch
/// buffers owned by [`BoardEstimator`] or by arena memory, ensuring zero heap
/// allocation on the hot path.
///
/// **Seed strategy**: minimal-sample seeds for RANSAC are computed inside the
/// solver by fitting a single DLT homography over all `4 × group_size` sample
/// correspondences and decomposing it with IPPE-Square (see
/// `solve_seed_from_sample_homography`).  Both Necker branches of the
/// metric homography are polished by Gauss-Newton and the lower-reprojection
/// branch wins — works uniformly for `group_size == 4` (per-tag corners) and
/// `group_size == 1` (ChAruco saddle points).
pub struct PointCorrespondences<'a> {
    /// Observed 2D image points (pixels). Length = M.
    pub image_points: &'a [Point2f],
    /// Corresponding 3D model points (board frame, metres). Length = M.
    pub object_points: &'a [[f64; 3]],
    /// Pre-inverted observation covariances (information matrices). Length = M.
    /// Must be positive semi-definite; use [`Matrix2::identity`] for isotropic
    /// unit weighting.
    pub information_matrices: &'a [Matrix2<f64>],
    /// Number of consecutive points forming one logical correspondence group.
    /// `M` must be an exact multiple of `group_size`.
    pub group_size: usize,
}

impl PointCorrespondences<'_> {
    /// Number of correspondence groups: `M / group_size`.
    #[inline]
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.image_points.len() / self.group_size
    }
}

// ── Sample-homography IPPE-Square RANSAC seed ─────────────────────────────

/// Joint reprojection RMSE (squared mean) of `pose` against the 4 × `group_size`
/// correspondences listed by `sample`.  Used to score IPPE-Square candidates
/// during minimal-sample seeding.
///
/// Returns a finite value monotonic with reprojection error; `f64::INFINITY`
/// for poses that fail the cheirality check (a point falls behind the camera).
fn sample_reprojection_score(
    pose: &Pose,
    sample: &[usize; 4],
    corr: &PointCorrespondences<'_>,
    intrinsics: &CameraIntrinsics,
) -> f64 {
    let gs = corr.group_size;
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;

    for &s_val in sample {
        let start = s_val * gs;
        for k in start..(start + gs) {
            let obj = corr.object_points[k];
            let p_world = Vector3::new(obj[0], obj[1], obj[2]);
            let p_cam = pose.rotation * p_world + pose.translation;
            if p_cam.z < 1e-4 {
                return f64::INFINITY;
            }
            let z_inv = 1.0 / p_cam.z;
            let u = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
            let v = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;
            let du = u - f64::from(corr.image_points[k].x);
            let dv = v - f64::from(corr.image_points[k].y);
            sum_sq += du * du + dv * dv;
            count += 1;
        }
    }

    if count == 0 {
        f64::INFINITY
    } else {
        sum_sq / count as f64
    }
}

/// One unweighted Gauss-Newton refinement step against the sub-sample listed
/// by `sample` (≤ 4 × `group_size` correspondences).
///
/// This is a sample-restricted variant of [`RobustPoseSolver::gn_step`] — used
/// inside [`solve_seed_from_sample_homography`] to polish each IPPE-Square
/// branch into its local minimum BEFORE scoring.  Polishing equalises the
/// noise floor across the 8 candidates so the joint-reprojection score
/// reliably separates the correct Necker branch from the wrong one.
///
/// Returns the input pose unchanged if the 6×6 normal equations are singular.
fn gn_step_on_sample(
    pose: &Pose,
    sample: &[usize; 4],
    corr: &PointCorrespondences<'_>,
    intrinsics: &CameraIntrinsics,
) -> Pose {
    let mut jtj = Matrix6::<f64>::zeros();
    let mut jtr = Vector6::<f64>::zeros();
    let gs = corr.group_size;

    for &s_val in sample {
        let start = s_val * gs;
        for k in start..(start + gs) {
            let obj = corr.object_points[k];
            let p_world = Vector3::new(obj[0], obj[1], obj[2]);
            let p_cam = pose.rotation * p_world + pose.translation;
            if p_cam.z < 1e-4 {
                continue;
            }
            let z_inv = 1.0 / p_cam.z;
            let x_z = p_cam.x * z_inv;
            let y_z = p_cam.y * z_inv;

            let u = intrinsics.fx * x_z + intrinsics.cx;
            let v = intrinsics.fy * y_z + intrinsics.cy;
            let res_u = f64::from(corr.image_points[k].x) - u;
            let res_v = f64::from(corr.image_points[k].y) - v;

            let (ju0, ju2, ju3, ju4, ju5, jv1, jv2, jv3, jv4, jv5) =
                projection_jacobian(x_z, y_z, z_inv, intrinsics);

            jtr[0] += ju0 * res_u;
            jtr[1] += jv1 * res_v;
            jtr[2] += ju2 * res_u + jv2 * res_v;
            jtr[3] += ju3 * res_u + jv3 * res_v;
            jtr[4] += ju4 * res_u + jv4 * res_v;
            jtr[5] += ju5 * res_u + jv5 * res_v;

            jtj[(0, 0)] += ju0 * ju0;
            jtj[(0, 2)] += ju0 * ju2;
            jtj[(0, 3)] += ju0 * ju3;
            jtj[(0, 4)] += ju0 * ju4;
            jtj[(0, 5)] += ju0 * ju5;

            jtj[(1, 1)] += jv1 * jv1;
            jtj[(1, 2)] += jv1 * jv2;
            jtj[(1, 3)] += jv1 * jv3;
            jtj[(1, 4)] += jv1 * jv4;
            jtj[(1, 5)] += jv1 * jv5;

            jtj[(2, 2)] += ju2 * ju2 + jv2 * jv2;
            jtj[(2, 3)] += ju2 * ju3 + jv2 * jv3;
            jtj[(2, 4)] += ju2 * ju4 + jv2 * jv4;
            jtj[(2, 5)] += ju2 * ju5 + jv2 * jv5;

            jtj[(3, 3)] += ju3 * ju3 + jv3 * jv3;
            jtj[(3, 4)] += ju3 * ju4 + jv3 * jv4;
            jtj[(3, 5)] += ju3 * ju5 + jv3 * jv5;

            jtj[(4, 4)] += ju4 * ju4 + jv4 * jv4;
            jtj[(4, 5)] += ju4 * ju5 + jv4 * jv5;

            jtj[(5, 5)] += ju5 * ju5 + jv5 * jv5;
        }
    }
    symmetrize_jtj6(&mut jtj);

    if let Some(chol) = jtj.cholesky() {
        let delta = chol.solve(&jtr);
        let twist = Vector3::new(delta[3], delta[4], delta[5]);
        let dq = UnitQuaternion::from_scaled_axis(twist);
        Pose {
            rotation: (dq * quat_from_so3(pose.rotation))
                .to_rotation_matrix()
                .into_inner(),
            translation: pose.translation + Vector3::new(delta[0], delta[1], delta[2]),
        }
    } else {
        *pose
    }
}

/// Fit a planar homography from board-frame `(X, Y, 1)` to normalised image
/// `(x_n, y_n, 1)` over ALL `4 × group_size` sample correspondences via
/// totals-least-squares (the smallest-eigenvalue eigenvector of the symmetric
/// 9 × 9 `A^T A`), then decompose into both Necker pose branches using
/// `solve_ippe_square`.
///
/// Returns both `(camera-from-board)` branches (in the sample-centroid-shifted
/// frame, which is later un-shifted by [`solve_seed_from_sample_homography`]),
/// the centroid offset that the caller must apply, or `None` on pathological
/// inputs (colinear points, singular `A^T A`).
///
/// **Why this complements per-tag IPPE**: per-tag IPPE-Square on 4 noisy
/// corners (≈ 20-30 px tag edges) can produce a pose biased by 50°+ — both
/// branches end up wrong-sided.  The 16-point DLT homography averages corner
/// noise across all 4 sampled tags and yields a much tighter homography
/// estimate; running IPPE-Square on it preserves the two-branch
/// disambiguation without losing the noise-averaging benefit.  This is the
/// rescue path for the "catastrophic per-tag collapse" frames documented in
/// `diagnostics/board_p99_investigation_2026-05-14/MEMO.md`.
#[allow(clippy::similar_names, clippy::many_single_char_names)]
fn ippe_branches_from_sample_homography(
    sample: &[usize; 4],
    corr: &PointCorrespondences<'_>,
    intrinsics: &CameraIntrinsics,
) -> Option<([Pose; 2], Vector3<f64>)> {
    let gs = corr.group_size;
    let n = 4 * gs;

    // Centroid of board-frame X, Y for numerical conditioning AND to feed
    // IPPE-Square (which expects metric coords centred at origin).
    let mut cx_o = 0.0f64;
    let mut cy_o = 0.0f64;
    for &s_val in sample {
        let start = s_val * gs;
        for k in start..(start + gs) {
            cx_o += corr.object_points[k][0];
            cy_o += corr.object_points[k][1];
        }
    }
    cx_o /= n as f64;
    cy_o /= n as f64;

    // Mean squared distance for object-side scale normalisation.
    let mut mean_dist = 0.0f64;
    for &s_val in sample {
        let start = s_val * gs;
        for k in start..(start + gs) {
            let dx = corr.object_points[k][0] - cx_o;
            let dy = corr.object_points[k][1] - cy_o;
            mean_dist += (dx * dx + dy * dy).sqrt();
        }
    }
    mean_dist /= n as f64;
    if mean_dist < 1e-12 {
        return None;
    }
    let scale_o = std::f64::consts::SQRT_2 / mean_dist;

    // Build A^T A (9 × 9) symmetric matrix in place.  For each correspondence
    // we accumulate 2 DLT rows (u-row and v-row).
    let mut ata = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U9, _>::zeros();
    for &s_val in sample {
        let start = s_val * gs;
        for k in start..(start + gs) {
            let x = (corr.object_points[k][0] - cx_o) * scale_o;
            let y = (corr.object_points[k][1] - cy_o) * scale_o;
            let u = (f64::from(corr.image_points[k].x) - intrinsics.cx) / intrinsics.fx;
            let v = (f64::from(corr.image_points[k].y) - intrinsics.cy) / intrinsics.fy;

            // DLT rows for the homography `[x_n, y_n, 1]^T ∝ H · [X, Y, 1]^T`.
            let r1 = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u];
            let r2 = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v];
            for i in 0..9 {
                for j in i..9 {
                    ata[(i, j)] += r1[i] * r1[j] + r2[i] * r2[j];
                }
            }
        }
    }
    // Mirror upper → lower.
    for i in 1..9 {
        for j in 0..i {
            ata[(i, j)] = ata[(j, i)];
        }
    }

    let eig = ata.symmetric_eigen();
    let mut min_idx = 0usize;
    let mut min_val = eig.eigenvalues[0];
    for i in 1..9 {
        if eig.eigenvalues[i] < min_val {
            min_val = eig.eigenvalues[i];
            min_idx = i;
        }
    }
    let h_vec = eig.eigenvectors.column(min_idx);
    let h_n = Matrix3::new(
        h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4], h_vec[5], h_vec[6], h_vec[7], h_vec[8],
    );

    // Un-normalise the object-side scaling: the homography we want maps
    // *centred-but-unscaled* board coords (X - cx_o, Y - cy_o, 1) → normalised
    // image.  The scaler we applied to (X, Y) was uniform `scale_o`, so we
    // undo it by scaling the first two columns of H_n.
    let mut h_metric = h_n;
    h_metric.column_mut(0).scale_mut(scale_o);
    h_metric.column_mut(1).scale_mut(scale_o);

    // Sign convention: ensure positive depth at the (now-centred) origin.
    // The centred origin maps to `h_metric * [0, 0, 1]^T = h_metric.col(2)`.
    if h_metric[(2, 2)] < 0.0 {
        h_metric *= -1.0;
    }

    let [pose_a, pose_b] = solve_ippe_square(&h_metric)?;
    if !pose_a
        .rotation
        .iter()
        .chain(pose_a.translation.iter())
        .chain(pose_b.rotation.iter())
        .chain(pose_b.translation.iter())
        .all(|v| v.is_finite())
    {
        return None;
    }

    let centroid_offset = Vector3::new(cx_o, cy_o, 0.0);
    Some(([pose_a, pose_b], centroid_offset))
}

/// Sample-homography IPPE-Square seed for a minimal RANSAC sample.
///
/// Fits a planar homography from board-frame `(X, Y)` to normalised image
/// coordinates via DLT over all `4 × group_size` sample correspondences, then
/// decomposes the homography with `solve_ippe_square` to obtain both Necker
/// branches.  Each branch is polished with 3 unweighted Gauss-Newton steps
/// over the same correspondences and scored by joint reprojection RMSE; the
/// lower-scoring branch wins.
///
/// The Necker (planar-pose) ambiguity is disambiguated *jointly*: the wrong
/// branch produces a higher reprojection score over the full sample, so the
/// scoring step selects the correct global pose without per-tag voting.
///
/// **Why DLT over all 4 × group_size points (not per-tag IPPE)**: per-tag
/// IPPE-Square on 4 noisy corners (≈ 20-30 px tag edges) can produce a pose
/// biased by 50°+ — both branches end up wrong-sided.  Fitting a single DLT
/// homography over ALL sample correspondences averages corner noise across
/// the 4 sampled tags / 16 saddle points and yields a much tighter homography
/// estimate; running IPPE-Square on it preserves the two-branch
/// disambiguation without losing the noise-averaging benefit.  See
/// `diagnostics/board_p99_investigation_2026-05-14/MEMO.md` for the
/// catastrophic-per-tag-collapse frames this rescues, and
/// `diagnostics/multi_ippe_seed_2026-05-14/MEMO.md` for the algorithm rationale.
///
/// **Why this replaces DLT + Zhang homography decomposition**: DLT minimises
/// algebraic error rather than geometric reprojection, but that's fine here
/// because IPPE-Square performs a geometric re-projection check.  The Zhang
/// decomposition `R = SO(3)-project(K^{-1} · H)` would have collapsed the
/// planar Necker ambiguity at the SVD step — producing one branch
/// unconditionally; IPPE-Square enumerates both branches and lets the joint
/// reprojection score pick the winner.
///
/// Returns `None` only on pathological inputs (NaN intrinsics, degenerate
/// sample) — callers should treat `None` as "skip this minimal sample".
#[allow(clippy::similar_names)]
fn solve_seed_from_sample_homography(
    sample: &[usize; 4],
    corr: &PointCorrespondences<'_>,
    intrinsics: &CameraIntrinsics,
) -> Option<Pose> {
    let (pose_branches, centroid) = ippe_branches_from_sample_homography(sample, corr, intrinsics)?;

    let mut best_pose: Option<Pose> = None;
    let mut best_score = f64::INFINITY;

    for pose_centred in pose_branches {
        // IPPE-Square returns poses in the centroid-shifted board frame
        // (origin = (cx_o, cy_o, 0)).  Un-shift to the original board frame:
        //   t_board = t_centred − R · centroid.
        let raw = Pose {
            rotation: pose_centred.rotation,
            translation: pose_centred.translation - pose_centred.rotation * centroid,
        };

        // Polish each branch with 3 unweighted GN steps over the
        // 4 × group_size sample correspondences (basin-convergent in 1-3
        // iterations).  Polishing equalises the noise floor between the two
        // branches so the joint-reprojection score reliably separates the
        // correct Necker branch.
        let mut polished = raw;
        for _ in 0..3 {
            let next = gn_step_on_sample(&polished, sample, corr, intrinsics);
            let dt = (next.translation - polished.translation).norm();
            polished = next;
            if dt < 1e-6 {
                break;
            }
        }

        let score = sample_reprojection_score(&polished, sample, corr, intrinsics);
        if score < best_score && score.is_finite() {
            best_score = score;
            best_pose = Some(polished);
        }
    }

    best_pose
}

// ── Robust Pose Solver ─────────────────────────────────────────────────────

/// Pure mathematical engine for robust, multi-correspondence board pose
/// estimation.
///
/// Completely decoupled from [`crate::batch::DetectionBatch`] and tag layout.  Accepts flat
/// [`PointCorrespondences`] slices and returns a verified [`BoardPose`].
///
/// **Algorithm**: LO-RANSAC (outer) → unweighted Gauss-Newton verification
/// (inner) → Anisotropic Weighted Levenberg-Marquardt final refinement.
#[derive(Default)]
pub struct RobustPoseSolver {
    /// LO-RANSAC hyper-parameters.
    pub lo_ransac: LoRansacConfig,
}

impl RobustPoseSolver {
    /// Creates a new solver with default LO-RANSAC parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: override the LO-RANSAC configuration.
    #[must_use]
    pub fn with_lo_ransac_config(mut self, cfg: LoRansacConfig) -> Self {
        self.lo_ransac = cfg;
        self
    }

    // ── Public entry point ───────────────────────────────────────────────

    /// Estimates a board pose from flat point correspondences.
    ///
    /// Returns `None` if fewer than 4 groups are present, or if LO-RANSAC
    /// cannot find a consensus set.
    ///
    /// When `outlier_drop_d2_threshold > 0.0`, the outlier-aware drop step
    /// runs on the final AW-LM output: if a single observation's d²
    /// exceeds the threshold and dominates the second-worst by ≥ 2×, the
    /// containing group is masked, the LM is re-run, and the new pose is
    /// kept iff its kept-group d² sum is strictly lower than the
    /// original's. Pass `0.0` to disable.
    #[must_use]
    pub fn estimate(
        &self,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        outlier_drop_d2_threshold: f64,
    ) -> Option<BoardPose> {
        if corr.num_groups() < 4 {
            return None;
        }

        // Phase 1: LO-RANSAC → verified IPPE seed + tight inlier count.
        let (best_pose, _tight_mask) = self.lo_ransac_loop(corr, intrinsics)?;

        // Phase 2: re-evaluate inliers at the generous tau_aw_lm on the
        // LO-verified pose.  The wider window maximises the AW-LM observation
        // count; Huber weighting (k = 1.345) handles any residual mild outliers.
        let (aw_lm_mask, _) =
            self.evaluate_inliers(&best_pose, corr, intrinsics, self.lo_ransac.tau_aw_lm_sq);

        // Phase 3: final AW-LM over the verified, relaxed inlier set.
        let (refined_pose, covariance) =
            self.refine_aw_lm(&best_pose, corr, intrinsics, &aw_lm_mask);

        // Opt-in: outlier-aware drop on the AW-LM result (N-observation
        // form of the per-tag mechanism). If the masked re-fit produced
        // a singular Hessian (NaN-sentinel cov), prefer the unmasked
        // covariance — the dropped-group pose is still selected because
        // `kept_3 < kept_4` proved its fit was better, but emitting a
        // NaN cov alongside a non-NaN one when both are available throws
        // away usable uncertainty.
        let (final_pose, final_cov) = match self.apply_outlier_drop_to_board_lm(
            &refined_pose,
            corr,
            intrinsics,
            &aw_lm_mask,
            outlier_drop_d2_threshold,
        ) {
            Some((pose_3, cov_3, _g_dropped)) => {
                let cov_3_finite = cov_3.iter().all(|v| v.is_finite());
                let cov_4_finite = covariance.iter().all(|v| v.is_finite());
                if cov_3_finite || !cov_4_finite {
                    (pose_3, cov_3)
                } else {
                    // pose_3 wins on fit, cov_4 wins on definedness.
                    (pose_3, covariance)
                }
            },
            None => (refined_pose, covariance),
        };

        Some(BoardPose {
            pose: final_pose,
            covariance: final_cov,
        })
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Core LO-RANSAC loop.
    ///
    /// Outer loop: random 4-group sampling → DLT-homography + IPPE-Square
    /// seed → outer-threshold evaluation.  Inner loop (LO): unweighted
    /// Gauss-Newton refinement + tight re-evaluation with monotonicity
    /// guard.  Dynamic stopping: `k` is updated after each tight-count
    /// improvement using the standard RANSAC formula
    /// `k = log(1-p) / log(1-ω⁴)` where `ω` is the verified tight inlier
    /// ratio from `lo_inner`.
    ///
    /// **Seed strategy**: each 4-group sample is fed to
    /// [`solve_seed_from_sample_homography`] which fits a DLT homography over
    /// the `4 × group_size` correspondences, decomposes it with IPPE-Square
    /// to obtain both Necker branches, polishes each by Gauss-Newton, and
    /// returns the lower-joint-reprojection branch.  IPPE-Square's two-branch
    /// enumeration disambiguates the planar pose ambiguity that a Zhang
    /// `R = SO(3)-project(K⁻¹H)` decomposition would silently collapse.
    #[allow(clippy::too_many_lines, clippy::cast_sign_loss)]
    fn lo_ransac_loop(
        &self,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
    ) -> Option<(Pose, [u64; 16])> {
        let cfg = &self.lo_ransac;
        let num_groups = corr.num_groups();

        let mut global_best_tight_count = 0usize;
        let mut global_best_seed: Option<Pose> = None;
        let mut dynamic_k = cfg.k_max;

        // Deterministic XOR-shift RNG (reproducible across frames).
        let mut rng_seed = 0x1337u32;

        for iter in 0..cfg.k_max {
            // ── Draw 4 distinct groups without replacement ────────────────
            let mut sample = [0usize; 4];
            let mut found = 0usize;
            let mut attempts = 0u32;
            while found < 4 && attempts < 1000 {
                attempts += 1;
                rng_seed ^= rng_seed << 13;
                rng_seed ^= rng_seed >> 17;
                rng_seed ^= rng_seed << 5;
                let s = (rng_seed as usize) % num_groups;
                if !sample[..found].contains(&s) {
                    sample[found] = s;
                    found += 1;
                }
            }
            if found < 4 {
                continue;
            }

            // ── DLT-homography + IPPE-Square branch-enumeration seed ─────
            // Fit one homography from all 4 × group_size sample
            // correspondences, decompose with IPPE-Square (both Necker
            // branches), pick the lower joint-reprojection branch after a
            // 3-step GN polish.
            let Some(seed_pose) = solve_seed_from_sample_homography(&sample, corr, intrinsics)
            else {
                continue;
            };

            // ── Evaluate consensus at the outer threshold ─────────────────
            let (best_outer_mask, outer_count) =
                self.evaluate_inliers(&seed_pose, corr, intrinsics, cfg.tau_outer_sq);
            if outer_count < cfg.min_inliers {
                continue;
            }

            // ── LO inner loop (verification gate) ────────────────────────
            // The unweighted GN pose produced inside lo_inner is discarded
            // (spec mandate) to prevent biasing the AW-LM initialisation.
            // Only tight_count governs global state and dynamic stopping.
            let (_gn_pose, _tight_mask, tight_count) =
                self.lo_inner(seed_pose, &best_outer_mask, corr, intrinsics);

            if tight_count > global_best_tight_count {
                global_best_tight_count = tight_count;
                global_best_seed = Some(seed_pose);

                let inlier_ratio = tight_count as f64 / num_groups as f64;
                if inlier_ratio >= 0.99 {
                    dynamic_k = cfg.k_min;
                } else {
                    let p_fail = 1.0 - cfg.confidence;
                    let p_good_sample = 1.0 - inlier_ratio.powi(4);
                    let k_compute = p_fail.ln() / p_good_sample.ln();
                    dynamic_k = (k_compute.max(0.0).ceil() as u32).clamp(cfg.k_min, cfg.k_max);
                }
            }

            // ── Bounded early termination ─────────────────────────────────
            if iter >= cfg.k_min && iter >= dynamic_k {
                break;
            }
        }

        let final_seed = global_best_seed?;
        Some((final_seed, [0u64; 16]))
    }

    /// LO inner loop: iteratively refines the pose with unweighted Gauss-Newton
    /// and re-evaluates inliers with the tight `tau_inner` threshold.
    ///
    /// **Monotonicity guard:** if the tight inlier count stops improving,
    /// the GN has reached the basin bottom — terminate early.
    fn lo_inner(
        &self,
        seed_pose: Pose,
        outer_mask: &[u64; 16],
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
    ) -> (Pose, [u64; 16], usize) {
        let cfg = &self.lo_ransac;

        let (init_inner_mask, init_inner_count) =
            self.evaluate_inliers(&seed_pose, corr, intrinsics, cfg.tau_inner_sq);

        let mut lo_pose = seed_pose;
        // First GN step uses the outer (wider) inlier mask for better conditioning.
        let mut lo_gn_mask = *outer_mask;
        let mut prev_inner_count = init_inner_count;

        let mut best_pose = seed_pose;
        let mut best_mask = init_inner_mask;
        let mut best_count = init_inner_count;

        for _lo_iter in 0..cfg.lo_max_iterations {
            let new_pose = self.gn_step(&lo_pose, corr, intrinsics, &lo_gn_mask);

            let (new_inner_mask, new_inner_count) =
                self.evaluate_inliers(&new_pose, corr, intrinsics, cfg.tau_inner_sq);

            // Monotonicity guard: tight consensus must strictly grow.
            if new_inner_count <= prev_inner_count {
                break;
            }

            prev_inner_count = new_inner_count;
            lo_pose = new_pose;
            // Subsequent GN steps operate on the tight inlier set.
            lo_gn_mask = new_inner_mask;
            best_pose = new_pose;
            best_mask = new_inner_mask;
            best_count = new_inner_count;
        }

        (best_pose, best_mask, best_count)
    }

    /// Projects all correspondence groups and classifies each group as an
    /// inlier if its mean squared reprojection error is below `tau_sq`.
    ///
    /// The threshold comparison avoids `sqrt`:
    /// `sum_sq < group_size * tau_sq  ⟺  mean_sq < tau_sq`.
    ///
    /// Returns a 1 024-bit group-level inlier bitmask (16 × u64) and the
    /// inlier count.
    ///
    /// If *any* point in a group has near-zero camera depth the whole group is
    /// rejected immediately (break + `valid_group = false`).  This differs from
    /// `gn_step` / `refine_aw_lm` which silently `continue` past degenerate
    /// points: those solvers accumulate whatever non-degenerate corners remain,
    /// whereas the inlier test must be conservative — a partial error sum would
    /// undercount reprojection error and admit a bad pose as an inlier.
    #[allow(clippy::unused_self)]
    fn evaluate_inliers(
        &self,
        pose: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        tau_sq: f64,
    ) -> ([u64; 16], usize) {
        let mut mask = [0u64; 16];
        let mut count = 0usize;
        let num_groups = corr.num_groups();
        let gs = corr.group_size;

        for g in 0..num_groups {
            let start = g * gs;
            let threshold = gs as f64 * tau_sq;

            let mut sum_sq = 0.0f64;
            let mut valid_group = true;

            for k in start..(start + gs) {
                let obj = corr.object_points[k];
                let p_world = Vector3::new(obj[0], obj[1], obj[2]);
                let p_cam = pose.rotation * p_world + pose.translation;
                if p_cam.z < 1e-4 {
                    valid_group = false;
                    break;
                }
                let z_inv = 1.0 / p_cam.z;
                let px = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
                let py = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;
                let dx = px - f64::from(corr.image_points[k].x);
                let dy = py - f64::from(corr.image_points[k].y);
                sum_sq += dx * dx + dy * dy;
            }

            if valid_group && sum_sq < threshold {
                count += 1;
                mask[g / 64] |= 1 << (g % 64);
            }
        }

        (mask, count)
    }

    /// One step of **unweighted** Gauss-Newton pose refinement over inlier groups.
    ///
    /// Solves `(J^T J) δ = J^T r` with the left-perturbation SE(3) Jacobian.
    /// No Marquardt damping, no information-matrix weighting.
    ///
    /// Returns the original pose unchanged if the normal equations are singular.
    #[allow(clippy::unused_self)]
    fn gn_step(
        &self,
        pose: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        inlier_mask: &[u64; 16],
    ) -> Pose {
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        let gs = corr.group_size;
        let num_groups = corr.num_groups();

        for g in 0..num_groups {
            if (inlier_mask[g / 64] & (1 << (g % 64))) == 0 {
                continue;
            }
            let start = g * gs;

            for k in start..(start + gs) {
                let obj = corr.object_points[k];
                let p_world = Vector3::new(obj[0], obj[1], obj[2]);
                let p_cam = pose.rotation * p_world + pose.translation;
                if p_cam.z < 1e-4 {
                    continue;
                }
                let z_inv = 1.0 / p_cam.z;
                let x_z = p_cam.x * z_inv;
                let y_z = p_cam.y * z_inv;

                let u = intrinsics.fx * x_z + intrinsics.cx;
                let v = intrinsics.fy * y_z + intrinsics.cy;

                let res_u = f64::from(corr.image_points[k].x) - u;
                let res_v = f64::from(corr.image_points[k].y) - v;

                // Left-perturbation SE(3) Jacobian (scalar accumulation — no
                // intermediate Matrix2x6; mirrors build_normal_equations in
                // pose_weighted.rs with identity information matrix / w=1).
                let (ju0, ju2, ju3, ju4, ju5, jv1, jv2, jv3, jv4, jv5) =
                    projection_jacobian(x_z, y_z, z_inv, intrinsics);

                jtr[0] += ju0 * res_u;
                jtr[1] += jv1 * res_v;
                jtr[2] += ju2 * res_u + jv2 * res_v;
                jtr[3] += ju3 * res_u + jv3 * res_v;
                jtr[4] += ju4 * res_u + jv4 * res_v;
                jtr[5] += ju5 * res_u + jv5 * res_v;

                jtj[(0, 0)] += ju0 * ju0;
                jtj[(0, 2)] += ju0 * ju2;
                jtj[(0, 3)] += ju0 * ju3;
                jtj[(0, 4)] += ju0 * ju4;
                jtj[(0, 5)] += ju0 * ju5;

                jtj[(1, 1)] += jv1 * jv1;
                jtj[(1, 2)] += jv1 * jv2;
                jtj[(1, 3)] += jv1 * jv3;
                jtj[(1, 4)] += jv1 * jv4;
                jtj[(1, 5)] += jv1 * jv5;

                jtj[(2, 2)] += ju2 * ju2 + jv2 * jv2;
                jtj[(2, 3)] += ju2 * ju3 + jv2 * jv3;
                jtj[(2, 4)] += ju2 * ju4 + jv2 * jv4;
                jtj[(2, 5)] += ju2 * ju5 + jv2 * jv5;

                jtj[(3, 3)] += ju3 * ju3 + jv3 * jv3;
                jtj[(3, 4)] += ju3 * ju4 + jv3 * jv4;
                jtj[(3, 5)] += ju3 * ju5 + jv3 * jv5;

                jtj[(4, 4)] += ju4 * ju4 + jv4 * jv4;
                jtj[(4, 5)] += ju4 * ju5 + jv4 * jv5;

                jtj[(5, 5)] += ju5 * ju5 + jv5 * jv5;
            }
        }
        symmetrize_jtj6(&mut jtj);

        // Solve the normal equations; return original pose if system is singular.
        if let Some(chol) = jtj.cholesky() {
            let delta = chol.solve(&jtr);
            let twist = Vector3::new(delta[3], delta[4], delta[5]);
            let dq = UnitQuaternion::from_scaled_axis(twist);
            Pose {
                rotation: (dq * quat_from_so3(pose.rotation))
                    .to_rotation_matrix()
                    .into_inner(),
                translation: pose.translation + Vector3::new(delta[0], delta[1], delta[2]),
            }
        } else {
            *pose
        }
    }

    /// Per-observation Mahalanobis d² under `corr.information_matrices`.
    /// Returns `None` for observations behind the camera (`z < 1e-4`); the
    /// caller treats those as having zero contribution (matches the
    /// [`Self::refine_aw_lm`] residual block, which `continue`s on the
    /// same condition).
    #[inline]
    fn observation_d2(
        pose: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        k: usize,
    ) -> Option<f64> {
        let obj = corr.object_points[k];
        let p_cam = pose.rotation * Vector3::new(obj[0], obj[1], obj[2]) + pose.translation;
        if p_cam.z < 1e-4 {
            return None;
        }
        let z_inv = 1.0 / p_cam.z;
        let u = intrinsics.fx * (p_cam.x * z_inv) + intrinsics.cx;
        let v = intrinsics.fy * (p_cam.y * z_inv) + intrinsics.cy;
        let res_u = f64::from(corr.image_points[k].x) - u;
        let res_v = f64::from(corr.image_points[k].y) - v;
        let info = corr.information_matrices[k];
        Some(
            res_u * (info[(0, 0)] * res_u + info[(0, 1)] * res_v)
                + res_v * (info[(1, 0)] * res_u + info[(1, 1)] * res_v),
        )
    }

    /// Walks the active groups in `inlier_mask` and returns the top-two
    /// per-observation d² values at `pose`, along with the global
    /// observation index of the maximum. Sentinel `(usize::MAX, 0.0, 0.0)`
    /// ⇒ no active observations.
    #[inline]
    fn top_two_observation_d2(
        pose: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        inlier_mask: &[u64; 16],
    ) -> (usize, f64, f64) {
        let gs = corr.group_size;
        let num_groups = corr.num_groups();

        let mut k_worst = usize::MAX;
        let mut d2_worst = 0.0_f64;
        let mut d2_second = 0.0_f64;

        for g in 0..num_groups {
            if (inlier_mask[g / 64] & (1 << (g % 64))) == 0 {
                continue;
            }
            let start = g * gs;
            for k in start..(start + gs) {
                let Some(d2) = Self::observation_d2(pose, corr, intrinsics, k) else {
                    continue;
                };
                if d2 > d2_worst {
                    d2_second = d2_worst;
                    d2_worst = d2;
                    k_worst = k;
                } else if d2 > d2_second {
                    d2_second = d2;
                }
            }
        }
        (k_worst, d2_worst, d2_second)
    }

    /// Sum of per-observation d² at two candidate poses over the same active
    /// groups in `inlier_mask`, *excluding* `skip_group`. Fused into a
    /// single traversal so the projection-residual block runs once per
    /// observation instead of twice — the self-rejection comparison in
    /// [`Self::apply_outlier_drop_to_board_lm`] needs both sums under the
    /// identical mask, so this is a free win when the drop fires.
    fn kept_group_d2_sums(
        pose_a: &Pose,
        pose_b: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        inlier_mask: &[u64; 16],
        skip_group: usize,
    ) -> (f64, f64) {
        let gs = corr.group_size;
        let num_groups = corr.num_groups();
        let mut sum_a = 0.0_f64;
        let mut sum_b = 0.0_f64;

        for g in 0..num_groups {
            if g == skip_group {
                continue;
            }
            if (inlier_mask[g / 64] & (1 << (g % 64))) == 0 {
                continue;
            }
            let start = g * gs;
            for k in start..(start + gs) {
                if let Some(d2) = Self::observation_d2(pose_a, corr, intrinsics, k) {
                    sum_a += d2;
                }
                if let Some(d2) = Self::observation_d2(pose_b, corr, intrinsics, k) {
                    sum_b += d2;
                }
            }
        }
        (sum_a, sum_b)
    }

    /// After [`Self::refine_aw_lm`] converges, identify the worst observation
    /// (per-observation Mahalanobis d² under the LM's info matrices), and if
    /// it both exceeds `threshold` and dominates the second-worst by ≥ 2×,
    /// mask the *group* containing it, re-run the LM warm-started from the
    /// 4-pose seed, and keep the masked pose iff its kept-group d² sum is
    /// strictly lower than the original pose's.
    ///
    /// Self-rejection compares d² over the *kept* groups (not all
    /// observations): under both isotropic and anisotropic Σ_c the LM
    /// minimises the same weighted residual, so a drop that genuinely
    /// improves the fit on the kept set is safe to commit.
    ///
    /// `threshold ≤ 0.0` short-circuits and returns `None`.
    #[inline]
    fn apply_outlier_drop_to_board_lm(
        &self,
        pose_4: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        inlier_mask: &[u64; 16],
        threshold: f64,
    ) -> Option<(Pose, Matrix6<f64>, usize)> {
        const DOMINANCE_RATIO: f64 = 2.0;

        if threshold <= 0.0 {
            return None;
        }

        let (k_worst, d2_worst, d2_second) =
            Self::top_two_observation_d2(pose_4, corr, intrinsics, inlier_mask);

        if k_worst == usize::MAX || d2_worst <= threshold {
            return None;
        }
        if d2_worst < DOMINANCE_RATIO * d2_second {
            return None;
        }

        // The dropped unit is the *group* containing the worst observation.
        // For AprilGrid (gs=4) this masks an entire tag; for ChAruco saddles
        // (gs=1) it is a single saddle point.
        let g_worst = k_worst / corr.group_size;
        let mut masked_mask = *inlier_mask;
        masked_mask[g_worst / 64] &= !(1u64 << (g_worst % 64));

        let (pose_3, cov_3) = self.refine_aw_lm(pose_4, corr, intrinsics, &masked_mask);

        // Self-rejection over the kept groups under the original
        // (unmasked) info matrices. Both sums are computed in a single
        // traversal — same active mask and skip_group for both poses.
        let (kept_4, kept_3) =
            Self::kept_group_d2_sums(pose_4, &pose_3, corr, intrinsics, inlier_mask, g_worst);

        if kept_3 < kept_4 {
            Some((pose_3, cov_3, g_worst))
        } else {
            None
        }
    }

    /// Final refinement: Anisotropic Weighted Levenberg-Marquardt over the
    /// verified inlier set.
    ///
    /// Uses pre-inverted information matrices from `corr.information_matrices`
    /// and Huber weighting (k = 1.345) for robustness against mild outliers
    /// inside the wide `tau_aw_lm` window.
    #[allow(clippy::too_many_lines, clippy::similar_names, clippy::unused_self)]
    fn refine_aw_lm(
        &self,
        initial_pose: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        inlier_mask: &[u64; 16],
    ) -> (Pose, Matrix6<f64>) {
        let mut pose = *initial_pose;
        let mut lambda = 1e-3;
        let mut nu = 2.0;

        let gs = corr.group_size;
        let num_groups = corr.num_groups();

        let compute_equations = |current_pose: &Pose| -> (f64, Matrix6<f64>, Vector6<f64>) {
            let mut jtj = Matrix6::<f64>::zeros();
            let mut jtr = Vector6::<f64>::zeros();
            let mut total_cost = 0.0;

            for g in 0..num_groups {
                if (inlier_mask[g / 64] & (1 << (g % 64))) == 0 {
                    continue;
                }
                let start = g * gs;

                for k in start..(start + gs) {
                    let obj = corr.object_points[k];
                    let p_world = Vector3::new(obj[0], obj[1], obj[2]);
                    let p_cam = current_pose.rotation * p_world + current_pose.translation;
                    if p_cam.z < 1e-4 {
                        total_cost += 1e6;
                        continue;
                    }
                    let z_inv = 1.0 / p_cam.z;
                    let x_z = p_cam.x * z_inv;
                    let y_z = p_cam.y * z_inv;

                    let u = intrinsics.fx * x_z + intrinsics.cx;
                    let v = intrinsics.fy * y_z + intrinsics.cy;

                    let res_u = f64::from(corr.image_points[k].x) - u;
                    let res_v = f64::from(corr.image_points[k].y) - v;

                    let info = corr.information_matrices[k];

                    let dist_sq = res_u * (info[(0, 0)] * res_u + info[(0, 1)] * res_v)
                        + res_v * (info[(1, 0)] * res_u + info[(1, 1)] * res_v);

                    let huber_k = 1.345;
                    let dist = dist_sq.sqrt();
                    let weight = if dist > huber_k { huber_k / dist } else { 1.0 };
                    total_cost += if dist > huber_k {
                        huber_k * (dist - 0.5 * huber_k)
                    } else {
                        0.5 * dist_sq
                    };

                    let (ju0, ju2, ju3, ju4, ju5, jv1, jv2, jv3, jv4, jv5) =
                        projection_jacobian(x_z, y_z, z_inv, intrinsics);

                    let w00 = info[(0, 0)] * weight;
                    let w01 = info[(0, 1)] * weight;
                    let w10 = info[(1, 0)] * weight;
                    let w11 = info[(1, 1)] * weight;

                    let k00 = ju0 * w00;
                    let k01 = ju0 * w01;
                    let k10 = jv1 * w10;
                    let k11 = jv1 * w11;
                    let k20 = ju2 * w00 + jv2 * w10;
                    let k21 = ju2 * w01 + jv2 * w11;
                    let k30 = ju3 * w00 + jv3 * w10;
                    let k31 = ju3 * w01 + jv3 * w11;
                    let k40 = ju4 * w00 + jv4 * w10;
                    let k41 = ju4 * w01 + jv4 * w11;
                    let k50 = ju5 * w00 + jv5 * w10;
                    let k51 = ju5 * w01 + jv5 * w11;

                    jtr[0] += k00 * res_u + k01 * res_v;
                    jtr[1] += k10 * res_u + k11 * res_v;
                    jtr[2] += k20 * res_u + k21 * res_v;
                    jtr[3] += k30 * res_u + k31 * res_v;
                    jtr[4] += k40 * res_u + k41 * res_v;
                    jtr[5] += k50 * res_u + k51 * res_v;

                    jtj[(0, 0)] += k00 * ju0;
                    jtj[(0, 1)] += k01 * jv1;
                    jtj[(0, 2)] += k00 * ju2 + k01 * jv2;
                    jtj[(0, 3)] += k00 * ju3 + k01 * jv3;
                    jtj[(0, 4)] += k00 * ju4 + k01 * jv4;
                    jtj[(0, 5)] += k00 * ju5 + k01 * jv5;

                    jtj[(1, 1)] += k11 * jv1;
                    jtj[(1, 2)] += k10 * ju2 + k11 * jv2;
                    jtj[(1, 3)] += k10 * ju3 + k11 * jv3;
                    jtj[(1, 4)] += k10 * ju4 + k11 * jv4;
                    jtj[(1, 5)] += k10 * ju5 + k11 * jv5;

                    jtj[(2, 2)] += k20 * ju2 + k21 * jv2;
                    jtj[(2, 3)] += k20 * ju3 + k21 * jv3;
                    jtj[(2, 4)] += k20 * ju4 + k21 * jv4;
                    jtj[(2, 5)] += k20 * ju5 + k21 * jv5;

                    jtj[(3, 3)] += k30 * ju3 + k31 * jv3;
                    jtj[(3, 4)] += k30 * ju4 + k31 * jv4;
                    jtj[(3, 5)] += k30 * ju5 + k31 * jv5;

                    jtj[(4, 4)] += k40 * ju4 + k41 * jv4;
                    jtj[(4, 5)] += k40 * ju5 + k41 * jv5;

                    jtj[(5, 5)] += k50 * ju5 + k51 * jv5;
                }
            }
            symmetrize_jtj6(&mut jtj);
            (total_cost, jtj, jtr)
        };

        let (mut cur_cost, mut cur_jtj, mut cur_jtr) = compute_equations(&pose);
        // See `pose_weighted.rs:refine_pose_lm_weighted_with_info` for
        // rationale: bail out after 3 consecutive Cholesky failures so a
        // persistently-singular damped Hessian doesn't burn MAX_ITERS
        // before triggering the NaN-cov sentinel.
        let mut consecutive_chol_failures: u8 = 0;

        for _iter in 0..20 {
            if cur_jtr.amax() < 1e-8 {
                break;
            }

            let mut jtj_damped = cur_jtj;
            let diag = cur_jtj.diagonal();
            for i in 0..6 {
                jtj_damped[(i, i)] += lambda * (diag[i] + 1e-6);
            }

            if let Some(chol) = jtj_damped.cholesky() {
                consecutive_chol_failures = 0;
                let delta = chol.solve(&cur_jtr);
                let twist = Vector3::new(delta[3], delta[4], delta[5]);
                let dq = UnitQuaternion::from_scaled_axis(twist);
                let new_pose = Pose {
                    rotation: (dq * quat_from_so3(pose.rotation))
                        .to_rotation_matrix()
                        .into_inner(),
                    translation: pose.translation + Vector3::new(delta[0], delta[1], delta[2]),
                };

                let (new_cost, new_jtj, new_jtr) = compute_equations(&new_pose);
                let rho = (cur_cost - new_cost)
                    / (0.5 * delta.dot(&(lambda * delta + cur_jtr)).max(1e-12));

                if rho > 0.0 {
                    pose = new_pose;
                    cur_cost = new_cost;
                    cur_jtj = new_jtj;
                    cur_jtr = new_jtr;
                    lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
                    nu = 2.0;
                    if delta.norm() < 1e-7 {
                        break;
                    }
                } else {
                    lambda *= nu;
                    nu *= 2.0;
                }
            } else {
                consecutive_chol_failures += 1;
                if consecutive_chol_failures >= 3 {
                    break;
                }
                lambda *= 10.0;
            }
        }

        // Singular OR near-singular `JᵀWJ` (e.g. degenerate inlier geometry
        // that the LO-RANSAC gate failed to reject) makes the
        // inverse-Hessian covariance ill-defined. `try_inverse` only
        // catches EXACT singularity (LU zero pivot); near-singular Hessians
        // invert to finite-but-absurd matrices (~1e15-1e30). Emit a
        // NaN-filled sentinel in both cases; downstream consumers branch
        // on `cov[(0,0)].is_nan()`.
        let covariance = match cur_jtj.try_inverse() {
            Some(inv) if inv.iter().all(|v| v.is_finite()) => inv,
            _ => Matrix6::from_element(f64::NAN),
        };
        (pose, covariance)
    }
}

// ── Board Estimator (AprilGrid adapter) ────────────────────────────────────

/// Estimator for multi-tag AprilGrid board poses.
///
/// Bridges the [`crate::batch::DetectionBatch`] SoA layout and [`AprilGridTopology`] marker
/// geometry with the tag-layout-agnostic [`RobustPoseSolver`].  All heavy pose
/// mathematics lives in the solver; this struct is responsible only for
/// constructing the flat [`PointCorrespondences`] view and retaining the
/// pre-allocated scratch buffers needed to do so without heap allocation.
pub struct BoardEstimator {
    /// Configuration of the board layout.
    pub config: Arc<AprilGridTopology>,
    /// The underlying robust pose solver (contains LO-RANSAC config).
    pub solver: RobustPoseSolver,
    // ── Pre-allocated scratch buffers (single heap allocation in new()) ──────
    // img/obj/info are per-point: MAX_CORR = MAX_CANDIDATES × CORNERS_PER_TAG.
    //
    // No per-group seed buffer: the solver constructs minimal-sample seeds
    // by enumerating both Necker branches of IPPE-Square for each tag in the
    // 4-tag minimal sample and scoring them by joint reprojection (see
    // [`solve_seed_from_sample_homography`] / [`RobustPoseSolver::lo_ransac_loop`]).
    scratch_img: Box<[Point2f]>,
    scratch_obj: Box<[[f64; 3]]>,
    scratch_info: Box<[Matrix2<f64>]>,
}

impl BoardEstimator {
    const CORNERS_PER_TAG: usize = 4;
    const MAX_CORR: usize = MAX_CANDIDATES * Self::CORNERS_PER_TAG;

    /// Creates a new `BoardEstimator`.
    ///
    /// Performs a single one-time heap allocation to back the scratch buffers.
    /// Reuse the same `BoardEstimator` across frames to amortise this cost and
    /// guarantee zero per-`estimate()` allocations.
    ///
    /// `config` is wrapped in [`Arc`] so multiple estimators or frames can share
    /// the same board geometry without cloning the marker table.
    #[must_use]
    pub fn new(config: Arc<AprilGridTopology>) -> Self {
        Self {
            config,
            solver: RobustPoseSolver::new(),
            scratch_img: vec![Point2f { x: 0.0, y: 0.0 }; Self::MAX_CORR].into_boxed_slice(),
            scratch_obj: vec![[0.0f64; 3]; Self::MAX_CORR].into_boxed_slice(),
            scratch_info: vec![Matrix2::zeros(); Self::MAX_CORR].into_boxed_slice(),
        }
    }

    /// Builder: override the LO-RANSAC configuration.
    #[must_use]
    pub fn with_lo_ransac_config(mut self, cfg: LoRansacConfig) -> Self {
        self.solver.lo_ransac = cfg;
        self
    }

    // ── Public entry point ───────────────────────────────────────────────

    /// Estimates the board pose from a batch of detections.
    ///
    /// Returns `None` if fewer than 4 valid tags match the board layout or if
    /// LO-RANSAC cannot find a consensus set.
    ///
    /// `outlier_drop_d2_threshold > 0.0` enables the post-LM outlier-aware
    /// drop step on the AW-LM result. Pass `0.0` to disable. Should be
    /// sourced from `DetectorConfig::outlier_drop_d2_threshold` so the
    /// profile-level opt-in stays consistent with the per-tag pose path.
    #[must_use]
    pub fn estimate(
        &mut self,
        batch: &crate::batch::DetectionBatchView<'_>,
        intrinsics: &CameraIntrinsics,
        outlier_drop_d2_threshold: f64,
    ) -> Option<BoardPose> {
        // Phase 1: flatten valid batch entries into the pre-allocated scratch
        // slices and invert per-corner covariances into information matrices.
        let num_groups = self.flatten_batch(batch);
        if num_groups < 4 {
            return None;
        }

        let m = num_groups * Self::CORNERS_PER_TAG;
        let corr = PointCorrespondences {
            image_points: &self.scratch_img[..m],
            object_points: &self.scratch_obj[..m],
            information_matrices: &self.scratch_info[..m],
            group_size: Self::CORNERS_PER_TAG,
        };

        // Phase 2–4: LO-RANSAC → GN verification → AW-LM refinement
        // (+ optional outlier-aware drop when threshold > 0.0).
        self.solver
            .estimate(&corr, intrinsics, outlier_drop_d2_threshold)
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Scans the batch and writes all valid, board-matched tag data into the
    /// pre-allocated scratch buffers.
    ///
    /// Returns the number of tag groups written (i.e. the value of `num_groups`
    /// for the subsequent `PointCorrespondences`).
    fn flatten_batch(&mut self, batch: &crate::batch::DetectionBatchView<'_>) -> usize {
        let mut g = 0usize;

        for i in 0..batch.len() {
            let id = batch.ids[i] as usize;
            if id >= self.config.obj_points.len() {
                continue;
            }
            let Some(obj) = self.config.obj_points[id] else {
                continue;
            };

            let base = g * Self::CORNERS_PER_TAG;
            for (j, obj_pt) in obj.iter().enumerate() {
                self.scratch_img[base + j] = batch.corners[i][j];
                self.scratch_obj[base + j] = *obj_pt;
                // Invert the per-corner covariance into an information matrix.
                // Fall back to identity if the covariance is singular.
                self.scratch_info[base + j] = Matrix2::new(
                    f64::from(batch.corner_covariances[i][j * 4]),
                    f64::from(batch.corner_covariances[i][j * 4 + 1]),
                    f64::from(batch.corner_covariances[i][j * 4 + 2]),
                    f64::from(batch.corner_covariances[i][j * 4 + 3]),
                )
                .try_inverse()
                .unwrap_or_else(Matrix2::identity);
            }

            g += 1;
        }

        g
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    missing_docs
)]
mod tests {
    use super::*;
    use crate::batch::{CandidateState, DetectionBatch, DetectionBatchView, Point2f};
    use nalgebra::Matrix3;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Standard synthetic intrinsics used across tests.
    fn test_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0)
    }

    /// Collect all corner points from an obj_points slice into a flat `Vec`.
    fn all_corners(obj_points: &[Option<[[f64; 3]; 4]>]) -> Vec<[f64; 3]> {
        obj_points
            .iter()
            .filter_map(|opt| *opt)
            .flat_map(std::iter::IntoIterator::into_iter)
            .collect()
    }

    /// Compute the centroid of a set of 3D points.
    fn centroid(pts: &[[f64; 3]]) -> [f64; 3] {
        let n = pts.len() as f64;
        let (sx, sy, sz) = pts.iter().fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
            (ax + p[0], ay + p[1], az + p[2])
        });
        [sx / n, sy / n, sz / n]
    }

    /// Build a `DetectionBatch` by projecting every marker in `obj_points` through
    /// `pose` and `intrinsics`.  Corners are stored as f32 (matching `Point2f`);
    /// per-tag pose data encodes the camera-space position of each tag's geometric center
    /// and the board rotation quaternion (historically consumed by per-tag seed bridges;
    /// now consumed only by sanity tests that still inspect `Pose6D`).
    /// Identity corner covariances → isotropic AW-LM weighting.
    fn build_synthetic_batch(
        obj_points: &[Option<[[f64; 3]; 4]>],
        pose: &Pose,
        intrinsics: &CameraIntrinsics,
    ) -> (DetectionBatch, usize) {
        let mut batch = DetectionBatch::new();
        let mut n = 0usize;

        let q = quat_from_so3(pose.rotation);

        for (tag_id, opt_pts) in obj_points.iter().enumerate() {
            let Some(obj) = opt_pts else { continue };

            for (j, pt) in obj.iter().enumerate() {
                let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                let proj = pose.project(&p_world, intrinsics);
                batch.corners[n][j] = Point2f {
                    x: proj[0] as f32,
                    y: proj[1] as f32,
                };
            }

            // Center-origin convention: encode the tag's geometric center in camera frame.
            let center = Vector3::new(
                (obj[0][0] + obj[2][0]) * 0.5,
                (obj[0][1] + obj[2][1]) * 0.5,
                (obj[0][2] + obj[2][2]) * 0.5,
            );
            let det_t = pose.rotation * center + pose.translation;
            batch.poses[n].data = [
                det_t.x as f32,
                det_t.y as f32,
                det_t.z as f32,
                q.i as f32,
                q.j as f32,
                q.k as f32,
                q.w as f32,
            ];

            for j in 0..4 {
                // corner_covariances[n] is 4 × 2×2 matrices packed row-major (16 f32).
                // j*4 = (0,0), j*4+3 = (1,1) — set identity for each corner.
                batch.corner_covariances[n][j * 4] = 1.0;
                batch.corner_covariances[n][j * 4 + 3] = 1.0;
            }

            batch.ids[n] = tag_id as u32;
            batch.status_mask[n] = CandidateState::Valid;
            n += 1;
        }

        (batch, n)
    }

    /// Flatten the first `num_valid` batch entries into `PointCorrespondences`
    /// buffers using unit information matrices.
    ///
    /// Returns the three backing `Vec`s so the caller can keep them alive for the
    /// lifetime of the `PointCorrespondences` view.
    #[allow(clippy::type_complexity)]
    fn build_correspondences_from_batch(
        obj_points: &[Option<[[f64; 3]; 4]>],
        view: &DetectionBatchView<'_>,
        _estimator: &BoardEstimator,
    ) -> (Vec<Point2f>, Vec<[f64; 3]>, Vec<Matrix2<f64>>) {
        let num_valid = view.len();
        let mut img = Vec::with_capacity(num_valid * 4);
        let mut obj = Vec::with_capacity(num_valid * 4);
        let mut info = Vec::with_capacity(num_valid * 4);

        for b_idx in 0..num_valid {
            let id = view.ids[b_idx] as usize;
            let pts = obj_points[id].unwrap();
            for (j, &obj_pt) in pts.iter().enumerate() {
                img.push(view.corners[b_idx][j]);
                obj.push(obj_pt);
                info.push(Matrix2::identity());
            }
        }

        (img, obj, info)
    }

    /// Per-corner mean squared reprojection error (in pixel²) for the first
    /// `num_valid` candidates in the batch.
    fn mean_reprojection_sq(
        pose: &Pose,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
        obj_points: &[Option<[[f64; 3]; 4]>],
        num_valid: usize,
    ) -> f64 {
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for i in 0..num_valid {
            let id = batch.ids[i] as usize;
            let obj = obj_points[id].unwrap();
            for (j, pt) in obj.iter().enumerate() {
                let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                let proj = pose.project(&p_world, intrinsics);
                let dx = proj[0] - f64::from(batch.corners[i][j].x);
                let dy = proj[1] - f64::from(batch.corners[i][j].y);
                sum_sq += dx * dx + dy * dy;
                count += 1;
            }
        }
        sum_sq / count.max(1) as f64
    }

    // ── Board layout tests ────────────────────────────────────────────────────

    #[test]
    fn test_charuco_board_marker_count() {
        // 6×6 grid: markers appear in cells where (r+c) is even → exactly 18 markers.
        let config = CharucoTopology::new(6, 6, 0.1, 0.08, usize::MAX).unwrap();
        let count = config.obj_points.iter().filter(|o| o.is_some()).count();
        assert_eq!(count, 18);
    }

    #[test]
    fn test_charuco_board_centroid_is_origin() {
        // For a symmetric ChAruco board the geometric centroid of all marker corners
        // must coincide with the board coordinate origin.
        let config = CharucoTopology::new(6, 6, 0.1, 0.08, usize::MAX).unwrap();
        let pts = all_corners(&config.obj_points);
        assert!(!pts.is_empty());
        let c = centroid(&pts);
        assert!(c[0].abs() < 1e-9, "centroid x = {}", c[0]);
        assert!(c[1].abs() < 1e-9, "centroid y = {}", c[1]);
        assert!(c[2].abs() < 1e-9, "all points must lie on z = 0");
    }

    #[test]
    fn test_charuco_corner_order_tl_tr_br_bl() {
        // For every marker the corners must follow the [TL, TR, BR, BL] order:
        //   TL.x < TR.x, TR.x == BR.x, BL.x == TL.x, BL.y > TL.y
        let config = CharucoTopology::new(4, 4, 0.1, 0.08, usize::MAX).unwrap();
        for opt in &config.obj_points {
            let [tl, tr, br, bl] = opt.unwrap();
            assert!(tl[0] < tr[0], "TL.x must be left of TR.x");
            assert!(
                (tl[1] - tr[1]).abs() < 1e-9,
                "TL and TR must share the same y"
            );
            assert!(
                (tr[0] - br[0]).abs() < 1e-9,
                "TR and BR must share the same x"
            );
            assert!(
                (bl[0] - tl[0]).abs() < 1e-9,
                "BL and TL must share the same x"
            );
            assert!(
                bl[1] > tl[1],
                "BL.y must be below TL.y (y increases downward)"
            );
            assert!(
                (bl[1] - br[1]).abs() < 1e-9,
                "BL and BR must share the same y"
            );
            for pt in &[tl, tr, br, bl] {
                assert!(pt[2].abs() < 1e-9, "all corners must lie on z = 0");
            }
        }
    }

    #[test]
    fn test_charuco_marker_size_matches_config() {
        // Each marker's width and height must equal `marker_length`.
        let marker_length = 0.08;
        let config = CharucoTopology::new(4, 4, 0.1, marker_length, usize::MAX).unwrap();
        for opt in &config.obj_points {
            let [tl, tr, _br, bl] = opt.unwrap();
            let width = tr[0] - tl[0];
            let height = bl[1] - tl[1];
            assert!((width - marker_length).abs() < 1e-9, "marker width {width}");
            assert!(
                (height - marker_length).abs() < 1e-9,
                "marker height {height}"
            );
        }
    }

    #[test]
    fn test_charuco_board_no_marker_overlap() {
        // No two markers may share a corner position.
        let config = CharucoTopology::new(4, 4, 0.1, 0.08, usize::MAX).unwrap();
        let mut corners: Vec<[f64; 3]> = config
            .obj_points
            .iter()
            .filter_map(|o| *o)
            .flat_map(IntoIterator::into_iter)
            .collect();
        corners.sort_by(|a, b| {
            a[0].partial_cmp(&b[0])
                .unwrap()
                .then(a[1].partial_cmp(&b[1]).unwrap())
        });
        for w in corners.windows(2) {
            assert!(
                (w[0][0] - w[1][0]).abs() > 1e-9 || (w[0][1] - w[1][1]).abs() > 1e-9,
                "duplicate corner: {:?}",
                w[0]
            );
        }
    }

    #[test]
    fn test_aprilgrid_board_marker_count() {
        // AprilGrid: every cell has a marker.
        let config = AprilGridTopology::new(4, 4, 0.01, 0.1, usize::MAX).unwrap();
        let count = config.obj_points.iter().filter(|o| o.is_some()).count();
        assert_eq!(count, 16);
    }

    #[test]
    fn test_aprilgrid_board_centroid_is_origin() {
        let config = AprilGridTopology::new(6, 6, 0.01, 0.1, usize::MAX).unwrap();
        let pts = all_corners(&config.obj_points);
        let c = centroid(&pts);
        assert!(c[0].abs() < 1e-9, "centroid x = {}", c[0]);
        assert!(c[1].abs() < 1e-9, "centroid y = {}", c[1]);
    }

    #[test]
    fn test_aprilgrid_adjacent_marker_step() {
        // Adjacent markers in the same row must be separated by marker_length + spacing.
        let marker_length = 0.1;
        let spacing = 0.02;
        let config = AprilGridTopology::new(2, 3, spacing, marker_length, usize::MAX).unwrap();
        let step = marker_length + spacing;

        let tl0 = config.obj_points[0].unwrap()[0];
        let tl1 = config.obj_points[1].unwrap()[0];
        assert!(
            (tl1[0] - tl0[0] - step).abs() < 1e-9,
            "expected step {step}, got {}",
            tl1[0] - tl0[0]
        );

        let tl_r0 = config.obj_points[0].unwrap()[0];
        let tl_r1 = config.obj_points[3].unwrap()[0]; // row 1, col 0 → index = 1*3+0 = 3
        assert!(
            (tl_r1[1] - tl_r0[1] - step).abs() < 1e-9,
            "expected row step {step}, got {}",
            tl_r1[1] - tl_r0[1]
        );
    }

    // ── Mathematical correctness tests ────────────────────────────────────────

    // AprilGrid config shared across the solver/estimator math tests.
    fn math_test_config() -> Arc<AprilGridTopology> {
        Arc::new(AprilGridTopology::new(4, 4, 0.02, 0.08, usize::MAX).unwrap())
    }

    #[test]
    fn test_evaluate_inliers_perfect_pose_all_inliers() {
        // Under the exact ground-truth pose the reprojection error is sub-pixel
        // (limited only by f32 quantisation ≈ 1e-5 px); tau_sq = 1.0 must admit all tags.
        let config = math_test_config();
        let estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, num_valid) = build_synthetic_batch(&config.obj_points, &pose, &intrinsics);
        let v = batch.partition(num_valid);
        let view = batch.view(v);

        let (img, obj, info) =
            build_correspondences_from_batch(&config.obj_points, &view, &estimator);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };

        let solver = RobustPoseSolver::new();
        let (_, count) = solver.evaluate_inliers(&pose, &corr, &intrinsics, 1.0);
        assert_eq!(count, v, "all tags must be inliers under perfect pose");
    }

    #[test]
    fn test_evaluate_inliers_bad_pose_no_inliers() {
        // A pose shifted 0.5 m in X produces large reprojection error;
        // even the generous tau_sq = 100 must reject all tags.
        let config = math_test_config();
        let estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, num_valid) =
            build_synthetic_batch(&config.obj_points, &true_pose, &intrinsics);
        let v = batch.partition(num_valid);
        let view = batch.view(v);

        let bad_pose = Pose::new(Matrix3::identity(), Vector3::new(0.5, 0.0, 1.0));
        let (img, obj, info) =
            build_correspondences_from_batch(&config.obj_points, &view, &estimator);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };

        let solver = RobustPoseSolver::new();
        let (_, count) = solver.evaluate_inliers(&bad_pose, &corr, &intrinsics, 100.0);
        assert_eq!(
            count, 0,
            "no tags should survive under a heavily shifted pose"
        );
    }

    #[test]
    fn test_evaluate_inliers_inlier_mask_consistency() {
        // The bitmask returned by evaluate_inliers must have exactly `count` bits set.
        let config = math_test_config();
        let estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, num_valid) = build_synthetic_batch(&config.obj_points, &pose, &intrinsics);
        let v = batch.partition(num_valid);
        let view = batch.view(v);

        let (img, obj, info) =
            build_correspondences_from_batch(&config.obj_points, &view, &estimator);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };

        let solver = RobustPoseSolver::new();
        let (mask, count) = solver.evaluate_inliers(&pose, &corr, &intrinsics, 1.0);

        let bits_set: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        assert_eq!(
            bits_set, count,
            "bitmask popcount must equal reported count"
        );
    }

    #[test]
    fn test_gn_step_reduces_reprojection_error() {
        // A single unweighted Gauss-Newton step from a 2 cm offset must strictly
        // reduce the mean squared reprojection error.
        let config = math_test_config();
        let estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, num_valid) =
            build_synthetic_batch(&config.obj_points, &true_pose, &intrinsics);
        let v = batch.partition(num_valid);
        let view = batch.view(v);

        let perturbed = Pose::new(Matrix3::identity(), Vector3::new(0.02, 0.0, 1.0));
        let (img, obj, info) =
            build_correspondences_from_batch(&config.obj_points, &view, &estimator);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };
        let all_inliers = [u64::MAX; 16];

        let solver = RobustPoseSolver::new();
        let before = mean_reprojection_sq(&perturbed, &batch, &intrinsics, &config.obj_points, v);
        let stepped = solver.gn_step(&perturbed, &corr, &intrinsics, &all_inliers);
        let after = mean_reprojection_sq(&stepped, &batch, &intrinsics, &config.obj_points, v);

        assert!(
            after < before,
            "GN step must reduce error: {before:.6} → {after:.6} px²"
        );
    }

    #[test]
    fn test_gn_step_singular_returns_original() {
        // With no inliers the normal equations are all-zero (singular);
        // gn_step must return the input pose unchanged.
        let config = math_test_config();
        let estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, num_valid) = build_synthetic_batch(&config.obj_points, &pose, &intrinsics);
        let v = batch.partition(num_valid);
        let view = batch.view(v);

        let (img, obj, info) =
            build_correspondences_from_batch(&config.obj_points, &view, &estimator);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };
        let no_inliers = [0u64; 16];

        let solver = RobustPoseSolver::new();
        let result = solver.gn_step(&pose, &corr, &intrinsics, &no_inliers);
        assert!(
            (result.translation - pose.translation).norm() < 1e-12,
            "pose must be unchanged when normal equations are singular"
        );
    }

    #[test]
    fn test_refine_aw_lm_converges_from_small_offset() {
        // AW-LM from a 2 cm / 1 cm offset must converge to within 0.1 mm.
        let config = math_test_config();
        let estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, num_valid) =
            build_synthetic_batch(&config.obj_points, &true_pose, &intrinsics);
        let v = batch.partition(num_valid);
        let view = batch.view(v);

        let perturbed = Pose::new(Matrix3::identity(), Vector3::new(0.02, -0.01, 1.0));
        let (img, obj, info) =
            build_correspondences_from_batch(&config.obj_points, &view, &estimator);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };
        let all_inliers = [u64::MAX; 16];

        let solver = RobustPoseSolver::new();
        let (refined, cov) = solver.refine_aw_lm(&perturbed, &corr, &intrinsics, &all_inliers);

        let t_error = (refined.translation - true_pose.translation).norm();
        assert!(
            t_error < 1e-4,
            "translation error {t_error} m exceeds 0.1 mm"
        );

        for i in 0..6 {
            assert!(
                cov[(i, i)] >= 0.0,
                "covariance diagonal [{i},{i}] must be non-negative"
            );
        }
    }

    #[test]
    fn test_refine_aw_lm_covariance_is_symmetric() {
        // The returned covariance (J^T J)^{-1} must be symmetric.
        let config = math_test_config();
        let estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, num_valid) = build_synthetic_batch(&config.obj_points, &pose, &intrinsics);
        let v = batch.partition(num_valid);
        let view = batch.view(v);

        let (img, obj, info) =
            build_correspondences_from_batch(&config.obj_points, &view, &estimator);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };
        let all_inliers = [u64::MAX; 16];

        let solver = RobustPoseSolver::new();
        let (_, cov) = solver.refine_aw_lm(&pose, &corr, &intrinsics, &all_inliers);

        for i in 0..6 {
            for j in (i + 1)..6 {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < 1e-12,
                    "covariance must be symmetric: [{i},{j}]={} ≠ [{j},{i}]={}",
                    cov[(i, j)],
                    cov[(j, i)]
                );
            }
        }
    }

    /// Empty inlier mask drives `JᵀWJ` to all-zero (rank 0). The covariance
    /// step must then produce a NaN-filled sentinel rather than the legacy
    /// zero matrix that falsely advertised a perfectly known pose.
    #[test]
    fn test_refine_aw_lm_singular_returns_nan_covariance() {
        let config = math_test_config();
        let estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, num_valid) = build_synthetic_batch(&config.obj_points, &pose, &intrinsics);
        let v = batch.partition(num_valid);
        let view = batch.view(v);

        let (img, obj, info) =
            build_correspondences_from_batch(&config.obj_points, &view, &estimator);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };
        // All-zero mask ⇒ zero `JᵀWJ` ⇒ singular.
        let no_inliers = [0u64; 16];

        let solver = RobustPoseSolver::new();
        let (_, cov) = solver.refine_aw_lm(&pose, &corr, &intrinsics, &no_inliers);

        assert!(
            cov[(0, 0)].is_nan(),
            "expected NaN-sentinel covariance for singular JᵀWJ, got cov[0,0]={}",
            cov[(0, 0)]
        );
        for r in 0..6 {
            for c in 0..6 {
                assert!(
                    cov[(r, c)].is_nan(),
                    "all entries must be NaN: cov[{r},{c}]={}",
                    cov[(r, c)]
                );
            }
        }
    }

    #[test]
    fn test_estimate_none_with_fewer_than_four_valid_tags() {
        // estimate() must return None when fewer than 4 board-matched tags are present.
        let config = math_test_config();
        let mut estimator = BoardEstimator::new(config);
        let intrinsics = test_intrinsics();

        for n_valid in 0..4 {
            let mut batch = DetectionBatch::new();
            for i in 0..n_valid {
                batch.ids[i] = i as u32;
                batch.status_mask[i] = CandidateState::Valid;
            }
            let v = batch.partition(n_valid);
            assert!(
                estimator
                    .estimate(&batch.view(v), &intrinsics, 0.0)
                    .is_none(),
                "expected None with {n_valid} valid tags"
            );
        }
    }

    #[test]
    fn test_estimate_end_to_end_recovers_translation() {
        // End-to-end: synthesise all markers of a 4×4 AprilGrid from a known pose
        // and verify that estimate() recovers the pose to within 1 mm / 0.1°.
        let config = math_test_config();
        let mut estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.05, -0.03, 1.5));
        let (mut batch, n) = build_synthetic_batch(&config.obj_points, &true_pose, &intrinsics);
        let v = batch.partition(n);

        let result = estimator.estimate(&batch.view(v), &intrinsics, 0.0);
        assert!(
            result.is_some(),
            "estimate() must succeed with all tags visible"
        );

        let board_pose = result.unwrap();
        let t_error = (board_pose.pose.translation - true_pose.translation).norm();
        assert!(t_error < 1e-3, "translation error {t_error} m exceeds 1 mm");

        let est_q = quat_from_so3(board_pose.pose.rotation);
        let true_q = quat_from_so3(true_pose.rotation);
        let r_error = est_q.angle_to(&true_q).to_degrees();
        assert!(r_error < 0.1, "rotation error {r_error}° exceeds 0.1°");
    }

    #[test]
    fn test_estimate_covariance_is_positive_definite() {
        // The covariance returned alongside a valid estimate must have a positive diagonal.
        let config = math_test_config();
        let mut estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (mut batch, n) = build_synthetic_batch(&config.obj_points, &pose, &intrinsics);
        let v = batch.partition(n);

        let result = estimator
            .estimate(&batch.view(v), &intrinsics, 0.0)
            .unwrap();
        for i in 0..6 {
            assert!(
                result.covariance[(i, i)] > 0.0,
                "covariance diagonal [{i},{i}] must be positive"
            );
        }
    }

    // ── Outlier-aware drop tests ──────────────────────────────────────────────

    /// Single-tag catastrophic outlier on a clean board: the drop should fire,
    /// mask the bad tag, and bring the pose closer to ground truth than the
    /// 4-corner LM that includes the outlier.
    ///
    /// The injected outlier magnitude is bracketed by two thresholds:
    ///   - [`LoRansacConfig::default`]'s `tau_aw_lm_sq` (per-group sum_sq
    ///     gate, multiplied by `group_size = 4` ⇒ 400 px²) — must NOT
    ///     exceed, or the bad tag is filtered before reaching the LM.
    ///   - The outlier-drop d² threshold (5σ² = 25.0 under the identity
    ///     info matrices `build_synthetic_batch` writes) — must exceed,
    ///     so the post-LM trigger fires.
    ///
    /// Both ends are derived constants so the test can't silently degrade
    /// if `LoRansacConfig` evolves.
    #[test]
    fn outlier_drop_fires_on_single_dominant_board_outlier() {
        // Derived from `LoRansacConfig::default().tau_aw_lm_sq` × `gs=4`.
        let inlier_group_threshold = LoRansacConfig::default().tau_aw_lm_sq * 4.0;
        const DROP_TRIGGER_D2: f64 = 25.0;
        // Pick an outlier magnitude squarely between the two gates.
        let outlier_px = (DROP_TRIGGER_D2.sqrt() + inlier_group_threshold.sqrt()) * 0.5;
        let outlier_d2 = outlier_px * outlier_px;
        assert!(
            outlier_d2 > DROP_TRIGGER_D2 && outlier_d2 < inlier_group_threshold,
            "test invariant: outlier must trigger drop yet survive inlier gate",
        );

        let config = math_test_config();
        let mut estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.05, -0.03, 1.5));
        let (mut batch, n) = build_synthetic_batch(&config.obj_points, &true_pose, &intrinsics);

        let outlier_tag = 0usize;
        assert!(batch.status_mask[outlier_tag] == CandidateState::Valid);
        batch.corners[outlier_tag][0].x += outlier_px as f32;

        let v = batch.partition(n);

        // Baseline (no outlier drop): the LM keeps the bad corner and biases
        // the pose. Threshold = 0.0 short-circuits the mechanism.
        let baseline = estimator
            .estimate(&batch.view(v), &intrinsics, 0.0)
            .expect("baseline pose must converge");

        // With outlier drop enabled, the mechanism should mask the bad
        // tag and the resulting pose should be closer to ground truth.
        let mut estimator2 = BoardEstimator::new(Arc::clone(&config));
        let dropped = estimator2
            .estimate(&batch.view(v), &intrinsics, DROP_TRIGGER_D2)
            .expect("outlier-drop pose must converge");

        let baseline_err = (baseline.pose.translation - true_pose.translation).norm();
        let dropped_err = (dropped.pose.translation - true_pose.translation).norm();
        assert!(
            dropped_err < baseline_err,
            "outlier drop must improve translation error: dropped={dropped_err}, baseline={baseline_err}",
        );
    }

    /// Clean observations (no outliers): drop must not fire, pose must be
    /// byte-identical to the threshold=0.0 baseline.
    #[test]
    fn outlier_drop_skips_clean_board() {
        let config = math_test_config();
        let mut estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.5));
        let (mut batch, n) = build_synthetic_batch(&config.obj_points, &pose, &intrinsics);
        let v = batch.partition(n);

        let baseline = estimator
            .estimate(&batch.view(v), &intrinsics, 0.0)
            .expect("baseline pose must converge");

        let mut estimator2 = BoardEstimator::new(Arc::clone(&config));
        let with_drop = estimator2
            .estimate(&batch.view(v), &intrinsics, 25.0)
            .expect("pose must converge");

        // On a noise-free synthetic board the drop trigger should not fire.
        assert!(
            (baseline.pose.translation - with_drop.pose.translation).norm() < 1e-12,
            "drop must not fire on clean observations",
        );
    }

    /// Two comparable outliers fail the dominance check (worst < 2× second-worst)
    /// — drop is suppressed, pose matches the threshold=0.0 baseline.
    #[test]
    fn outlier_drop_dominance_rejects_two_outliers() {
        let config = math_test_config();
        let mut estimator = BoardEstimator::new(Arc::clone(&config));
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.5));
        let (mut batch, n) = build_synthetic_batch(&config.obj_points, &true_pose, &intrinsics);

        // Inject two comparable +20 px outliers on two different tags.
        // worst d² and second-worst d² are now of similar magnitude — the
        // 2× dominance ratio should reject the drop.
        batch.corners[0][0].x += 20.0;
        batch.corners[1][0].x += 20.0;

        let v = batch.partition(n);

        let baseline = estimator
            .estimate(&batch.view(v), &intrinsics, 0.0)
            .expect("baseline pose must converge");

        let mut estimator2 = BoardEstimator::new(Arc::clone(&config));
        let with_drop = estimator2
            .estimate(&batch.view(v), &intrinsics, 25.0)
            .expect("pose must converge");

        // With two comparable outliers and the 2× dominance gate, the drop
        // must NOT fire — pose must equal the baseline.
        assert!(
            (baseline.pose.translation - with_drop.pose.translation).norm() < 1e-12,
            "dominance check must reject ambiguous-outlier drops",
        );
    }

    // ── ChAruco saddle topology tests ─────────────────────────────────────────

    #[test]
    fn test_charuco_saddle_count() {
        // A 6×6 square grid has (6-1)×(6-1) = 25 interior corners.
        let config = CharucoTopology::new(6, 6, 0.04, 0.03, usize::MAX).unwrap();
        assert_eq!(config.saddle_points.len(), 25);
    }

    #[test]
    fn test_charuco_saddle_coords_on_grid() {
        // Saddle points must lie exactly on the integer-multiple square grid.
        let sq = 0.04_f64;
        let config = CharucoTopology::new(4, 4, sq, 0.03, usize::MAX).unwrap();
        let offset_x = -4.0 * sq / 2.0;
        let offset_y = -4.0 * sq / 2.0;
        let s0 = config.saddle_points[0];
        assert!(
            (s0[0] - (offset_x + sq)).abs() < 1e-12,
            "saddle x: {}",
            s0[0]
        );
        assert!(
            (s0[1] - (offset_y + sq)).abs() < 1e-12,
            "saddle y: {}",
            s0[1]
        );
        assert!(s0[2].abs() < 1e-12, "saddle z must be 0");
        let saddle_cols = 3usize;
        let s = config.saddle_points[saddle_cols + 2];
        assert!(
            (s[0] - (offset_x + 3.0 * sq)).abs() < 1e-12,
            "saddle x: {}",
            s[0]
        );
        assert!(
            (s[1] - (offset_y + 2.0 * sq)).abs() < 1e-12,
            "saddle y: {}",
            s[1]
        );
    }

    #[test]
    fn test_charuco_saddle_adjacency_interior_marker() {
        // 6×6 board, marker at (r=2, c=2) has index 7 when counting (r+c)%2==0 cells.
        // All 4 adjacent saddles must be non-None for this interior marker.
        let config = CharucoTopology::new(6, 6, 0.04, 0.03, usize::MAX).unwrap();
        let adj = config.tag_cell_corners[7];
        assert!(adj[0].is_some(), "TL saddle of interior marker must exist");
        assert!(adj[1].is_some(), "TR saddle of interior marker must exist");
        assert!(adj[2].is_some(), "BR saddle of interior marker must exist");
        assert!(adj[3].is_some(), "BL saddle of interior marker must exist");
    }

    #[test]
    fn test_charuco_saddle_adjacency_corner_marker() {
        // Marker at (r=0, c=0) → only the BR corner of the square is an interior saddle.
        let config = CharucoTopology::new(4, 4, 0.04, 0.03, usize::MAX).unwrap();
        let adj = config.tag_cell_corners[0];
        assert!(adj[0].is_none(), "TL: (r-1,c-1) = (-1,-1) is out of bounds");
        assert!(adj[1].is_none(), "TR: (r-1,c) = (-1,0) is out of bounds");
        assert!(adj[2].is_some(), "BR: (r=0,c=0) is a valid interior saddle");
        assert!(adj[3].is_none(), "BL: (r,c-1) = (0,-1) is out of bounds");
    }

    #[test]
    fn test_dictionary_bounds_check_charuco() {
        // Requesting more markers than the dictionary has IDs must fail.
        // 10×10 board needs 50 markers; a limit of 49 must reject it.
        let err = CharucoTopology::new(10, 10, 0.04, 0.03, 49);
        assert!(err.is_err(), "must fail when markers > max_tag_id");
    }

    #[test]
    fn test_dictionary_bounds_check_aprilgrid() {
        // 5×5 AprilGrid needs 25 markers; a limit of 24 must reject it.
        let err = AprilGridTopology::new(5, 5, 0.01, 0.04, 24);
        assert!(err.is_err(), "must fail when markers > max_tag_id");
    }

    #[test]
    fn test_tag_family_max_id_count() {
        // Spot-check known dictionary sizes.
        use crate::config::TagFamily;
        assert_eq!(TagFamily::ArUco4x4_50.max_id_count(), 50);
        assert_eq!(TagFamily::ArUco4x4_100.max_id_count(), 100);
    }

    // ── Sample-homography + IPPE-Square seed tests ───────────────────────

    /// Helper: builds a noise-free correspondence set for the first 4 tags of
    /// `obj_points`, projected through `pose`, and returns the backing Vecs.
    #[allow(clippy::type_complexity)]
    fn build_four_tag_corr(
        obj_points: &[Option<[[f64; 3]; 4]>],
        pose: &Pose,
        intrinsics: &CameraIntrinsics,
    ) -> (Vec<Point2f>, Vec<[f64; 3]>, Vec<Matrix2<f64>>) {
        let mut img = Vec::with_capacity(16);
        let mut obj = Vec::with_capacity(16);
        let mut info = Vec::with_capacity(16);
        for opt in obj_points.iter().take(4) {
            let corners = opt.expect("first 4 tags must exist");
            for c in &corners {
                let proj = pose.project(&Vector3::new(c[0], c[1], c[2]), intrinsics);
                img.push(Point2f {
                    x: proj[0] as f32,
                    y: proj[1] as f32,
                });
                obj.push(*c);
                info.push(Matrix2::identity());
            }
        }
        (img, obj, info)
    }

    #[test]
    fn test_solve_seed_from_sample_homography_recovers_identity_pose() {
        // Identity rotation, translation along z = 1 m → seed must recover the
        // pose within a sub-degree / sub-cm tolerance (RANSAC inner threshold).
        let config = math_test_config();
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (img, obj, info) = build_four_tag_corr(&config.obj_points, &true_pose, &intrinsics);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };
        let sample = [0usize, 1, 2, 3];
        let seed = solve_seed_from_sample_homography(&sample, &corr, &intrinsics)
            .expect("clean synthetic correspondences must produce a seed");
        let t_err = (seed.translation - true_pose.translation).norm();
        assert!(
            t_err < 1e-3,
            "translation error {t_err} m exceeds 1 mm on noise-free input"
        );
        let q_true = quat_from_so3(true_pose.rotation);
        let q_est = quat_from_so3(seed.rotation);
        let r_err = q_true.angle_to(&q_est).to_degrees();
        assert!(
            r_err < 0.5,
            "rotation error {r_err}° exceeds 0.5° on noise-free input"
        );
    }

    #[test]
    fn test_solve_seed_from_sample_homography_recovers_oblique_pose() {
        // Oblique 20° pitch + lateral translation; the joint-reprojection score
        // must pick the correct Necker branch per tag.
        let config = math_test_config();
        let intrinsics = test_intrinsics();
        let rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 20f64.to_radians());
        let true_pose = Pose::new(
            *rot.to_rotation_matrix().matrix(),
            Vector3::new(0.05, -0.02, 1.5),
        );
        let (img, obj, info) = build_four_tag_corr(&config.obj_points, &true_pose, &intrinsics);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };
        let sample = [0usize, 1, 2, 3];
        let seed = solve_seed_from_sample_homography(&sample, &corr, &intrinsics)
            .expect("oblique noise-free correspondences must produce a seed");
        let t_err = (seed.translation - true_pose.translation).norm();
        assert!(
            t_err < 5e-3,
            "translation error {t_err} m exceeds 5 mm on oblique input"
        );
        let q_true = quat_from_so3(true_pose.rotation);
        let q_est = quat_from_so3(seed.rotation);
        let r_err = q_true.angle_to(&q_est).to_degrees();
        assert!(
            r_err < 1.0,
            "rotation error {r_err}° exceeds 1° on oblique input — likely wrong Necker branch"
        );
    }

    #[test]
    fn test_solve_seed_from_sample_homography_pixel_noise_robust() {
        // 0.5 px Gaussian-equivalent pixel noise: the seed must still land
        // within the RANSAC outer threshold (tau_outer = 5 px ≈ 25 px²) so
        // that LO inner-loop GN can polish to the AW-LM basin.
        let config = math_test_config();
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.02, -0.01, 1.2));
        let (mut img, obj, info) = build_four_tag_corr(&config.obj_points, &true_pose, &intrinsics);

        // Deterministic XOR-shift pseudo-noise (~0.5 px stddev).
        let mut rng = 0xBEEFu32;
        for p in &mut img {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let nx = f64::from(rng & 0xFFFF) / 65535.0 - 0.5;
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let ny = f64::from(rng & 0xFFFF) / 65535.0 - 0.5;
            p.x += nx as f32;
            p.y += ny as f32;
        }

        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
        };
        let sample = [0usize, 1, 2, 3];
        let seed = solve_seed_from_sample_homography(&sample, &corr, &intrinsics)
            .expect("0.5 px noise must not break IPPE seeding");
        let t_err = (seed.translation - true_pose.translation).norm();
        assert!(
            t_err < 0.05,
            "translation error {t_err} m exceeds 5 cm at 0.5 px noise"
        );
    }

    #[test]
    fn test_solve_seed_from_sample_homography_saddle_path() {
        // group_size = 1 (ChAruco saddle path): the seed function must produce
        // a pose from 4 coplanar saddle correspondences via the DLT+IPPE-Square
        // path — the only candidate pool — so the LO inner loop has a
        // sub-degree starting point.
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let obj_pts: [[f64; 3]; 4] = [
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.05, 0.05, 0.0],
            [-0.05, 0.05, 0.0],
        ];
        let img_pts: Vec<Point2f> = obj_pts
            .iter()
            .map(|p| {
                let proj = true_pose.project(&Vector3::new(p[0], p[1], p[2]), &intrinsics);
                Point2f {
                    x: proj[0] as f32,
                    y: proj[1] as f32,
                }
            })
            .collect();
        let info: Vec<Matrix2<f64>> = (0..4).map(|_| Matrix2::identity()).collect();
        let corr = PointCorrespondences {
            image_points: &img_pts,
            object_points: &obj_pts,
            information_matrices: &info,
            group_size: 1,
        };
        let sample = [0usize, 1, 2, 3];
        let seed = solve_seed_from_sample_homography(&sample, &corr, &intrinsics)
            .expect("clean saddle-path correspondences must produce a seed");
        let t_err = (seed.translation - true_pose.translation).norm();
        assert!(
            t_err < 1e-3,
            "saddle-path translation error {t_err} m exceeds 1 mm on noise-free input"
        );
    }
}
