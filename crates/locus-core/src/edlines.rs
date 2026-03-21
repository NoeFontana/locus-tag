//! Hybrid quad extraction: Angular Arc Boundary → Huber IRLS → Sub-Pixel Parabola.
//!
//! Pipeline:
//!   1. **Angular Arc Boundary** — collect all 4-connected outer boundary pixels of
//!      the component.  The four extremal boundary pixels (topmost T, rightmost R,
//!      bottommost B, leftmost L) partition the boundary into four CW arcs, one arc
//!      per edge of the quadrilateral.  This correctly handles any rotation angle.
//!   2. **Huber IRLS** — fit a robust TLS line to each arc.  Three iterations of
//!      Iteratively Reweighted Least Squares with a Huber loss suppress corner-bleed
//!      outliers.
//!   3. **Micro-Ray Gradient Parabola** — sample the full-resolution grayscale image
//!      along short normals (±2 px) to each IRLS line.  Fit a 3-point parabola to
//!      the 1-D gradient profile; extract the sub-pixel peak.  This escapes the
//!      ±0.5 px binary quantization limit.
//!   4. **Sub-Pixel IRLS Re-fit + Intersection** — re-fit lines to the continuous
//!      edge points (tighter Huber δ), then intersect adjacent pairs to produce an
//!      initial four-corner estimate.
//!   5. **Joint Gauss-Newton** — jointly optimise all eight corner DOFs (4 corners ×
//!      2 coordinates) by minimising the sum of squared perpendicular distances from
//!      the sub-pixel observations to their respective edge.  An unrolled 8×8
//!      Cholesky solver (all stack, zero allocations) converges in 1–3 iterations.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]

use std::f64::consts::PI;

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

use crate::gwlf::MomentAccumulator;
use crate::image::ImageView;
use crate::quad::Point;
use crate::segmentation::ComponentStats;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the hybrid EDLines quad extractor.
pub(crate) struct EdLinesConfig {
    /// Huber δ for Phase 2 binary IRLS (pixels). Default: 1.5.
    pub huber_delta: f64,
    /// Number of IRLS iterations for Phase 2. Default: 3.
    pub irls_iters: usize,
    /// Huber δ for Phase 4 sub-pixel IRLS (pixels in gray-image space). Default: 0.5.
    pub sp_huber_delta: f64,
    /// Number of IRLS iterations for Phase 4. Default: 2.
    pub sp_irls_iters: usize,
    /// Distance between ray probes along the tangent direction (pixels). Default: 1.5.
    pub sample_step: f64,
    /// Minimum absolute gradient at the probe centre to accept a sub-pixel point.
    /// Default: 8.0 (intensity units).
    pub grad_min_mag: f64,
    /// Minimum number of sub-pixel points per side to attempt a Phase 4 re-fit.
    /// Default: 5.
    pub min_edge_pts: usize,
    /// Number of Joint Gauss-Newton iterations for Phase 5 corner refinement.
    /// Default: 3.  Set to 0 to disable.
    pub gn_iters: usize,
}

impl EdLinesConfig {
    /// Construct from the detector config (hard-coded defaults pending empirical tuning).
    #[must_use]
    pub fn from_detector_config(_cfg: &crate::config::DetectorConfig) -> Self {
        Self {
            huber_delta: 1.5,
            irls_iters: 3,
            sp_huber_delta: 0.5,
            sp_irls_iters: 2,
            sample_step: 1.5,
            grad_min_mag: 8.0,
            min_edge_pts: 5,
            gn_iters: 3,
        }
    }
}

// ── Internal types ─────────────────────────────────────────────────────────────

/// A homogeneous line `nx·x + ny·y + d = 0` together with its weighted centroid.
#[derive(Clone, Copy)]
struct Line {
    /// Unit normal (nx² + ny² = 1).
    nx: f64,
    /// Unit normal (nx² + ny² = 1).
    ny: f64,
    /// Homogeneous offset.
    d: f64,
    /// Centroid — a point known to lie on the line, used to parameterise the tangent.
    cx: f64,
    cy: f64,
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Fit a weighted TLS line to `points` using `weights` (parallel slices).
///
/// Returns `None` if the weighted sum is degenerate.
fn fit_line_weighted(points: &[(f64, f64)], weights: &[f64]) -> Option<Line> {
    debug_assert_eq!(points.len(), weights.len());
    let mut acc = MomentAccumulator::new();
    for (&(x, y), &w) in points.iter().zip(weights.iter()) {
        if w > 1e-12 {
            acc.add(x, y, w);
        }
    }

    let cov = acc.covariance()?;
    let centroid = acc.centroid()?;

    let a = cov[(0, 0)];
    let b = cov[(0, 1)];
    let c = cov[(1, 1)];

    // Minimum-variance eigenvector = line normal.
    let trace = a + c;
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let lambda_min = (trace - disc) / 2.0;

    let (nx, ny) = if b.abs() > 1e-9 {
        let vx = b;
        let vy = lambda_min - a;
        let len = (vx * vx + vy * vy).sqrt();
        if len < 1e-12 {
            return None;
        }
        (vx / len, vy / len)
    } else if a < c {
        (1.0_f64, 0.0_f64)
    } else {
        (0.0_f64, 1.0_f64)
    };

    let d = -(nx * centroid.x + ny * centroid.y);
    Some(Line {
        nx,
        ny,
        d,
        cx: centroid.x,
        cy: centroid.y,
    })
}

/// Iteratively Reweighted Least Squares (Huber loss) line fit.
///
/// Allocates the weight vector in `arena` to avoid heap allocation.
fn fit_line_irls(
    arena: &Bump,
    points: &[(f64, f64)],
    huber_delta: f64,
    iters: usize,
) -> Option<Line> {
    if points.len() < 2 {
        return None;
    }

    let n = points.len();
    // Arena-allocated weights: zero `Vec::new()` cost.
    let weights: &mut [f64] = arena.alloc_slice_fill_copy(n, 1.0_f64);
    let mut line: Option<Line> = None;

    for _ in 0..iters {
        let l = fit_line_weighted(points, weights)?;
        // Huber weight update.
        for (i, &(x, y)) in points.iter().enumerate() {
            let r = (l.nx * x + l.ny * y + l.d).abs();
            weights[i] = if r < huber_delta {
                1.0
            } else {
                huber_delta / r.max(1e-9)
            };
        }
        line = Some(l);
    }
    line
}

/// Intersect two homogeneous lines via the cross-product rule.
///
/// Returns `None` if the lines are (nearly) parallel.
fn intersect_lines(l1: &Line, l2: &Line) -> Option<(f64, f64)> {
    let ww = l1.nx * l2.ny - l1.ny * l2.nx;
    if ww.abs() < 1e-9 {
        return None;
    }
    let wx = l1.ny * l2.d - l1.d * l2.ny;
    let wy = l1.d * l2.nx - l1.nx * l2.d;
    Some((wx / ww, wy / ww))
}

/// Signed shoelace area of a 4-point polygon.
///
/// Positive ⟺ the polygon is CCW in standard (y-up) math, which equals CW in
/// image (y-down) coordinates.
#[inline]
fn shoelace_area(pts: &[(f64, f64); 4]) -> f64 {
    let mut area = 0.0_f64;
    for i in 0..4 {
        let j = (i + 1) % 4;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    area * 0.5
}

// ── Phase 5 helpers: 8×8 Cholesky + Joint Gauss-Newton ───────────────────────

/// Solve the 8×8 symmetric positive-definite system H·x = b via LL^T Cholesky.
///
/// All computation is on the stack.  Returns `None` if the system is degenerate
/// (any diagonal pivot drops below `1e-12`).
#[must_use]
#[allow(clippy::many_single_char_names)]
/// Cholesky factorisation of a symmetric positive-definite 8×8 matrix.
///
/// Returns the lower-triangular factor L (row-major) such that H = L·L^T,
/// or `None` if any diagonal pivot drops below 1e-12.
fn cholesky_factor_8x8(h_in: &[f64; 64]) -> Option<[f64; 64]> {
    let mut l = *h_in;
    for i in 0..8_usize {
        // Diagonal pivot: L[i][i]² = H[i][i] − Σ_{k<i} L[i][k]²
        let mut s = l[i * 8 + i];
        for k in 0..i {
            s -= l[i * 8 + k] * l[i * 8 + k];
        }
        if s < 1e-12 {
            return None; // not positive-definite
        }
        let l_ii = s.sqrt();
        l[i * 8 + i] = l_ii;
        let inv_lii = 1.0 / l_ii;

        // Off-diagonal: L[j][i] = (H[j][i] − Σ_{k<i} L[j][k]·L[i][k]) / L[i][i]
        for j in (i + 1)..8 {
            let mut sum = l[j * 8 + i]; // lower-triangular read of H (symmetric)
            for k in 0..i {
                sum -= l[j * 8 + k] * l[i * 8 + k];
            }
            l[j * 8 + i] = sum * inv_lii;
        }
    }
    Some(l)
}

/// Solve L·L^T·x = b given the Cholesky factor L (forward + backward substitution).
#[allow(clippy::many_single_char_names)]
fn cholesky_solve_with_factor(l: &[f64; 64], b: &[f64; 8]) -> [f64; 8] {
    // Forward substitution: L·y = b
    let mut y = [0.0_f64; 8];
    for i in 0..8 {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * 8 + k] * y[k];
        }
        y[i] = s / l[i * 8 + i];
    }

    // Backward substitution: L^T·x = y
    let mut x = [0.0_f64; 8];
    for i in (0..8).rev() {
        let mut s = y[i];
        for k in (i + 1)..8 {
            s -= l[k * 8 + i] * x[k]; // L^T[i][k] = L[k][i] = l[k*8+i]
        }
        x[i] = s / l[i * 8 + i];
    }
    x
}

fn cholesky_solve_8x8(h_in: &[f64; 64], b: &[f64; 8]) -> Option<[f64; 8]> {
    let l = cholesky_factor_8x8(h_in)?;
    Some(cholesky_solve_with_factor(&l, b))
}

/// Compute the full 8×8 inverse from a Cholesky factor L (lower-triangular, row-major).
///
/// Solves `L·L^T·x = e_col` for each column of the identity matrix.
/// Returns H⁻¹ as a flat 8×8 row-major array.
fn cholesky_inverse_8x8(l: &[f64; 64]) -> [f64; 64] {
    let mut inv = [0.0_f64; 64];
    let mut e = [0.0_f64; 8];
    for col in 0..8_usize {
        e.fill(0.0);
        e[col] = 1.0;
        let x = cholesky_solve_with_factor(l, &e);
        for i in 0..8 {
            inv[i * 8 + col] = x[i];
        }
    }
    inv
}

/// Joint Gauss-Newton refinement of all four corners.
///
/// Jointly minimises Σ_k Σ_{q∈edge_k} r_q² where r_q is the signed perpendicular
/// distance from sub-pixel observation q (on edge k) to the line connecting the
/// current corners[k] and corners[(k+1)%4].
///
/// State θ = [x₀,y₀,x₁,y₁,x₂,y₂,x₃,y₃].  Edge k connects corner k to corner
/// (k+1)%4.  sp[k] is the slice of (x, y) sub-pixel observations for edge k.
///
/// Returns `(refined_corners, covariances)`.  The covariance is `Some` when
/// the solver converged normally, containing per-corner 2×2 covariances
/// `[[σ_xx, σ_xy, σ_yx, σ_yy]; 4]` extracted from the Hessian inverse.
/// Returns `None` covariance if the solver diverged and reverted to Phase-4.
#[allow(clippy::too_many_lines, clippy::type_complexity)]
fn refine_corners_gauss_newton(
    mut corners: [(f64, f64); 4],
    sp: [&[(f64, f64)]; 4],
    max_iters: usize,
) -> ([(f64, f64); 4], Option<[[f64; 4]; 4]>) {
    const TOL: f64 = 0.005; // convergence: max corner displacement < 0.005 px

    let original = corners;

    // Track residual stats from the last iteration for covariance estimation.
    let mut last_h = [0.0_f64; 64];
    let mut total_residual_sq = 0.0_f64;
    let mut n_obs: usize = 0;

    for _iter in 0..max_iters {
        let mut h = [0.0_f64; 64]; // 8×8 symmetric Gauss-Newton Hessian (J^T J)
        let mut g = [0.0_f64; 8]; // gradient vector (J^T r)
        let mut iter_residual_sq = 0.0_f64;
        let mut iter_n_obs: usize = 0;

        for k in 0..4_usize {
            let ck = corners[k];
            let ck1 = corners[(k + 1) % 4];

            let ex = ck1.0 - ck.0;
            let ey = ck1.1 - ck.1;
            let len = (ex * ex + ey * ey).sqrt();
            if len < 1e-6 {
                continue;
            }

            // Unit tangent and normal for the current edge direction.
            let tx = ex / len;
            let ty = ey / len;
            let nx = -ty; // unit normal (CCW rotation of tangent)
            let ny = tx;

            // State-vector indices for the two endpoints of edge k.
            let ik_x = 2 * k;
            let ik_y = 2 * k + 1;
            let ik1_x = 2 * ((k + 1) % 4);
            let ik1_y = 2 * ((k + 1) % 4) + 1;

            for &(qx, qy) in sp[k] {
                // Projection parameter α ∈ [0,1] along edge (0 = corner k, 1 = corner k+1).
                let alpha = ((qx - ck.0) * tx + (qy - ck.1) * ty) / len;

                // Exclusion zone: discard observations too close to corners (corner bleed).
                if !(0.05_f64..=0.95_f64).contains(&alpha) {
                    continue;
                }

                // Perpendicular residual: signed distance from q to the current edge line.
                let r = (qx - ck.0) * nx + (qy - ck.1) * ny;

                // Sparse Jacobian row (4 non-zero elements out of 8):
                //   ∂r/∂x_k       = -(1-α)·nₓ
                //   ∂r/∂y_k       = -(1-α)·nᵧ
                //   ∂r/∂x_{k+1}   =    -α ·nₓ
                //   ∂r/∂y_{k+1}   =    -α ·nᵧ
                let a_k = 1.0 - alpha;
                let js = [
                    (ik_x, -a_k * nx),
                    (ik_y, -a_k * ny),
                    (ik1_x, -alpha * nx),
                    (ik1_y, -alpha * ny),
                ];

                // Accumulate H += J^T J and g += J^T r (unit weight).
                // Populate the full 8×8 matrix (both triangles) so that the
                // Cholesky can read either the upper or lower half consistently.
                for &(a, ja) in &js {
                    g[a] += ja * r;
                    for &(b, jb) in &js {
                        h[a * 8 + b] += ja * jb;
                    }
                }

                iter_residual_sq += r * r;
                iter_n_obs += 1;
            }
        }

        // Snapshot residual stats (scalars only; H is saved below only on exit).
        total_residual_sq = iter_residual_sq;
        n_obs = iter_n_obs;

        // Tikhonov regularisation: H += λ·I.  Prevents singular systems when
        // an edge has too few valid observations after exclusion-zone filtering.
        // Save the un-regularised H first for covariance extraction.
        last_h = h;
        for i in 0..8 {
            h[i * 8 + i] += 1e-6;
        }

        // Solve H·Δθ = −g.
        let neg_g = g.map(|x| -x);
        let Some(delta) = cholesky_solve_8x8(&h, &neg_g) else {
            break; // singular — keep current estimate
        };

        // Apply update and check convergence.
        let mut max_step = 0.0_f64;
        for i in 0..4 {
            corners[i].0 += delta[2 * i];
            corners[i].1 += delta[2 * i + 1];
            max_step = max_step.max(delta[2 * i].abs()).max(delta[2 * i + 1].abs());
        }

        if max_step < TOL {
            break;
        }
    }

    // Sanity check: if any corner moved more than 5 px from the initial estimate,
    // the solver likely diverged — revert to the Phase-4 intersections.
    for i in 0..4 {
        let dx = corners[i].0 - original[i].0;
        let dy = corners[i].1 - original[i].1;
        if dx * dx + dy * dy > 25.0 {
            return (original, None);
        }
    }

    // Extract per-corner 2×2 covariances from H⁻¹.
    // σ² = Σ r² / (n_obs - 8) is the residual variance estimate.
    // Σ_c[k] = σ² · H⁻¹[2k:2k+2, 2k:2k+2].
    let covs = if n_obs > 8 {
        // Add regularisation to last_h before inversion (same λ as solve step).
        for i in 0..8 {
            last_h[i * 8 + i] += 1e-6;
        }
        cholesky_factor_8x8(&last_h).map(|l| {
            let h_inv = cholesky_inverse_8x8(&l);
            let sigma_sq = total_residual_sq / (n_obs - 8) as f64;
            std::array::from_fn(|k| {
                let r = 2 * k;
                [
                    sigma_sq * h_inv[r * 8 + r],             // σ_xx
                    sigma_sq * h_inv[r * 8 + (r + 1)],       // σ_xy
                    sigma_sq * h_inv[(r + 1) * 8 + r],       // σ_yx
                    sigma_sq * h_inv[(r + 1) * 8 + (r + 1)], // σ_yy
                ]
            })
        })
    } else {
        None
    };

    (corners, covs)
}

// ── Phase 1: Angular Arc Boundary ─────────────────────────────────────────────

/// Extract outer boundary pixels and assign them to four edge groups using
/// angular arc segmentation.
///
/// **Two-stage approach** that combines the strengths of monotone scanning
/// and angular arc segmentation:
///
/// *Stage 1 — Monotone scan* (inner-bit filtering):
/// For each column, record only the **topmost** and **bottommost** foreground
/// pixel.  For each row, record only the **leftmost** and **rightmost**.
/// This naturally selects the convex outer boundary of the component, discarding
/// interior dark data-bit pixels that would otherwise bias the line fits.
///
/// *Stage 2 — Angular arc assignment* (rotation awareness):
/// The four extremal outer-boundary pixels (T=topmost, R=rightmost,
/// B=bottommost, L=leftmost) divide the collected pixels into four CW arcs
/// in image (y-down) coordinates, one arc per edge of the quadrilateral:
///
///   edge\[0\]: T → R  (top-right facing edge)
///   edge\[1\]: R → B  (bottom-right facing edge)
///   edge\[2\]: B → L  (bottom-left facing edge)
///   edge\[3\]: L → T  (top-left facing edge, wraps around)
///
/// This correctly handles any tag rotation angle without interior contamination.
#[allow(clippy::too_many_lines)]
fn extract_boundary_segments<'a>(
    arena: &'a Bump,
    labels: &[u32],
    img_width: usize,
    img_height: usize,
    comp_label: u32,
    stat: &ComponentStats,
) -> [BumpVec<'a, (f64, f64)>; 4] {
    // Centroid from moments accumulated during the LSL pass.
    let m00 = f64::from(stat.pixel_count).max(1.0);
    let cx = stat.m10 as f64 / m00 + 0.5;
    let cy = stat.m01 as f64 / m00 + 0.5;

    let min_x = stat.min_x as usize;
    let max_x = stat.max_x as usize;
    let min_y = stat.min_y as usize;
    let max_y = stat.max_y as usize;

    // Stage 1: monotone scan — collect outer boundary pixels only.
    // For each column: topmost + bottommost foreground pixel.
    // For each row:    leftmost + rightmost foreground pixel.
    // Use a BumpVec of (px, py, angle) to feed Stage 2.
    let mut all_bnd: BumpVec<(f64, f64, f64)> = BumpVec::new_in(arena);

    // Track TWO sets of 4 extremal pixels for arc boundary selection.
    //
    // Axis-aligned: T=min_y, R=max_x, B=max_y, L=min_x.
    //   For axis-aligned tags these map to the 4 edge midpoints, giving 4 clean
    //   arcs. Degenerate when two adjacent corners share the same axis extremal.
    //
    // Diagonal (45°-rotated): NW=min(x+y), NE=min(y-x), SE=max(x+y), SW=max(y-x).
    //   Always corresponds to the actual 4 corners, but degenerate for a different
    //   class of quadrilateral geometries.
    //
    // Stage 2 will pick whichever system has no degenerate (zero-width) arc.
    let mut t_y = usize::MAX;
    let mut t_x = 0usize; // min y (then min x)
    let mut r_x = 0usize;
    let mut r_y = usize::MAX; // max x (then min y)
    let mut b_y = 0usize;
    let mut b_x = 0usize; // max y (then max x)
    let mut l_x = usize::MAX;
    let mut l_y = 0usize; // min x (then max y)

    let mut nw_sum = isize::MAX;
    let mut nw_x = 0usize;
    let mut nw_y = 0usize;
    let mut ne_dif = isize::MAX;
    let mut ne_x = 0usize;
    let mut ne_y = 0usize;
    let mut se_sum = isize::MIN;
    let mut se_x = 0usize;
    let mut se_y = 0usize;
    let mut sw_dif = isize::MIN;
    let mut sw_x = 0usize;
    let mut sw_y = 0usize;

    let push_pt = |x: usize,
                   y: usize,
                   t_y: &mut usize,
                   t_x: &mut usize,
                   r_x: &mut usize,
                   r_y: &mut usize,
                   b_y: &mut usize,
                   b_x: &mut usize,
                   l_x: &mut usize,
                   l_y: &mut usize,
                   nw_sum: &mut isize,
                   nw_x: &mut usize,
                   nw_y: &mut usize,
                   ne_dif: &mut isize,
                   ne_x: &mut usize,
                   ne_y: &mut usize,
                   se_sum: &mut isize,
                   se_x: &mut usize,
                   se_y: &mut usize,
                   sw_dif: &mut isize,
                   sw_x: &mut usize,
                   sw_y: &mut usize,
                   all_bnd: &mut BumpVec<(f64, f64, f64)>| {
        // Axis-aligned extremals
        if y < *t_y || (y == *t_y && x < *t_x) {
            *t_y = y;
            *t_x = x;
        }
        if x > *r_x || (x == *r_x && y < *r_y) {
            *r_x = x;
            *r_y = y;
        }
        if y > *b_y || (y == *b_y && x > *b_x) {
            *b_y = y;
            *b_x = x;
        }
        if x < *l_x || (x == *l_x && y > *l_y) {
            *l_x = x;
            *l_y = y;
        }
        // Diagonal extremals
        let ix = x as isize;
        let iy = y as isize;
        let sum = ix + iy;
        let dif = iy - ix;
        if sum < *nw_sum {
            *nw_sum = sum;
            *nw_x = x;
            *nw_y = y;
        }
        if dif < *ne_dif {
            *ne_dif = dif;
            *ne_x = x;
            *ne_y = y;
        }
        if sum > *se_sum {
            *se_sum = sum;
            *se_x = x;
            *se_y = y;
        }
        if dif > *sw_dif {
            *sw_dif = dif;
            *sw_x = x;
            *sw_y = y;
        }
        let px = x as f64 + 0.5;
        let py = y as f64 + 0.5;
        let angle = (py - cy).atan2(px - cx);
        all_bnd.push((px, py, angle));
    };

    // Column scan: topmost and bottommost foreground pixel per column.
    for x in min_x..=max_x {
        let mut top_y: Option<usize> = None;
        let mut bot_y: Option<usize> = None;
        for y in min_y..=max_y {
            if labels[y * img_width + x] == comp_label {
                if top_y.is_none() {
                    top_y = Some(y);
                }
                bot_y = Some(y);
            }
        }
        if let Some(ty) = top_y {
            push_pt(
                x,
                ty,
                &mut t_y,
                &mut t_x,
                &mut r_x,
                &mut r_y,
                &mut b_y,
                &mut b_x,
                &mut l_x,
                &mut l_y,
                &mut nw_sum,
                &mut nw_x,
                &mut nw_y,
                &mut ne_dif,
                &mut ne_x,
                &mut ne_y,
                &mut se_sum,
                &mut se_x,
                &mut se_y,
                &mut sw_dif,
                &mut sw_x,
                &mut sw_y,
                &mut all_bnd,
            );
        }
        if let Some(by) = bot_y
            && top_y != bot_y
        {
            push_pt(
                x,
                by,
                &mut t_y,
                &mut t_x,
                &mut r_x,
                &mut r_y,
                &mut b_y,
                &mut b_x,
                &mut l_x,
                &mut l_y,
                &mut nw_sum,
                &mut nw_x,
                &mut nw_y,
                &mut ne_dif,
                &mut ne_x,
                &mut ne_y,
                &mut se_sum,
                &mut se_x,
                &mut se_y,
                &mut sw_dif,
                &mut sw_x,
                &mut sw_y,
                &mut all_bnd,
            );
        }
    }

    // Row scan: leftmost and rightmost foreground pixel per row.
    for y in min_y..=max_y {
        let row_off = y * img_width;
        let mut lft_x: Option<usize> = None;
        let mut rgt_x: Option<usize> = None;
        for x in min_x..=max_x {
            if labels[row_off + x] == comp_label {
                if lft_x.is_none() {
                    lft_x = Some(x);
                }
                rgt_x = Some(x);
            }
        }
        if let Some(lx) = lft_x {
            push_pt(
                lx,
                y,
                &mut t_y,
                &mut t_x,
                &mut r_x,
                &mut r_y,
                &mut b_y,
                &mut b_x,
                &mut l_x,
                &mut l_y,
                &mut nw_sum,
                &mut nw_x,
                &mut nw_y,
                &mut ne_dif,
                &mut ne_x,
                &mut ne_y,
                &mut se_sum,
                &mut se_x,
                &mut se_y,
                &mut sw_dif,
                &mut sw_x,
                &mut sw_y,
                &mut all_bnd,
            );
        }
        if let Some(rx) = rgt_x
            && lft_x != rgt_x
        {
            push_pt(
                rx,
                y,
                &mut t_y,
                &mut t_x,
                &mut r_x,
                &mut r_y,
                &mut b_y,
                &mut b_x,
                &mut l_x,
                &mut l_y,
                &mut nw_sum,
                &mut nw_x,
                &mut nw_y,
                &mut ne_dif,
                &mut ne_x,
                &mut ne_y,
                &mut se_sum,
                &mut se_x,
                &mut se_y,
                &mut sw_dif,
                &mut sw_x,
                &mut sw_y,
                &mut all_bnd,
            );
        }
    }

    let empty: [BumpVec<(f64, f64)>; 4] = [
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
    ];

    if t_y == usize::MAX || all_bnd.len() < 8 {
        return empty;
    }

    // Stage 2: angular arc assignment.
    // Try axis-aligned extremals (T/R/B/L) first; fall back to diagonal
    // extremals (NW/NE/SE/SW) if any axis-aligned arc is degenerate.
    // Both systems together cover all convex quadrilateral orientations:
    // for any parallelogram geometry, at least one system will have 4 distinct
    // non-degenerate arcs.
    //
    // Min-arc-width threshold: 5° (0.087 rad). Narrower arcs produce too few
    // points on one edge for reliable IRLS convergence.
    #[allow(clippy::items_after_statements)]
    const MIN_ARC: f64 = 5.0 * PI / 180.0;

    // Compute axis-aligned arc widths (CW: T→R→B→L→T).
    let t_angle = ((t_y as f64 + 0.5) - cy).atan2((t_x as f64 + 0.5) - cx);
    let r_angle = ((r_y as f64 + 0.5) - cy).atan2((r_x as f64 + 0.5) - cx);
    let b_angle = ((b_y as f64 + 0.5) - cy).atan2((b_x as f64 + 0.5) - cx);
    let l_angle = ((l_y as f64 + 0.5) - cy).atan2((l_x as f64 + 0.5) - cx);
    let r_norm = (r_angle - t_angle).rem_euclid(2.0 * PI);
    let b_norm = (b_angle - t_angle).rem_euclid(2.0 * PI);
    let l_norm = (l_angle - t_angle).rem_euclid(2.0 * PI);
    let trbl_ok = r_norm >= MIN_ARC && (b_norm - r_norm) >= MIN_ARC && (l_norm - b_norm) >= MIN_ARC;

    // Compute diagonal arc widths (CW: NW→NE→SE→SW→NW).
    let nw_angle = ((nw_y as f64 + 0.5) - cy).atan2((nw_x as f64 + 0.5) - cx);
    let ne_angle = ((ne_y as f64 + 0.5) - cy).atan2((ne_x as f64 + 0.5) - cx);
    let se_angle = ((se_y as f64 + 0.5) - cy).atan2((se_x as f64 + 0.5) - cx);
    let sw_angle = ((sw_y as f64 + 0.5) - cy).atan2((sw_x as f64 + 0.5) - cx);
    let ne_norm = (ne_angle - nw_angle).rem_euclid(2.0 * PI);
    let se_norm = (se_angle - nw_angle).rem_euclid(2.0 * PI);
    let sw_norm = (sw_angle - nw_angle).rem_euclid(2.0 * PI);
    let diag_ok =
        ne_norm >= MIN_ARC && (se_norm - ne_norm) >= MIN_ARC && (sw_norm - se_norm) >= MIN_ARC;

    if !trbl_ok && !diag_ok {
        return empty;
    }

    // Prefer axis-aligned arcs; use diagonal only when axis-aligned is degenerate.
    let (base_angle, a1, a2, a3) = if trbl_ok {
        (t_angle, r_norm, b_norm, l_norm)
    } else {
        (nw_angle, ne_norm, se_norm, sw_norm)
    };

    // Assign each outer-boundary pixel to one of four arc groups.
    let mut edges: [BumpVec<(f64, f64)>; 4] = [
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
    ];

    // The column/row scans may emit the same corner pixel twice; duplicates are
    // harmless for IRLS (Huber re-weighting handles repeated points gracefully).
    for &(px, py, angle) in &all_bnd {
        let a_norm = (angle - base_angle).rem_euclid(2.0 * PI);
        let group = if a_norm < a1 {
            0
        } else if a_norm < a2 {
            1
        } else if a_norm < a3 {
            2
        } else {
            3
        };
        edges[group].push((px, py));
    }

    // Suppress the `img_height` unused-variable warning in the new code path
    // (still needed for the 4-connected check in the old path — kept as a param
    // for API stability).
    let _ = img_height;

    edges
}

// ── Phase 3: Micro-Ray Parabolic Sub-Pixel Refinement ─────────────────────────

/// For each probe point along the IRLS line (in binary-image space), cast a
/// short ray (±2 integer steps) along the normal in *gray-image space*, fit a
/// parabola to the 1-D gradient profile, and collect the sub-pixel edge
/// position in gray-image space.
///
/// `line` carries coordinates in binary-image space.  `dec` scales those to
/// gray-image space (`dec = gray.width / binary.width`).  The returned points
/// are in gray-image space.
fn refine_edge_subpixel<'a>(
    arena: &'a Bump,
    gray: &ImageView,
    line: &Line,
    dec: f64,
    min_x_bin: f64,
    max_x_bin: f64,
    min_y_bin: f64,
    max_y_bin: f64,
    sample_step: f64,
    grad_min_mag: f64,
) -> BumpVec<'a, (f64, f64)> {
    let mut result: BumpVec<(f64, f64)> = BumpVec::new_in(arena);

    // Tangent direction (perpendicular to normal).
    let tx = -line.ny;
    let ty = line.nx;

    // Find the tangent-coordinate range of the line within the bounding box.
    // Project all 4 bbox corners onto the tangent.
    let corners = [
        (min_x_bin, min_y_bin),
        (max_x_bin, min_y_bin),
        (max_x_bin, max_y_bin),
        (min_x_bin, max_y_bin),
    ];
    let t_vals: [f64; 4] = corners.map(|(bx, by)| (bx - line.cx) * tx + (by - line.cy) * ty);

    let t_min = t_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = t_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let gray_w = gray.width as f64;
    let gray_h = gray.height as f64;

    // Normal direction in gray-image space (same direction, just scaled image).
    let nx = line.nx;
    let ny = line.ny;

    let mut t = t_min;
    while t <= t_max {
        // Probe centre in binary-image space.
        let px_bin = line.cx + t * tx;
        let py_bin = line.cy + t * ty;

        // Scale to gray-image space (pixel-center convention preserved).
        let px = px_bin * dec;
        let py = py_bin * dec;

        // Sample 5 intensities at k ∈ {-2, -1, 0, 1, 2} along the normal.
        // k=0 should be near the edge by construction (the IRLS line tracks the edge).
        // intensities[k+2] for k in {-2,-1,0,1,2}
        let mut intensities = [0.0_f64; 5];
        let mut all_in_bounds = true;
        for (ki, slot) in intensities.iter_mut().enumerate() {
            let k = ki as f64 - 2.0; // k ∈ {-2,-1,0,1,2}
            let sx = px + k * nx;
            let sy = py + k * ny;
            if sx < 0.5 || sy < 0.5 || sx >= gray_w - 0.5 || sy >= gray_h - 0.5 {
                all_in_bounds = false;
                break;
            }
            *slot = gray.sample_bilinear(sx, sy);
        }

        if all_in_bounds {
            // Central-difference gradient at positions k = -1, 0, +1.
            // g[j] = (intensities[j+1] - intensities[j-1]) / 2, at j = 1,2,3 (k = -1,0,+1).
            let g_neg1 = (intensities[2] - intensities[0]) * 0.5; // gradient at k = -1
            let g_0 = (intensities[3] - intensities[1]) * 0.5; // gradient at k =  0
            let g_pos1 = (intensities[4] - intensities[2]) * 0.5; // gradient at k = +1

            // Require a minimum gradient magnitude at the probe centre.
            if g_0.abs() >= grad_min_mag {
                // 3-point parabolic peak estimator.
                // f(k) = a·k² + b·k + c fitted to (g_neg1, g_0, g_pos1) at k = (-1,0,1).
                // Vertex at k* = -b/(2a) = -(g_pos1 - g_neg1) / (g_pos1 + g_neg1 - 2·g_0).
                let denom = g_pos1 + g_neg1 - 2.0 * g_0;
                let k_star = if denom.abs() > 1e-6 {
                    (-(g_pos1 - g_neg1) / denom).clamp(-1.5, 1.5)
                } else {
                    0.0 // gradient nearly linear; no clear peak — stay at centre
                };

                // Sub-pixel edge location in gray-image space.
                result.push((px + k_star * nx, py + k_star * ny));
            }
        }

        t += sample_step;
    }

    result
}

// ── Main entry ─────────────────────────────────────────────────────────────────

/// Extract a quad using the hybrid angular-boundary / IRLS / sub-pixel pipeline.
///
/// `binary` is the binarised (decimated) image; `gray` is the full-resolution
/// grayscale image for sub-pixel refinement.  `labels` and `comp_label` identify
/// the specific connected component to process.
///
/// Returns corners in **binary-image (decimated) coordinates** so that the
/// caller's `corners * decimation` conversion produces full-resolution positions,
/// consistent with the [`ContourRdp`] path.
#[allow(clippy::too_many_lines)]
pub(crate) fn extract_quad_edlines(
    arena: &Bump,
    binary: &ImageView,
    gray: &ImageView,
    labels: &[u32],
    comp_label: u32,
    stat: &ComponentStats,
    cfg: &EdLinesConfig,
) -> Option<([Point; 4], crate::quad::CornerCovariances)> {
    let bbox_w = (stat.max_x - stat.min_x) as usize + 1;
    let bbox_h = (stat.max_y - stat.min_y) as usize + 1;
    if bbox_w < 5 || bbox_h < 5 {
        return None;
    }

    // ── Phase 1: Angular arc boundary segmentation ────────────────────────────
    // edges[0]: T→R arc, edges[1]: R→B arc, edges[2]: B→L arc, edges[3]: L→T arc
    let edges =
        extract_boundary_segments(arena, labels, binary.width, binary.height, comp_label, stat);

    for e in &edges {
        if e.len() < 2 {
            return None;
        }
    }

    // ── Phase 2: Huber IRLS on binary boundary points ─────────────────────────
    let line0 = fit_line_irls(arena, edges[0].as_slice(), cfg.huber_delta, cfg.irls_iters)?;
    let line1 = fit_line_irls(arena, edges[1].as_slice(), cfg.huber_delta, cfg.irls_iters)?;
    let line2 = fit_line_irls(arena, edges[2].as_slice(), cfg.huber_delta, cfg.irls_iters)?;
    let line3 = fit_line_irls(arena, edges[3].as_slice(), cfg.huber_delta, cfg.irls_iters)?;
    let bin_lines = [line0, line1, line2, line3];

    // Decimation scale factor: gray-image pixels per binary-image pixel.
    // When decimation = 1, dec = 1.0 and all coordinate conversions are no-ops.
    let dec = gray.width as f64 / binary.width as f64;

    // Bounding box in binary-image pixel-centre coordinates.
    let min_x_bin = f64::from(stat.min_x) + 0.5;
    let max_x_bin = f64::from(stat.max_x) + 0.5;
    let min_y_bin = f64::from(stat.min_y) + 0.5;
    let max_y_bin = f64::from(stat.max_y) + 0.5;

    // ── Phase 3: Sub-pixel refinement (results in gray-image space) ───────────
    let sp: [BumpVec<(f64, f64)>; 4] = [
        refine_edge_subpixel(
            arena,
            gray,
            &bin_lines[0],
            dec,
            min_x_bin,
            max_x_bin,
            min_y_bin,
            max_y_bin,
            cfg.sample_step,
            cfg.grad_min_mag,
        ),
        refine_edge_subpixel(
            arena,
            gray,
            &bin_lines[1],
            dec,
            min_x_bin,
            max_x_bin,
            min_y_bin,
            max_y_bin,
            cfg.sample_step,
            cfg.grad_min_mag,
        ),
        refine_edge_subpixel(
            arena,
            gray,
            &bin_lines[2],
            dec,
            min_x_bin,
            max_x_bin,
            min_y_bin,
            max_y_bin,
            cfg.sample_step,
            cfg.grad_min_mag,
        ),
        refine_edge_subpixel(
            arena,
            gray,
            &bin_lines[3],
            dec,
            min_x_bin,
            max_x_bin,
            min_y_bin,
            max_y_bin,
            cfg.sample_step,
            cfg.grad_min_mag,
        ),
    ];

    // ── Phase 4: Sub-pixel IRLS re-fit, then intersect ───────────────────────
    // Convert a binary-space Line to gray-image space (scale centroid by dec;
    // unit normal is direction-only and does not scale).
    let to_gray = |l: &Line| -> Line {
        let cx_g = l.cx * dec;
        let cy_g = l.cy * dec;
        Line {
            nx: l.nx,
            ny: l.ny,
            d: -(l.nx * cx_g + l.ny * cy_g),
            cx: cx_g,
            cy: cy_g,
        }
    };

    // Try Phase-4 IRLS on the sub-pixel point set; fall back to scaled Phase-2 line.
    let try_refit = |pts: &[(f64, f64)], fallback: Line| -> Line {
        if pts.len() >= cfg.min_edge_pts {
            fit_line_irls(arena, pts, cfg.sp_huber_delta, cfg.sp_irls_iters).unwrap_or(fallback)
        } else {
            fallback
        }
    };

    let fl: [Line; 4] = [
        try_refit(sp[0].as_slice(), to_gray(&bin_lines[0])),
        try_refit(sp[1].as_slice(), to_gray(&bin_lines[1])),
        try_refit(sp[2].as_slice(), to_gray(&bin_lines[2])),
        try_refit(sp[3].as_slice(), to_gray(&bin_lines[3])),
    ];

    // Intersect adjacent edge pairs to get 4 initial corners (gray-image space):
    //   corner_T = edge[3] ∩ edge[0]   (L→T arc meets T→R arc)
    //   corner_R = edge[0] ∩ edge[1]   (T→R arc meets R→B arc)
    //   corner_B = edge[1] ∩ edge[2]   (R→B arc meets B→L arc)
    //   corner_L = edge[2] ∩ edge[3]   (B→L arc meets L→T arc)
    let ct = intersect_lines(&fl[3], &fl[0])?;
    let cr = intersect_lines(&fl[0], &fl[1])?;
    let cb = intersect_lines(&fl[1], &fl[2])?;
    let cl = intersect_lines(&fl[2], &fl[3])?;

    // ── Phase 5: Joint Gauss-Newton corner refinement ─────────────────────────
    // Jointly optimise all 8 corner DOFs by minimising the sum of squared
    // perpendicular distances from the Phase-3 sub-pixel observations to their
    // respective edge.  The 8×8 normal equations are solved via an unrolled
    // Cholesky on the stack (zero allocations).  Initialised from Phase-4
    // intersections; falls back to Phase-4 if the solver diverges.
    let sp_slices = [
        sp[0].as_slice(),
        sp[1].as_slice(),
        sp[2].as_slice(),
        sp[3].as_slice(),
    ];
    let (refined, gn_covs) = refine_corners_gauss_newton([ct, cr, cb, cl], sp_slices, cfg.gn_iters);
    let [ct, cr, cb, cl] = refined;

    // Validate: all corners within the expanded bbox (gray-image space).
    let margin_x = (max_x_bin - min_x_bin) * dec * 0.25;
    let margin_y = (max_y_bin - min_y_bin) * dec * 0.25;
    let g_min_x = min_x_bin * dec - margin_x;
    let g_max_x = max_x_bin * dec + margin_x;
    let g_min_y = min_y_bin * dec - margin_y;
    let g_max_y = max_y_bin * dec + margin_y;

    for &(qx, qy) in &[ct, cr, cb, cl] {
        if qx < g_min_x || qx > g_max_x || qy < g_min_y || qy > g_max_y {
            return None;
        }
    }

    // Validate signed area (shoelace in gray-image space).
    // `pixel_count` is in binary pixels; scale the threshold to gray pixels².
    let pts4 = [ct, cr, cb, cl];
    let area = shoelace_area(&pts4);
    if area.abs() < f64::from(stat.pixel_count) * dec * dec * 0.1 {
        return None;
    }

    // ── Return corners in binary-image (decimated) space ──────────────────────
    // The caller in `quad.rs` multiplies by `decimation as f64` to get full-res
    // coordinates; we must undo our own `dec` scaling before returning.
    let inv_dec = 1.0 / dec;

    // Arc assignment guarantees CW traversal [T, R, B, L] in y-down image coords,
    // which gives positive shoelace area.  If unexpectedly negative, reverse to
    // restore CW winding.
    let to_pt = |(x, y): (f64, f64)| Point {
        x: x * inv_dec,
        y: y * inv_dec,
    };

    // GN covariances are in gray-image space.  Since corners are converted via
    // inv_dec, then later multiplied by d (decimation) in quad.rs, and d × inv_dec = 1,
    // the covariances are already in full-res pixel² units — no scaling needed.
    // Cast f64 → f32 for batch storage.
    let covs_f32: [[f32; 4]; 4] = match gn_covs {
        Some(c) => std::array::from_fn(|i| c[i].map(|v| v as f32)),
        None => [[0.0; 4]; 4],
    };

    Some(if area >= 0.0 {
        ([to_pt(ct), to_pt(cr), to_pt(cb), to_pt(cl)], covs_f32)
    } else {
        // Reverse winding: corners go [T, L, B, R], so reorder covariances too.
        // Original order: [T=0, R=1, B=2, L=3] → reversed: [T=0, L=3, B=2, R=1]
        (
            [to_pt(ct), to_pt(cl), to_pt(cb), to_pt(cr)],
            [covs_f32[0], covs_f32[3], covs_f32[2], covs_f32[1]],
        )
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::image::ImageView;
    use crate::segmentation::ComponentStats;

    /// Build a grayscale image with a filled dark square on a bright background.
    fn make_square_image(canvas: usize, sq_x0: usize, sq_y0: usize, sq_size: usize) -> Vec<u8> {
        let mut img = vec![200u8; canvas * canvas];
        for y in sq_y0..sq_y0 + sq_size {
            for x in sq_x0..sq_x0 + sq_size {
                img[y * canvas + x] = 30;
            }
        }
        img
    }

    /// Build a label map with `comp_label = 1` inside the square.
    fn make_labels(canvas: usize, sq_x0: usize, sq_y0: usize, sq_size: usize) -> Vec<u32> {
        let mut labels = vec![0u32; canvas * canvas];
        for y in sq_y0..sq_y0 + sq_size {
            for x in sq_x0..sq_x0 + sq_size {
                labels[y * canvas + x] = 1;
            }
        }
        labels
    }

    /// Build ComponentStats with correct moment accumulators for a filled square.
    fn make_stats_square(sq_x0: u16, sq_y0: u16, sq_size: u16) -> ComponentStats {
        let x1 = u64::from(sq_x0);
        let y1 = u64::from(sq_y0);
        let n = u64::from(sq_size);
        // Σ(x = x1..x1+n-1) x = n * (2*x1 + n - 1) / 2
        let m10 = n * (2 * x1 + n - 1) / 2 * n; // n rows, each row contributes Σ x
        let m01 = n * (2 * y1 + n - 1) / 2 * n; // n cols, each col contributes Σ y
        ComponentStats {
            min_x: sq_x0,
            min_y: sq_y0,
            max_x: sq_x0 + sq_size - 1,
            max_y: sq_y0 + sq_size - 1,
            pixel_count: u32::from(sq_size) * u32::from(sq_size),
            first_pixel_x: sq_x0,
            first_pixel_y: sq_y0,
            m10,
            m01,
            m20: 0,
            m02: 0,
            m11: 0,
        }
    }

    #[test]
    fn test_edlines_square_returns_corners() {
        let canvas = 100usize;
        let sq_x0 = 30u16;
        let sq_y0 = 30u16;
        let sq_size = 40u16;

        let pixels = make_square_image(canvas, sq_x0 as usize, sq_y0 as usize, sq_size as usize);
        let labels = make_labels(canvas, sq_x0 as usize, sq_y0 as usize, sq_size as usize);
        let img = ImageView::new(&pixels, canvas, canvas, canvas).unwrap();
        let stats = make_stats_square(sq_x0, sq_y0, sq_size);
        let cfg = EdLinesConfig {
            huber_delta: 1.5,
            irls_iters: 3,
            sp_huber_delta: 0.5,
            sp_irls_iters: 2,
            sample_step: 1.5,
            grad_min_mag: 4.0, // lower threshold for synthetic hard-edge image
            min_edge_pts: 5,
            gn_iters: 3,
        };
        let arena = Bump::new();
        // Use the same image as both binary and gray (dec = 1.0).
        let result = extract_quad_edlines(&arena, &img, &img, &labels, 1, &stats, &cfg);
        assert!(
            result.is_some(),
            "EDLines should detect a quad in a clean synthetic square"
        );

        // All returned corners must be within 4 px of one of the true square corners.
        let true_corners = [
            [f64::from(sq_x0), f64::from(sq_y0)],
            [f64::from(sq_x0 + sq_size), f64::from(sq_y0)],
            [f64::from(sq_x0 + sq_size), f64::from(sq_y0 + sq_size)],
            [f64::from(sq_x0), f64::from(sq_y0 + sq_size)],
        ];
        let (corners, _covs) = result.unwrap();
        let mut matched = [false; 4];
        for corner in &corners {
            let mut found = false;
            for (i, tc) in true_corners.iter().enumerate() {
                if !matched[i] {
                    let dist = ((corner.x - tc[0]).powi(2) + (corner.y - tc[1]).powi(2)).sqrt();
                    if dist < 4.0 {
                        matched[i] = true;
                        found = true;
                        break;
                    }
                }
            }
            assert!(
                found,
                "Corner ({:.1}, {:.1}) is not within 4 px of any true corner",
                corner.x, corner.y
            );
        }
    }

    #[test]
    fn test_edlines_tiny_roi_returns_none() {
        let pixels = vec![100u8; 10 * 10];
        let img = ImageView::new(&pixels, 10, 10, 10).unwrap();
        let stats = ComponentStats {
            min_x: 1,
            min_y: 1,
            max_x: 3,
            max_y: 3,
            pixel_count: 9,
            first_pixel_x: 1,
            first_pixel_y: 1,
            m10: 0,
            m01: 0,
            m20: 0,
            m02: 0,
            m11: 0,
        };
        let cfg = EdLinesConfig {
            huber_delta: 1.5,
            irls_iters: 3,
            sp_huber_delta: 0.5,
            sp_irls_iters: 2,
            sample_step: 1.5,
            grad_min_mag: 8.0,
            min_edge_pts: 5,
            gn_iters: 3,
        };
        let arena = Bump::new();
        // labels and comp_label are not accessed because the bbox guard fires first.
        let result = extract_quad_edlines(&arena, &img, &img, &[], 0, &stats, &cfg);
        assert!(result.is_none(), "Tiny ROI must return None");
    }

    /// Verify that the GN solver returns non-zero, diagonal-dominant covariances
    /// when observations have small perpendicular noise (simulating real sub-pixel
    /// edge detection).  With zero noise the residual variance σ² = 0, which
    /// correctly yields zero covariances; we test the realistic case here.
    #[test]
    fn test_gn_covariance_nonzero() {
        let corners: [(f64, f64); 4] = [(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0)];

        let mut sp: [Vec<(f64, f64)>; 4] = [vec![], vec![], vec![], vec![]];
        let n = 20_usize;
        for k in 0..4 {
            let (ax, ay) = corners[k];
            let (bx, by) = corners[(k + 1) % 4];
            let ex = bx - ax;
            let ey = by - ay;
            let len = (ex * ex + ey * ey).sqrt();
            let nx = -ey / len;
            let ny = ex / len;
            for i in 0..n {
                let t = (i as f64 + 0.5) / n as f64;
                // Add small deterministic perpendicular noise (~0.1 px).
                let noise = 0.1 * ((i * 7 + k * 13) as f64).sin();
                sp[k].push((
                    ax + t * (bx - ax) + noise * nx,
                    ay + t * (by - ay) + noise * ny,
                ));
            }
        }

        let sp_refs: [&[(f64, f64)]; 4] = [&sp[0], &sp[1], &sp[2], &sp[3]];
        let (refined, cov_opt) = refine_corners_gauss_newton(corners, sp_refs, 5);

        // Solver should converge (observations are close to the edges).
        for (i, &rc) in refined.iter().enumerate() {
            let dx = rc.0 - corners[i].0;
            let dy = rc.1 - corners[i].1;
            assert!(
                dx.abs() < 0.5 && dy.abs() < 0.5,
                "Corner {i} moved too far: ({dx:.4}, {dy:.4})"
            );
        }

        // Covariances must be Some and non-zero (noise → σ² > 0).
        let covs = cov_opt.expect("GN should return covariances for a converged solution");
        for (k, cov) in covs.iter().enumerate() {
            let sigma_xx = cov[0];
            let sigma_yy = cov[3];
            assert!(
                sigma_xx > 0.0 && sigma_yy > 0.0,
                "Corner {k}: diagonal covariances must be positive, got σ_xx={sigma_xx}, σ_yy={sigma_yy}"
            );
            // Diagonal-dominant: |σ_xx| and |σ_yy| should exceed cross-terms.
            let sigma_xy = cov[1];
            assert!(
                sigma_xx.abs() >= sigma_xy.abs(),
                "Corner {k}: expected diagonal dominance"
            );
        }
    }

    /// Verify that adding noise to observations increases the covariance magnitudes.
    #[test]
    fn test_gn_covariance_scales_with_noise() {
        let corners: [(f64, f64); 4] = [(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0)];

        let n = 30_usize;
        let mut sp_clean: [Vec<(f64, f64)>; 4] = [vec![], vec![], vec![], vec![]];
        let mut sp_noisy: [Vec<(f64, f64)>; 4] = [vec![], vec![], vec![], vec![]];

        // Simple deterministic "noise" pattern using sin to avoid RNG dependency.
        for k in 0..4 {
            let (ax, ay) = corners[k];
            let (bx, by) = corners[(k + 1) % 4];
            let ex = bx - ax;
            let ey = by - ay;
            let len = (ex * ex + ey * ey).sqrt();
            // Normal direction (perpendicular to edge).
            let nx = -ey / len;
            let ny = ex / len;
            for i in 0..n {
                let t = (i as f64 + 0.5) / n as f64;
                let px = ax + t * (bx - ax);
                let py = ay + t * (by - ay);
                sp_clean[k].push((px, py));
                // Add perpendicular noise of ~0.5 px amplitude.
                let noise = 0.5 * ((i * 7 + k * 13) as f64).sin();
                sp_noisy[k].push((px + noise * nx, py + noise * ny));
            }
        }

        let refs_clean: [&[(f64, f64)]; 4] =
            [&sp_clean[0], &sp_clean[1], &sp_clean[2], &sp_clean[3]];
        let refs_noisy: [&[(f64, f64)]; 4] =
            [&sp_noisy[0], &sp_noisy[1], &sp_noisy[2], &sp_noisy[3]];

        let (_, cov_clean) = refine_corners_gauss_newton(corners, refs_clean, 5);
        let (_, cov_noisy) = refine_corners_gauss_newton(corners, refs_noisy, 5);

        let covs_c = cov_clean.expect("clean should converge");
        let covs_n = cov_noisy.expect("noisy should converge");

        // Sum of diagonal variances should be larger for the noisy case.
        let trace_clean: f64 = covs_c.iter().map(|c| c[0] + c[3]).sum();
        let trace_noisy: f64 = covs_n.iter().map(|c| c[0] + c[3]).sum();
        assert!(
            trace_noisy > trace_clean,
            "Noisy observations should yield larger covariances: clean={trace_clean:.6}, noisy={trace_noisy:.6}"
        );
    }
}
