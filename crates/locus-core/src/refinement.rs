//! Corner-refinement dispatch.
//!
//! Owns the [`crate::config::CornerRefinementMode`] decision across the
//! pipeline's two refinement stages: [`refine_quad_corners`] runs per
//! candidate during quad extraction, and [`apply_detector_gwlf`] runs
//! batch-wise after the first homography pass. Both honour the
//! per-candidate route from [`crate::quad::resolve_route`], so
//! `AdaptivePpb` policies fire GWLF on the routes that ask for it.

#![allow(clippy::cast_possible_wrap)]

use bumpalo::Bump;

use crate::Point;
use crate::batch::{DetectionBatch, ROUTED_TO_HIGH};
use crate::config::{
    CornerRefinementMode, DetectorConfig, QuadExtractionMode, QuadExtractionPolicy,
};
use crate::image::ImageView;
use crate::quad::{CornerCovariances, fit_edge_line, refine_edge_erf};

/// Per-candidate corner refinement dispatch.
///
/// `route_extraction` and `route_refinement` come from
/// [`crate::quad::resolve_route`]. Behaviour by cell:
///
/// | extractor    | mode  | quad-stage action                              |
/// |--------------|-------|------------------------------------------------|
/// | any          | None  | passthrough (propagates GN covariances)        |
/// | any          | Erf   | per-corner PSF Gauss-Newton fit                |
/// | ContourRdp   | Gwlf  | gradient-peak warm-start for GWLF              |
/// | EdLines      | Gwlf  | passthrough (GN corners already sub-pixel)     |
///
/// The split on `Gwlf` is empirical: `ContourRdp`'s integer-precision
/// corners need a sub-pixel warm-start before GWLF in
/// [`apply_detector_gwlf`] converges reliably (no warm-start regresses
/// mean RMSE +15 % and p90 rotation +210 % on the 1080p render-tag
/// hub). EdLines' Gauss-Newton corners are already sub-pixel and a
/// gradient-peak refit only degrades them.
#[expect(
    clippy::too_many_arguments,
    reason = "per-frame corner-refinement dispatch; arena, image, quad, covariances, the two route enums, sigma and decimation are distinct pipeline inputs and grouping them into a struct only adds indirection on the refinement hot path"
)]
#[inline]
pub(crate) fn refine_quad_corners(
    arena: &Bump,
    refinement_img: &ImageView,
    quad_pts: [Point; 4],
    gn_covs: CornerCovariances,
    route_extraction: QuadExtractionMode,
    route_refinement: CornerRefinementMode,
    sigma: f64,
    decimation: usize,
) -> ([Point; 4], CornerCovariances) {
    match (route_extraction, route_refinement) {
        (_, CornerRefinementMode::None)
        | (QuadExtractionMode::EdLines, CornerRefinementMode::Gwlf) => (quad_pts, gn_covs),
        (_, CornerRefinementMode::Erf) => {
            let corners =
                refine_all_quad_corners(arena, refinement_img, quad_pts, sigma, decimation, true);
            (corners, [[0.0; 4]; 4])
        },
        (QuadExtractionMode::ContourRdp, CornerRefinementMode::Gwlf) => {
            let corners =
                refine_all_quad_corners(arena, refinement_img, quad_pts, sigma, decimation, false);
            (corners, [[0.0; 4]; 4])
        },
    }
}

/// Refine each of a quad's four corners using its two cyclic
/// neighbours.
///
/// Indices: corner `i` is refined using `(i-1, i, i+1)` mod 4. With
/// CW-ordered corners this gives the conventional `(prev, current,
/// next)` triplet that `refine_corner` expects. The result equals four
/// `refine_corner` calls, but each edge line is fitted once: corner `i`
/// uses the lines of edges `(i-1, i)` and `(i, i+1)`, which corner `i±1`
/// fits from the same unrefined endpoints.
pub(crate) fn refine_all_quad_corners(
    arena: &Bump,
    img: &ImageView,
    pts: [Point; 4],
    sigma: f64,
    decimation: usize,
    use_erf: bool,
) -> [Point; 4] {
    let lines: [Option<(f64, f64, f64)>; 4] = std::array::from_fn(|i| {
        edge_line(
            arena,
            img,
            pts[i],
            pts[(i + 1) % 4],
            sigma,
            decimation,
            use_erf,
        )
    });
    std::array::from_fn(|i| intersect_corner(pts[i], lines[(i + 3) % 4], lines[i], decimation))
}

/// Single-corner refinement: intersect two edge-line fits at point `p`,
/// using its neighbours `p_prev` and `p_next` to define the edges.
///
/// `use_erf = true` runs the PSF-blurred Gauss-Newton fit and falls
/// back to the gradient-peak fit on sample shortfall. `use_erf = false`
/// runs only the gradient-peak fit.
#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "single-corner refinement primitive; the point, its two neighbours, arena, image, sigma, decimation and use_erf flag are each distinct geometric or tuning inputs with no natural struct grouping"
)]
pub(crate) fn refine_corner(
    arena: &Bump,
    img: &ImageView,
    p: Point,
    p_prev: Point,
    p_next: Point,
    sigma: f64,
    decimation: usize,
    use_erf: bool,
) -> Point {
    let line1 = edge_line(arena, img, p_prev, p, sigma, decimation, use_erf);
    let line2 = edge_line(arena, img, p, p_next, sigma, decimation, use_erf);
    intersect_corner(p, line1, line2, decimation)
}

/// Line fit of the edge `a → b`: the ERF fit with a gradient-peak fallback, or the
/// gradient-peak fit alone.
fn edge_line(
    arena: &Bump,
    img: &ImageView,
    a: Point,
    b: Point,
    sigma: f64,
    decimation: usize,
    use_erf: bool,
) -> Option<(f64, f64, f64)> {
    if use_erf {
        refine_edge_erf(arena, img, a, b, sigma, decimation)
            .or_else(|| fit_edge_line(img, a, b, decimation))
    } else {
        fit_edge_line(img, a, b, decimation)
    }
}

/// Intersection of two edge lines as the refined corner, or `p` unchanged when the lines are
/// missing, near-parallel, or meet farther than the sanity radius from `p` (2 px, plus the
/// decimation factor when decimated).
pub(crate) fn intersect_corner(
    p: Point,
    line1: Option<(f64, f64, f64)>,
    line2: Option<(f64, f64, f64)>,
    decimation: usize,
) -> Point {
    if let (Some(l1), Some(l2)) = (line1, line2) {
        let det = l1.0 * l2.1 - l2.0 * l1.1;
        if det.abs() > 1e-6 {
            let x = (l1.1 * l2.2 - l2.1 * l1.2) / det;
            let y = (l2.0 * l1.2 - l1.0 * l2.2) / det;

            let dist_sq = (x - p.x).powi(2) + (y - p.y).powi(2);
            let max_dist = if decimation > 1 {
                (decimation as f64) + 2.0
            } else {
                2.0
            };
            if dist_sq < max_dist * max_dist {
                return Point { x, y };
            }
        }
    }

    p
}

/// Resolves the per-candidate refinement mode from the persisted
/// `route_label`. Mirrors [`crate::quad::resolve_route`]'s refinement
/// branch for the post-extraction stage, where PPB is no longer in
/// scope.
#[inline]
fn resolve_route_refinement(config: &DetectorConfig, route_label: u8) -> CornerRefinementMode {
    match config.quad_extraction_policy {
        QuadExtractionPolicy::Static => {
            debug_assert_eq!(
                route_label,
                crate::batch::ROUTED_TO_STATIC,
                "Static policy candidates must carry the ROUTED_TO_STATIC label",
            );
            config.refinement_mode
        },
        QuadExtractionPolicy::AdaptivePpb(cfg) => {
            if route_label == ROUTED_TO_HIGH {
                cfg.high_refinement
            } else {
                cfg.low_refinement
            }
        },
    }
}

/// Returns `true` if any route under the active policy resolves to
/// `Gwlf`. Used as the fast-exit gate of [`apply_detector_gwlf`].
#[inline]
fn any_route_uses_gwlf(config: &DetectorConfig) -> bool {
    match config.quad_extraction_policy {
        QuadExtractionPolicy::Static => config.refinement_mode == CornerRefinementMode::Gwlf,
        QuadExtractionPolicy::AdaptivePpb(cfg) => {
            cfg.low_refinement == CornerRefinementMode::Gwlf
                || cfg.high_refinement == CornerRefinementMode::Gwlf
        },
    }
}

/// Detector-level GWLF refinement pass.
///
/// Iterates the batch and runs GWLF on every candidate whose
/// route-resolved refinement is `Gwlf`. On success, overwrites corners
/// and writes the calibrated 2×2 covariances. On failure, leaves the
/// quad-stage corners in place — those are already extractor-appropriate
/// (Edge-warm-started for `ContourRdp+Gwlf`; pristine Gauss-Newton for
/// `EdLines+Gwlf`).
///
/// Returns `Some((fallback_count, avg_delta))` when at least one
/// candidate routed to `Gwlf`; the caller must then recompute
/// homographies, since corners may have moved. Returns `None`
/// otherwise.
pub(crate) fn apply_detector_gwlf(
    batch: &mut DetectionBatch,
    n: usize,
    refinement_img: &ImageView,
    config: &DetectorConfig,
) -> Option<(usize, f32)> {
    if !any_route_uses_gwlf(config) {
        return None;
    }

    let mut gwlf_fallback_count: usize = 0;
    let mut total_delta = 0.0f32;
    let mut count: usize = 0;

    for i in 0..n {
        // Candidates the contrast funnel already rejected are never decoded: skip them.
        if batch.status_mask[i] != crate::batch::CandidateState::Active
            || resolve_route_refinement(config, batch.routed_to[i]) != CornerRefinementMode::Gwlf
        {
            continue;
        }

        let coarse = [
            [batch.corners[i][0].x, batch.corners[i][0].y],
            [batch.corners[i][1].x, batch.corners[i][1].y],
            [batch.corners[i][2].x, batch.corners[i][2].y],
            [batch.corners[i][3].x, batch.corners[i][3].y],
        ];

        if let Some((refined, covs)) = crate::gwlf::refine_quad_gwlf_with_cov(
            refinement_img,
            &coarse,
            config.gwlf_transversal_alpha,
        ) {
            for j in 0..4 {
                let dx = refined[j][0] - coarse[j][0];
                let dy = refined[j][1] - coarse[j][1];
                total_delta += (dx * dx + dy * dy).sqrt();
                count += 1;

                batch.corners[i][j].x = refined[j][0];
                batch.corners[i][j].y = refined[j][1];

                batch.corner_covariances[i][j * 4] = covs[j][(0, 0)] as f32;
                batch.corner_covariances[i][j * 4 + 1] = covs[j][(0, 1)] as f32;
                batch.corner_covariances[i][j * 4 + 2] = covs[j][(1, 0)] as f32;
                batch.corner_covariances[i][j * 4 + 3] = covs[j][(1, 1)] as f32;
            }
        } else {
            // Quad-stage refinement already placed sensible corners in
            // the batch (Edge warm-start for ContourRdp+Gwlf, GN pristine
            // for EdLines+Gwlf). Leave them alone; just count the failure.
            gwlf_fallback_count += 1;
        }
    }

    if count == 0 && gwlf_fallback_count == 0 {
        return None;
    }

    let gwlf_avg_delta = if count > 0 {
        total_delta / count as f32
    } else {
        0.0
    };

    Some((gwlf_fallback_count, gwlf_avg_delta))
}

/// Window half-width bounds of [`corner_subpix`] (px); see [`corner_subpix_half_windows`].
const SUBPIX_MIN_HALF: u32 = 2;
const SUBPIX_MAX_HALF: u32 = 4;
/// Distance (px) the smallest [`corner_subpix`] patch reaches from its corner: the window
/// half-width plus the ring its central-difference gradients read.
pub(crate) const MIN_CORNER_SUPPORT_PX: f32 = (SUBPIX_MIN_HALF + 1) as f32;
/// Candidate window half-widths as fractions of the marker cell.
const SUBPIX_CELL_FRACTIONS: [f64; 3] = [0.3, 0.5, 0.75];
/// Largest fraction of a cell the smallest (2 px) window may cover: markers with cells under
/// 2 / 0.6 ≈ 3.3 px keep their seed corners. Measured on ChArUco boards of 2.3–2.9 px cells,
/// where the 2 px window degraded board rotation (0.07° → 0.21° on 2.9 px cells); neutral on
/// the ICRA, render-tag, AprilGrid, Liu4K and EuRoC benchmarks.
const SUBPIX_MIN_WINDOW_CELL_FRACTION: f64 = 0.6;
/// Uncertainty ratio past which a smaller window is preferred over a larger one. Both
/// uncertainties are residual-variance estimates from a few dozen effective samples, whose
/// ratio is F-distributed; a factor of 2 is about where a difference stops being noise. Taking
/// the plain minimum instead picks small windows on noise and costs EuRoC LOO 0.30 → 0.39 px.
const SUBPIX_SIGNIFICANT_RATIO: f64 = 2.0;
/// Patch side for the largest window: `2·(MAX + 1) + 1` samples, one ring beyond the window
/// for the central-difference gradients.
const SUBPIX_PATCH_MAX: usize = 2 * (SUBPIX_MAX_HALF as usize + 1) + 1;

/// Candidate window half-widths of [`corner_subpix`] for a marker whose side spans `cells`
/// cells over `side_px` pixels: `clamp(round(f·cell), 2, 4)` for `f` in 0.3, 0.5, 0.75 cell,
/// deduplicated, ascending. Returns the array and how many entries are used.
///
/// The bounds come from the corner model. The estimator assumes an ideal L-junction, so
/// gradients near the blurred apex are not orthogonal to `p − c`; the window must reach past
/// the blur (2 px floor, 4 px cap covering the PSFs measured on the benchmarks). The model
/// holds only within the black border cell and the quiet zone, so the window stays under a
/// cell. Inside those bounds the best size depends on how close other structure sits to the
/// corner, which only the image shows, so [`subpix_marker_corners`] tries each candidate and
/// keeps the most certain result.
pub(crate) fn corner_subpix_half_windows(side_px: f64, cells: usize) -> ([u32; 3], usize) {
    let cell = side_px / cells.max(1) as f64;
    let mut out = [0u32; 3];
    let mut n = 0;
    // The smallest window must fit well inside a cell, or the junction model cannot hold
    // anywhere in it: no candidate, so the corner keeps its seed.
    if f64::from(SUBPIX_MIN_HALF) > SUBPIX_MIN_WINDOW_CELL_FRACTION * cell {
        return (out, 0);
    }
    for f in SUBPIX_CELL_FRACTIONS {
        // In [2, 4] after the clamp, so the cast is exact.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let half = (f * cell)
            .round()
            .clamp(f64::from(SUBPIX_MIN_HALF), f64::from(SUBPIX_MAX_HALF))
            as u32;
        if n == 0 || out[n - 1] != half {
            out[n] = half;
            n += 1;
        }
    }
    (out, n)
}
/// Iteration cap and step tolerance (px) of [`corner_subpix`] (the `cv::cornerSubPix` values
/// aruco_nano uses).
const SUBPIX_MAX_ITER: u32 = 12;
const SUBPIX_EPS: f64 = 0.005;

/// Gradient-orthogonality refinement of a decoded marker's four corners (the
/// `decoder.corner_subpix` pass), keeping only moves the marker model accepts.
///
/// Each candidate window ([`corner_subpix_half_windows`]) is tried, and the largest accepted one
/// whose [`Subpix::uncertainty`] is within `SUBPIX_SIGNIFICANT_RATIO` of the smallest is kept:
/// a larger clean window averages more gradients, while structure entering the window inflates
/// the residual by far more than that ratio. The estimator converges
/// on any junction in its window, including structure outside the marker: a ChArUco
/// chessboard corner next to the marker's white square, or clutter beside a small tag. A
/// refined corner is therefore kept only if the image around it still shows the corner of this
/// marker's black border, as checked by [`marker_corner_consistent`]. A rejected corner keeps
/// its seed. Also returns which corners moved (bit `j` = corner `j`).
///
/// **Fusion with the whole-edge corner.** The junction estimate sees only the corner's
/// neighbourhood. Each edge also carries information along its whole length:
/// - [`fit_marker_edge`] fits it as a line with a covariance from its own scatter;
/// - [`intersect_edges`] gives the corner as the intersection of its two edges.
///
/// At an L-corner the two estimates are combined by inverse covariance when they agree
/// ([`fuse_corner`]); both covariances are statistically calibrated. At an X-junction
/// ([`junction_is_x`]: AprilGrid connectors) the edge lines carry the photometric edge offset
/// and the junction point does not, so the junction estimate stands alone. Fusion assumes a
/// pinhole image, where the edges are straight; the decoder only calls this on one.
pub(crate) fn subpix_marker_corners(
    img: &ImageView,
    seed: [[f64; 2]; 4],
    cells: usize,
) -> ([[f64; 2]; 4], u8) {
    let side = (0..4)
        .map(|j| {
            let (p, q) = (seed[j], seed[(j + 1) % 4]);
            (q[0] - p[0]).hypot(q[1] - p[1])
        })
        .sum::<f64>()
        * 0.25;
    let (halves, count) = corner_subpix_half_windows(side, cells);
    let probe = (0.5 * side / cells.max(1) as f64).max(1.0);
    let mut out = seed;
    let mut refined_bits = 0u8;
    let mut picks = [None::<Subpix>; 4];
    for j in 0..4 {
        let (prev, next) = (seed[(j + 3) % 4], seed[(j + 1) % 4]);
        if let Some(pick) = marker_junction(img, seed[j], prev, next, &halves[..count], probe) {
            out[j] = pick.corner;
            refined_bits |= 1 << j;
            picks[j] = Some(pick);
        }
    }
    // A corner the junction model rejects while both neighbours pass is usually not near the
    // marker's corner at all: quad extraction cut across a blurred apex or a touching square,
    // and no window reaches the junction. Its two edges still run straight from the good
    // neighbours, so they place it. A repaired corner can make its neighbour repairable, so
    // the sweep repeats until no corner changes.
    let mut repaired = count > 0;
    while repaired {
        repaired = false;
        for j in 0..4 {
            let (p, n) = ((j + 3) % 4, (j + 1) % 4);
            if picks[j].is_none()
                && picks[p].is_some()
                && picks[n].is_some()
                && let Some(pick) =
                    repair_corner(img, &out, j, side, cells, &halves[..count], probe)
            {
                out[j] = pick.corner;
                refined_bits |= 1 << j;
                picks[j] = Some(pick);
                repaired = true;
            }
        }
    }
    // Fuse each L-corner with the intersection of its two whole-edge lines.
    let cell = side / cells.max(1) as f64;
    // The band spans the blur and the refined corners' error, and stays within half a cell.
    let half = (0.5 * cell).clamp(2.0, EDGE_MAX_HALF_PX);
    // Edges between the locally refined corners: their directions are already corrected.
    let anchors = out;
    let edges: [Option<EdgeLine>; 4] = edge_lines(img, &anchors, half);
    // All four corners or none: a marker whose corners mix the two estimators (they differ by
    // the junction model's apex bias) is no longer a consistent square, which the pose turns
    // into rotation error.
    let mut fused = [[0.0f64; 2]; 4];
    for j in 0..4 {
        let Some(local) = picks[j] else {
            return (out, refined_bits);
        };
        let (prev, next) = (anchors[(j + 3) % 4], anchors[(j + 1) % 4]);
        if junction_is_x(img, local.corner, prev, next, anchors[j], probe) {
            return (out, refined_bits);
        }
        let (Some(before), Some(after)) = (edges[(j + 3) % 4], edges[j]) else {
            return (out, refined_bits);
        };
        let Some((line_corner, line_cov)) = intersect_edges(&before, &after) else {
            return (out, refined_bits);
        };
        let moved = (line_corner[0] - anchors[j][0]).hypot(line_corner[1] - anchors[j][1]);
        if moved > LINE_CORNER_MAX_MOVE_PX {
            return (out, refined_bits);
        }
        let Some(corner) = fuse_corner(local.corner, local.cov, line_corner, line_cov) else {
            return (out, refined_bits);
        };
        fused[j] = corner;
    }
    out = fused;
    (out, refined_bits)
}

/// The marker junction near `seed` (adjacent corners `prev`, `next`): [`corner_subpix`] at each
/// window half-width in `halves` (ascending), keeping solutions that pass
/// [`marker_corner_consistent`], and of those the largest window that is not significantly
/// less certain than the best one.
fn marker_junction(
    img: &ImageView,
    seed: [f64; 2],
    prev: [f64; 2],
    next: [f64; 2],
    halves: &[u32],
    probe: f64,
) -> Option<Subpix> {
    let mut accepted = [None::<Subpix>; 3];
    for (slot, &half) in accepted.iter_mut().zip(halves) {
        *slot = corner_subpix(img, seed, half).filter(|refined| {
            marker_corner_consistent(img, refined.corner, prev, next, seed, probe)
        });
    }
    let least = accepted
        .iter()
        .flatten()
        .map(|r| r.uncertainty)
        .fold(f64::INFINITY, f64::min);
    accepted
        .iter()
        .flatten()
        .rev()
        .find(|r| r.uncertainty <= SUBPIX_SIGNIFICANT_RATIO * least)
        .copied()
}

/// Largest repair, as a fraction of the marker side: a quarter side is two to three cells.
const REPAIR_MAX_SIDE_FRACTION: f64 = 0.25;
/// Edge refits of a repair, and the corner step (px) under which they stop.
const REPAIR_ITERATIONS: usize = 4;
const REPAIR_TOLERANCE_PX: f64 = 0.05;

/// Re-places corner `j` of `quad`, whose neighbours are good, at the crossing of its two edges.
///
/// Each edge is fitted on its half next to the good neighbour: a seed `e` px off moves the
/// seed line at most `0.43·e` from the true edge there, while the half next to the bad seed
/// may miss the edge entirely; the fit is repeated from each new crossing. The result must
/// then be confirmed as a marker junction by [`marker_junction`], so a repair is never weaker
/// evidence than an ordinary refined corner.
fn repair_corner(
    img: &ImageView,
    quad: &[[f64; 2]; 4],
    j: usize,
    side: f64,
    cells: usize,
    halves: &[u32],
    probe: f64,
) -> Option<Subpix> {
    let (prev, seed, next) = (quad[(j + 3) % 4], quad[j], quad[(j + 1) % 4]);
    let half = (0.5 * side / cells.max(1) as f64).clamp(2.0, EDGE_MAX_HALF_PX);
    let mid = |a: [f64; 2], b: [f64; 2]| [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
    let sample = |x: f64, y: f64| img.sample_bilinear(x, y);
    // An edge near the rim of the band pulls the station centroids towards the band centre, so
    // one fit only moves part way; refitting from each new crossing converges.
    let mut corner = seed;
    for _ in 0..REPAIR_ITERATIONS {
        let before = fit_marker_edge(&sample, prev, mid(prev, corner), half)?;
        let after = fit_marker_edge(&sample, mid(corner, next), next, half)?;
        let (crossing, _) = intersect_edges(&before, &after)?;
        let step = (crossing[0] - corner[0]).hypot(crossing[1] - corner[1]);
        corner = crossing;
        if step < REPAIR_TOLERANCE_PX {
            break;
        }
    }
    if (corner[0] - seed[0]).hypot(corner[1] - seed[1]) > REPAIR_MAX_SIDE_FRACTION * side {
        return None;
    }
    marker_junction(img, corner, prev, next, halves, probe)
}

/// Inverse-covariance fusion of two corner estimates `a`, `b` with covariances `[xx, xy, yy]`.
///
/// `None` unless both covariances are positive definite, the estimates agree under their summed
/// covariance (Mahalanobis d² within `FUSION_GATE_D2`, the χ²₂ 99 % point), and the fused corner
/// lies between them (within their separation of each), so a degenerate weight cannot throw
/// it away.
fn fuse_corner(a: [f64; 2], ca: [f64; 3], b: [f64; 2], cb: [f64; 3]) -> Option<[f64; 2]> {
    let inv = |c: [f64; 3]| {
        let det = c[0] * c[2] - c[1] * c[1];
        (c[0] > 0.0 && det > 1e-18 && det.is_finite())
            .then(|| [c[2] / det, -c[1] / det, c[0] / det])
    };
    let (ia, ib) = (inv(ca)?, inv(cb)?);
    let s = inv([ca[0] + cb[0], ca[1] + cb[1], ca[2] + cb[2]])?;
    let d = [b[0] - a[0], b[1] - a[1]];
    let d2 = d[0] * (s[0] * d[0] + s[1] * d[1]) + d[1] * (s[1] * d[0] + s[2] * d[1]);
    if d2 > FUSION_GATE_D2 {
        return None;
    }
    let cov = inv([ia[0] + ib[0], ia[1] + ib[1], ia[2] + ib[2]])?;
    let rhs = [
        ia[0] * a[0] + ia[1] * a[1] + ib[0] * b[0] + ib[1] * b[1],
        ia[1] * a[0] + ia[2] * a[1] + ib[1] * b[0] + ib[2] * b[1],
    ];
    let fused = [
        cov[0] * rhs[0] + cov[1] * rhs[1],
        cov[1] * rhs[0] + cov[2] * rhs[1],
    ];
    let span = d[0].hypot(d[1]) + 1e-9;
    let near = |p: [f64; 2]| (fused[0] - p[0]).hypot(fused[1] - p[1]) <= span;
    (near(a) && near(b)).then_some(fused)
}

/// χ²₂ 99 % point: the agreement test of [`fuse_corner`].
const FUSION_GATE_D2: f64 = 9.21;
/// Largest distance (px) a whole-edge line corner may sit from its seed and still be fused.
const LINE_CORNER_MAX_MOVE_PX: f64 = 3.0;
/// Fraction of each edge, at each end, left out of the line fit: the corner regions, where the
/// neighbouring edge and the junction blur the profile.
const EDGE_END_MARGIN: f64 = 0.15;
/// Spacing (px) of the edge-normal samples of one station.
const EDGE_PROFILE_STEP: f64 = 0.5;
/// Profile samples on each side of the central difference: 1 × 0.5 px, a 1 px baseline.
const EDGE_DERIVATIVE_TAPS: usize = 1;
/// Widest band half-width (px): the blur measured on the benchmarks plus the refined corners'
/// error. A wider band only adds samples and clutter.
const EDGE_MAX_HALF_PX: f64 = 4.0;
/// Profile buffer size: the widest band plus the taps.
const EDGE_MAX_PROFILE: usize = 2 * (8 + EDGE_DERIVATIVE_TAPS) + 1;
/// Stations per pixel of edge length. Half of 0.7 measured equal on ICRA, ChArUco and
/// render-tag, at 15 % less latency on dense frames.
const EDGE_STATION_DENSITY: f64 = 0.35;
/// Most stations sampled along one edge (stack buffer size).
const EDGE_MAX_STATIONS: usize = 256;

/// A marker edge as a straight line fitted through its edge stations, with the line's
/// uncertainty from the stations' own scatter.
#[derive(Clone, Copy)]
struct EdgeLine {
    /// Centroid of the stations.
    point: [f64; 2],
    /// Unit direction along the edge.
    dir: [f64; 2],
    /// Unit normal.
    normal: [f64; 2],
    /// Variance (px²) of the line's offset at `point`.
    var_offset: f64,
    /// Variance (rad²) of the line's angle.
    var_angle: f64,
}

/// Fits the marker edge from `p0` to `p1`.
///
/// Stations are spread over the middle of the edge (`EDGE_END_MARGIN` left out at each end).
/// At each, the edge's position is the `|∇I·n|`-weighted centroid of samples across the edge
/// within `±half` px. A total-least-squares line through the station positions gives the edge;
/// the residual variance `s²` of the stations about it gives `Var(offset) = s²/N` and
/// `Var(angle) = s²/Σt²`. Bit edges crossing the band, lens curvature or clutter scatter the
/// stations and so inflate the line's own uncertainty, which is what keeps a contaminated edge
/// from dominating a fusion.
/// [`fit_marker_edge`] for the four sides of `quad`. When every profile sample is inside the
/// image, as for nearly every marker, the samples skip the per-sample bounds check.
fn edge_lines(img: &ImageView, quad: &[[f64; 2]; 4], half: f64) -> [Option<EdgeLine>; 4] {
    // Profiles stay within `half` plus the derivative taps of the sides.
    let reach = half + EDGE_PROFILE_STEP * EDGE_DERIVATIVE_TAPS as f64 + 1.0;
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in quad {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k] - reach);
            hi[k] = hi[k].max(p[k] + reach);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let inside = lo[0] >= 1.0
        && lo[1] >= 1.0
        && hi[0] <= img.width as f64 - 2.0
        && hi[1] <= img.height as f64 - 2.0;
    if inside {
        #[allow(
            unsafe_code,
            reason = "the bounds of every sample are checked once per marker above, so the per-sample checks of the safe sampler are redundant on this hot path"
        )]
        // SAFETY: every sample point lies in `[1, width − 2] × [1, height − 2]` (bounded above),
        // so after the sampler's −0.5 shift both bilinear taps are valid pixel indices.
        let sample = |x: f64, y: f64| unsafe { img.sample_bilinear_unchecked(x, y) };
        core::array::from_fn(|e| fit_marker_edge(&sample, quad[e], quad[(e + 1) % 4], half))
    } else {
        let sample = |x: f64, y: f64| img.sample_bilinear(x, y);
        core::array::from_fn(|e| fit_marker_edge(&sample, quad[e], quad[(e + 1) % 4], half))
    }
}

fn fit_marker_edge(
    sample: &impl Fn(f64, f64) -> f64,
    p0: [f64; 2],
    p1: [f64; 2],
    half: f64,
) -> Option<EdgeLine> {
    let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let len = dx.hypot(dy);
    if len < 8.0 {
        return None;
    }
    let (ux, uy) = (dx / len, dy / len);
    let (nx, ny) = (-uy, ux);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let stations = ((EDGE_STATION_DENSITY * len).round() as usize).clamp(6, EDGE_MAX_STATIONS);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let steps = (half / EDGE_PROFILE_STEP).round().max(1.0) as usize;
    // The intensity profile extends one derivative baseline beyond the band on each side.
    let reach = steps + EDGE_DERIVATIVE_TAPS;
    let samples = 2 * reach + 1;
    if samples > EDGE_MAX_PROFILE {
        return None;
    }
    let mut profile = [0.0f64; EDGE_MAX_PROFILE];
    let mut points = [[0.0f64; 2]; EDGE_MAX_STATIONS];
    let mut count = 0usize;
    for k in 0..stations {
        let t = EDGE_END_MARGIN + (1.0 - 2.0 * EDGE_END_MARGIN) * k as f64 / (stations - 1) as f64;
        let (cx, cy) = (p0[0] + t * dx, p0[1] + t * dy);
        for (idx, slot) in profile[..samples].iter_mut().enumerate() {
            let o = (idx as f64 - reach as f64) * EDGE_PROFILE_STEP;
            *slot = sample(cx + o * nx, cy + o * ny);
        }
        // Gradient magnitude across the edge as a 1 px central difference of the profile.
        let (mut sum_w, mut sum_wo) = (0.0, 0.0);
        for at in EDGE_DERIVATIVE_TAPS..samples - EDGE_DERIVATIVE_TAPS {
            let w = (profile[at + EDGE_DERIVATIVE_TAPS] - profile[at - EDGE_DERIVATIVE_TAPS]).abs();
            let o = (at as f64 - reach as f64) * EDGE_PROFILE_STEP;
            sum_w += w;
            sum_wo += w * o;
        }
        if sum_w > 1e-9 {
            let o = sum_wo / sum_w;
            points[count] = [cx + o * nx, cy + o * ny];
            count += 1;
        }
    }
    if count < 5 {
        return None;
    }
    let pts = &points[..count];
    let n_pts = count as f64;
    let mx = pts.iter().map(|p| p[0]).sum::<f64>() / n_pts;
    let my = pts.iter().map(|p| p[1]).sum::<f64>() / n_pts;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (ex, ey) = (p[0] - mx, p[1] - my);
        sxx += ex * ex;
        sxy += ex * ey;
        syy += ey * ey;
    }
    // Principal direction of the 2×2 scatter matrix.
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let dir = [theta.cos(), theta.sin()];
    let normal = [-dir[1], dir[0]];
    let (mut sum_rr, mut sum_tt) = (0.0, 0.0);
    for p in pts {
        let (ex, ey) = (p[0] - mx, p[1] - my);
        let across = ex * normal[0] + ey * normal[1];
        let along = ex * dir[0] + ey * dir[1];
        sum_rr += across * across;
        sum_tt += along * along;
    }
    // Upper 95 % bound on the residual variance: with few stations the sample variance is
    // itself uncertain (χ² with n − 2 degrees of freedom), and an underestimate would make a
    // short edge over-confident in the fusion.
    let dof = n_pts - 2.0;
    let s2 = sum_rr / dof * chi2_variance_inflation(dof);
    (sum_tt > 1e-9).then_some(EdgeLine {
        point: [mx, my],
        dir,
        normal,
        var_offset: s2 / n_pts,
        var_angle: s2 / sum_tt,
    })
}

/// `k / χ²₀.₀₅(k)`: the factor taking a sample variance with `k` degrees of freedom to the
/// upper end of its one-sided 95 % confidence interval. The χ² quantile uses the Wilson–Hilferty
/// cube-root normal approximation (within 1 % for `k ≥ 3`).
fn chi2_variance_inflation(k: f64) -> f64 {
    const Z_05: f64 = -1.644_853_6;
    let h = 2.0 / (9.0 * k);
    let quantile = k * (1.0 - h + Z_05 * h.sqrt()).powi(3);
    if quantile > 0.0 {
        k / quantile
    } else {
        f64::INFINITY
    }
}

/// Intersection of two edge lines and its covariance `[xx, xy, yy]`.
///
/// Each line constrains the corner along its normal, with variance `Var(offset) +
/// Var(angle)·ℓ²` at distance `ℓ` from its centroid along the line; the two constraints map to
/// image coordinates through the inverse of the normals matrix.
fn intersect_edges(a: &EdgeLine, b: &EdgeLine) -> Option<([f64; 2], [f64; 3])> {
    let det = a.normal[0] * b.normal[1] - a.normal[1] * b.normal[0];
    if det.abs() < 1e-6 {
        return None;
    }
    let ra = a.normal[0] * a.point[0] + a.normal[1] * a.point[1];
    let rb = b.normal[0] * b.point[0] + b.normal[1] * b.point[1];
    // Inverse of [[a.n], [b.n]].
    let inv = [
        [b.normal[1] / det, -a.normal[1] / det],
        [-b.normal[0] / det, a.normal[0] / det],
    ];
    let corner = [
        inv[0][0] * ra + inv[0][1] * rb,
        inv[1][0] * ra + inv[1][1] * rb,
    ];
    let along =
        |l: &EdgeLine| (corner[0] - l.point[0]) * l.dir[0] + (corner[1] - l.point[1]) * l.dir[1];
    let va = a.var_offset + a.var_angle * along(a).powi(2);
    let vb = b.var_offset + b.var_angle * along(b).powi(2);
    let cov = [
        inv[0][0] * inv[0][0] * va + inv[0][1] * inv[0][1] * vb,
        inv[0][0] * inv[1][0] * va + inv[0][1] * inv[1][1] * vb,
        inv[1][0] * inv[1][0] * va + inv[1][1] * inv[1][1] * vb,
    ];
    Some((corner, cov))
}

/// Whether the marker corner at `corner` is an X-junction: the outward diagonal reads as dark
/// as the inward one (an AprilGrid connector square touches the corner). There the whole-edge
/// lines carry the photometric edge offset while the junction point does not, so the two
/// estimators disagree by a bias rather than by noise and are not fused. Same probe geometry as
/// [`marker_corner_consistent`].
fn junction_is_x(
    img: &ImageView,
    corner: [f64; 2],
    prev: [f64; 2],
    next: [f64; 2],
    at: [f64; 2],
    probe: f64,
) -> bool {
    let unit = |to: [f64; 2]| {
        let (dx, dy) = (to[0] - at[0], to[1] - at[1]);
        let n = dx.hypot(dy);
        [dx / n, dy / n]
    };
    let (u, v) = (unit(prev), unit(next));
    let at_offset = |a: f64, b: f64| {
        img.sample_bilinear(
            corner[0] + probe * (a * u[0] + b * v[0]),
            corner[1] + probe * (a * u[1] + b * v[1]),
        )
    };
    let inward = at_offset(1.0, 1.0);
    let outward = at_offset(-1.0, -1.0);
    let bright = 0.5 * (at_offset(1.0, -1.0) + at_offset(-1.0, 1.0));
    outward < 0.5 * (inward + bright)
}

/// Whether `corner` looks like the corner of a dark-bordered marker whose adjacent corners are
/// `prev` and `next` (seed positions; `at` is the seed of this corner, for the edge directions).
///
/// With unit vectors `u`, `v` along the two edges from the corner, the image is sampled half a
/// cell along the inward diagonal (`u + v`, inside the black border cell) and the two side
/// diagonals (`u − v`, `v − u`, in the quiet zone beside each edge). The inward sample must be
/// darker than the side samples, and the side samples must agree within half that contrast.
/// The outward diagonal is not constrained: on an AprilGrid board the tag corner touches a
/// black connector square there, and the corner is still the true junction. A capture by
/// outside structure fails this test, because the inward probe then lands in the bright quiet
/// zone or one side probe lands on the other structure.
fn marker_corner_consistent(
    img: &ImageView,
    corner: [f64; 2],
    prev: [f64; 2],
    next: [f64; 2],
    at: [f64; 2],
    probe: f64,
) -> bool {
    let unit = |to: [f64; 2]| {
        let (dx, dy) = (to[0] - at[0], to[1] - at[1]);
        let n = dx.hypot(dy);
        [dx / n, dy / n]
    };
    let (u, v) = (unit(prev), unit(next));
    let at_offset = |a: f64, b: f64| {
        img.sample_bilinear(
            corner[0] + probe * (a * u[0] + b * v[0]),
            corner[1] + probe * (a * u[1] + b * v[1]),
        )
    };
    let inward = at_offset(1.0, 1.0);
    let (side_u, side_v) = (at_offset(1.0, -1.0), at_offset(-1.0, 1.0));
    let contrast = 0.5 * (side_u + side_v) - inward;
    contrast > 0.0 && (side_u - side_v).abs() < 0.5 * contrast
}

/// Gradient-orthogonality corner refinement, the `cv::cornerSubPix` model.
///
/// At a corner `c`, every gradient `∇I(p)` in the neighbourhood is orthogonal to `p − c`: on a
/// flat patch the gradient vanishes and on an edge through `c` it is normal to the edge. So
/// `c` solves `(Σ w·∇I ∇Iᵀ) c = Σ w·∇I ∇Iᵀ p` over the `(2·half + 1)²` window, with Gaussian
/// weights `w = exp(−(dx² + dy²)/half²)`. The window is re-centred and the system re-solved
/// until the step is under `SUBPIX_EPS` px (at most `SUBPIX_MAX_ITER` times). Gradients are
/// central differences of a bilinearly resampled patch, as OpenCV computes them.
///
/// Returns `None` when the result leaves the `half`-px box around the seed, the window leaves
/// the image, or the normal matrix is singular (a flat or single-edge patch). Locus pixel
/// convention (+0.5 centres), like the rest of the crate. `half` must be in
/// `1..=SUBPIX_MAX_HALF`.
pub(crate) fn corner_subpix(img: &ImageView, seed: [f64; 2], half: u32) -> Option<Subpix> {
    debug_assert!((1..=SUBPIX_MAX_HALF).contains(&half));
    let hw = half as usize;
    let span = 2 * hw + 1; // window side
    let side = span + 2; // plus one gradient ring
    let mut weights = [0.0f64; SUBPIX_PATCH_MAX];
    let inv_hw2 = 1.0 / f64::from(half * half);
    for (k, weight) in weights.iter_mut().enumerate().take(span) {
        let off = k as f64 - hw as f64;
        *weight = (-off * off * inv_hw2).exp();
    }
    let mut patch = [0.0f64; SUBPIX_PATCH_MAX * SUBPIX_PATCH_MAX];
    let reach = (hw + 1) as f64;

    let mut corner = seed;
    let mut uncertainty = f64::INFINITY;
    let mut cov = [f64::INFINITY, 0.0, f64::INFINITY];
    for _ in 0..SUBPIX_MAX_ITER {
        // Patch sample (row, col) sits at corner + (col − hw − 1, row − hw − 1); in array
        // coordinates (pixel centres at integers) its top-left is corner − 0.5 − (hw + 1).
        let x0 = corner[0] - 0.5 - reach;
        let y0 = corner[1] - 0.5 - reach;
        if !(x0 >= 0.0 && y0 >= 0.0)
            || x0 + (side as f64) >= img.width as f64
            || y0 + (side as f64) >= img.height as f64
        {
            return None;
        }
        // Non-negative and in range: checked just above.
        #[allow(clippy::cast_sign_loss)]
        let (ix, iy) = (x0.floor() as usize, y0.floor() as usize);
        let (fx, fy) = (x0 - ix as f64, y0 - iy as f64);
        let (w00, w10, w01, w11) = (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        );
        for row in 0..side {
            let top = &img.data[(iy + row) * img.stride + ix..];
            let bottom = &img.data[(iy + row + 1) * img.stride + ix..];
            for col in 0..side {
                patch[row * side + col] = w00 * f64::from(top[col])
                    + w10 * f64::from(top[col + 1])
                    + w01 * f64::from(bottom[col])
                    + w11 * f64::from(bottom[col + 1]);
            }
        }

        // Normal equations: [sxx sxy; sxy syy] · step = [rhs_x; rhs_y], with q = ∇I·(p − c)
        // per sample; `sqq` = Σ w·q² gives the residual at the solution in closed form.
        let (mut sxx, mut sxy, mut syy, mut rhs_x, mut rhs_y) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut sqq = 0.0;
        let (mut bxx, mut bxy, mut byy, mut sw) = (0.0, 0.0, 0.0, 0.0);
        for wy in 0..span {
            let py = wy as f64 - hw as f64;
            let row = (wy + 1) * side;
            for wx in 0..span {
                let px = wx as f64 - hw as f64;
                let at = row + wx + 1;
                let gx = patch[at + 1] - patch[at - 1];
                let gy = patch[at + side] - patch[at - side];
                let weight = weights[wy] * weights[wx];
                let (gxx, gxy, gyy) = (gx * gx * weight, gx * gy * weight, gy * gy * weight);
                sxx += gxx;
                sxy += gxy;
                syy += gyy;
                rhs_x += gxx * px + gxy * py;
                rhs_y += gxy * px + gyy * py;
                let q = gx * px + gy * py;
                sqq += weight * q * q;
                let w2 = weight * weight;
                bxx += w2 * gx * gx;
                bxy += w2 * gx * gy;
                byy += w2 * gy * gy;
                sw += weight;
            }
        }
        let det = sxx * syy - sxy * sxy;
        if det.abs() <= f64::EPSILON * f64::EPSILON {
            return None;
        }
        let step = [
            (syy * rhs_x - sxy * rhs_y) / det,
            (sxx * rhs_y - sxy * rhs_x) / det,
        ];
        corner = [corner[0] + step[0], corner[1] + step[1]];
        // Σ w·(∇I·(c' − p))² = sᵀAs − 2sᵀb + Σ w·q² for the step s from window centre c.
        let residual = step[0] * (sxx * step[0] + sxy * step[1])
            + step[1] * (sxy * step[0] + syy * step[1])
            - 2.0 * (step[0] * rhs_x + step[1] * rhs_y)
            + sqq;
        uncertainty = residual.max(0.0) * (sxx + syy) / det;
        cov = sandwich_covariance([sxx, sxy, syy], [bxx, bxy, byy], det, residual, sw);
        if step[0] * step[0] + step[1] * step[1] <= SUBPIX_EPS * SUBPIX_EPS {
            break;
        }
    }
    let limit = f64::from(half);
    if (corner[0] - seed[0]).abs() <= limit && (corner[1] - seed[1]).abs() <= limit {
        Some(Subpix {
            corner,
            uncertainty,
            cov,
        })
    } else {
        None
    }
}

/// Weighted least-squares sandwich covariance `σ²·A⁻¹BA⁻¹` (`[xx, xy, yy]`) for the normal
/// matrix `a = [sxx, sxy, syy]` (determinant `det`), `b = Σw²∇I∇Iᵀ` and `σ² = residual / sum_w`.
fn sandwich_covariance(a: [f64; 3], b: [f64; 3], det: f64, residual: f64, sum_w: f64) -> [f64; 3] {
    let (ixx, ixy, iyy) = (a[2] / det, -a[1] / det, a[0] / det);
    let (mxx, mxy, myx, myy) = (
        ixx * b[0] + ixy * b[1],
        ixx * b[1] + ixy * b[2],
        ixy * b[0] + iyy * b[1],
        ixy * b[1] + iyy * b[2],
    );
    let sigma2 = residual.max(0.0) / sum_w.max(f64::EPSILON);
    [
        sigma2 * (mxx * ixx + mxy * ixy),
        sigma2 * (mxx * ixy + mxy * iyy),
        sigma2 * (myx * ixy + myy * iyy),
    ]
}

/// A converged [`corner_subpix`] solution.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Subpix {
    /// The refined corner (Locus pixel convention).
    pub(crate) corner: [f64; 2],
    /// `Σ w·r² · tr(A⁻¹)`: the weighted residual of the gradient-orthogonality fit times the
    /// trace of the inverse normal matrix, i.e. the trace of the corner covariance up to the
    /// window's weight normalisation. Comparable across windows on the same corner.
    pub(crate) uncertainty: f64,
    /// Corner covariance (px²; `[xx, xy, yy]`): the weighted least-squares sandwich
    /// `σ²·A⁻¹BA⁻¹` with `A = Σw∇I∇Iᵀ`, `B = Σw²∇I∇Iᵀ` and `σ² = Σw·r²/Σw`.
    pub(crate) cov: [f64; 3],
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used
)]
mod subpix_tests {
    use super::corner_subpix;
    use crate::image::ImageView;

    const W: usize = 64;

    /// A dark wedge with its apex at `apex` (Locus convention), bounded by the rays at
    /// `theta` and `theta + 90°`, on a bright background; 8×8 supersampled pixel coverage.
    fn wedge(apex: [f64; 2], theta: f64) -> Vec<u8> {
        let (u, v) = ([theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]);
        let mut img = vec![0u8; W * W];
        for y in 0..W {
            for x in 0..W {
                let mut dark = 0;
                for sy in 0..8 {
                    for sx in 0..8 {
                        let px = x as f64 + (f64::from(sx) + 0.5) / 8.0 - apex[0];
                        let py = y as f64 + (f64::from(sy) + 0.5) / 8.0 - apex[1];
                        if px * u[0] + py * u[1] > 0.0 && px * v[0] + py * v[1] > 0.0 {
                            dark += 1;
                        }
                    }
                }
                img[y * W + x] = (220.0 - 180.0 * f64::from(dark) / 64.0).round() as u8;
            }
        }
        img
    }

    /// Parity with `cv2.cornerSubPix(img, seed, (4, 4), (-1, -1), (MAX_ITER | EPS, 12, 0.005))`
    /// (OpenCV 5.0, run on the same images; its corners shifted to Locus' +0.5 convention).
    /// Agreement is to OpenCV's float32 rounding, not to the apex: the gradient-orthogonality
    /// model itself leaves ~0.2 px on an unblurred, box-filtered wedge.
    #[test]
    fn matches_opencv_corner_subpix() {
        let opencv = [
            [31.246_89, 30.818_348],
            [31.649_41, 30.665_356],
            [31.308_928, 30.592_333],
            [31.444_061, 30.542_173],
        ];
        for (k, theta) in [0.0f64, 0.3, 0.9, 1.4].into_iter().enumerate() {
            let apex = [31.27 + 0.11 * k as f64, 30.64 - 0.07 * k as f64];
            let data = wedge(apex, theta);
            let img = ImageView::new(&data, W, W, W).unwrap();
            let got = corner_subpix(&img, [apex[0] + 1.3, apex[1] - 0.9], 4)
                .unwrap()
                .corner;
            let d = (got[0] - opencv[k][0]).hypot(got[1] - opencv[k][1]);
            assert!(
                d < 0.01,
                "theta={theta}: {got:?} vs OpenCV {:?} ({d:.4} px)",
                opencv[k]
            );
        }
    }

    #[test]
    fn candidate_windows_follow_the_cell_within_the_blur_bounds() {
        use super::corner_subpix_half_windows as hw;
        let used = |(h, n): ([u32; 3], usize)| h[..n].to_vec();
        // 8-cell markers: cell = side / 8; candidates round(0.3/0.5/0.75 · cell) in [2, 4].
        assert_eq!(used(hw(16.0, 8)), Vec::<u32>::new()); // cell 2 px: no window fits
        assert_eq!(used(hw(24.0, 8)), Vec::<u32>::new()); // cell 3 px: under 3.3 px
        assert_eq!(used(hw(28.0, 8)), vec![2, 3]); // cell 3.5 px: 1.05, 1.75, 2.6
        assert_eq!(used(hw(32.0, 8)), vec![2, 3]); // cell 4 px: 1.2, 2, 3
        assert_eq!(used(hw(48.0, 8)), vec![2, 3, 4]); // cell 6 px: 1.8, 3, 4.5
        assert_eq!(used(hw(400.0, 8)), vec![4]);
        assert_eq!(used(hw(0.0, 8)), Vec::<u32>::new());
    }

    /// Bright canvas with the given dark axis-aligned squares `[x0, x1) × [y0, y1)`.
    fn squares(dark: &[[usize; 4]]) -> Vec<u8> {
        let mut img = vec![220u8; W * W];
        for &[x0, x1, y0, y1] in dark {
            for y in y0..y1 {
                for x in x0..x1 {
                    img[y * W + x] = 40;
                }
            }
        }
        img
    }

    #[test]
    fn marker_corner_check_accepts_marker_corners_and_rejects_outside_junctions() {
        use super::marker_corner_consistent as consistent;
        // Marker [20, 44)²; its top-left corner sits at (20, 20) in Locus coordinates, with
        // adjacent corners (20, 44) and (44, 20).
        let (at, prev, next) = ([20.0, 20.0], [20.0, 44.0], [44.0, 20.0]);
        let marker = squares(&[[20, 44, 20, 44]]);
        let img = ImageView::new(&marker, W, W, W).unwrap();
        assert!(consistent(&img, at, prev, next, at, 3.0));

        // AprilGrid: a black connector square touches the corner from outside.
        let grid = squares(&[[20, 44, 20, 44], [8, 20, 8, 20]]);
        let img = ImageView::new(&grid, W, W, W).unwrap();
        assert!(consistent(&img, at, prev, next, at, 3.0));

        // ChArUco-like capture: the corner of an outside square in the quiet zone.
        let board = squares(&[[20, 44, 20, 44], [4, 14, 4, 14]]);
        let img = ImageView::new(&board, W, W, W).unwrap();
        assert!(!consistent(&img, [14.0, 14.0], prev, next, at, 3.0));
    }

    /// A marker seeded with one corner 4 px off its junction, along an edge (quad extraction
    /// cutting across a blurred apex): the junction model rejects that seed, and the corner
    /// is re-placed from its two edges; the good corners are untouched.
    #[test]
    fn repairs_a_corner_seeded_off_the_junction() {
        use super::subpix_marker_corners;
        // 8-cell marker [16, 48)² (4 px cells): a one-cell black ring around a bright payload
        // with one dark bit, on a bright quiet zone.
        let mut data = squares(&[[16, 48, 16, 48]]);
        for y in 20..44 {
            for x in 20..44 {
                data[y * W + x] = if (28..32).contains(&x) && (24..28).contains(&y) {
                    40
                } else {
                    220
                };
            }
        }
        let img = ImageView::new(&data, W, W, W).unwrap();
        let truth = [[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]];
        let mut seed = truth;
        seed[2] = [48.0, 44.0];
        let (got, bits) = subpix_marker_corners(&img, seed, 8);
        assert_eq!(bits, 0b1111);
        for (g, t) in got.iter().zip(&truth) {
            let d = (g[0] - t[0]).hypot(g[1] - t[1]);
            assert!(d < 0.3, "{got:?} vs {truth:?}");
        }
    }

    /// A tilted, anti-aliased straight edge: dark where `(p − origin)·normal > 0`.
    fn tilted_edge(origin: [f64; 2], angle: f64) -> Vec<u8> {
        let normal = [-angle.sin(), angle.cos()];
        let mut img = vec![0u8; W * W];
        for y in 0..W {
            for x in 0..W {
                let mut dark = 0;
                for sy in 0..8 {
                    for sx in 0..8 {
                        let px = x as f64 + (f64::from(sx) + 0.5) / 8.0 - origin[0];
                        let py = y as f64 + (f64::from(sy) + 0.5) / 8.0 - origin[1];
                        if px * normal[0] + py * normal[1] > 0.0 {
                            dark += 1;
                        }
                    }
                }
                img[y * W + x] = (220.0 - 180.0 * f64::from(dark) / 64.0).round() as u8;
            }
        }
        img
    }

    #[test]
    fn edge_line_recovers_a_tilted_edge() {
        use super::fit_marker_edge;
        let (origin, angle) = ([31.3, 30.8], 0.21f64);
        let data = tilted_edge(origin, angle);
        let img = ImageView::new(&data, W, W, W).unwrap();
        let dir = [angle.cos(), angle.sin()];
        // Seed endpoints 0.6 px off the edge, along the normal.
        let off = [-dir[1] * 0.6, dir[0] * 0.6];
        let p0 = [
            origin[0] - 20.0 * dir[0] + off[0],
            origin[1] - 20.0 * dir[1] + off[1],
        ];
        let p1 = [
            origin[0] + 20.0 * dir[0] + off[0],
            origin[1] + 20.0 * dir[1] + off[1],
        ];
        let sample = |x: f64, y: f64| img.sample_bilinear(x, y);
        let line = fit_marker_edge(&sample, p0, p1, 2.0).expect("edge fits");
        let dist = (origin[0] - line.point[0]) * line.normal[0]
            + (origin[1] - line.point[1]) * line.normal[1];
        assert!(dist.abs() < 0.05, "line {dist:.3} px off the edge");
        let cross = line.dir[0] * dir[1] - line.dir[1] * dir[0];
        assert!(cross.abs() < 0.003, "line direction off by {cross:.4} rad");
        assert!(line.var_offset > 0.0 && line.var_angle > 0.0);
    }

    #[test]
    fn chi2_inflation_matches_tabulated_quantiles() {
        use super::chi2_variance_inflation as f;
        // k / χ²₀.₀₅(k) from tables: k = 8 → 8/2.733, 28 → 28/16.928, 100 → 100/77.929.
        for (k, q) in [(8.0, 2.733), (28.0, 16.928), (100.0, 77.929)] {
            assert!((f(k) - k / q).abs() / (k / q) < 0.01, "k = {k}: {}", f(k));
        }
    }

    #[test]
    fn edge_lines_intersect_at_their_crossing() {
        use super::{EdgeLine, intersect_edges};
        let line = |point: [f64; 2], dir: [f64; 2]| EdgeLine {
            point,
            dir,
            normal: [-dir[1], dir[0]],
            var_offset: 0.01,
            var_angle: 1e-4,
        };
        let a = line([10.0, 5.0], [1.0, 0.0]);
        let b = line([3.0, 20.0], [0.0, 1.0]);
        let (corner, cov) = intersect_edges(&a, &b).expect("crossing lines");
        assert!((corner[0] - 3.0).abs() < 1e-9 && (corner[1] - 5.0).abs() < 1e-9);
        // Each axis carries its line's offset variance plus the angle term at its lever arm.
        assert!((cov[0] - (0.01 + 1e-4 * 225.0)).abs() < 1e-9, "{cov:?}");
        assert!((cov[2] - (0.01 + 1e-4 * 49.0)).abs() < 1e-9, "{cov:?}");
        assert!(cov[1].abs() < 1e-12);
    }

    #[test]
    fn junction_test_tells_x_corners_from_l_corners() {
        use super::junction_is_x;
        let (at, prev, next) = ([20.0, 20.0], [20.0, 44.0], [44.0, 20.0]);
        let marker = squares(&[[20, 44, 20, 44]]);
        let img = ImageView::new(&marker, W, W, W).unwrap();
        assert!(!junction_is_x(&img, at, prev, next, at, 3.0));
        let grid = squares(&[[20, 44, 20, 44], [8, 20, 8, 20]]);
        let img = ImageView::new(&grid, W, W, W).unwrap();
        assert!(junction_is_x(&img, at, prev, next, at, 3.0));
    }

    #[test]
    fn fusion_weights_by_covariance_and_refuses_disagreement() {
        use super::fuse_corner;
        // Equal isotropic covariances: the midpoint.
        let f = fuse_corner([0.0, 0.0], [0.04, 0.0, 0.04], [0.2, 0.0], [0.04, 0.0, 0.04]).unwrap();
        assert!((f[0] - 0.1).abs() < 1e-12 && f[1].abs() < 1e-12);
        // A four times more certain estimate gets four times the weight.
        let f = fuse_corner([0.0, 0.0], [0.01, 0.0, 0.01], [0.5, 0.0], [0.04, 0.0, 0.04]).unwrap();
        assert!((f[0] - 0.1).abs() < 1e-12);
        // 1 px apart with 0.1 px standard deviations: inconsistent, not fused.
        assert!(
            fuse_corner([0.0, 0.0], [0.01, 0.0, 0.01], [1.0, 0.0], [0.01, 0.0, 0.01]).is_none()
        );
        // A covariance that is not positive definite is refused.
        assert!(
            fuse_corner(
                [0.0, 0.0],
                [0.01, 0.02, 0.01],
                [0.1, 0.0],
                [0.01, 0.0, 0.01]
            )
            .is_none()
        );
    }

    #[test]
    fn declines_without_a_corner() {
        let flat = vec![128u8; W * W];
        let img = ImageView::new(&flat, W, W, W).unwrap();
        assert!(corner_subpix(&img, [32.2, 31.7], 4).is_none());

        // A single straight edge constrains one direction only: the normal matrix is singular.
        let edge: Vec<u8> = (0..W * W)
            .map(|i| if i % W < 32 { 40 } else { 220 })
            .collect();
        let img = ImageView::new(&edge, W, W, W).unwrap();
        assert!(corner_subpix(&img, [32.0, 30.5], 4).is_none());
    }

    #[test]
    fn declines_when_the_window_leaves_the_image() {
        let data = wedge([3.0, 3.0], 0.0);
        let img = ImageView::new(&data, W, W, W).unwrap();
        assert!(corner_subpix(&img, [3.4, 2.6], 4).is_none());
    }
}
