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
#[allow(clippy::too_many_arguments)]
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
/// next)` triplet that [`refine_corner`] expects.
pub(crate) fn refine_all_quad_corners(
    arena: &Bump,
    img: &ImageView,
    pts: [Point; 4],
    sigma: f64,
    decimation: usize,
    use_erf: bool,
) -> [Point; 4] {
    [
        refine_corner(
            arena, img, pts[0], pts[3], pts[1], sigma, decimation, use_erf,
        ),
        refine_corner(
            arena, img, pts[1], pts[0], pts[2], sigma, decimation, use_erf,
        ),
        refine_corner(
            arena, img, pts[2], pts[1], pts[3], sigma, decimation, use_erf,
        ),
        refine_corner(
            arena, img, pts[3], pts[2], pts[0], sigma, decimation, use_erf,
        ),
    ]
}

/// Single-corner refinement: intersect two edge-line fits at point `p`,
/// using its neighbours `p_prev` and `p_next` to define the edges.
///
/// `use_erf = true` runs the PSF-blurred Gauss-Newton fit and falls
/// back to the gradient-peak fit on sample shortfall. `use_erf = false`
/// runs only the gradient-peak fit.
#[allow(clippy::too_many_arguments)]
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
    let line1 = if use_erf {
        refine_edge_erf(arena, img, p_prev, p, sigma, decimation)
            .or_else(|| fit_edge_line(img, p_prev, p, decimation))
    } else {
        fit_edge_line(img, p_prev, p, decimation)
    };

    let line2 = if use_erf {
        refine_edge_erf(arena, img, p, p_next, sigma, decimation)
            .or_else(|| fit_edge_line(img, p, p_next, decimation))
    } else {
        fit_edge_line(img, p, p_next, decimation)
    };

    if let (Some(l1), Some(l2)) = (line1, line2) {
        let det = l1.0 * l2.1 - l2.0 * l1.1;
        if det.abs() > 1e-6 {
            let x = (l1.1 * l2.2 - l2.1 * l1.2) / det;
            let y = (l2.0 * l1.2 - l1.0 * l2.2) / det;

            let dist_sq = (x - p.x).powi(2) + (y - p.y).powi(2);
            // Base displacement gate. The ERF (2-DOF) path can legitimately
            // move a corner further from the seed than the gradient-peak path,
            // because 2-DOF also corrects the seed's 1–3° rotation (the seed is
            // a Douglas-Peucker chord midpoint): it gets 3.0 px. The
            // gradient-peak path (`use_erf == false`) keeps the original 2.0 px
            // gate — widening it gains nothing there and only admits more
            // runaway. (For ERF: 2.5 clipped the tag16h5 P99 rotation wins back
            // to ~27°; 4.0 caused board_charuco P99 translation +241 %; 3.0 is
            // the empirical sweet spot across the insta suite.)
            let base_max_dist = if use_erf { 3.0 } else { 2.0 };
            let max_dist = if decimation > 1 {
                (decimation as f64) + base_max_dist
            } else {
                base_max_dist
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
        if resolve_route_refinement(config, batch.routed_to[i]) != CornerRefinementMode::Gwlf {
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
