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
/// `route_refinement` comes from [`crate::quad::resolve_route`]. `None`
/// and `Gwlf` both pass corners through and propagate any GN
/// covariances; `Gwlf`'s actual refinement happens later in
/// [`apply_detector_gwlf`]. `Erf` overwrites corners (and invalidates
/// the GN covariances) with the PSF-blurred Gauss-Newton fit.
pub(crate) fn refine_quad_corners(
    arena: &Bump,
    refinement_img: &ImageView,
    quad_pts: [Point; 4],
    gn_covs: CornerCovariances,
    route_refinement: CornerRefinementMode,
    sigma: f64,
    decimation: usize,
) -> ([Point; 4], CornerCovariances) {
    if matches!(
        route_refinement,
        CornerRefinementMode::None | CornerRefinementMode::Gwlf
    ) {
        return (quad_pts, gn_covs);
    }

    debug_assert_eq!(route_refinement, CornerRefinementMode::Erf);
    let corners = refine_all_quad_corners(arena, refinement_img, quad_pts, sigma, decimation, true);

    (corners, [[0.0; 4]; 4])
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

/// Resolves the `(extraction, refinement)` pair for a candidate from
/// the persisted `route_label`. Mirrors [`crate::quad::resolve_route`]
/// for the post-extraction stage, where PPB is no longer in scope.
#[inline]
fn resolve_candidate_route(
    config: &DetectorConfig,
    route_label: u8,
) -> (QuadExtractionMode, CornerRefinementMode) {
    match config.quad_extraction_policy {
        QuadExtractionPolicy::Static => (config.quad_extraction_mode, config.refinement_mode),
        QuadExtractionPolicy::AdaptivePpb(cfg) => {
            if route_label == ROUTED_TO_HIGH {
                (cfg.high_extraction, cfg.high_refinement)
            } else {
                (cfg.low_extraction, cfg.low_refinement)
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
/// route-resolved refinement is `Gwlf`. On failure the fallback
/// depends on the candidate's extractor:
///
/// * `ContourRdp` falls back to `Erf` on the un-refined `quad_pts`.
/// * `EdLines` trusts the Gauss-Newton corners (they are already
///   sub-pixel; the per-corner ERF fit can only do worse here).
///
/// Returns `Some((fallback_count, avg_delta))` when at least one
/// candidate routed to `Gwlf`; the caller must then recompute
/// homographies, since corners may have moved. Returns `None`
/// otherwise.
pub(crate) fn apply_detector_gwlf(
    arena: &Bump,
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
        let (extraction, refinement) = resolve_candidate_route(config, batch.routed_to[i]);
        if refinement != CornerRefinementMode::Gwlf {
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
            gwlf_fallback_count += 1;
            apply_gwlf_failure_fallback(
                arena,
                refinement_img,
                batch,
                i,
                &coarse,
                extraction,
                config.subpixel_refinement_sigma,
            );
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

/// Per-candidate fallback when [`crate::gwlf::refine_quad_gwlf_with_cov`]
/// fails. `coarse` is the candidate's pre-GWLF corners (the un-refined
/// `quad_pts`, since [`refine_quad_corners`] takes the passthrough
/// branch for `Gwlf`).
///
/// Decimation is always `1` at this stage: corners are in full-res
/// image space by the time the detector-level GWLF pass runs.
fn apply_gwlf_failure_fallback(
    arena: &Bump,
    refinement_img: &ImageView,
    batch: &mut DetectionBatch,
    i: usize,
    coarse: &[[f32; 2]; 4],
    extraction: QuadExtractionMode,
    sigma: f64,
) {
    match extraction {
        QuadExtractionMode::ContourRdp => {
            let pts = [
                Point {
                    x: f64::from(coarse[0][0]),
                    y: f64::from(coarse[0][1]),
                },
                Point {
                    x: f64::from(coarse[1][0]),
                    y: f64::from(coarse[1][1]),
                },
                Point {
                    x: f64::from(coarse[2][0]),
                    y: f64::from(coarse[2][1]),
                },
                Point {
                    x: f64::from(coarse[3][0]),
                    y: f64::from(coarse[3][1]),
                },
            ];
            let refined = refine_all_quad_corners(arena, refinement_img, pts, sigma, 1, true);
            for (j, p) in refined.iter().enumerate() {
                batch.corners[i][j].x = p.x as f32;
                batch.corners[i][j].y = p.y as f32;
                for k in 0..4 {
                    batch.corner_covariances[i][j * 4 + k] = 0.0;
                }
            }
        },
        // EdLines GN corners are already sub-pixel; per-corner ERF
        // refits routinely degrade them (see `lessons.md §4.1`).
        QuadExtractionMode::EdLines => {},
    }
}
