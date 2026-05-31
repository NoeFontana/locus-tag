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

/// Per-corner empirical noise estimate (σ² in px²) drawn from the ERF
/// edge-fit residual MSE, indexed by quad corner (`[c0, c1, c2, c3]`).
/// `0.0` sentinel means "no ERF measurement on either adjacent edge"
/// (e.g. ContourRdp+Gwlf route, or both ERF fits failed) — downstream
/// `finalize_corner_covariance` falls back to the constant `σ_n²`.
/// See `docs/engineering/pose_covariance_followup_2026-05-22.md` §4.
pub(crate) type CornerEmpiricalNoise = [f32; 4];

/// "No empirical noise available" sentinel — Phase D's
/// `finalize_corner_covariance` treats this as `max(σ_n², 0) = σ_n²`,
/// preserving today's structure-tensor-only behaviour.
pub(crate) const NO_EMPIRICAL_NOISE_SENTINEL: CornerEmpiricalNoise = [0.0; 4];

/// Combine the two adjacent edges' MSEs into a single per-corner σ_n².
/// Arithmetic mean when both fits produced finite residuals; falls back
/// to the single available estimate if only one edge ran ERF; returns
/// `0.0` (the "no empirical evidence" sentinel) when neither did.
///
/// No ceiling clamp here — the pose-stage finalizer applies the
/// `min(empirical, 16·σ_n²)` cap where `σ_n²` is naturally in scope.
#[inline]
fn combine_edge_mses(mse_e1: f64, mse_e2: f64) -> f32 {
    let m1 = if mse_e1.is_finite() {
        Some(mse_e1)
    } else {
        None
    };
    let m2 = if mse_e2.is_finite() {
        Some(mse_e2)
    } else {
        None
    };
    match (m1, m2) {
        (Some(a), Some(b)) => (0.5 * (a + b)) as f32,
        (Some(a), None) | (None, Some(a)) => a as f32,
        (None, None) => 0.0,
    }
}

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
) -> ([Point; 4], CornerCovariances, CornerEmpiricalNoise) {
    match (route_extraction, route_refinement) {
        (_, CornerRefinementMode::None)
        | (QuadExtractionMode::EdLines, CornerRefinementMode::Gwlf) => {
            (quad_pts, gn_covs, NO_EMPIRICAL_NOISE_SENTINEL)
        },
        (_, CornerRefinementMode::Erf) => {
            let (corners, empirical) =
                refine_all_quad_corners(arena, refinement_img, quad_pts, sigma, decimation, true);
            (corners, [[0.0; 4]; 4], empirical)
        },
        (QuadExtractionMode::ContourRdp, CornerRefinementMode::Gwlf) => {
            // GWLF route uses gradient-peak edge fits (use_erf=false); no
            // ERF MSE is produced. Empirical noise stays at the sentinel
            // so Phase D falls back to constant σ_n².
            let (corners, _) =
                refine_all_quad_corners(arena, refinement_img, quad_pts, sigma, decimation, false);
            (corners, [[0.0; 4]; 4], NO_EMPIRICAL_NOISE_SENTINEL)
        },
    }
}

/// Refine each of a quad's four corners using its two cyclic
/// neighbours.
///
/// Indices: corner `i` is refined using `(i-1, i, i+1)` mod 4. With
/// CW-ordered corners this gives the conventional `(prev, current,
/// next)` triplet that [`refine_corner`] expects.
///
/// Returns the four refined points alongside the per-corner empirical
/// noise estimate (`0.0` sentinel when `use_erf=false` — ERF didn't
/// run, no MSE to draw from).
pub(crate) fn refine_all_quad_corners(
    arena: &Bump,
    img: &ImageView,
    pts: [Point; 4],
    sigma: f64,
    decimation: usize,
    use_erf: bool,
) -> ([Point; 4], CornerEmpiricalNoise) {
    let (p0, n0) = refine_corner(
        arena, img, pts[0], pts[3], pts[1], sigma, decimation, use_erf,
    );
    let (p1, n1) = refine_corner(
        arena, img, pts[1], pts[0], pts[2], sigma, decimation, use_erf,
    );
    let (p2, n2) = refine_corner(
        arena, img, pts[2], pts[1], pts[3], sigma, decimation, use_erf,
    );
    let (p3, n3) = refine_corner(
        arena, img, pts[3], pts[2], pts[0], sigma, decimation, use_erf,
    );
    ([p0, p1, p2, p3], [n0, n1, n2, n3])
}

/// Single-corner refinement: intersect two edge-line fits at point `p`,
/// using its neighbours `p_prev` and `p_next` to define the edges.
///
/// `use_erf = true` runs the PSF-blurred Gauss-Newton fit and falls
/// back to the gradient-peak fit on sample shortfall. `use_erf = false`
/// runs only the gradient-peak fit.
///
/// Returns the refined point alongside the per-corner empirical σ_n²
/// drawn from the two adjacent ERF edges' residual MSEs. Sentinel `0.0`
/// when `use_erf=false` or both ERF fits fell through to the
/// gradient-peak fallback (no MSE produced) — downstream
/// `finalize_corner_covariance` then falls back to constant `σ_n²`.
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
) -> (Point, f32) {
    let (line1, mse1) = if use_erf {
        match refine_edge_erf(arena, img, p_prev, p, sigma, decimation) {
            Some((line, mse)) => (Some(line), mse),
            None => (fit_edge_line(img, p_prev, p, decimation), f64::NAN),
        }
    } else {
        (fit_edge_line(img, p_prev, p, decimation), f64::NAN)
    };

    let (line2, mse2) = if use_erf {
        match refine_edge_erf(arena, img, p, p_next, sigma, decimation) {
            Some((line, mse)) => (Some(line), mse),
            None => (fit_edge_line(img, p, p_next, decimation), f64::NAN),
        }
    } else {
        (fit_edge_line(img, p, p_next, decimation), f64::NAN)
    };

    let empirical_sigma_sq = combine_edge_mses(mse1, mse2);

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
                return (Point { x, y }, empirical_sigma_sq);
            }
        }
    }

    // Intersection rejected (degenerate / outside sanity radius) — keep
    // the coarse point. Empirical noise is still meaningful: it reflects
    // the edge fits' agreement with the image, not the intersection's
    // success. But on this fallback path the corner's σ² should be the
    // sentinel — we have no evidence the kept point is well localised.
    (p, 0.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 4 combination formula: both edges contributed a finite MSE.
    /// Equal-weight arithmetic mean, no clamp at this stage (clamp lives
    /// in `pose_weighted::effective_sigma_n_sq`).
    #[test]
    #[allow(clippy::float_cmp)]
    fn combine_edge_mses_both_finite_averages() {
        assert_eq!(combine_edge_mses(4.0, 16.0), 10.0);
        assert_eq!(combine_edge_mses(0.0, 8.0), 4.0);
        // Symmetry: combine is commutative.
        assert_eq!(combine_edge_mses(1.5, 7.5), combine_edge_mses(7.5, 1.5));
    }

    /// Degenerate case: one edge skipped ERF (NaN sentinel — e.g. its
    /// `ErfEdgeFitter::new` returned `None` and we fell back to
    /// `fit_edge_line`'s gradient-peak fit). The other edge's MSE
    /// should pass through unchanged — discarding the single available
    /// measurement just because its sibling is missing would be lossy.
    #[test]
    #[allow(clippy::float_cmp)]
    fn combine_edge_mses_one_nan_passes_through_finite() {
        assert_eq!(combine_edge_mses(8.0, f64::NAN), 8.0);
        assert_eq!(combine_edge_mses(f64::NAN, 8.0), 8.0);
        // `0.0` MSE is a valid measurement (perfect fit), distinct from
        // NaN ("no measurement"): it should still feed the average.
        assert_eq!(combine_edge_mses(0.0, f64::NAN), 0.0);
    }

    /// Both edges' MSE is NaN ⇒ "no empirical evidence" sentinel `0.0`,
    /// which the pose-stage finalizer treats as `max(σ_n², 0) = σ_n²`
    /// (i.e. recovers today's structure-tensor-only behaviour).
    #[test]
    #[allow(clippy::float_cmp)]
    fn combine_edge_mses_all_nan_emits_zero_sentinel() {
        assert_eq!(combine_edge_mses(f64::NAN, f64::NAN), 0.0);
        // Infinity (an ill-conditioned fit) is also rejected and treated
        // as "no measurement" — defensive, since infinity would defeat
        // the ceiling clamp in `effective_sigma_n_sq` if it leaked.
        assert_eq!(combine_edge_mses(f64::INFINITY, f64::NAN), 0.0);
    }
}
