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
/// missing, near-parallel, or meet farther than the sanity radius from `p`.
fn intersect_corner(
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

/// Patch side for the largest window: `2·(MAX + 1) + 1` samples, one ring beyond the window
/// for the central-difference gradients.
const SUBPIX_PATCH_MAX: usize = 2 * (crate::config::MAX_CORNER_SUBPIX_HALF_WINDOW as usize + 1) + 1;
/// Iteration cap and step tolerance (px) of [`corner_subpix`] (the `cv::cornerSubPix` values
/// aruco_nano uses).
const SUBPIX_MAX_ITER: u32 = 12;
const SUBPIX_EPS: f64 = 0.005;

/// Gradient-orthogonality corner refinement, the `cv::cornerSubPix` model.
///
/// At a corner `c`, every gradient `∇I(p)` in the neighbourhood is orthogonal to `p − c`: on a
/// flat patch the gradient vanishes and on an edge through `c` it is normal to the edge. So
/// `c` solves `(Σ w·∇I ∇Iᵀ) c = Σ w·∇I ∇Iᵀ p` over the `(2·half + 1)²` window, with Gaussian
/// weights `w = exp(−(dx² + dy²)/half²)`. The window is re-centred and the system re-solved
/// until the step is under `SUBPIX_EPS` px (at most `SUBPIX_MAX_ITER` times). Gradients are
/// central differences of a bilinearly resampled patch, as OpenCV computes them.
///
/// Returns `seed` when the result leaves the `half`-px box around it, the window leaves the
/// image, or the normal matrix is singular (a flat or single-edge patch). Locus pixel
/// convention (+0.5 centres), like the rest of the crate. `half` must be in
/// `1..=MAX_CORNER_SUBPIX_HALF_WINDOW`.
pub(crate) fn corner_subpix(img: &ImageView, seed: [f64; 2], half: u32) -> [f64; 2] {
    debug_assert!((1..=crate::config::MAX_CORNER_SUBPIX_HALF_WINDOW).contains(&half));
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
    for _ in 0..SUBPIX_MAX_ITER {
        // Patch sample (row, col) sits at corner + (col − hw − 1, row − hw − 1); in array
        // coordinates (pixel centres at integers) its top-left is corner − 0.5 − (hw + 1).
        let x0 = corner[0] - 0.5 - reach;
        let y0 = corner[1] - 0.5 - reach;
        if !(x0 >= 0.0 && y0 >= 0.0)
            || x0 + (side as f64) >= img.width as f64
            || y0 + (side as f64) >= img.height as f64
        {
            return seed;
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

        // Normal equations: [sxx sxy; sxy syy] · step = [rhs_x; rhs_y].
        let (mut sxx, mut sxy, mut syy, mut rhs_x, mut rhs_y) = (0.0, 0.0, 0.0, 0.0, 0.0);
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
            }
        }
        let det = sxx * syy - sxy * sxy;
        if det.abs() <= f64::EPSILON * f64::EPSILON {
            return seed;
        }
        let step = [
            (syy * rhs_x - sxy * rhs_y) / det,
            (sxx * rhs_y - sxy * rhs_x) / det,
        ];
        corner = [corner[0] + step[0], corner[1] + step[1]];
        if step[0] * step[0] + step[1] * step[1] <= SUBPIX_EPS * SUBPIX_EPS {
            break;
        }
    }
    let limit = f64::from(half);
    if (corner[0] - seed[0]).abs() <= limit && (corner[1] - seed[1]).abs() <= limit {
        corner
    } else {
        seed
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    // The fallbacks return `seed` itself, so the comparison is exact by contract.
    clippy::float_cmp
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
            let got = corner_subpix(&img, [apex[0] + 1.3, apex[1] - 0.9], 4);
            let d = (got[0] - opencv[k][0]).hypot(got[1] - opencv[k][1]);
            assert!(
                d < 0.01,
                "theta={theta}: {got:?} vs OpenCV {:?} ({d:.4} px)",
                opencv[k]
            );
        }
    }

    #[test]
    fn keeps_the_seed_without_a_corner() {
        let flat = vec![128u8; W * W];
        let img = ImageView::new(&flat, W, W, W).unwrap();
        assert_eq!(corner_subpix(&img, [32.2, 31.7], 4), [32.2, 31.7]);

        // A single straight edge constrains one direction only: the normal matrix is singular.
        let edge: Vec<u8> = (0..W * W)
            .map(|i| if i % W < 32 { 40 } else { 220 })
            .collect();
        let img = ImageView::new(&edge, W, W, W).unwrap();
        assert_eq!(corner_subpix(&img, [32.0, 30.5], 4), [32.0, 30.5]);
    }

    #[test]
    fn keeps_the_seed_when_the_window_leaves_the_image() {
        let data = wedge([3.0, 3.0], 0.0);
        let img = ImageView::new(&data, W, W, W).unwrap();
        assert_eq!(corner_subpix(&img, [3.4, 2.6], 4), [3.4, 2.6]);
    }
}
