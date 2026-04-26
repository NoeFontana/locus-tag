//! Phase C.5 — post-decode edge-fit corner re-refinement.
//!
//! For each `Valid` candidate, fit each of the 4 outer tag edges with the
//! shared [`ErfEdgeFitter`] (PSF-blurred step-function model), then intersect
//! adjacent edges to recover 4 sub-pixel corners.
//!
//! Decoded corners are already sub-pixel accurate, so we skip the
//! `scan_initial_d` pre-step. Per-iteration A/B re-estimation drives the last
//! fraction of a pixel under variable lighting. Behind
//! [`DetectorConfig::post_decode_refinement`] (default `false`); profiles that
//! don't opt in are byte-identical when off.
//!
//! Why edge-fit-then-intersect rather than a per-corner saddle point: a 2D
//! saddle window inside a tag straddles internal bit boundaries that bias the
//! solve toward whichever neighbour is brighter. Fitting the four outer edges
//! independently localises each to its own black-to-light transition without
//! interior contamination.
//!
//! Covariance handling: this pass deliberately leaves `corner_covariances`
//! untouched. The Cramér-Rao bound `σ_n² / JᵀJ` from the line fit is overly
//! optimistic for synthetic-PSF imagery (Blender renders) where the true
//! intensity profile differs from the Gaussian-erf model — over-trusting the
//! refined corners pulls the weighted LM solver's pose tail. Phase A's
//! GWLF / structure-tensor covariance is calibrated against pose RMSE and is
//! the correct prior to keep.

#![allow(clippy::cast_possible_truncation)]

use bumpalo::Bump;

use crate::batch::{CandidateState, DetectionBatch, Point2f};
use crate::config::DetectorConfig;
use crate::decoder::Homography;
use crate::edge_refinement::{ErfEdgeFitter, RefineConfig, SampleConfig};
use crate::image::ImageView;

/// PSF blur σ assumed for the ERF step model. Matches the value used by the
/// quad-extraction GWLF pass and the decoder ERF refinement.
const SIGMA: f64 = 0.6;
/// Reject a refit step larger than this (in px) on any single corner. The
/// decoded corners are already accurate to ~0.5 px after Phase A's GWLF
/// refinement; any larger displacement almost always indicates the line fit
/// locked onto an interior bit boundary instead of the outer edge.
const MAX_STEP_PX: f32 = 1.5;
/// Minimum quad edge length (px) for the refit to engage. Below this the
/// 2.5-px perpendicular sample window leaks samples from the adjacent edge
/// of the L-corner, biasing both line fits. 16 px also guarantees ≥ ~12
/// usable samples per edge for stable Gauss-Newton convergence.
const MIN_EDGE_FOR_REFIT: f32 = 16.0;
/// `|sin(angle)|` lower bound between adjacent edges for the 2×2 line-system
/// solve. 0.05 corresponds to ~2.87°, well below the smallest deviation a
/// well-formed quad will exhibit; the gate exists to reject pathological
/// line-fit drifts that make adjacent edges nearly parallel.
const MIN_LINE_DET: f64 = 0.05;
/// Lower bound on the per-line `JᵀJ` to consider the fit converged. Below
/// this the line variance becomes meaningless and the intersection covariance
/// would explode; we bail rather than emit a useless prior.
const MIN_LINE_JTJ: f64 = 1.0;

/// Run Phase C.5 on the Valid slice of `batch`. No-op if the flag is off,
/// `n == 0`, or no candidate clears the edge-length gate.
///
/// **Privileges:** Read `status_mask`, `corners` (input), and the image
/// tensor. Write `corners` and `homographies` only at indices where
/// `status_mask[i] == Valid`. `corner_covariances` is in the allowed write
/// set per the SoA contract but this pass intentionally preserves the prior
/// (Phase A) covariance — see module docs.
pub fn refit_valid_corners(
    batch: &mut DetectionBatch,
    n: usize,
    img: &ImageView,
    config: &DetectorConfig,
    arena: &Bump,
) {
    if !config.post_decode_refinement {
        return;
    }
    let n_clamped = n.min(batch.capacity());
    let sample_cfg = SampleConfig::for_decoder();
    let refine_cfg = RefineConfig::post_decode_style(SIGMA);

    for i in 0..n_clamped {
        if batch.status_mask[i] != CandidateState::Valid {
            continue;
        }
        if min_edge_length_sq(&batch.corners[i]) < MIN_EDGE_FOR_REFIT * MIN_EDGE_FOR_REFIT {
            continue;
        }
        refit_one_tag(batch, i, img, &sample_cfg, &refine_cfg, arena);
    }
}

/// Squared length of the shortest of the 4 quad edges. Rotation-invariant
/// lower bound on tag side length, compared against `MIN_EDGE_FOR_REFIT²` so
/// the gate avoids a sqrt per Valid candidate.
#[inline]
fn min_edge_length_sq(corners: &[Point2f; 4]) -> f32 {
    let mut min_sq = f32::INFINITY;
    for k in 0..4 {
        let a = &corners[k];
        let b = &corners[(k + 1) % 4];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len_sq = dx * dx + dy * dy;
        if len_sq < min_sq {
            min_sq = len_sq;
        }
    }
    min_sq
}

fn refit_one_tag(
    batch: &mut DetectionBatch,
    i: usize,
    img: &ImageView,
    sample_cfg: &SampleConfig,
    refine_cfg: &RefineConfig,
    arena: &Bump,
) {
    let corners_in = batch.corners[i];

    // lines[k] = [nx, ny, d] for the edge from corner k to corner (k+1) % 4.
    let mut lines = [[0.0f64; 3]; 4];
    for k in 0..4 {
        let p1 = [f64::from(corners_in[k].x), f64::from(corners_in[k].y)];
        let p2 = [
            f64::from(corners_in[(k + 1) % 4].x),
            f64::from(corners_in[(k + 1) % 4].y),
        ];
        let Some(mut fitter) = ErfEdgeFitter::new(img, p1, p2, true) else {
            return;
        };
        let samples = fitter.collect_samples(arena, sample_cfg);
        if samples.len() < 10 {
            return;
        }
        fitter.refine(&samples, refine_cfg);
        if fitter.line_jtj() < MIN_LINE_JTJ {
            return;
        }
        let (nx, ny, d) = fitter.line_params();
        lines[k] = [nx, ny, d];
    }

    // Intersect adjacent edges. Corner k = lines[(k+3) % 4] ∩ lines[k].
    let mut new_corners = [[0.0f64; 2]; 4];
    for k in 0..4 {
        let Some((x, y)) = intersect_lines(&lines[(k + 3) % 4], &lines[k]) else {
            return;
        };
        let dx = (x as f32) - corners_in[k].x;
        let dy = (y as f32) - corners_in[k].y;
        if dx * dx + dy * dy > MAX_STEP_PX * MAX_STEP_PX {
            return;
        }
        new_corners[k] = [x, y];
    }

    // Re-solve the homography. `square_to_quad` validates a 1e-4 roundtrip;
    // a refit that pushed corners off-plane keeps the original homography.
    if let Some(h_new) = Homography::square_to_quad(&new_corners) {
        let h = h_new.h;
        for r in 0..3 {
            for c in 0..3 {
                batch.homographies[i].data[r * 3 + c] = h[(r, c)] as f32;
            }
        }
    }

    for (j, corner) in new_corners.iter().enumerate() {
        batch.corners[i][j].x = corner[0] as f32;
        batch.corners[i][j].y = corner[1] as f32;
    }
}

/// Solve the 2×2 line-intersection. `M = [[nx_a, ny_a], [nx_b, ny_b]]`,
/// `[x; y] = -M⁻¹ [d_a; d_b]`. Returns `None` for near-parallel adjacent
/// edges (degenerate quad after refit).
#[inline]
#[allow(clippy::similar_names)]
fn intersect_lines(line_a: &[f64; 3], line_b: &[f64; 3]) -> Option<(f64, f64)> {
    let [nxa, nya, da] = *line_a;
    let [nxb, nyb, db] = *line_b;
    let det = nxa * nyb - nxb * nya;
    if !det.is_finite() || det.abs() < MIN_LINE_DET {
        return None;
    }
    let inv_det = 1.0 / det;
    let x = (-da * nyb + db * nya) * inv_det;
    let y = (da * nxb - db * nxa) * inv_det;
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some((x, y))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::batch::{Matrix3x3, Pose6D};

    fn make_image(size: usize, draw: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        let mut data = vec![255u8; size * size];
        for y in 0..size {
            for x in 0..size {
                data[y * size + x] = draw(x, y);
            }
        }
        data
    }

    fn synthetic_batch(corners: [Point2f; 4]) -> Box<DetectionBatch> {
        let mut batch = Box::new(DetectionBatch::new());
        batch.corners[0] = corners;
        batch.status_mask[0] = CandidateState::Valid;
        batch.homographies[0] = Matrix3x3 {
            data: [1.0; 9],
            padding: [0.0; 7],
        };
        batch.poses[0] = Pose6D::default();
        batch
    }

    /// Build a synthetic dark-square-on-light-background image of side `size`,
    /// with the dark region at `bounds_x × bounds_y`.
    fn dark_square(
        size: usize,
        bounds_x: std::ops::Range<usize>,
        bounds_y: std::ops::Range<usize>,
    ) -> Vec<u8> {
        make_image(size, |x, y| {
            if bounds_x.contains(&x) && bounds_y.contains(&y) {
                30
            } else {
                220
            }
        })
    }

    /// Run `refit_valid_corners` and assert every corner of candidate 0 was
    /// preserved bit-for-bit.
    fn assert_corners_unchanged(
        size: usize,
        corners: [Point2f; 4],
        bounds_x: std::ops::Range<usize>,
        bounds_y: std::ops::Range<usize>,
        cfg: &DetectorConfig,
    ) {
        let data = dark_square(size, bounds_x, bounds_y);
        let img = ImageView::new(&data, size, size, size).expect("image");
        let mut batch = synthetic_batch(corners);
        let before = batch.corners[0];
        let arena = Bump::new();

        refit_valid_corners(&mut batch, 1, &img, cfg, &arena);

        for (k, c) in before.iter().enumerate() {
            assert_eq!(batch.corners[0][k].x, c.x);
            assert_eq!(batch.corners[0][k].y, c.y);
        }
    }

    #[test]
    fn flag_off_is_noop_even_on_valid_candidate() {
        let cfg = DetectorConfig::default();
        assert!(!cfg.post_decode_refinement);
        assert_corners_unchanged(
            64,
            [
                Point2f { x: 20.5, y: 20.5 },
                Point2f { x: 43.5, y: 20.5 },
                Point2f { x: 43.5, y: 43.5 },
                Point2f { x: 20.5, y: 43.5 },
            ],
            20..44,
            20..44,
            &cfg,
        );
    }

    #[test]
    fn refines_perturbed_corners_toward_true_edge() {
        let size = 80usize;
        // Black square with corners exactly at integer grid lines.
        let data = make_image(size, |x, y| {
            if (16..64).contains(&x) && (16..64).contains(&y) {
                30
            } else {
                220
            }
        });
        let img = ImageView::new(&data, size, size, size).expect("image");

        // Seed corners perturbed by ~0.4 px from the true positions.
        let mut batch = synthetic_batch([
            Point2f { x: 16.4, y: 16.3 },
            Point2f { x: 63.7, y: 16.5 },
            Point2f { x: 63.6, y: 63.4 },
            Point2f { x: 16.3, y: 63.7 },
        ]);
        let arena = Bump::new();

        let cfg = DetectorConfig {
            post_decode_refinement: true,
            ..Default::default()
        };
        refit_valid_corners(&mut batch, 1, &img, &cfg, &arena);

        // The refit should pull all 4 corners well within 0.2 px of the true
        // edge intersections at (16, 16), (64, 16), (64, 64), (16, 64).
        let truth = [
            (16.0_f32, 16.0_f32),
            (64.0_f32, 16.0_f32),
            (64.0_f32, 64.0_f32),
            (16.0_f32, 64.0_f32),
        ];
        for (k, (gx, gy)) in truth.iter().enumerate() {
            let dx = batch.corners[0][k].x - gx;
            let dy = batch.corners[0][k].y - gy;
            let err = (dx * dx + dy * dy).sqrt();
            assert!(err < 0.2, "corner {k}: error {err} px exceeds 0.2");
        }
    }

    #[test]
    fn skips_when_edge_is_below_min_length() {
        // 8-px edges — below MIN_EDGE_FOR_REFIT (16).
        let cfg = DetectorConfig {
            post_decode_refinement: true,
            ..Default::default()
        };
        assert_corners_unchanged(
            32,
            [
                Point2f { x: 10.0, y: 10.0 },
                Point2f { x: 18.0, y: 10.0 },
                Point2f { x: 18.0, y: 18.0 },
                Point2f { x: 10.0, y: 18.0 },
            ],
            10..18,
            10..18,
            &cfg,
        );
    }

    #[test]
    fn intersect_lines_rejects_near_parallel() {
        // Two near-parallel horizontal lines: |sin(angle)| ≈ 1e-3, well below
        // MIN_LINE_DET = 0.05.
        let line_a = [1.0, 0.0, 0.0];
        let line_b = [1.0, 1e-3, 0.0];
        assert!(intersect_lines(&line_a, &line_b).is_none());
    }

    #[test]
    fn rejects_refit_when_corners_drift_beyond_max_step() {
        // Tag corners seeded ≥ 5 px off-true: each line will fit the true edge
        // and the intersection will jump > MAX_STEP_PX = 1.5 from the seeded
        // corner, triggering the reject branch. Result: corners preserved.
        let cfg = DetectorConfig {
            post_decode_refinement: true,
            ..Default::default()
        };
        assert_corners_unchanged(
            80,
            [
                Point2f { x: 11.0, y: 11.0 },
                Point2f { x: 69.0, y: 11.0 },
                Point2f { x: 69.0, y: 69.0 },
                Point2f { x: 11.0, y: 69.0 },
            ],
            16..64,
            16..64,
            &cfg,
        );
    }
}
