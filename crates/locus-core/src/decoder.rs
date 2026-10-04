//! Tag decoding, homography computation, and bit sampling.
//!
//! This module handles the final stage of the pipeline:
//! 1. **Homography**: Computing the projection from canonical tag space to image pixels.
//! 2. **Bit Sampling**: Bilinear interpolation of intensities at grid points, retried at
//!    0.9× and 1.1× quad scale.
//! 3. **Error Correction**: Correcting bit flips using tag-family specific Hamming distances.
//! 4. **Marker evidence**: a match must also show the marker's dark border ring
//!    ([`ring_evidence`]), within a per-family error budget.
//! 5. **Corner refinement**: the decoded quad's ERF refinement must still decode. Under
//!    decode-first ordering ([`crate::config::DetectorConfig::decode_first`]) the quad stage
//!    hands over unrefined contour corners, so a match stands only once the refined quad
//!    verifies it. The accepted corners then go through the sub-pixel pass
//!    ([`crate::refinement::subpix_marker_corners`]) and the photometric calibration
//!    ([`crate::marker_inset`]).
//! 6. **Recovery**: a near miss within the family's recovery window whose seed shows the
//!    border ring gets the refinement decode-first skipped, then a coarse corner-nudge search.

#![allow(unsafe_code, clippy::cast_sign_loss)]
use crate::batch::{Matrix3x3, Point2f, quad_to_f32, quad_to_f64};
use crate::config;
use crate::simd::math::{bilinear_interpolate_fixed, rcp_nr};
use crate::simd::roi::RoiCache;
#[cfg(any(test, feature = "bench-internals"))]
use bumpalo::Bump;
use multiversion::multiversion;
use nalgebra::{SMatrix, SVector};

use crate::workspace::WORKSPACE_ARENA;

/// A 3x3 Homography matrix.
pub struct Homography {
    /// The 3x3 homography matrix.
    pub h: SMatrix<f64, 3, 3>,
}

/// A Digital Differential Analyzer (DDA) for incremental homography projection.
///
/// This avoids expensive matrix multiplications by using discrete partial derivatives
/// when stepping through a uniform grid in tag space.
// See `Homography::to_dda` for the dead-code rationale.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct HomographyDda {
    /// Current numerator for X coordinate.
    pub nx: f64,
    /// Current numerator for Y coordinate.
    pub ny: f64,
    /// Current denominator (perspective divide).
    pub d: f64,
    /// Partial derivative of nx with respect to u.
    pub dnx_du: f64,
    /// Partial derivative of ny with respect to u.
    pub dny_du: f64,
    /// Partial derivative of d with respect to u.
    pub dd_du: f64,
    /// Partial derivative of nx with respect to v.
    pub dnx_dv: f64,
    /// Partial derivative of ny with respect to v.
    pub dny_dv: f64,
    /// Partial derivative of d with respect to v.
    pub dd_dv: f64,
}

impl Homography {
    /// The `f32` [`Matrix3x3`] the batch stores: column-major, zero padding.
    #[must_use]
    pub(crate) fn to_matrix3x3(&self) -> Matrix3x3 {
        let mut m = Matrix3x3::default();
        for (slot, &val) in m.data.iter_mut().zip(self.h.iter()) {
            *slot = val as f32;
        }
        m
    }

    /// A stored [`Matrix3x3`] widened back to `f64`.
    #[must_use]
    pub(crate) fn from_matrix3x3(m: &Matrix3x3) -> Self {
        Self {
            h: SMatrix::<f64, 3, 3>::from_column_slice(&m.data.map(f64::from)),
        }
    }

    /// Convert the homography into a DDA state for a grid with step size (du, dv).
    /// Initial state is computed at (u0, v0) in canonical tag space.
    // Dead-code lint runs without target_feature gating, so the AVX2/NEON-only
    // consumer in `sample_grid_values_dda_simd` is invisible to it.
    #[allow(dead_code)]
    #[must_use]
    pub fn to_dda(&self, u0: f64, v0: f64, du: f64, dv: f64) -> HomographyDda {
        let h = self.h;
        let nx = h[(0, 0)] * u0 + h[(0, 1)] * v0 + h[(0, 2)];
        let ny = h[(1, 0)] * u0 + h[(1, 1)] * v0 + h[(1, 2)];
        let d = h[(2, 0)] * u0 + h[(2, 1)] * v0 + h[(2, 2)];

        HomographyDda {
            nx,
            ny,
            d,
            dnx_du: h[(0, 0)] * du,
            dny_du: h[(1, 0)] * du,
            dd_du: h[(2, 0)] * du,
            dnx_dv: h[(0, 1)] * dv,
            dny_dv: h[(1, 1)] * dv,
            dd_dv: h[(2, 1)] * dv,
        }
    }

    /// Compute homography from 4 source points to 4 destination points using DLT.
    /// Points are [x, y].
    #[cfg(any(test, feature = "bench-internals"))]
    #[must_use]
    pub fn from_pairs(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<Self> {
        let mut a = SMatrix::<f64, 8, 9>::zeros();

        for i in 0..4 {
            let sx = src[i][0];
            let sy = src[i][1];
            let dx = dst[i][0];
            let dy = dst[i][1];

            a[(i * 2, 0)] = -sx;
            a[(i * 2, 1)] = -sy;
            a[(i * 2, 2)] = -1.0;
            a[(i * 2, 6)] = sx * dx;
            a[(i * 2, 7)] = sy * dx;
            a[(i * 2, 8)] = dx;

            a[(i * 2 + 1, 3)] = -sx;
            a[(i * 2 + 1, 4)] = -sy;
            a[(i * 2 + 1, 5)] = -1.0;
            a[(i * 2 + 1, 6)] = sx * dy;
            a[(i * 2 + 1, 7)] = sy * dy;
            a[(i * 2 + 1, 8)] = dy;
        }

        let mut b = SVector::<f64, 8>::zeros();
        let mut m = SMatrix::<f64, 8, 8>::zeros();
        for i in 0..8 {
            for j in 0..8 {
                m[(i, j)] = a[(i, j)];
            }
            b[i] = -a[(i, 8)];
        }

        m.lu().solve(&b).and_then(|h_vec| {
            let mut h = SMatrix::<f64, 3, 3>::identity();
            h[(0, 0)] = h_vec[0];
            h[(0, 1)] = h_vec[1];
            h[(0, 2)] = h_vec[2];
            h[(1, 0)] = h_vec[3];
            h[(1, 1)] = h_vec[4];
            h[(1, 2)] = h_vec[5];
            h[(2, 0)] = h_vec[6];
            h[(2, 1)] = h_vec[7];
            h[(2, 2)] = 1.0;
            let res = Self { h };
            for i in 0..4 {
                let p_proj = res.project(src[i]);
                let err_sq = (p_proj[0] - dst[i][0]).powi(2) + (p_proj[1] - dst[i][1]).powi(2);
                if !err_sq.is_finite() || err_sq > 1e-4 {
                    return None;
                }
            }
            Some(res)
        })
    }

    /// Optimized homography computation from canonical unit square to a quad.
    /// Source points are assumed to be: `[(-1,-1), (1,-1), (1,1), (-1,1)]`.
    #[must_use]
    pub fn square_to_quad(dst: &[[f64; 2]; 4]) -> Option<Self> {
        let mut b = SVector::<f64, 8>::zeros();
        let mut m = SMatrix::<f64, 8, 8>::zeros();

        // Hardcoded coefficients for src = [(-1,-1), (1,-1), (1,1), (-1,1)]
        // Point 0: (-1, -1) -> (x0, y0)
        let x0 = dst[0][0];
        let y0 = dst[0][1];
        // h0 + h1 - h2 - x0*h6 - x0*h7 = -x0  =>  1, 1, -1, ..., -x0, -x0
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 1.0;
        m[(0, 2)] = -1.0;
        m[(0, 6)] = -x0;
        m[(0, 7)] = -x0;
        b[0] = -x0;
        // h3 + h4 - h5 - y0*h6 - y0*h7 = -y0  =>  ..., 1, 1, -1, -y0, -y0
        m[(1, 3)] = 1.0;
        m[(1, 4)] = 1.0;
        m[(1, 5)] = -1.0;
        m[(1, 6)] = -y0;
        m[(1, 7)] = -y0;
        b[1] = -y0;

        // Point 1: (1, -1) -> (x1, y1)
        let x1 = dst[1][0];
        let y1 = dst[1][1];
        // -h0 + h1 + h2 + x1*h6 - x1*h7 = -x1
        m[(2, 0)] = -1.0;
        m[(2, 1)] = 1.0;
        m[(2, 2)] = -1.0;
        m[(2, 6)] = x1;
        m[(2, 7)] = -x1;
        b[2] = -x1;
        m[(3, 3)] = -1.0;
        m[(3, 4)] = 1.0;
        m[(3, 5)] = -1.0;
        m[(3, 6)] = y1;
        m[(3, 7)] = -y1;
        b[3] = -y1;

        // Point 2: (1, 1) -> (x2, y2)
        let x2 = dst[2][0];
        let y2 = dst[2][1];
        // -h0 - h1 + h2 + x2*h6 + x2*h7 = -x2
        m[(4, 0)] = -1.0;
        m[(4, 1)] = -1.0;
        m[(4, 2)] = -1.0;
        m[(4, 6)] = x2;
        m[(4, 7)] = x2;
        b[4] = -x2;
        m[(5, 3)] = -1.0;
        m[(5, 4)] = -1.0;
        m[(5, 5)] = -1.0;
        m[(5, 6)] = y2;
        m[(5, 7)] = y2;
        b[5] = -y2;

        // Point 3: (-1, 1) -> (x3, y3)
        let x3 = dst[3][0];
        let y3 = dst[3][1];
        // h0 - h1 + h2 - x3*h6 + x3*h7 = -x3
        m[(6, 0)] = 1.0;
        m[(6, 1)] = -1.0;
        m[(6, 2)] = -1.0;
        m[(6, 6)] = -x3;
        m[(6, 7)] = x3;
        b[6] = -x3;
        m[(7, 3)] = 1.0;
        m[(7, 4)] = -1.0;
        m[(7, 5)] = -1.0;
        m[(7, 6)] = -y3;
        m[(7, 7)] = y3;
        b[7] = -y3;

        m.lu().solve(&b).and_then(|h_vec| {
            let mut h = SMatrix::<f64, 3, 3>::identity();
            h[(0, 0)] = h_vec[0];
            h[(0, 1)] = h_vec[1];
            h[(0, 2)] = h_vec[2];
            h[(1, 0)] = h_vec[3];
            h[(1, 1)] = h_vec[4];
            h[(1, 2)] = h_vec[5];
            h[(2, 0)] = h_vec[6];
            h[(2, 1)] = h_vec[7];
            h[(2, 2)] = 1.0;
            let res = Self { h };
            let src_unit = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
            for i in 0..4 {
                let p_proj = res.project(src_unit[i]);
                let err_sq = (p_proj[0] - dst[i][0]).powi(2) + (p_proj[1] - dst[i][1]).powi(2);
                if err_sq > 1e-4 {
                    return None;
                }
            }
            Some(res)
        })
    }

    /// Project a point using the homography.
    #[must_use]
    pub fn project(&self, p: [f64; 2]) -> [f64; 2] {
        let res = self.h * SVector::<f64, 3>::new(p[0], p[1], 1.0);
        // Guard the perspective divide: a near-zero `w` (point on the camera plane /
        // degenerate homography) would yield a non-finite result. Clamp to a tiny
        // sign-preserving epsilon so the divide stays finite — the projection becomes
        // large-but-finite, which callers reject via their reprojection-residual gate
        // (mirrors the `d.abs() < 1e-8` guard in `sample_grid_values_distorted`).
        let w = res[2];
        let w = if w.abs() < 1e-12 {
            1e-12_f64.copysign(w)
        } else {
            w
        };
        [res[0] / w, res[1] / w]
    }
}

/// Square-to-quad homography of `quad` as the batch's `f32` matrix; `None` when degenerate.
fn homography_matrix(quad: &[[f64; 2]; 4]) -> Option<Matrix3x3> {
    Homography::square_to_quad(quad).map(|h| h.to_matrix3x3())
}

/// `quad` scaled by `scale` about its centroid.
fn scale_about_centroid(quad: &[[f64; 2]; 4], scale: f64) -> [[f64; 2]; 4] {
    let c = [0, 1].map(|k| quad.iter().map(|p| p[k]).sum::<f64>() / 4.0);
    quad.map(|p| [c[0] + (p[0] - c[0]) * scale, c[1] + (p[1] - c[1]) * scale])
}

/// Compute homographies for all active quads in the batch using a pure-function SoA approach.
///
/// This uses `rayon` for data-parallel computation of the square-to-quad homographies.
/// Quads are defined by 4 corners in `corners` for each candidate index.
#[tracing::instrument(skip_all, name = "pipeline::homography_pass")]
pub fn compute_homographies_soa(
    corners: &[[Point2f; 4]],
    status_mask: &[crate::batch::CandidateState],
    homographies: &mut [Matrix3x3],
) {
    use crate::batch::CandidateState;
    use rayon::prelude::*;

    // Each homography maps from canonical square [(-1,-1), (1,-1), (1,1), (-1,1)] to image quads.
    homographies
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, h_out)| {
            *h_out = if status_mask[i] == CandidateState::Active {
                homography_matrix(&quad_to_f64(&corners[i])).unwrap_or_default()
            } else {
                Matrix3x3::default()
            };
        });
}

/// Decode-first ordering skipped the quad-stage corner refinement and the edge-contrast gate
/// on its output; run both on a decoded (or near-miss) seed, then the decoder's ERF pass, so a
/// candidate is accepted with the same corners, and only if, refine-first ordering would
/// have accepted it. `None` when the refined quad fails the gate.
fn refine_decode_first_seed(
    arena: &bumpalo::Bump,
    img: &crate::image::ImageView,
    seed: &[[f64; 2]; 4],
    config: &crate::config::DetectorConfig,
) -> Option<[[f64; 2]; 4]> {
    let sigma = config.subpixel_refinement_sigma;
    let pts = seed.map(|p| crate::Point { x: p[0], y: p[1] });
    let quad =
        crate::refinement::refine_all_quad_corners(arena, img, pts, sigma, config.decimation);
    crate::quad::edge_contrast_exceeds(img, quad, &[0.0], config.quad_min_edge_score)
        .then(|| refine_corners_erf(arena, img, &quad.map(|p| [p.x, p.y]), sigma))
}

/// Refine corners using "Erf-Fit" (Gaussian fit to intensity profile).
///
/// This assumes the edge intensity profile is an Error Function (convolution of step edge with Gaussian PSF).
/// We minimize the photometric error between the image and the ERF model using Gauss-Newton.
pub(crate) fn refine_corners_erf(
    arena: &bumpalo::Bump,
    img: &crate::image::ImageView,
    corners: &[[f64; 2]; 4],
    sigma: f64,
) -> [[f64; 2]; 4] {
    use crate::edge_refinement::{ErfEdgeFitter, RefineConfig, SampleConfig};

    let mut lines = [(0.0f64, 0.0f64, 0.0f64); 4];
    let mut line_valid = [false; 4];
    let sample_cfg = SampleConfig::for_decoder();
    let refine_cfg = RefineConfig::decoder_style(sigma);

    for i in 0..4 {
        let next = (i + 1) % 4;
        let p1 = corners[i];
        let p2 = corners[next];

        if let Some(mut fitter) = ErfEdgeFitter::new(img, p1, p2, false)
            && fitter.fit(arena, &sample_cfg, &refine_cfg)
        {
            lines[i] = fitter.line_params();
            line_valid[i] = true;
        }
    }

    if !line_valid.iter().all(|&v| v) {
        return *corners;
    }

    // Each corner moves to the intersection of its two edge lines, within 2 px.
    core::array::from_fn(|i| {
        let p = crate::Point {
            x: corners[i][0],
            y: corners[i][1],
        };
        let q = crate::refinement::intersect_corner(p, Some(lines[(i + 3) % 4]), Some(lines[i]), 1);
        [q.x, q.y]
    })
}

/// Returns the threshold that maximizes inter-class variance.
pub(crate) fn compute_otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 128.0;
    }

    let n = values.len() as f64;
    let total_sum: f64 = values.iter().sum();

    let min_val = values.iter().copied().fold(f64::MAX, f64::min);
    let max_val = values.iter().copied().fold(f64::MIN, f64::max);

    if (max_val - min_val) < 1.0 {
        return f64::midpoint(min_val, max_val);
    }

    let mut best_threshold = f64::midpoint(min_val, max_val);
    let mut best_variance = 0.0;

    // Otsu-style search: maximize inter-class variance over 16 candidate splits.
    for i in 1..16 {
        let t = min_val + (max_val - min_val) * (f64::from(i) / 16.0);

        let mut w0 = 0.0;
        let mut sum0 = 0.0;

        for &v in values {
            if v <= t {
                w0 += 1.0;
                sum0 += v;
            }
        }

        let w1 = n - w0;
        if w0 < 1.0 || w1 < 1.0 {
            continue;
        }

        let mean0 = sum0 / w0;
        let mean1 = (total_sum - sum0) / w1;

        let variance = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);

        if variance > best_variance {
            best_variance = variance;
            best_threshold = t;
        }
    }

    best_threshold
}

/// Maximum number of bits in a supported tag family payload.
const MAX_BIT_COUNT: usize = 64;

/// Near-miss recovery (decode-first refinement, corner nudging) costs tens of decodes per
/// candidate, so it runs only for a best match within the family's recovery window: the largest
/// Hamming distance that uniformly random bits (texture) reach with probability at most this.
const RECOVERY_FALSE_TRIGGER_RATE: f64 = 0.1;

/// Recovery also requires the seed quad to show a marker's dark border ring: at most this
/// fraction of ring cells may read bright (see [`ring_evidence`]). On Liu4K, ICRA, EuRoC and
/// render-tag, every successful recovery but one had under 10 % bright cells, while texture
/// near-misses spread over the whole range; this skips 75–90 % of the futile ones.
const RECOVERY_RING_MAX_ERROR_RATE: f32 = 0.2;

/// Whether a decoded outline is observed rather than cut by the frame of a `width × height`
/// image (pixel `(i, j)` spans `[i, i + 1] × [j, j + 1]`).
///
/// A marker the frame cuts reports the frame edge as one of its sides and a clipped corner
/// where the printed one lies outside (EuRoC: 4–5 px off on 10 of 11 border false positives).
/// So every corner must lie in the image, and every side's middle at least the reach of the
/// smallest corner window from the border: closer, the side's outer half is not seen and it
/// cannot be told from the frame edge. A corner may touch the border: there its two edges
/// still place it (tag16h5 render: a corner 0.5 px from the border, 0.1 px from the truth).
#[allow(clippy::cast_precision_loss)]
fn outline_observed(corners: &[Point2f; 4], width: usize, height: usize) -> bool {
    let (w, h) = (width as f32, height as f32);
    let margin = |x: f32, y: f32| x.min(y).min(w - x).min(h - y);
    (0..4).all(|j| {
        let (a, b) = (corners[j], corners[(j + 1) % 4]);
        margin(a.x, a.y) >= 0.0
            && margin(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
                >= crate::refinement::MIN_CORNER_SUPPORT_PX
    })
}

/// Largest `h` with `4 · num_codes · Σ_{k ≤ h} C(bit_count, k) / 2^bit_count ≤
/// RECOVERY_FALSE_TRIGGER_RATE`: the union bound, over every rotated code, on the probability
/// that random bits land within `h` of one. 36h11 → 6 (0.08), ArUcoMip36h12 → 6 (0.03),
/// tag16h5 → 1 (0.03), ArUco 4x4_100 → 0.
fn recovery_window(bit_count: usize, num_codes: usize) -> u32 {
    let n = bit_count as u32;
    let scale = 4.0 * num_codes as f64 * (-f64::from(n)).exp2();
    let (mut window, mut binom, mut cum) = (0, 1.0_f64, 0.0_f64);
    for k in 0..=n {
        cum += binom;
        if cum * scale > RECOVERY_FALSE_TRIGGER_RATE {
            break;
        }
        window = k;
        binom = binom * f64::from(n - k) / f64::from(k + 1);
    }
    window
}

/// Capacity of the per-decoder scratch arrays: one decoder per tag family at most (the
/// detector registers each family once).
pub(crate) const MAX_DECODERS: usize = 8;
const _: () = assert!(crate::config::TagFamily::all().len() <= MAX_DECODERS);

/// Border-ring cells of the largest supported family: `4·(d + 1)` for a `d×d` payload.
const MAX_RING_CELLS: usize = 4 * (8 + 1);

/// A ring cell counts as bright (an error) only once it reaches this fraction of the way from
/// the payload's dark class mean to its bright class mean.
///
/// On small markers (≲ 20 px, ≈ 2 px per cell) the PSF pulls ring cells next to the white
/// surround well above the class midpoint, while a textured false positive has ring cells
/// that are genuinely bright. Measured with `cargo xtask sota` (LocalMean front end, zero
/// error budget), moving the cut from the midpoint (0.5) to 0.9 keeps tag16h5 false
/// positives at 3 (112 without the check) and recovers ICRA `forward` recall from 64.6 %
/// to 72.7 % (73.4 % without the check). Setup: full datasets, release build, 1 thread per
/// detector, AMD EPYC-Milan (see `docs/engineering/benchmarking/sota_scoreboard_20261002.md`).
const RING_BRIGHT_FRACTION: f64 = 0.9;

/// Marker evidence beyond the codeword: the one-cell black border around the payload.
///
/// The canonical square `[-1, 1]²` spans the `d×d` data cells plus one border cell on each
/// side, so border-cell centres sit at `±(d + 1)/(d + 2)`. `sample` maps canonical points to
/// image intensities (pinhole or distortion-aware). Payload cells are classified dark/bright
/// exactly as decoding does (adaptive per-cell thresholds), and a ring cell counts as an
/// error when it reaches [`RING_BRIGHT_FRACTION`] of the way from the dark to the bright class
/// mean. Returns `(errors, ring_cells)`, or `None` when the evidence cannot be evaluated (a
/// sample outside the image, a single-class payload, an unsupported grid size).
///
/// A codeword match alone is weak evidence for small dictionaries: with `N` codes of `n` bits
/// and Hamming budget `h`, a random candidate decodes with probability
/// `4·N·Σ_{k≤h} C(n, k) / 2ⁿ` (≈ 1.8e-3 for tag16h5 at h = 0), so frames with hundreds of
/// textured candidates produce false positives. A uniformly dark ring of `4·(d + 1)` cells is
/// independent evidence that texture rarely supplies.
fn ring_evidence(
    decoder: &(impl TagDecoder + ?Sized),
    mut sample: impl FnMut(&[(f64, f64)], &mut [f64]) -> bool,
) -> Option<(u32, u32)> {
    let d = decoder.dimension();
    let n_ring = 4 * (d + 1);
    if n_ring > MAX_RING_CELLS {
        return None;
    }
    let pitch = 2.0 / (d + 2) as f64;
    let centre = |k: usize| -1.0 + pitch * (k as f64 + 0.5);
    let mut ring_pts = [(0.0f64, 0.0f64); MAX_RING_CELLS];
    let mut m = 0;
    for k in 0..d + 2 {
        for l in 0..d + 2 {
            if k == 0 || l == 0 || k == d + 1 || l == d + 1 {
                ring_pts[m] = (centre(l), centre(k));
                m += 1;
            }
        }
    }
    debug_assert_eq!(m, n_ring);
    let mut ring = [0.0f64; MAX_RING_CELLS];
    if !sample(&ring_pts[..n_ring], &mut ring[..n_ring]) {
        return None;
    }

    let points = decoder.sample_points();
    let n = points.len().min(MAX_BIT_COUNT);
    let mut data = [0.0f64; MAX_BIT_COUNT];
    if !sample(&points[..n], &mut data[..n]) {
        return None;
    }
    let thresholds = compute_adaptive_thresholds(&data[..n], &points[..n]);
    let (mut dark, mut n_dark, mut bright, mut n_bright) = (0.0, 0u32, 0.0, 0u32);
    for (&v, &t) in data[..n].iter().zip(&thresholds[..n]) {
        if v > t {
            bright += v;
            n_bright += 1;
        } else {
            dark += v;
            n_dark += 1;
        }
    }
    if n_dark == 0 || n_bright == 0 {
        return None;
    }
    let (dark_mean, bright_mean) = (dark / f64::from(n_dark), bright / f64::from(n_bright));
    let bright_cut = dark_mean + RING_BRIGHT_FRACTION * (bright_mean - dark_mean);
    let errors = ring[..n_ring].iter().filter(|&&v| v > bright_cut).count() as u32;
    Some((errors, n_ring as u32))
}

/// [`ring_evidence`] through a pinhole homography with the ROI-cached sampler.
fn rectified_ring_evidence(
    img: &crate::image::ImageView,
    roi: &RoiCache,
    h: &Homography,
    decoder: &(impl TagDecoder + ?Sized),
) -> Option<(u32, u32)> {
    ring_evidence(decoder, |pts, out| {
        sample_grid_values_optimized(img, h, roi, pts, out, pts.len())
    })
}

/// Whether ring evidence fits the budget `max_error_rate · ring_cells`. Evidence that cannot
/// be evaluated (`None`) does not reject: the check only ever removes candidates it can see.
fn ring_budget_ok(evidence: Option<(u32, u32)>, max_error_rate: f32) -> bool {
    evidence.is_none_or(|(errors, cells)| {
        // The epsilon absorbs the f32 → f64 widening error (0.35f32 · 20 = 6.99999988).
        f64::from(errors) <= (f64::from(max_error_rate) * f64::from(cells) + 1e-6).floor()
    })
}

/// Border-ring check: a rate `>= 1` disables it, otherwise `evidence` (evaluated only then)
/// must fit [`ring_budget_ok`].
fn ring_ok(max_error_rate: f32, evidence: impl FnOnce() -> Option<(u32, u32)>) -> bool {
    max_error_rate >= 1.0 || ring_budget_ok(evidence(), max_error_rate)
}

/// Pinhole [`ring_ok`] for one stored homography.
fn border_ring_ok(
    img: &crate::image::ImageView,
    roi: &RoiCache,
    homography: &Matrix3x3,
    decoder: &(impl TagDecoder + ?Sized),
    max_error_rate: f32,
) -> bool {
    ring_ok(max_error_rate, || {
        rectified_ring_evidence(img, roi, &Homography::from_matrix3x3(homography), decoder)
    })
}

/// Border-ring error budget of a family whose decodes accept up to `max_h` errors in `bits`
/// bits: the configured rate, or by default the codeword's own error density `max_h / bits`
/// (one ring cell of 28 for tag36h11 at h = 2, none for tag16h5 at h = 0).
#[allow(clippy::cast_precision_loss)]
fn ring_error_rate(config: &crate::config::DetectorConfig, max_h: u32, bits: usize) -> f32 {
    config
        .decoder_max_border_error_rate
        .unwrap_or_else(|| max_h as f32 / bits.max(1) as f32)
}

/// [`Homography::to_dda`] for a decoder's row-major sample grid: it starts at the first sample
/// and steps one cell along a row (`u`) and one row down (`v`).
// See `Homography::to_dda` for the dead-code rationale.
#[allow(dead_code)]
fn grid_dda(h: &Homography, points: &[(f64, f64)], dim: usize) -> HomographyDda {
    let (du, dv) = if dim > 1 {
        (points[1].0 - points[0].0, points[dim].1 - points[0].1)
    } else {
        (0.0, 0.0)
    };
    h.to_dda(points[0].0, points[0].1, du, dv)
}

/// Per-decoder Hamming budget (`max_hamming_error`, else the family default) and border-ring
/// error rate ([`ring_error_rate`]), resolved once per batch so the per-candidate loops read
/// plain values. The arrays stay on the stack; the detector registers at most one decoder per
/// family.
fn decoder_budgets(
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
) -> ([u32; MAX_DECODERS], [f32; MAX_DECODERS]) {
    debug_assert!(
        decoders.len() <= MAX_DECODERS,
        "more decoders than tag families"
    );
    let mut max_h = [0u32; MAX_DECODERS];
    let mut ring_rate = [0.0f32; MAX_DECODERS];
    for (idx, d) in decoders.iter().enumerate() {
        max_h[idx] = config
            .max_hamming_error
            .unwrap_or_else(|| d.default_max_hamming());
        ring_rate[idx] = ring_error_rate(config, max_h[idx], d.bit_count());
    }
    (max_h, ring_rate)
}

/// Sample values from the image using DDA-based coordinate generation and SIMD bilinear sampling.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn sample_grid_values_dda_simd(
    img: &crate::image::ImageView,
    roi: &RoiCache,
    h: &Homography,
    decoder: &(impl TagDecoder + ?Sized),
    intensities: &mut [f64],
) -> bool {
    let n = decoder.bit_count();
    let points = decoder.sample_points();
    if points.is_empty() {
        return false;
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    // SAFETY:
    // 1. AVX2 + FMA intrinsics (`_mm256_*`, `_mm256_fmadd_ps`) are sound on
    //    x86_64 because the enclosing `#[cfg(all(target_arch = "x86_64",
    //    target_feature = "avx2", target_feature = "fma"))]` gate above
    //    guarantees both features are available at compile time.
    // 2. The per-chunk early-out at `(mask & ((1 << count) - 1)) != ((1 << count) - 1)`
    //    (computed from `mask_x` ∧ `mask_y`, which require
    //    `0 <= img_x < width - 1` and `0 <= img_y < height - 1`) returns `false`
    //    if any of the `count` active lanes lie outside the safe 2×2 bilinear
    //    neighbourhood. `sample_bilinear_v8` is therefore only invoked when
    //    every active lane's gather offset `(viy * stride + vix)` is guaranteed
    //    in-range for the 4 pixels of the bilinear footprint.
    // 3. `sample_bilinear_v8` uses `_mm256_i32gather_epi32` to fetch 8-bit
    //    pixels via 32-bit loads, which can read up to 3 bytes past the last
    //    pixel of the underlying buffer. The required ≥3-byte SIMD-gather
    //    padding contract is documented in `docs/engineering/constraints.md`
    //    §2 and `docs/engineering/ffi_contracts.md` §1, and is the
    //    caller-side responsibility of the FFI buffer-preparation layer
    //    (`prepare_image_view` in `crates/locus-py/src/lib.rs`).
    //    `sample_bilinear_v8` additionally guards its gather path with
    //    `ImageView::has_simd_padding()` and falls back to scalar sampling
    //    when the contract is not met.
    unsafe {
        use crate::simd::math::rcp_nr_v8;
        use crate::simd::sampler::sample_bilinear_v8;
        use std::arch::x86_64::*;

        let dim = decoder.dimension();
        let dda = grid_dda(h, points, dim);

        let w_limit = _mm256_set1_ps(img.width as f32 - 1.0);
        let h_limit = _mm256_set1_ps(img.height as f32 - 1.0);

        let mut current_nx_row = dda.nx as f32;
        let mut current_ny_row = dda.ny as f32;
        let mut current_d_row = dda.d as f32;

        let dnx_du = dda.dnx_du as f32;
        let dny_du = dda.dny_du as f32;
        let dd_du = dda.dd_du as f32;

        let v_dnx_du = _mm256_set1_ps(dnx_du);
        let v_dny_du = _mm256_set1_ps(dny_du);
        let v_dd_du = _mm256_set1_ps(dd_du);
        let v_steps = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
        let v_half = _mm256_set1_ps(0.5);

        let mut idx = 0;
        for _y in 0..dim {
            let mut nx_start = current_nx_row;
            let mut ny_start = current_ny_row;
            let mut d_start = current_d_row;

            for _x in (0..dim).step_by(8) {
                let count = (dim - _x).min(8);

                // Vectorized coordinate generation (DDA)
                // x[i] = (nx + i*dnx_du)
                let v_nx_simd = _mm256_fmadd_ps(v_steps, v_dnx_du, _mm256_set1_ps(nx_start));
                let v_ny_simd = _mm256_fmadd_ps(v_steps, v_dny_du, _mm256_set1_ps(ny_start));
                let v_d_simd = _mm256_fmadd_ps(v_steps, v_dd_du, _mm256_set1_ps(d_start));

                // Perspective divide: (nx/d, ny/d)
                let v_winv = rcp_nr_v8(v_d_simd);
                let v_img_x_raw = _mm256_mul_ps(v_nx_simd, v_winv);
                let v_img_y_raw = _mm256_mul_ps(v_ny_simd, v_winv);

                // Offset by -0.5 to match bilinear logic center alignment
                let v_img_x = _mm256_sub_ps(v_img_x_raw, v_half);
                let v_img_y = _mm256_sub_ps(v_img_y_raw, v_half);

                // Bounds check: must be in [0, width - 1) for safe 2x2 bilinear fetch
                let v_zero = _mm256_setzero_ps();
                let mask_x = _mm256_and_ps(
                    _mm256_cmp_ps(v_img_x, v_zero, _CMP_GE_OQ),
                    _mm256_cmp_ps(v_img_x, w_limit, _CMP_LT_OQ),
                );
                let mask_y = _mm256_and_ps(
                    _mm256_cmp_ps(v_img_y, v_zero, _CMP_GE_OQ),
                    _mm256_cmp_ps(v_img_y, h_limit, _CMP_LT_OQ),
                );
                let mask = _mm256_movemask_ps(_mm256_and_ps(mask_x, mask_y));

                if (mask & ((1 << count) - 1)) != ((1 << count) - 1) {
                    return false;
                }

                // Restore original (non-subtracted) coords for sample_bilinear_v8 which handles the offset
                let mut v_img_x_arr = [0.0f32; 8];
                let mut v_img_y_arr = [0.0f32; 8];
                _mm256_storeu_ps(v_img_x_arr.as_mut_ptr(), v_img_x_raw);
                _mm256_storeu_ps(v_img_y_arr.as_mut_ptr(), v_img_y_raw);

                let mut sampled = [0.0f32; 8];
                sample_bilinear_v8(img, &v_img_x_arr, &v_img_y_arr, &mut sampled);

                for i in 0..count {
                    intensities[idx] = f64::from(sampled[i]);
                    idx += 1;
                }

                // Advance scalar row starts for next SIMD chunk
                nx_start += 8.0 * dnx_du;
                ny_start += 8.0 * dny_du;
                d_start += 8.0 * dd_du;
            }

            current_nx_row += dda.dnx_dv as f32;
            current_ny_row += dda.dny_dv as f32;
            current_d_row += dda.dd_dv as f32;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[expect(
        unsafe_code,
        reason = "workspace denies unsafe_code; this NEON-gated block uses aarch64 SIMD intrinsics and its soundness is justified in the SAFETY comment below"
    )]
    // SAFETY:
    // 1. NEON intrinsics (`vfmaq_f32`, `vrecpeq_f32`, `vld1q_f32`,
    //    `vst1q_f32`, etc.) are sound on aarch64 because the enclosing
    //    `#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]`
    //    gate above guarantees the NEON feature is available at compile
    //    time.
    // 2. Per-lane bounds for the sampled coordinates are enforced inside
    //    `sample_bilinear_v8`'s NEON branch (`crates/locus-core/src/simd/
    //    sampler.rs`), which clamps each (x, y) to `[0, width - 2] ×
    //    [0, height - 2]` via `vmaxq_f32`/`vminq_f32` before truncating to
    //    `i32` and computing the 2×2 footprint offsets. This guarantees the
    //    four scalar pixel loads (`img.data[base]`, `+1`, `+stride`,
    //    `+stride + 1`) are in-range. Unlike the AVX2 path above, no
    //    early-out is needed here because the NEON sampler does not use a
    //    SIMD gather instruction — it uses bounds-clamped scalar loads.
    // 3. `sample_bilinear_v8` is shared between the AVX2 and NEON paths
    //    and documents a ≥3-byte SIMD-gather padding contract for its AVX2
    //    branch (caller responsibility, enforced upstream at
    //    `prepare_image_view` in `crates/locus-py/src/lib.rs`; see
    //    `docs/engineering/constraints.md` §2 and
    //    `docs/engineering/ffi_contracts.md` §1). The NEON branch uses
    //    bounds-clamped scalar loads rather than a 32-bit gather of 8-bit
    //    pixels, so the 3-byte slack is not strictly required on aarch64;
    //    nevertheless the contract is enforced at the FFI boundary
    //    irrespective of the target architecture, which keeps a future
    //    aarch64 gather variant sound by construction.
    unsafe {
        use crate::simd::sampler::sample_bilinear_v8;
        use std::arch::aarch64::*;

        let dim = decoder.dimension();
        let dda = grid_dda(h, points, dim);

        let mut current_nx_row = dda.nx as f32;
        let mut current_ny_row = dda.ny as f32;
        let mut current_d_row = dda.d as f32;

        let dnx_du = dda.dnx_du as f32;
        let dny_du = dda.dny_du as f32;
        let dd_du = dda.dd_du as f32;

        let v_dnx_du = vdupq_n_f32(dnx_du);
        let v_dny_du = vdupq_n_f32(dny_du);
        let v_dd_du = vdupq_n_f32(dd_du);
        let v_steps_low = vld1q_f32([0.0, 1.0, 2.0, 3.0].as_ptr());
        let v_steps_high = vld1q_f32([4.0, 5.0, 6.0, 7.0].as_ptr());

        let mut idx = 0;
        for _y in 0..dim {
            let mut nx_start = current_nx_row;
            let mut ny_start = current_ny_row;
            let mut d_start = current_d_row;

            for _x in (0..dim).step_by(8) {
                let count = (dim - _x).min(8);

                // NEON perspective divide using vrecpeq_f32 + vrecpsq_f32
                let mut v_img_x = [0.0f32; 8];
                let mut v_img_y = [0.0f32; 8];

                for (chunk, v_steps) in [v_steps_low, v_steps_high].into_iter().enumerate() {
                    let v_nx_c = vfmaq_f32(vdupq_n_f32(nx_start), v_steps, v_dnx_du);
                    let v_ny_c = vfmaq_f32(vdupq_n_f32(ny_start), v_steps, v_dny_du);
                    let v_d_c = vfmaq_f32(vdupq_n_f32(d_start), v_steps, v_dd_du);

                    let v_winv = vrecpeq_f32(v_d_c);
                    let v_winv = vmulq_f32(v_winv, vrecpsq_f32(v_d_c, v_winv));

                    let img_x = vmulq_f32(v_nx_c, v_winv);
                    let img_y = vmulq_f32(v_ny_c, v_winv);

                    let offset = chunk * 4;
                    vst1q_f32(v_img_x.as_mut_ptr().add(offset), img_x);
                    vst1q_f32(v_img_y.as_mut_ptr().add(offset), img_y);
                }

                let mut sampled = [0.0f32; 8];
                sample_bilinear_v8(img, &v_img_x, &v_img_y, &mut sampled);

                for i in 0..count {
                    intensities[idx] = f64::from(sampled[i]);
                    idx += 1;
                }

                nx_start += 8.0 * dnx_du;
                ny_start += 8.0 * dny_du;
                d_start += 8.0 * dd_du;
            }

            current_nx_row += dda.dnx_dv as f32;
            current_ny_row += dda.dny_dv as f32;
            current_d_row += dda.dd_dv as f32;
        }
    }

    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    return sample_grid_values_optimized(img, h, roi, points, intensities, n);

    #[cfg(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    true
}

/// Sample values from the image using SIMD-optimized Fast-Math and ROI caching.
///
/// # Panics
/// Panics if the number of sample points exceeds `MAX_BIT_COUNT`.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn sample_grid_values_optimized(
    img: &crate::image::ImageView,
    h: &Homography,
    roi: &RoiCache,
    points: &[(f64, f64)],
    intensities: &mut [f64],
    n: usize,
) -> bool {
    let h00 = h.h[(0, 0)] as f32;
    let h01 = h.h[(0, 1)] as f32;
    let h02 = h.h[(0, 2)] as f32;
    let h10 = h.h[(1, 0)] as f32;
    let h11 = h.h[(1, 1)] as f32;
    let h12 = h.h[(1, 2)] as f32;
    let h20 = h.h[(2, 0)] as f32;
    let h21 = h.h[(2, 1)] as f32;
    let h22 = h.h[(2, 2)] as f32;

    let w_limit = (img.width - 1) as f32;
    let h_limit = (img.height - 1) as f32;

    for (i, &p) in points.iter().take(n).enumerate() {
        let px = p.0 as f32;
        let py = p.1 as f32;

        // Fast-Math Reciprocal
        let wz = h20 * px + h21 * py + h22;
        let winv = rcp_nr(wz);

        let img_x = (h00 * px + h01 * py + h02) * winv - 0.5;
        let img_y = (h10 * px + h11 * py + h12) * winv - 0.5;

        if img_x < 0.0 || img_x >= w_limit || img_y < 0.0 || img_y >= h_limit {
            return false;
        }

        let ix = img_x.floor() as usize;
        let iy = img_y.floor() as usize;

        // Sample from ROI cache using fixed-point bilinear
        let v00 = roi.get(ix, iy);
        let v10 = roi.get(ix + 1, iy);
        let v01 = roi.get(ix, iy + 1);
        let v11 = roi.get(ix + 1, iy + 1);

        intensities[i] = f64::from(bilinear_interpolate_fixed(img_x, img_y, v00, v10, v01, v11));
    }
    true
}

/// Sample the bit grid of `detection` (its corners' homography) for `decoder`'s points.
///
/// Samples bilinearly and classifies each bit against the decoder's adaptive threshold (a
/// blend of the grid's Otsu split and its quadrant means), as [`sample_grid_soa_precomputed`]
/// does in the pipeline. `None` when a sample falls outside the image.
///
/// # Panics
/// Panics if the number of sample points exceeds `MAX_BIT_COUNT`.
#[cfg(any(test, feature = "bench-internals"))]
#[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
pub fn sample_grid_generic(
    img: &crate::image::ImageView,
    arena: &Bump,
    detection: &crate::Detection,
    decoder: &(impl TagDecoder + ?Sized),
) -> Option<u64> {
    let (min_x, min_y, max_x, max_y) = detection.aabb();
    let roi = RoiCache::new(img, arena, min_x, min_y, max_x, max_y);

    let homography = Homography::square_to_quad(&detection.corners)?;

    let points = decoder.sample_points();
    // Stack-allocated buffer for up to 64 sample points (covers all standard tag families)
    let mut intensities = [0.0f64; MAX_BIT_COUNT];
    let n = points.len().min(MAX_BIT_COUNT);
    assert!(
        points.len() <= MAX_BIT_COUNT,
        "Tag bit count ({}) exceeds static buffer size ({})",
        points.len(),
        MAX_BIT_COUNT
    );

    if !sample_grid_values_dda_simd(img, &roi, &homography, decoder, &mut intensities) {
        return None;
    }

    Some(crate::strategy::bits_from_intensities(
        &intensities[..n],
        &compute_adaptive_thresholds(&intensities[..n], points),
    ))
}

/// Sample the bit grid using Structure of Arrays (SoA) data and a precomputed ROI cache.
///
/// # Panics
/// Panics if the number of sample points exceeds `MAX_BIT_COUNT`.
pub fn sample_grid_soa_precomputed(
    img: &crate::image::ImageView,
    roi: &RoiCache,
    homography: &Matrix3x3,
    decoder: &(impl TagDecoder + ?Sized),
) -> Option<u64> {
    let homography_obj = Homography::from_matrix3x3(homography);

    let points = decoder.sample_points();
    let mut intensities = [0.0f64; MAX_BIT_COUNT];
    let n = points.len().min(MAX_BIT_COUNT);
    assert!(
        points.len() <= MAX_BIT_COUNT,
        "Tag bit count ({}) exceeds static buffer size ({})",
        points.len(),
        MAX_BIT_COUNT
    );

    if !sample_grid_values_dda_simd(img, roi, &homography_obj, decoder, &mut intensities) {
        return None;
    }

    Some(crate::strategy::bits_from_intensities(
        &intensities[..n],
        &compute_adaptive_thresholds(&intensities[..n], points),
    ))
}

/// Internal helper to compute adaptive thresholds for a grid of intensities.
fn compute_adaptive_thresholds(intensities: &[f64], points: &[(f64, f64)]) -> [f64; MAX_BIT_COUNT] {
    let n = intensities.len();
    let global_threshold = compute_otsu_threshold(intensities);

    let mut quad_sums = [0.0; 4];
    let mut quad_counts = [0; 4];
    for (i, p) in points.iter().take(n).enumerate() {
        let qi = if p.0 < 0.0 {
            usize::from(p.1 >= 0.0)
        } else {
            2 + usize::from(p.1 >= 0.0)
        };
        quad_sums[qi] += intensities[i];
        quad_counts[qi] += 1;
    }

    let mut thresholds = [0.0f64; MAX_BIT_COUNT];
    for (i, p) in points.iter().take(n).enumerate() {
        let qi = if p.0 < 0.0 {
            usize::from(p.1 >= 0.0)
        } else {
            2 + usize::from(p.1 >= 0.0)
        };
        let quad_avg = if quad_counts[qi] > 0 {
            quad_sums[qi] / f64::from(quad_counts[qi])
        } else {
            global_threshold
        };

        // Blend global Otsu and local mean (0.7 / 0.3 weighting is common for fiducials)
        thresholds[i] = 0.7 * global_threshold + 0.3 * quad_avg;
    }
    thresholds
}

/// Rotate a square bit grid 90 degrees clockwise.
/// This is an O(1) bitwise operation but conceptually represents rotating the N x N pixel grid.
#[cfg(any(test, feature = "bench-internals"))]
#[must_use]
pub fn rotate90(bits: u64, dim: usize) -> u64 {
    let mut res = 0u64;
    for y in 0..dim {
        for x in 0..dim {
            if (bits >> (y * dim + x)) & 1 != 0 {
                let nx = dim - 1 - y;
                let ny = x;
                res |= 1 << (ny * dim + nx);
            }
        }
    }
    res
}

/// Sample the bit grid using scalar bilinear interpolation with distortion remapping.
///
/// Projects each canonical tag sample point through the ideal homography `h_ideal`
/// (computed from undistorted corners), then applies the camera distortion map to
/// convert the ideal pixel coordinate to the actual coordinate in the distorted image,
/// finally sampling the distorted image via bilinear interpolation.
///
/// This path is only called for non-rectified cameras (`!C::IS_RECTIFIED`). For rectified
/// cameras the faster SIMD path in [`sample_grid_soa_precomputed`] is used instead.
#[cfg(feature = "non_rectified")]
fn sample_grid_values_distorted<C: crate::camera::CameraModel>(
    img: &crate::image::ImageView,
    h_ideal: &Homography,
    decoder: &(impl TagDecoder + ?Sized),
    intrinsics: &crate::pose::CameraIntrinsics,
    model: &C,
    intensities: &mut [f64; MAX_BIT_COUNT],
) -> bool {
    let points = decoder.sample_points();
    let n = points.len().min(MAX_BIT_COUNT);
    sample_points_distorted(
        img,
        h_ideal,
        &points[..n],
        intrinsics,
        model,
        &mut intensities[..n],
    )
}

/// Sample arbitrary canonical points through the ideal homography and the distortion map
/// (see [`sample_grid_values_distorted`]).
#[cfg(feature = "non_rectified")]
#[expect(
    clippy::similar_names,
    reason = "paired coordinate-component bindings (px/py, xn/yn, xd/yd, ix/iy, nx/ny) follow the x/y and ideal-vs-distorted math notation and are intentionally similar"
)]
fn sample_points_distorted<C: crate::camera::CameraModel>(
    img: &crate::image::ImageView,
    h_ideal: &Homography,
    points: &[(f64, f64)],
    intrinsics: &crate::pose::CameraIntrinsics,
    model: &C,
    intensities: &mut [f64],
) -> bool {
    if points.is_empty() {
        return false;
    }
    let hm = &h_ideal.h;
    let w_limit = (img.width as f64) - 1.0 - 1e-4;
    let h_limit = (img.height as f64) - 1.0 - 1e-4;

    for (i, (u, v)) in points.iter().enumerate() {
        let u = *u;
        let v = *v;

        let nx = hm[(0, 0)] * u + hm[(0, 1)] * v + hm[(0, 2)];
        let ny = hm[(1, 0)] * u + hm[(1, 1)] * v + hm[(1, 2)];
        let d = hm[(2, 0)] * u + hm[(2, 1)] * v + hm[(2, 2)];

        if d.abs() < 1e-8 {
            return false;
        }

        let px_ideal = nx / d;
        let py_ideal = ny / d;

        // Convert ideal pixel → normalized → apply distortion → distorted pixel.
        let xn = (px_ideal - intrinsics.cx) / intrinsics.fx;
        let yn = (py_ideal - intrinsics.cy) / intrinsics.fy;
        let [xd, yd] = model.distort(xn, yn);
        let px = xd * intrinsics.fx + intrinsics.cx;
        let py = yd * intrinsics.fy + intrinsics.cy;

        if px < 0.0 || px > w_limit || py < 0.0 || py > h_limit {
            return false;
        }

        let ix = px.floor() as usize;
        let iy = py.floor() as usize;
        let stride = img.stride;
        // SAFETY: bounds checked above; ix <= w_limit - 1 < width - 1, iy <= h_limit - 1 < height - 1.
        let v00 = unsafe { *img.data.get_unchecked(iy * stride + ix) };
        // SAFETY: bounds checked above; ix+1 <= w_limit < width, iy within height.
        let v10 = unsafe { *img.data.get_unchecked(iy * stride + ix + 1) };
        // SAFETY: bounds checked above; iy+1 <= h_limit < height, ix within width.
        let v01 = unsafe { *img.data.get_unchecked((iy + 1) * stride + ix) };
        // SAFETY: bounds checked above; ix+1 <= w_limit < width, iy+1 <= h_limit < height.
        let v11 = unsafe { *img.data.get_unchecked((iy + 1) * stride + ix + 1) };
        intensities[i] = f64::from(bilinear_interpolate_fixed(
            px as f32, py as f32, v00, v10, v01, v11,
        ));
    }
    true
}

/// Distortion-aware decode for a single candidate using scalar sampling.
///
/// Undistorts the detected corners to compute an ideal homography, then samples the
/// distorted image at the correctly distortion-mapped coordinates for each bit sample
/// point. Called by [`decode_batch_soa_with_camera`] for non-rectified cameras.
#[cfg(feature = "non_rectified")]
fn decode_candidate_distorted<C: crate::camera::CameraModel>(
    img: &crate::image::ImageView,
    corners: &[Point2f; 4],
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
    intrinsics: &crate::pose::CameraIntrinsics,
    model: &C,
) -> (crate::batch::CandidateState, u32, u8, u64, f32) {
    use crate::batch::CandidateState;

    let ideal: [[f64; 2]; 4] = core::array::from_fn(|j| {
        intrinsics.undistort_pixel(f64::from(corners[j].x), f64::from(corners[j].y))
    });

    let center = [
        (ideal[0][0] + ideal[1][0] + ideal[2][0] + ideal[3][0]) * 0.25,
        (ideal[0][1] + ideal[1][1] + ideal[2][1] + ideal[3][1]) * 0.25,
    ];

    let mut best_h = u32::MAX;
    let mut best_bits = 0u64;
    // Lowest-Hamming candidate that is within its own decoder's budget and shows the
    // border ring: `(id, rotation, code, hamming)`.
    let mut accepted: Option<(u32, u8, u64, u32)> = None;
    // Ring evidence on the reported (unscaled) quad, at most once per decoder.
    let h_report = Homography::square_to_quad(&ideal);
    let mut ring_cache = [None::<bool>; MAX_DECODERS];

    let (decoder_max_h, decoder_ring_rate) = decoder_budgets(decoders, config);

    for &scale in &[1.0f64, 0.9, 1.1] {
        let scaled: [[f64; 2]; 4] = core::array::from_fn(|j| {
            [
                center[0] + (ideal[j][0] - center[0]) * scale,
                center[1] + (ideal[j][1] - center[1]) * scale,
            ]
        });

        let Some(h_ideal) = Homography::square_to_quad(&scaled) else {
            continue;
        };

        let mut intensities = [0.0f64; MAX_BIT_COUNT];

        for (decoder_idx, decoder) in decoders.iter().enumerate() {
            let n = decoder.bit_count();
            if !sample_grid_values_distorted::<C>(
                img,
                &h_ideal,
                decoder.as_ref(),
                intrinsics,
                model,
                &mut intensities,
            ) {
                continue;
            }

            let pts = decoder.sample_points();
            let thresholds = compute_adaptive_thresholds(&intensities[..n], pts);
            let code = crate::strategy::bits_from_intensities(&intensities[..n], &thresholds);

            if let Some((id, hamming, rot)) = decoder.decode_full(code, 255) {
                if hamming < best_h {
                    best_h = hamming;
                    best_bits = code;
                }
                let improves = accepted.is_none_or(|(_, _, _, h)| hamming < h);
                if improves
                    && hamming <= decoder_max_h[decoder_idx]
                    && *ring_cache[decoder_idx].get_or_insert_with(|| {
                        ring_ok(decoder_ring_rate[decoder_idx], || {
                            h_report.as_ref().and_then(|h| {
                                ring_evidence(decoder.as_ref(), |pts, out| {
                                    sample_points_distorted(img, h, pts, intrinsics, model, out)
                                })
                            })
                        })
                    })
                {
                    accepted = Some((id, rot, code, hamming));
                }
            }
            if accepted.is_some_and(|(_, _, _, h)| h == 0) {
                break;
            }
        }
        if accepted.is_some_and(|(_, _, _, h)| h == 0) {
            break;
        }
    }

    if let Some((id, rot, code, hamming)) = accepted {
        (CandidateState::Valid, id, rot, code, hamming as f32)
    } else {
        (
            CandidateState::FailedDecode,
            0,
            0,
            if best_h == u32::MAX { 0 } else { best_bits },
            if best_h == u32::MAX {
                0.0
            } else {
                best_h as f32
            },
        )
    }
}

/// Distortion-aware batch decode for non-rectified cameras.
#[cfg(feature = "non_rectified")]
fn decode_batch_soa_with_camera_inner<C: crate::camera::CameraModel>(
    batch: &mut crate::batch::DetectionBatch,
    n: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
    intrinsics: &crate::pose::CameraIntrinsics,
    model: &C,
) {
    use crate::batch::CandidateState;
    use rayon::prelude::*;

    // Split the batch into disjoint per-column mutable slices so each rayon
    // worker can write its target SoA cells directly, eliminating the
    // per-frame `Vec<TupleN>` heap allocation that the previous
    // collect-then-drain pattern paid outside the arena every frame.
    // Each worker owns its index slot via rayon's `Zip`; reads from
    // `corners_slot` happen before writes within a single closure body.
    let status_out = &mut batch.status_mask[..n];
    let ids_out = &mut batch.ids[..n];
    let payloads_out = &mut batch.payloads[..n];
    let error_rates_out = &mut batch.error_rates[..n];
    let corners_out = &mut batch.corners[..n];
    let homographies_out = &mut batch.homographies[..n];

    // Rayon `Zip` truncates to the shortest input; guard the disjoint-slice
    // contract so a future off-by-one fix on any column fails loudly in
    // debug rather than silently dropping the last candidate.
    debug_assert_eq!(status_out.len(), n);
    debug_assert_eq!(ids_out.len(), n);
    debug_assert_eq!(payloads_out.len(), n);
    debug_assert_eq!(error_rates_out.len(), n);
    debug_assert_eq!(corners_out.len(), n);
    debug_assert_eq!(homographies_out.len(), n);

    status_out
        .par_iter_mut()
        .zip(ids_out.par_iter_mut())
        .zip(payloads_out.par_iter_mut())
        .zip(error_rates_out.par_iter_mut())
        .zip(corners_out.par_iter_mut())
        .zip(homographies_out.par_iter_mut())
        .for_each(
            |(((((status_slot, id_slot), payload_slot), err_slot), corners_slot), h_slot)| {
                if *status_slot != CandidateState::Active {
                    // Bypass: preserve status_mask and error_rates (no-op
                    // writes in the original drain); zero ids/payloads to
                    // match the `(state, 0, 0, 0, error_rates)` tuple the
                    // original closure returned.
                    *id_slot = 0;
                    *payload_slot = 0;
                    return;
                }

                let (state, id, rot, bits, err) = decode_candidate_distorted::<C>(
                    img,
                    corners_slot,
                    decoders,
                    config,
                    intrinsics,
                    model,
                );

                *status_slot = state;
                *id_slot = id;
                *payload_slot = bits;
                *err_slot = err;

                // Reorder corners based on the decoded rotation (same convention as the SIMD path).
                if state == CandidateState::Valid && rot > 0 {
                    let mut tmp = [Point2f::default(); 4];
                    for (j, item) in tmp.iter_mut().enumerate() {
                        let src = (j + usize::from(rot)) % 4;
                        *item = corners_slot[src];
                    }
                    *corners_slot = tmp;

                    // Recompute the homography for the rotated corners. The
                    // pre-decoding homography mapped canonical (-1,-1) → old
                    // corners[0]; after the rotation, that point now lives at
                    // new corners[(4 - rot) % 4]. Downstream consumers
                    // (e.g. `CharucoRefiner` projecting saddle predictions
                    // through this homography) require it to remain aligned with
                    // the canonical TL/TR/BR/BL convention of `corners[]`.
                    // Without this recompute, `H(canonical_TL)` lands at the
                    // wrong image corner and any extrapolated point (saddle,
                    // bit-grid sample, etc.) lands at a rotated image position.
                    if let Some(h_new) = homography_matrix(&quad_to_f64(&tmp)) {
                        h_slot.data = h_new.data;
                    }
                }
            },
        );
}

/// Distortion-aware entry point for [`decode_batch_soa`].
///
/// When `C::IS_RECTIFIED = true` (i.e., [`PinholeModel`](crate::camera::PinholeModel)),
/// this delegates to the existing SIMD pipeline with zero overhead — the compiler
/// eliminates the distortion branch entirely via monomorphization.
///
/// When `C::IS_RECTIFIED = false`, each candidate's corners are undistorted to compute
/// an ideal homography. The bit grid is then sampled by projecting through the ideal
/// homography and applying the distortion map to get coordinates in the raw distorted
/// image, ensuring accurate bit sampling even under large fisheye distortion.
///
/// [`PinholeModel`]: crate::camera::PinholeModel
#[cfg(feature = "non_rectified")]
#[tracing::instrument(skip_all, name = "pipeline::decoding_pass_distortion")]
pub fn decode_batch_soa_with_camera<C: crate::camera::CameraModel>(
    batch: &mut crate::batch::DetectionBatch,
    n: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
    intrinsics: Option<&crate::pose::CameraIntrinsics>,
    model: &C,
) {
    if C::IS_RECTIFIED {
        // Zero-overhead path for rectified images: delegate to the existing SIMD pipeline.
        // The `if C::IS_RECTIFIED` is a compile-time constant; for PinholeModel the
        // compiler eliminates the else branch entirely via dead-code elimination.
        decode_batch_soa(batch, n, img, decoders, config);
    } else if let Some(intrinsics) = intrinsics {
        decode_batch_soa_with_camera_inner::<C>(batch, n, img, decoders, config, intrinsics, model);
    } else {
        // No intrinsics provided — fall back to the standard path.
        decode_batch_soa(batch, n, img, decoders, config);
    }
}

/// Decode all active candidates in the batch using the Structure of Arrays (SoA) layout, for a
/// pinhole camera (no lens distortion).
///
/// This phase executes SIMD bilinear interpolation and Hamming error correction, then refines
/// the accepted corners (sub-pixel pass and photometric calibration). If a candidate fails
/// decoding, its `status_mask` is flipped to `FailedDecode`.
#[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
#[tracing::instrument(skip_all, name = "pipeline::decoding_pass")]
pub fn decode_batch_soa(
    batch: &mut crate::batch::DetectionBatch,
    n: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
) {
    use crate::batch::CandidateState;
    use rayon::prelude::*;

    // `None` for `max_hamming_error` means "use family defaults"; an explicit `Some(n)`
    // overrides every family uniformly.
    let (decoder_max_h_buf, decoder_ring_rate_buf) = decoder_budgets(decoders, config);
    let decoder_max_h = &decoder_max_h_buf[..decoders.len()];
    let decoder_ring_rate = &decoder_ring_rate_buf[..decoders.len()];
    // Decode-first matches come from unrefined contour corners, about a pixel off on small
    // markers, so a ring sample can land in the white surround. Such a match is verified on the
    // refined quad with the full budget; the seed only has to look like a marker, with the
    // recovery gate's tolerance.
    let mut seed_ring_rate_buf = decoder_ring_rate_buf;
    if config.decode_first() {
        for rate in &mut seed_ring_rate_buf[..decoders.len()] {
            *rate = rate.max(RECOVERY_RING_MAX_ERROR_RATE);
        }
    }
    let seed_ring_rate = &seed_ring_rate_buf[..decoders.len()];
    // Looser frame-level floor used by the recovery-refinement gate
    // ("did we fail to accept anywhere?"). In single-family setups this
    // equals the family default; in multi-family setups it preserves
    // today's behaviour of still considering recovery.
    let frame_max_h_floor = decoder_max_h.iter().copied().max().unwrap_or(0);
    let recovery_max_h = decoders
        .iter()
        .map(|d| recovery_window(d.bit_count(), d.num_codes()))
        .max()
        .unwrap_or(0);

    // Split the batch into disjoint per-column mutable slices so each
    // rayon worker can write its target SoA cells directly, eliminating
    // the per-frame `Vec<TupleN>` heap allocation that the previous
    // collect-then-drain pattern paid outside the arena every frame. The
    // closure snapshots `corners_slot` / `h_slot` into stack-allocated
    // locals before performing any work, so the eventual writes back to
    // those slots at end-of-closure preserve the original sequential
    // semantics (refined-corners overwrite first, then optional rotation
    // permutation + homography recompute).
    let status_out = &mut batch.status_mask[..n];
    let ids_out = &mut batch.ids[..n];
    let payloads_out = &mut batch.payloads[..n];
    let error_rates_out = &mut batch.error_rates[..n];
    let corners_out = &mut batch.corners[..n];
    let homographies_out = &mut batch.homographies[..n];
    let refined_out = &mut batch.corner_refined[..n];

    // Rayon `Zip` truncates to the shortest input; guard the disjoint-slice
    // contract so a future off-by-one fix on any column fails loudly in
    // debug rather than silently dropping the last candidate.
    debug_assert_eq!(status_out.len(), n);
    debug_assert_eq!(ids_out.len(), n);
    debug_assert_eq!(payloads_out.len(), n);
    debug_assert_eq!(error_rates_out.len(), n);
    debug_assert_eq!(corners_out.len(), n);
    debug_assert_eq!(homographies_out.len(), n);

    status_out
        .par_iter_mut()
        .zip(ids_out.par_iter_mut())
        .zip(payloads_out.par_iter_mut())
        .zip(error_rates_out.par_iter_mut())
        .zip(corners_out.par_iter_mut())
        .zip(homographies_out.par_iter_mut())
        .zip(refined_out.par_iter_mut())
        .for_each(
            |(
                (((((status_slot, id_slot), payload_slot), err_slot), corners_slot), h_slot),
                refined_slot,
            )| {
                if *status_slot != CandidateState::Active {
                    // Bypass: preserve status_mask and error_rates (no-op
                    // writes in the original drain); zero ids/payloads to
                    // match the `(state, 0, 0, 0, error_rates, None)` tuple
                    // the original closure returned.
                    *id_slot = 0;
                    *payload_slot = 0;
                    return;
                }

                // Snapshot the input corners and homography into locals so
                // they cannot alias the later write-backs to the same SoA
                // slots. The original sequential closure read `&batch.corners[i]`
                // / `&batch.homographies[i]` throughout, untouched until the
                // post-collect drain wrote the refined/rotated values back.
                let input_corners: [Point2f; 4] = *corners_slot;
                let input_homography: Matrix3x3 = *h_slot;

                // Cells across the matched family's outline (payload plus the one-cell border):
                // the layout the sub-pixel stage and the photometric calibration read.
                let mut cells = 0usize;
                let (state, id, rot, payload, error_rate, refined_corners) = WORKSPACE_ARENA
                    .with_borrow_mut(|arena| {
                        arena.reset();

                        let corners = &input_corners;
                        let homography = &input_homography;

                        // Compute AABB for RoiCache ONCE per candidate.
                        // We expand it slightly (10%) to ensure scaled versions (0.9, 1.1) still fit.
                        let mut min_x = f32::MAX;
                        let mut min_y = f32::MAX;
                        let mut max_x = f32::MIN;
                        let mut max_y = f32::MIN;
                        for p in corners {
                            min_x = min_x.min(p.x);
                            min_y = min_y.min(p.y);
                            max_x = max_x.max(p.x);
                            max_y = max_y.max(p.y);
                        }
                        let w_aabb = max_x - min_x;
                        let h_aabb = max_y - min_y;
                        let roi = RoiCache::new(
                            img,
                            arena,
                            ((min_x - w_aabb * 0.1).floor() as i32).max(0) as usize,
                            ((min_y - h_aabb * 0.1).floor() as i32).max(0) as usize,
                            (((max_x + w_aabb * 0.1).ceil() as i32).min(img.width as i32 - 1))
                                .max(0) as usize,
                            (((max_y + h_aabb * 0.1).ceil() as i32).min(img.height as i32 - 1))
                                .max(0) as usize,
                        );

                        let mut best_h = u32::MAX;
                        let mut best_code = None;
                        let mut best_id = 0;
                        let mut best_rot = 0;
                        // Ring evidence is evaluated on the reported (unscaled) quad, so it
                        // depends only on the decoder: compute it at most once per decoder.
                        let mut ring_cache = [None::<bool>; MAX_DECODERS];
                        // The quad-stage refinement of a decode-first seed depends only on the
                        // seed corners: run it at most once per candidate.
                        let seed_corners = quad_to_f64(corners);
                        let mut decode_first_refined = None::<Option<[[f64; 2]; 4]>>;

                        let scales = [1.0, 0.9, 1.1];
                        let center = [
                            (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4.0,
                            (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4.0,
                        ];

                        for scale in scales {
                            let scaled_h_mat;
                            let current_homography = if (scale - 1.0f32).abs() > 1e-4 {
                                // The scaled quad stays in f32, like the batch corners.
                                let scaled_corners = corners.map(|p| Point2f {
                                    x: center[0] + (p.x - center[0]) * scale,
                                    y: center[1] + (p.y - center[1]) * scale,
                                });
                                // Degenerate scale: skip it.
                                let Some(h) = homography_matrix(&quad_to_f64(&scaled_corners))
                                else {
                                    continue;
                                };
                                scaled_h_mat = h;
                                &scaled_h_mat
                            } else {
                                homography
                            };

                            let mut best_h_in_scale = u32::MAX;
                            let mut best_match_in_scale: Option<(u32, u32, u8, u64, usize)> = None;
                            for (decoder_idx, decoder) in decoders.iter().enumerate() {
                                let Some(code) = sample_grid_soa_precomputed(
                                    img,
                                    &roi,
                                    current_homography,
                                    decoder.as_ref(),
                                ) else {
                                    continue;
                                };
                                let Some((id, hamming, rot)) = decoder.decode_full(code, 255)
                                else {
                                    continue;
                                };
                                best_h = best_h.min(hamming);

                                // Lowest Hamming distance wins across decoders. Evidence must
                                // hold for the geometry that is reported: the unscaled quad
                                // (`homography`), not the scaled one that happened to decode.
                                // A quiet zone's outer contour decodes at scale 0.9 but its
                                // ring is white.
                                if hamming <= decoder_max_h[decoder_idx]
                                    && hamming < best_h_in_scale
                                    && *ring_cache[decoder_idx].get_or_insert_with(|| {
                                        border_ring_ok(
                                            img,
                                            &roi,
                                            homography,
                                            decoder.as_ref(),
                                            seed_ring_rate[decoder_idx],
                                        )
                                    })
                                {
                                    best_h_in_scale = hamming;
                                    best_match_in_scale =
                                        Some((id, hamming, rot, code, decoder_idx));
                                }
                            }
                            if let Some((id, hamming, rot, code, decoder_idx)) = best_match_in_scale
                            {
                                let decoder = decoders[decoder_idx].as_ref();
                                cells = decoder.dimension() + 2;

                                // Always perform ERF refinement for finalists if requested
                                if config.refinement_mode
                                    == crate::config::CornerRefinementMode::Erf
                                {
                                    let refined_corners = if !config.decode_first() {
                                        refine_corners_erf(
                                            arena,
                                            img,
                                            &seed_corners,
                                            config.subpixel_refinement_sigma,
                                        )
                                    } else if let Some(refined) = *decode_first_refined
                                        .get_or_insert_with(|| {
                                            refine_decode_first_seed(
                                                arena,
                                                img,
                                                &seed_corners,
                                                config,
                                            )
                                        })
                                    {
                                        refined
                                    } else {
                                        continue;
                                    };
                                    let refined_corners_f32 = quad_to_f32(&refined_corners);

                                    // Verify that the refined corners still decode.
                                    let Some(ref_h_mat) = homography_matrix(&refined_corners)
                                    else {
                                        // Degenerate refinement. Refine-first ordering keeps
                                        // the unrefined match unless a later scale decodes; a
                                        // decode-first match stands only once its refined quad
                                        // verifies it, so it never reaches the acceptance
                                        // after the scale loop with its contour corners.
                                        if !config.decode_first() {
                                            best_code = Some(code);
                                            best_id = id;
                                            best_rot = rot;
                                        }
                                        continue;
                                    };

                                    if config.decode_first() {
                                        // Decode-first: the match came from unrefined corners,
                                        // so it stands only if the refined quad decodes the
                                        // same id within budget at the scale that matched and
                                        // shows its border ring; otherwise it is discarded.
                                        let verified = homography_matrix(&scale_about_centroid(
                                            &refined_corners,
                                            f64::from(scale),
                                        ))
                                        .and_then(|m| {
                                            sample_grid_soa_precomputed(img, &roi, &m, decoder)
                                        })
                                        .and_then(|code_ref| {
                                            decoder.decode_full(code_ref, 255).map(
                                                |(id_ref, hamming_ref, rot_ref)| {
                                                    (code_ref, id_ref, hamming_ref, rot_ref)
                                                },
                                            )
                                        })
                                        .filter(|&(_, id_ref, hamming_ref, _)| {
                                            id_ref == id
                                                && hamming_ref <= decoder_max_h[decoder_idx]
                                                && border_ring_ok(
                                                    img,
                                                    &roi,
                                                    &ref_h_mat,
                                                    decoder,
                                                    decoder_ring_rate[decoder_idx],
                                                )
                                        });
                                        if let Some((code_ref, _, hamming_ref, rot_ref)) = verified
                                        {
                                            return (
                                                CandidateState::Valid,
                                                id,
                                                rot_ref,
                                                code_ref,
                                                hamming_ref as f32,
                                                Some(refined_corners_f32),
                                            );
                                        }
                                        continue;
                                    }

                                    // Keep the refined corners if they decode the same tag with
                                    // a Hamming distance that is not worse.
                                    if let Some(code_ref) =
                                        sample_grid_soa_precomputed(img, &roi, &ref_h_mat, decoder)
                                        && let Some((id_ref, hamming_ref, _)) =
                                            decoder.decode_full(code_ref, 255)
                                        && id_ref == id
                                        && hamming_ref <= hamming
                                    {
                                        return (
                                            CandidateState::Valid,
                                            id,
                                            rot,
                                            code_ref,
                                            hamming_ref as f32,
                                            Some(refined_corners_f32),
                                        );
                                    }
                                }

                                return (
                                    CandidateState::Valid,
                                    id,
                                    rot,
                                    code,
                                    hamming as f32,
                                    None,
                                );
                            }

                            if best_h == 0 {
                                break;
                            }
                        }

                        // Stage 2: Configurable Corner Refinement (Recovery for near-misses).
                        // `best_h <= recovery_max_h` implies some decoder sampled and decoded.
                        if best_h > frame_max_h_floor && best_h <= recovery_max_h && {
                            let seed_h = Homography::from_matrix3x3(homography);
                            decoders.iter().any(|d| {
                                ring_budget_ok(
                                    rectified_ring_evidence(img, &roi, &seed_h, d.as_ref()),
                                    RECOVERY_RING_MAX_ERROR_RATE,
                                )
                            })
                        } {
                            match config.refinement_mode {
                                crate::config::CornerRefinementMode::None => {},
                                crate::config::CornerRefinementMode::Erf => {
                                    let nudge = 0.2;
                                    let mut current_corners = *corners;

                                    // Decode-first ordering left these corners unrefined: a
                                    // near miss gets the refinement it skipped, then one more
                                    // decode at each scale, before the coarse nudge search.
                                    if config.decode_first()
                                        && let Some(refined) =
                                            *decode_first_refined.get_or_insert_with(|| {
                                                refine_decode_first_seed(
                                                    arena,
                                                    img,
                                                    &seed_corners,
                                                    config,
                                                )
                                            })
                                        // Replay the scale retries on the refined quad, as
                                        // refine-first ordering does; the ring is checked on
                                        // the reported (unscaled) quad.
                                        && let Some(h_unscaled) = homography_matrix(&refined)
                                    {
                                        let refined_f32 = quad_to_f32(&refined);
                                        for scale in scales {
                                            let Some(h_mat) = homography_matrix(
                                                &scale_about_centroid(&refined, f64::from(scale)),
                                            ) else {
                                                continue;
                                            };
                                            for (decoder_idx, decoder) in
                                                decoders.iter().enumerate()
                                            {
                                                let Some(code) = sample_grid_soa_precomputed(
                                                    img,
                                                    &roi,
                                                    &h_mat,
                                                    decoder.as_ref(),
                                                ) else {
                                                    continue;
                                                };
                                                let Some((id, hamming, rot)) =
                                                    decoder.decode_full(code, 255)
                                                else {
                                                    continue;
                                                };
                                                if hamming < best_h {
                                                    best_h = hamming;
                                                    current_corners = refined_f32;
                                                }
                                                if hamming <= decoder_max_h[decoder_idx]
                                                    && border_ring_ok(
                                                        img,
                                                        &roi,
                                                        &h_unscaled,
                                                        decoder.as_ref(),
                                                        decoder_ring_rate[decoder_idx],
                                                    )
                                                {
                                                    cells = decoder.dimension() + 2;
                                                    return (
                                                        CandidateState::Valid,
                                                        id,
                                                        rot,
                                                        code,
                                                        hamming as f32,
                                                        Some(refined_f32),
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    for _pass in 0..2 {
                                        let mut pass_improved = false;
                                        for c_idx in 0..4 {
                                            for (dx, dy) in [
                                                (nudge, 0.0),
                                                (-nudge, 0.0),
                                                (0.0, nudge),
                                                (0.0, -nudge),
                                            ] {
                                                let mut test_corners = current_corners;
                                                test_corners[c_idx].x += dx;
                                                test_corners[c_idx].y += dy;

                                                // Must recompute homography for the nudged corners
                                                let Some(h_mat) =
                                                    homography_matrix(&quad_to_f64(&test_corners))
                                                else {
                                                    continue;
                                                };
                                                for (decoder_idx, decoder) in
                                                    decoders.iter().enumerate()
                                                {
                                                    let Some(code) = sample_grid_soa_precomputed(
                                                        img,
                                                        &roi,
                                                        &h_mat,
                                                        decoder.as_ref(),
                                                    ) else {
                                                        continue;
                                                    };
                                                    let Some((id, hamming, rot)) =
                                                        decoder.decode_full(code, 255)
                                                    else {
                                                        continue;
                                                    };
                                                    if hamming >= best_h {
                                                        continue;
                                                    }
                                                    best_h = hamming;
                                                    current_corners = test_corners;
                                                    pass_improved = true;
                                                    if hamming <= decoder_max_h[decoder_idx]
                                                        && border_ring_ok(
                                                            img,
                                                            &roi,
                                                            &h_mat,
                                                            decoder.as_ref(),
                                                            decoder_ring_rate[decoder_idx],
                                                        )
                                                    {
                                                        cells = decoder.dimension() + 2;
                                                        return (
                                                            CandidateState::Valid,
                                                            id,
                                                            rot,
                                                            code,
                                                            best_h as f32,
                                                            Some(current_corners),
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                        if !pass_improved {
                                            break;
                                        }
                                    }
                                },
                            }
                        }

                        if let Some(code) = best_code {
                            (
                                CandidateState::Valid,
                                best_id,
                                best_rot,
                                code,
                                best_h as f32,
                                None,
                            )
                        } else {
                            // Even on failure, return the best hamming distance found for debugging.
                            // If no code was sampled at all, best_h will be u32::MAX.
                            (
                                CandidateState::FailedDecode,
                                0,
                                0,
                                0,
                                if best_h == u32::MAX {
                                    0.0
                                } else {
                                    best_h as f32
                                },
                                None,
                            )
                        }
                    });

                *status_slot = state;
                *id_slot = id;
                *payload_slot = payload;
                *err_slot = error_rate;

                let decoder_refined = refined_corners.is_some();
                if let Some(decoded_corners) = refined_corners {
                    *corners_slot = decoded_corners;
                }

                // Gradient-orthogonality refinement of the accepted corners, after the
                // configured refinement mode (`decoder.corner_subpix`).
                let subpix = state == CandidateState::Valid && config.decoder_corner_subpix;
                let mut refined_bits = 0u8;
                if subpix {
                    let seed = quad_to_f64(corners_slot);
                    let (subpix_corners, bits) =
                        crate::refinement::subpix_marker_corners(img, seed, cells);
                    // Remove the photometric inset the decoded marker's bit edges measure.
                    let final_corners =
                        crate::marker_inset::calibrate_marker_corners(img, subpix_corners, cells)
                            .unwrap_or(subpix_corners);
                    refined_bits = bits;
                    *corners_slot = quad_to_f32(&final_corners);
                }

                if *status_slot == CandidateState::Valid
                    && !outline_observed(corners_slot, img.width, img.height)
                {
                    *status_slot = CandidateState::FailedDecode;
                }

                // Apply rotation reorder, if any.
                let valid = *status_slot == CandidateState::Valid;
                if valid && rot > 0 {
                    let mut temp_corners = [Point2f::default(); 4];
                    for (j, item) in temp_corners.iter_mut().enumerate() {
                        let src_idx = (j + usize::from(rot)) % 4;
                        *item = corners_slot[src_idx];
                    }
                    *corners_slot = temp_corners;
                    // Corner j now holds the old corner (j + rot) % 4.
                    let r = u32::from(rot) % 4;
                    refined_bits = ((refined_bits >> r) | (refined_bits << (4 - r))) & 0x0F;
                }
                *refined_slot = refined_bits;

                // Recompute the homography whenever corners changed — ERF or sub-pixel
                // refinement, *or* rotation. Without the ERF branch, a
                // canonical-orientation refined candidate (`rot == 0`,
                // `refined_corners.is_some()`) would carry the pre-refinement
                // homography forward into Phase D and `CharucoRefiner`, which
                // projects saddle predictions through `batch.homographies[i]`.
                // The stale-`h_slot` failure mode is the same class as
                // `memory/project_refine_saddle_noop.md`.
                //
                // A degenerate quad (zero area, collinear after rotation) has no homography:
                // `h_slot` then retains the previous one, mirroring pre-existing best-effort
                // behaviour. A stricter design would downgrade to `FailedDecode` here — left for
                // a follow-up that can weigh the recall trade-off against benchmarks.
                if valid
                    && (decoder_refined || subpix || rot > 0)
                    && let Some(h_new) = homography_matrix(&quad_to_f64(corners_slot))
                {
                    h_slot.data = h_new.data;
                }
            },
        );
}

/// A trait for decoding binary payloads from extracted tags.
pub trait TagDecoder: Send + Sync {
    /// Returns the name of the decoder family (e.g., "AprilTag36h11").
    fn name(&self) -> &str;
    /// Returns the dimension of the tag grid (e.g., 6 for 36h11).
    fn dimension(&self) -> usize;
    /// Returns the active number of bits in the tag (e.g., 41 for 41h12).
    fn bit_count(&self) -> usize;
    /// Returns the ideal sample points in canonical coordinates [-1, 1].
    fn sample_points(&self) -> &[(f64, f64)];
    /// Decodes the extracted bits into a tag ID, hamming distance, and rotation count.
    ///
    /// Returns `Some((id, hamming, rotation))` if decoding is successful, `None` otherwise.
    /// `rotation` is 0-3, representing 90-degree CW increments.
    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)>; // (id, hamming, rotation)
    /// Decodes with custom maximum hamming distance.
    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)>;
    /// Get the original code for a given ID (useful for testing/simulation).
    fn get_code(&self, id: u16) -> Option<u64>;
    /// Returns the total number of codes in the dictionary.
    fn num_codes(&self) -> usize;
    /// Returns all rotated versions of all codes in the dictionary: (bits, id, rotation)
    fn rotated_codes(&self) -> &[(u64, u16, u8)];
    /// Family-specific maximum Hamming budget used when `DetectorConfig`
    /// leaves `max_hamming_error` unset. Empirically tuned per family:
    /// 36h11 = 2 (code distance 11), 16h5 = 0 (code distance 5; admitting
    /// h≤1 floods the rendered tag16h5 1080p suite with false positives
    /// at unchanged recall — see `regression_hub_tag16h5_1080p`),
    /// ArUco4x4_* = 1 (dense codebooks), ArUco6x6_250 = 2, ArUcoMip36h12 = 2 (code distance 12).
    fn default_max_hamming(&self) -> u32;
}

/// Decoder for the AprilTag 36h11 family.
pub struct AprilTag36h11;

impl TagDecoder for AprilTag36h11 {
    fn name(&self) -> &'static str {
        "36h11"
    }
    fn dimension(&self) -> usize {
        6
    } // 6x6 grid of bits (excluding border)
    fn bit_count(&self) -> usize {
        36
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_APRILTAG36H11
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        // Use the pre-calculated dictionary with O(1) exact match + cached rotations.
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11)
            .decode(bits, 4) // Allow up to 4 bit errors for maximum recall
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[] // Removed from runtime struct, only used by testing/simulation which we will adjust later.
    }
    fn default_max_hamming(&self) -> u32 {
        2
    }
}

/// Decoder for the AprilTag 16h5 family.
pub struct AprilTag16h5;

impl TagDecoder for AprilTag16h5 {
    fn name(&self) -> &'static str {
        "16h5"
    }
    fn dimension(&self) -> usize {
        4
    }
    fn bit_count(&self) -> usize {
        16
    }
    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_APRILTAG16H5
    }
    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5)
            .decode(bits, 1)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }
    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }
    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5).get_code(id)
    }
    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5).len()
    }
    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }
    fn default_max_hamming(&self) -> u32 {
        0
    }
}

/// Decoder for the ArUco 4x4_50 family.
pub struct ArUco4x4_50;

impl TagDecoder for ArUco4x4_50 {
    fn name(&self) -> &'static str {
        "4X4_50"
    }
    fn dimension(&self) -> usize {
        4
    }
    fn bit_count(&self) -> usize {
        16
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_ARUCO4X4_50
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50)
            .decode(bits, 2)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }
    fn default_max_hamming(&self) -> u32 {
        1
    }
}

/// Decoder for the ArUco 4x4_100 family.
pub struct ArUco4x4_100;

impl TagDecoder for ArUco4x4_100 {
    fn name(&self) -> &'static str {
        "4X4_100"
    }
    fn dimension(&self) -> usize {
        4
    }
    fn bit_count(&self) -> usize {
        16
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_ARUCO4X4_100
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100)
            .decode(bits, 2)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }
    fn default_max_hamming(&self) -> u32 {
        1
    }
}

/// Decoder for the ArUco 6x6_250 family.
pub struct ArUco6x6_250;

impl TagDecoder for ArUco6x6_250 {
    fn name(&self) -> &'static str {
        "6X6_250"
    }
    fn dimension(&self) -> usize {
        6
    }
    fn bit_count(&self) -> usize {
        36
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_ARUCO6X6_250
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250)
            .decode(bits, 4)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }
    fn default_max_hamming(&self) -> u32 {
        2
    }
}

/// Decoder for the ArUco MIP 36h12 family.
pub struct ArUcoMip36h12;

impl TagDecoder for ArUcoMip36h12 {
    fn name(&self) -> &'static str {
        "MIP_36h12"
    }
    fn dimension(&self) -> usize {
        6
    }
    fn bit_count(&self) -> usize {
        36
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_ARUCOMIP36H12
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUcoMip36h12)
            .decode(bits, 4)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUcoMip36h12)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUcoMip36h12).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUcoMip36h12).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }
    fn default_max_hamming(&self) -> u32 {
        2
    }
}

/// Convert a TagFamily enum to a boxed decoder instance.
#[must_use]
pub fn family_to_decoder(family: config::TagFamily) -> Box<dyn TagDecoder + Send + Sync> {
    match family {
        config::TagFamily::AprilTag16h5 => Box::new(AprilTag16h5),
        config::TagFamily::AprilTag36h11 => Box::new(AprilTag36h11),
        config::TagFamily::ArUco4x4_50 => Box::new(ArUco4x4_50),
        config::TagFamily::ArUco4x4_100 => Box::new(ArUco4x4_100),
        config::TagFamily::ArUco6x6_250 => Box::new(ArUco6x6_250),
        config::TagFamily::ArUcoMip36h12 => Box::new(ArUcoMip36h12),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn outline_observed_rejects_markers_the_frame_cuts() {
        let p = |x: f32, y: f32| Point2f { x, y };
        let (w, h) = (64, 48);
        // A corner touching the border is still observed.
        assert!(outline_observed(
            &[p(20.0, 0.5), p(40.0, 12.0), p(28.0, 30.0), p(8.0, 18.0)],
            w,
            h
        ));
        // A corner outside the image.
        assert!(!outline_observed(
            &[p(20.0, -0.5), p(40.0, 12.0), p(28.0, 30.0), p(8.0, 18.0)],
            w,
            h
        ));
        // A side running along the bottom border: the frame edge, not a marker edge.
        assert!(!outline_observed(
            &[p(10.0, 20.0), p(30.0, 20.0), p(30.0, 46.0), p(10.0, 46.5)],
            w,
            h
        ));
        assert!(outline_observed(
            &[p(10.0, 20.0), p(30.0, 20.0), p(30.0, 44.0), p(10.0, 44.5)],
            w,
            h
        ));
    }

    proptest! {
        #[test]
        fn test_rotation_invariants(bits in 0..u64::MAX) {
            let dim = 6;
            let r1 = rotate90(bits, dim);
            let r2 = rotate90(r1, dim);
            let r3 = rotate90(r2, dim);
            let r4 = rotate90(r3, dim);

            // Mask to dim*dim bits to avoid noise in upper bits
            let mask = (1u64 << (dim * dim)) - 1;
            prop_assert_eq!(bits & mask, r4 & mask);
        }

        #[test]
        fn test_hamming_robustness(
            id_idx in 0usize..10,
            rotation in 0..4usize,
            flip1 in 0..36usize,
            flip2 in 0..36usize
        ) {
            let decoder = AprilTag36h11;
            let orig_id = id_idx as u16;
            let dict = crate::dictionaries::get_dictionary(config::TagFamily::AprilTag36h11);

            let mut test_bits = dict.codes[(id_idx * 4) + rotation];

            // Flip bits
            test_bits ^= 1 << flip1;
            test_bits ^= 1 << flip2;

            let result = decoder.decode(test_bits);
            prop_assert!(result.is_some());
            let (decoded_id, _, _) = result.expect("Should decode valid pattern");
            prop_assert_eq!(decoded_id, u32::from(orig_id));
        }

        #[test]
        fn test_false_positive_resistance(bits in 0..u64::MAX) {
            let decoder = AprilTag36h11;
            // Random bitstreams should rarely match any of the 587 codes
            if let Some((_id, hamming, _rot)) = decoder.decode(bits) {
                // If it decodes, it must have low hamming distance
                prop_assert!(hamming <= 4);
            }
        }

        #[test]
        fn prop_homography_projection(
            src in prop::collection::vec((-100.0..100.0, -100.0..100.0), 4),
            dst in prop::collection::vec((0.0..1000.0, 0.0..1000.0), 4)
        ) {
            let src_pts = [
                [src[0].0, src[0].1],
                [src[1].0, src[1].1],
                [src[2].0, src[2].1],
                [src[3].0, src[3].1],
            ];
            let dst_pts = [
                [dst[0].0, dst[0].1],
                [dst[1].0, dst[1].1],
                [dst[2].0, dst[2].1],
                [dst[3].0, dst[3].1],
            ];

            if let Some(h) = Homography::from_pairs(&src_pts, &dst_pts) {
                for i in 0..4 {
                    let p = h.project(src_pts[i]);
                    // Check for reasonable accuracy. 1e-4 is conservative for float precision
                    // issues in near-singular cases where from_pairs still returns Some.
                    prop_assert!((p[0] - dst_pts[i][0]).abs() < 1e-3,
                        "Point {}: project({:?}) -> {:?}, expected {:?}", i, src_pts[i], p, dst_pts[i]);
                    prop_assert!((p[1] - dst_pts[i][1]).abs() < 1e-3);
                }
            }
        }
    }

    #[test]
    fn recovery_window_bounds_texture_false_trigger_rate() {
        use crate::config::TagFamily;
        let window = |f| {
            let d = family_to_decoder(f);
            recovery_window(d.bit_count(), d.num_codes())
        };
        assert_eq!(window(TagFamily::AprilTag36h11), 6);
        assert_eq!(window(TagFamily::ArUcoMip36h12), 6);
        assert_eq!(window(TagFamily::ArUco6x6_250), 6);
        assert_eq!(window(TagFamily::AprilTag16h5), 1);
        assert_eq!(window(TagFamily::ArUco4x4_50), 1);
        assert_eq!(window(TagFamily::ArUco4x4_100), 0);
    }

    #[test]
    fn test_all_codes_decode() {
        let decoder = AprilTag36h11;
        for id in 0..587u16 {
            let code = crate::dictionaries::DICT_APRILTAG36H11
                .get_code(id)
                .expect("valid ID");
            let result = decoder.decode(code);
            assert!(result.is_some());
            let (id_out, _, _) = result.unwrap();
            assert_eq!(id_out, u32::from(id));
        }
    }
    #[test]
    fn test_grid_sampling() {
        let width = 64;
        let height = 64;
        let mut data = vec![0u8; width * height];
        // 8x8 grid, 36x36px tag centered at 32,32 => corners [14, 50]
        // TL=(14,14), TR=(50,14), BR=(50,50), BL=(14,50)

        // Border:
        for gy in 0..8 {
            for gx in 0..8 {
                if gx == 0 || gx == 7 || gy == 0 || gy == 7 {
                    for y in 0..4 {
                        for x in 0..4 {
                            let px = 14 + (f64::from(gx) * 4.5) as usize + x;
                            let py = 14 + (f64::from(gy) * 4.5) as usize + y;
                            if px < 64 && py < 64 {
                                data[py * width + px] = 0;
                            }
                        }
                    }
                }
            }
        }
        // Bit 0 (cell 1,1) -> White (canonical p = -0.625, -0.625)
        for y in 0..4 {
            for x in 0..4 {
                let px = 14 + (1.0 * 4.5) as usize + x;
                let py = 14 + (1.0 * 4.5) as usize + y;
                data[py * width + px] = 255;
            }
        }
        // Bit 35 (cell 6,6) -> Black (canonical p = 0.625, 0.625)
        for y in 0..4 {
            for x in 0..4 {
                let px = 14 + (6.0 * 4.5) as usize + x;
                let py = 14 + (6.0 * 4.5) as usize + y;
                data[py * width + px] = 0;
            }
        }

        let img = crate::image::ImageView::new(&data, width, height, width).unwrap();

        let decoder = AprilTag36h11;
        let arena = Bump::new();
        let cand = crate::Detection {
            corners: [[14.0, 14.0], [50.0, 14.0], [50.0, 50.0], [14.0, 50.0]],
            ..Default::default()
        };
        let bits =
            sample_grid_generic(&img, &arena, &cand, &decoder).expect("Should sample successfully");

        // bit 0 should be 1 (high intensity)
        assert_eq!(bits & 1, 1, "Bit 0 should be 1");
        // bit 35 should be 0 (low intensity)
        assert_eq!((bits >> 35) & 1, 0, "Bit 35 should be 0");
    }

    /// 8×8-cell AprilTag36h11-layout tag, 10 px per cell, at (20, 20) on a 120×120 page;
    /// payload cells alternate so both classes are present. `white_ring` lists ring cells
    /// `(gx, gy)` to paint white.
    fn ring_probe(white_ring: &[(usize, usize)]) -> Vec<u8> {
        let (w, cell, origin) = (120usize, 10usize, 20usize);
        let mut data = vec![220u8; w * w];
        for gy in 0..8 {
            for gx in 0..8 {
                let ring = gx == 0 || gy == 0 || gx == 7 || gy == 7;
                let white = if ring {
                    white_ring.contains(&(gx, gy))
                } else {
                    (gx + gy) % 2 == 0
                };
                for y in 0..cell {
                    for x in 0..cell {
                        let (px, py) = (origin + gx * cell + x, origin + gy * cell + y);
                        data[py * w + px] = if white { 220 } else { 20 };
                    }
                }
            }
        }
        data
    }

    fn ring_errors_of(data: &[u8]) -> Option<(u32, u32)> {
        let img = crate::image::ImageView::new(data, 120, 120, 120).unwrap();
        let arena = Bump::new();
        let roi = RoiCache::new(&img, &arena, 0, 0, 119, 119);
        let h = Homography::square_to_quad(&[
            [20.0, 20.0],
            [100.0, 20.0],
            [100.0, 100.0],
            [20.0, 100.0],
        ])
        .unwrap();
        let m = h.to_matrix3x3();
        rectified_ring_evidence(&img, &roi, &Homography::from_matrix3x3(&m), &AprilTag36h11)
    }

    #[test]
    fn border_ring_counts_bright_ring_cells() {
        assert_eq!(ring_errors_of(&ring_probe(&[])), Some((0, 28)));
        let corrupted = ring_probe(&[(3, 0), (7, 4), (2, 7), (0, 0)]);
        assert_eq!(ring_errors_of(&corrupted), Some((4, 28)));
    }

    #[test]
    fn border_ring_budget_is_a_floor_of_the_rate() {
        let data = ring_probe(&[(3, 0), (7, 4), (2, 7)]);
        let img = crate::image::ImageView::new(&data, 120, 120, 120).unwrap();
        let arena = Bump::new();
        let roi = RoiCache::new(&img, &arena, 0, 0, 119, 119);
        let h = Homography::square_to_quad(&[
            [20.0, 20.0],
            [100.0, 20.0],
            [100.0, 100.0],
            [20.0, 100.0],
        ])
        .unwrap();
        let m = h.to_matrix3x3();
        // 3 errors of 28: floor(0.1 · 28) = 2 rejects, floor(0.11 · 28) = 3 accepts.
        assert!(!border_ring_ok(&img, &roi, &m, &AprilTag36h11, 0.1));
        assert!(border_ring_ok(&img, &roi, &m, &AprilTag36h11, 0.11));
        assert!(border_ring_ok(&img, &roi, &m, &AprilTag36h11, 1.0));
    }

    #[test]
    fn ring_budget_survives_f32_rounding_and_skips_unevaluable() {
        // 0.35f32 · 20 = 6.99999988 in f64: the budget must still be 7 cells.
        assert!(ring_budget_ok(Some((7, 20)), 0.35));
        assert!(!ring_budget_ok(Some((8, 20)), 0.35));
        // Evidence that cannot be evaluated never rejects.
        assert!(ring_budget_ok(None, 0.0));
    }

    /// The distortion-aware decode path honours the ring budget too (zero distortion, so the
    /// probe renders exactly as in the pinhole test).
    #[cfg(feature = "non_rectified")]
    #[test]
    fn distorted_decode_applies_the_ring_budget() {
        let model = crate::camera::BrownConradyModel {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        };
        let intrinsics = crate::pose::CameraIntrinsics::new(100.0, 100.0, 60.0, 60.0);
        let ideal = [[20.0, 20.0], [100.0, 20.0], [100.0, 100.0], [20.0, 100.0]];
        let h = Homography::square_to_quad(&ideal).unwrap();
        for (white, expect) in [
            (&[][..], Some((0, 28))),
            (&[(3, 0), (7, 4), (2, 7)][..], Some((3, 28))),
        ] {
            let data = ring_probe(white);
            let img = crate::image::ImageView::new(&data, 120, 120, 120).unwrap();
            let got = ring_evidence(&AprilTag36h11, |pts, out| {
                sample_points_distorted(&img, &h, pts, &intrinsics, &model, out)
            });
            assert_eq!(got, expect);
        }
    }

    #[test]
    fn test_homography_dlt() {
        let src = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let dst = [[10.0, 10.0], [20.0, 11.0], [19.0, 21.0], [9.0, 20.0]];

        let h = Homography::from_pairs(&src, &dst).expect("DLT should succeed");
        for i in 0..4 {
            let p = h.project(src[i]);
            assert!((p[0] - dst[i][0]).abs() < 1e-6);
            assert!((p[1] - dst[i][1]).abs() < 1e-6);
        }
    }

    use crate::config::TagFamily;
    use crate::image::ImageView;
    use crate::quad::extract_quads_fast;
    use crate::segmentation::label_components_with_stats;
    use crate::test_utils::{TestImageParams, generate_test_image_with_params};
    use crate::threshold::ThresholdEngine;
    use bumpalo::Bump;

    /// Run full pipeline from image to decoded tags.
    fn run_full_pipeline(tag_size: usize, canvas_size: usize, tag_id: u16) -> Vec<(u32, u32)> {
        let params = TestImageParams {
            family: TagFamily::AprilTag36h11,
            id: tag_id,
            tag_size,
            canvas_size,
            ..Default::default()
        };

        let (data, _corners) = generate_test_image_with_params(&params);
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

        let arena = Bump::new();
        let engine = ThresholdEngine::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let mut binary = vec![0u8; canvas_size * canvas_size];
        engine.apply_threshold(&arena, &img, &stats, &mut binary);
        let label_result =
            label_components_with_stats(&arena, &binary, canvas_size, canvas_size, true);
        let detections = extract_quads_fast(&arena, &img, &label_result);

        let decoder = AprilTag36h11;
        let mut results = Vec::new();

        for quad in &detections {
            if let Some(bits) = sample_grid_generic(&img, &arena, quad, &decoder)
                && let Some((id, hamming, _rot)) = decoder.decode(bits)
            {
                results.push((id, hamming));
            }
        }

        results
    }

    /// Test E2E pipeline decodes correctly at varying sizes.
    #[test]
    fn test_e2e_decoding_at_varying_sizes() {
        let canvas_size = 640;
        let tag_sizes = [64, 100, 150, 200, 300];
        let test_id: u16 = 42;

        for tag_size in tag_sizes {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            let found = decoded.iter().any(|(id, _)| *id == u32::from(test_id));

            if tag_size >= 64 {
                assert!(found, "Tag size {tag_size}: ID {test_id} not found");
            }

            if found {
                let (_, hamming) = decoded
                    .iter()
                    .find(|(id, _)| *id == u32::from(test_id))
                    .unwrap();
                println!("Tag size {tag_size:>3}px: ID {test_id} with hamming {hamming}");
            }
        }
    }

    /// Test that multiple tag IDs decode correctly.
    #[test]
    fn test_e2e_multiple_ids() {
        let canvas_size = 400;
        let tag_size = 150;
        let test_ids: [u16; 5] = [0, 42, 100, 200, 500];

        for &test_id in &test_ids {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            let found = decoded.iter().any(|(id, _)| *id == u32::from(test_id));
            assert!(found, "ID {test_id} not decoded");

            let (_, hamming) = decoded
                .iter()
                .find(|(id, _)| *id == u32::from(test_id))
                .unwrap();
            assert_eq!(*hamming, 0, "ID {test_id} should have 0 hamming");
            println!("ID {test_id:>3}: Decoded with hamming {hamming}");
        }
    }

    /// Test decoding with edge ID values.
    #[test]
    fn test_e2e_edge_ids() {
        let canvas_size = 400;
        let tag_size = 150;
        let edge_ids: [u16; 2] = [0, 586];

        for &test_id in &edge_ids {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            let found = decoded.iter().any(|(id, _)| *id == u32::from(test_id));
            assert!(found, "Edge ID {test_id} not decoded");
            println!("Edge ID {test_id}: Decoded");
        }
    }

    /// Test ArUco 6x6_250 dictionary integration.
    #[test]
    fn test_aruco_6x6_250_roundtrip() {
        let decoder = ArUco6x6_250;
        let test_ids = [0, 42, 100, 249];

        for &id in &test_ids {
            let code = decoder.get_code(id).expect("code should exist");
            let (decoded_id, hamming, rot) = decoder.decode(code).expect("should decode");
            assert_eq!(decoded_id, u32::from(id));
            assert_eq!(hamming, 0);
            assert_eq!(rot, 0);
        }
    }

    /// Test ArUco MIP 36h12 dictionary integration (250 codes, rotation round-trip).
    #[test]
    fn test_aruco_mip_36h12_roundtrip() {
        let decoder = ArUcoMip36h12;
        assert_eq!(decoder.num_codes(), 250);
        for id in [0u16, 42, 100, 249] {
            let code = decoder.get_code(id).expect("code should exist");
            let (decoded_id, hamming, rot) = decoder.decode(code).expect("should decode");
            assert_eq!(decoded_id, u32::from(id));
            assert_eq!(hamming, 0);
            assert_eq!(rot, 0);
        }
    }

    #[test]
    fn test_project_stays_finite_at_near_zero_w() {
        // Last row all-zero → the perspective weight w = 0 for every point, so an
        // unguarded `res/w` would be ±inf/NaN. The sign-preserving epsilon clamp must
        // keep the projection finite (callers then reject it via reprojection residual).
        let h = SMatrix::<f64, 3, 3>::new(1.0, 0.0, 5.0, 0.0, 1.0, 7.0, 0.0, 0.0, 0.0);
        let hom = Homography { h };
        for p in [[0.0, 0.0], [3.0, -2.0], [1e6, 1e6]] {
            let out = hom.project(p);
            assert!(
                out[0].is_finite() && out[1].is_finite(),
                "project must stay finite at w≈0, got {out:?} for {p:?}"
            );
        }
    }
}
