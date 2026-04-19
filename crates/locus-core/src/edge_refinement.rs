//! Unified sub-pixel edge refinement using the Error Function (ERF) intensity model.
//!
//! This module centralizes the Gauss-Newton optimization for fitting a PSF-blurred
//! step function to image intensities along a line:
//!
//! $$I(x,y) = \frac{A+B}{2} + \frac{B-A}{2} \cdot \mathrm{erf}\!\left(\frac{d}{\sigma}\right)$$
//!
//! where $A,B$ are the dark/light intensities, $d$ is the signed perpendicular distance
//! to the edge line, and $\sigma$ is the Gaussian blur parameter.
//!
//! Both the quad extraction stage and the decoder refinement stage share this solver,
//! ensuring consistent Jacobian evaluation and benefiting from the same SIMD acceleration.

#![allow(unsafe_code, clippy::cast_sign_loss)]

use crate::image::ImageView;
use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use multiversion::multiversion;

/// Precomputed `1 / sqrt(pi)` — the Jacobian of `erf(x)` at `x` is
/// `(2 / sqrt(pi)) * exp(-x^2)`; factoring the constant saves a `sqrt` per GN iteration.
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// Configuration for sample collection along an edge.
#[derive(Clone, Debug)]
pub struct SampleConfig {
    /// Half-width of the search band perpendicular to the edge (in pixels).
    pub window: f64,
    /// Pixel stride for subsampling (1 = every pixel, 2 = skip alternate).
    pub stride: usize,
    /// Parametric range `(t_min, t_max)` along the edge segment for inclusion.
    /// `t=0` at `p1`, `t=1` at `p2`. Values outside `[0,1]` extend beyond endpoints.
    pub t_range: (f64, f64),
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            window: 2.5,
            stride: 1,
            t_range: (-0.1, 1.1),
        }
    }
}

impl SampleConfig {
    /// Sampling config for the quad-extraction path: widen the window under
    /// decimation, stride through very long edges, and clip `t_range` tightly
    /// enough that samples from the adjacent leg of an L-corner don't leak in.
    #[must_use]
    pub fn for_quad(edge_len: f64, decimation: usize) -> Self {
        let window = if decimation > 1 {
            decimation as f64 + 1.0
        } else {
            2.5
        };
        let stride = if edge_len > 100.0 { 2 } else { 1 };
        Self {
            window,
            stride,
            t_range: (-0.1, 1.1),
        }
    }

    /// Sampling config for the decoder ERF path (same as [`SampleConfig::default`]).
    #[must_use]
    pub fn for_decoder() -> Self {
        Self::default()
    }
}

/// Configuration for the Gauss-Newton refinement loop.
#[derive(Clone, Debug)]
pub struct RefineConfig {
    /// Gaussian blur sigma for the ERF model.
    pub sigma: f64,
    /// Maximum number of Gauss-Newton iterations.
    pub max_iterations: usize,
    /// If true, re-estimate A/B intensities each iteration (decoder style).
    /// If false, estimate A/B once before the GN loop (quad style).
    pub re_estimate_ab: bool,
    /// If true, pre-scan initial `d` by maximizing gradient alignment before GN.
    /// Decoder-style priors warrant this; quad-style priors already sit within the
    /// 2 px sanity gate and scanning risks pushing refined corners past the gate.
    pub scan_initial: bool,
    /// Convergence threshold: stop if |step| < this value.
    pub convergence_threshold: f64,
    /// Singularity threshold for J^T J.
    pub singular_threshold: f64,
    /// Maximum step size per iteration (clamped).
    pub step_clamp: f64,
    /// Minimum |B - A| contrast; break early if below this.
    pub min_contrast: f64,
}

impl RefineConfig {
    /// One-shot A/B estimation, no minimum contrast check, no initial `d` scan.
    /// Used by the quad extraction corner-refinement path.
    #[must_use]
    pub fn quad_style(sigma: f64) -> Self {
        Self {
            sigma,
            max_iterations: 15,
            re_estimate_ab: false,
            scan_initial: false,
            convergence_threshold: 1e-4,
            singular_threshold: 1e-10,
            step_clamp: 0.5,
            min_contrast: 0.0,
        }
    }

    /// Per-iteration A/B refinement, pre-refine scan, early exit on low contrast.
    /// Used by the decoder corner-refinement path.
    #[must_use]
    pub fn decoder_style(sigma: f64) -> Self {
        Self {
            sigma,
            max_iterations: 15,
            re_estimate_ab: true,
            scan_initial: true,
            convergence_threshold: 1e-4,
            singular_threshold: 1e-6,
            step_clamp: 0.5,
            min_contrast: 5.0,
        }
    }
}

/// Unified ERF-based edge fitter.
///
/// Encapsulates the line parameters `(nx, ny, d)` defining the implicit line
/// `nx * x + ny * y + d = 0`, along with edge geometry for sample collection.
///
/// # Normal Convention
///
/// The normal `(nx, ny)` points from the dark side toward the light side.
/// The convention is `nx = -dy/len, ny = dx/len` (left-hand normal of the
/// directed edge p1 → p2 for CW-wound quads).
pub struct ErfEdgeFitter<'a> {
    img: &'a ImageView<'a>,
    p1: [f64; 2],
    dx: f64,
    dy: f64,
    len: f64,
    nx: f64,
    ny: f64,
    d: f64,
}

impl<'a> ErfEdgeFitter<'a> {
    /// Create a new fitter for the edge from `p1` to `p2`.
    ///
    /// Returns `None` if the edge is shorter than 4 pixels.
    ///
    /// # Normal Convention
    ///
    /// `init_from_midpoint = true`: computes d from the edge midpoint (quad-style).
    /// `init_from_midpoint = false`: computes d from p1 (decoder-style).
    #[must_use]
    pub fn new(
        img: &'a ImageView<'a>,
        p1: [f64; 2],
        p2: [f64; 2],
        init_from_midpoint: bool,
    ) -> Option<Self> {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 4.0 {
            return None;
        }

        // Consistent normal convention: left-hand normal of directed edge p1→p2.
        let nx = -dy / len;
        let ny = dx / len;

        let d = if init_from_midpoint {
            let mid_x = (p1[0] + p2[0]) * 0.5;
            let mid_y = (p1[1] + p2[1]) * 0.5;
            -(nx * mid_x + ny * mid_y)
        } else {
            -(nx * p1[0] + ny * p1[1])
        };

        Some(Self {
            img,
            p1,
            dx,
            dy,
            len,
            nx,
            ny,
            d,
        })
    }

    /// Scan for the optimal initial `d` by maximizing gradient alignment.
    ///
    /// Tests 13 offsets in `[-2.4, +2.4]` pixels and picks the one where the
    /// projected gradient magnitude along the normal is highest.
    pub fn scan_initial_d(&mut self) {
        let window = 2.5;
        let (x0, x1, y0, y1) = self.get_scan_bounds(window);

        let mut best_offset = 0.0;
        let mut best_grad = 0.0;

        for k in -6..=6 {
            let offset = f64::from(k) * 0.4;
            let scan_d = self.d + offset;

            let (sum_g, count) =
                project_gradients_optimized(self.img, self.nx, self.ny, x0, x1, y0, y1, scan_d);

            if count > 0 && sum_g > best_grad {
                best_grad = sum_g;
                best_offset = offset;
            }
        }
        self.d += best_offset;
    }

    /// Collect pixel samples within a window of the edge.
    ///
    /// Returns arena-allocated tuples `(x, y, intensity)` for pixels that fall
    /// within the perpendicular band and parametric range of the edge segment.
    #[must_use]
    pub fn collect_samples(
        &self,
        arena: &'a Bump,
        config: &SampleConfig,
    ) -> BumpVec<'a, (f64, f64, f64)> {
        let (x0, x1, y0, y1) = self.get_scan_bounds(config.window);

        if config.stride == 1 {
            // Fast SIMD path (no subsampling)
            collect_samples_optimized(
                self.img,
                self.nx,
                self.ny,
                self.d,
                self.p1,
                self.dx,
                self.dy,
                self.len,
                x0,
                x1,
                y0,
                y1,
                config.window,
                config.t_range,
                arena,
            )
        } else {
            // Subsampled path (for large edges in decimated images)
            collect_samples_strided(
                self.img,
                self.nx,
                self.ny,
                self.d,
                self.p1,
                self.dx,
                self.dy,
                self.len,
                x0,
                x1,
                y0,
                y1,
                config.window,
                config.t_range,
                config.stride,
                arena,
            )
        }
    }

    /// Run the Gauss-Newton refinement on the collected samples.
    ///
    /// Updates `self.d` in place to minimize the residual between observed
    /// intensities and the ERF model.
    pub fn refine(&mut self, samples: &[(f64, f64, f64)], config: &RefineConfig) {
        if samples.len() < 10 {
            return;
        }

        let inv_sigma = 1.0 / config.sigma;

        // Initial A/B estimation
        let (mut a, mut b) = if config.re_estimate_ab {
            (128.0, 128.0)
        } else {
            estimate_ab_oneshot(samples, self.nx, self.ny, self.d)
        };

        // Early exit if one-shot A/B estimation fails
        if !config.re_estimate_ab && (b - a).abs() < 1.0 {
            return;
        }

        for _ in 0..config.max_iterations {
            // Per-iteration A/B refinement (decoder style).
            // Preserve previous (a, b) on zero-weight iterations, matching legacy.
            if config.re_estimate_ab
                && let Some((new_a, new_b)) =
                    estimate_ab_per_iter(self.img, samples, self.nx, self.ny, self.d)
            {
                a = new_a;
                b = new_b;
            }

            if config.min_contrast > 0.0 && (b - a).abs() < config.min_contrast {
                break;
            }

            let (sum_jtj, sum_jt_res) =
                refine_accumulate_optimized(samples, self.nx, self.ny, self.d, a, b, inv_sigma);

            if sum_jtj < config.singular_threshold {
                break;
            }

            let step = sum_jt_res / sum_jtj;
            self.d += step.clamp(-config.step_clamp, config.step_clamp);

            if step.abs() < config.convergence_threshold {
                break;
            }
        }
    }

    /// End-to-end fit: optional initial-`d` scan, sample collection, and GN refinement.
    ///
    /// Returns `true` if samples were sufficient to run the GN loop. When `false`,
    /// `(nx, ny, d)` retain their values from `new(...)` (and, if applicable, the scan).
    pub fn fit(
        &mut self,
        arena: &'a Bump,
        sample_cfg: &SampleConfig,
        refine_cfg: &RefineConfig,
    ) -> bool {
        if refine_cfg.scan_initial {
            self.scan_initial_d();
        }
        let samples = self.collect_samples(arena, sample_cfg);
        if samples.len() < 10 {
            return false;
        }
        self.refine(&samples, refine_cfg);
        true
    }

    /// Get the result as `(nx, ny, d)`.
    #[inline]
    #[must_use]
    pub fn line_params(&self) -> (f64, f64, f64) {
        (self.nx, self.ny, self.d)
    }

    /// Get the edge length in pixels.
    #[inline]
    #[must_use]
    pub fn edge_len(&self) -> f64 {
        self.len
    }

    #[inline]
    fn get_scan_bounds(&self, window: f64) -> (usize, usize, usize, usize) {
        let p2_0 = self.p1[0] + self.dx;
        let p2_1 = self.p1[1] + self.dy;

        let w_limit = (self.img.width - 2) as f64;
        let h_limit = (self.img.height - 2) as f64;

        let x0 = (self.p1[0].min(p2_0) - window - 0.5).clamp(1.0, w_limit) as usize;
        let x1 = (self.p1[0].max(p2_0) + window + 0.5).clamp(1.0, w_limit) as usize;
        let y0 = (self.p1[1].min(p2_1) - window - 0.5).clamp(1.0, h_limit) as usize;
        let y1 = (self.p1[1].max(p2_1) + window + 0.5).clamp(1.0, h_limit) as usize;
        (x0, x1, y0, y1)
    }
}

// ── A/B Intensity Estimation ─────────────────────────────────────────────────

/// One-shot A/B estimation from sample distances (quad-style).
///
/// Uses distance-weighted averaging: pixels far from the edge contribute more,
/// reducing contamination from the transition zone.
fn estimate_ab_oneshot(samples: &[(f64, f64, f64)], nx: f64, ny: f64, d: f64) -> (f64, f64) {
    let mut dark_sum = 0.0;
    let mut dark_weight = 0.0;
    let mut light_sum = 0.0;
    let mut light_weight = 0.0;

    for &(x, y, intensity) in samples {
        let signed_dist = nx * x + ny * y + d;
        if signed_dist < -1.0 {
            let w = (-signed_dist - 1.0).min(2.0);
            dark_sum += intensity * w;
            dark_weight += w;
        } else if signed_dist > 1.0 {
            let w = (signed_dist - 1.0).min(2.0);
            light_sum += intensity * w;
            light_weight += w;
        }
    }

    if dark_weight < 1.0 || light_weight < 1.0 {
        return (128.0, 128.0); // Insufficient samples
    }

    (dark_sum / dark_weight, light_sum / light_weight)
}

/// Per-iteration A/B estimation using bilinear sampling (decoder-style).
///
/// Re-reads intensities via bilinear interpolation each iteration to track
/// the line as `d` evolves. Returns `None` when either side lacks weight —
/// callers are expected to keep the previous iteration's estimate in that
/// case, matching the legacy decoder behavior.
fn estimate_ab_per_iter(
    img: &ImageView,
    samples: &[(f64, f64, f64)],
    nx: f64,
    ny: f64,
    d: f64,
) -> Option<(f64, f64)> {
    let mut dark_sum = 0.0;
    let mut dark_weight = 0.0;
    let mut light_sum = 0.0;
    let mut light_weight = 0.0;

    for &(x, y, _) in samples {
        let dist = nx * x + ny * y + d;
        let val = img.sample_bilinear(x, y);
        if dist < -1.0 {
            let w = (-dist - 0.5).clamp(0.1, 2.0);
            dark_sum += val * w;
            dark_weight += w;
        } else if dist > 1.0 {
            let w = (dist - 0.5).clamp(0.1, 2.0);
            light_sum += val * w;
            light_weight += w;
        }
    }

    if dark_weight <= 0.0 || light_weight <= 0.0 {
        return None;
    }

    Some((dark_sum / dark_weight, light_sum / light_weight))
}

// ── SIMD-Accelerated Gauss-Newton Accumulation ───────────────────────────────

/// Accumulate J^T J and J^T r for the Gauss-Newton step.
///
/// The model is `I(d) = (A+B)/2 + (B-A)/2 * erf(d/sigma)` with Jacobian
/// `dI/dd = (B-A) / (sqrt(pi) * sigma) * exp(-s^2)` where `s = d/sigma`.
///
/// On AVX2+FMA targets, uses vectorized erf_approx_v4 and FMA accumulation.
#[multiversion(targets(
    "x86_64+avx2+fma+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn refine_accumulate_optimized(
    samples: &[(f64, f64, f64)],
    nx: f64,
    ny: f64,
    d: f64,
    a: f64,
    b: f64,
    inv_sigma: f64,
) -> (f64, f64) {
    let mut sum_jtj = 0.0;
    let mut sum_jt_res = 0.0;
    let k = (b - a) * inv_sigma * INV_SQRT_PI;

    let mut i = 0;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
        // SAFETY: All AVX2/FMA intrinsics are guarded by cfg and runtime dispatch.
        unsafe {
            use std::arch::x86_64::*;
            let v_nx = _mm256_set1_pd(nx);
            let v_ny = _mm256_set1_pd(ny);
            let v_d = _mm256_set1_pd(d);
            let v_a = _mm256_set1_pd(a);
            let v_b = _mm256_set1_pd(b);
            let v_inv_sigma = _mm256_set1_pd(inv_sigma);
            let v_k = _mm256_set1_pd(k);
            let v_half = _mm256_set1_pd(0.5);
            let v_abs_mask = _mm256_set1_pd(-0.0);

            let mut v_sum_jtj = _mm256_setzero_pd();
            let mut v_sum_jt_res = _mm256_setzero_pd();

            while i + 4 <= samples.len() {
                let s0 = samples[i];
                let s1 = samples[i + 1];
                let s2 = samples[i + 2];
                let s3 = samples[i + 3];

                let v_x = _mm256_set_pd(s3.0, s2.0, s1.0, s0.0);
                let v_y = _mm256_set_pd(s3.1, s2.1, s1.1, s0.1);
                let v_img_val = _mm256_set_pd(s3.2, s2.2, s1.2, s0.2);

                let v_dist = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                    v_d,
                );
                let v_s = _mm256_mul_pd(v_dist, v_inv_sigma);

                let v_abs_s = _mm256_andnot_pd(v_abs_mask, v_s);
                let v_range_mask = _mm256_cmp_pd(v_abs_s, _mm256_set1_pd(3.0), _CMP_LE_OQ);

                if _mm256_movemask_pd(v_range_mask) != 0 {
                    // Vectorized ERF + Gauss-Newton accumulation.
                    // SAFETY: erf_approx_v4 requires AVX2+FMA, guaranteed by multiversion target.
                    let v_erf = crate::simd::math::erf_approx_v4(v_s);

                    // model = (a+b)*0.5 + (b-a)*0.5 * erf(s)
                    let v_ab_half = _mm256_mul_pd(_mm256_add_pd(v_a, v_b), v_half);
                    let v_ba_half = _mm256_mul_pd(_mm256_sub_pd(v_b, v_a), v_half);
                    let v_model = _mm256_fmadd_pd(v_ba_half, v_erf, v_ab_half);

                    // residual = img_val - model
                    let v_residual = _mm256_sub_pd(v_img_val, v_model);

                    // jac = k * exp(-s^2)
                    // SAFETY: transmute is safe for same-size SIMD ↔ array conversions.
                    let v_neg_s2 = _mm256_mul_pd(v_s, v_s);
                    let v_neg_s2 = _mm256_xor_pd(v_neg_s2, v_abs_mask);
                    let ns2: [f64; 4] = std::mem::transmute(v_neg_s2);
                    let v_exp =
                        _mm256_set_pd(ns2[3].exp(), ns2[2].exp(), ns2[1].exp(), ns2[0].exp());
                    let v_jac = _mm256_mul_pd(v_k, v_exp);

                    // Masked accumulation
                    let v_jac_masked = _mm256_and_pd(v_jac, v_range_mask);
                    let v_res_masked = _mm256_and_pd(v_residual, v_range_mask);

                    v_sum_jtj = _mm256_fmadd_pd(v_jac_masked, v_jac_masked, v_sum_jtj);
                    v_sum_jt_res = _mm256_fmadd_pd(v_jac_masked, v_res_masked, v_sum_jt_res);
                }
                i += 4;
            }

            // Horizontal reduction
            // SAFETY: transmute is safe for same-size SIMD ↔ array conversions.
            let jtj_lanes: [f64; 4] = std::mem::transmute(v_sum_jtj);
            let jtr_lanes: [f64; 4] = std::mem::transmute(v_sum_jt_res);
            sum_jtj += jtj_lanes[0] + jtj_lanes[1] + jtj_lanes[2] + jtj_lanes[3];
            sum_jt_res += jtr_lanes[0] + jtr_lanes[1] + jtr_lanes[2] + jtr_lanes[3];
        }
    }

    // Scalar tail
    while i < samples.len() {
        let (x, y, img_val) = samples[i];
        let dist = nx * x + ny * y + d;
        let s = dist * inv_sigma;
        if s.abs() <= 3.0 {
            let model = (a + b) * 0.5 + (b - a) * 0.5 * crate::simd::math::erf_approx(s);
            let residual = img_val - model;
            let jac = k * (-s * s).exp();
            sum_jtj += jac * jac;
            sum_jt_res += jac * residual;
        }
        i += 1;
    }

    (sum_jtj, sum_jt_res)
}

// ── SIMD-Accelerated Sample Collection ───────────────────────────────────────

/// Gradient projection for d-scan initialization.
///
/// Computes the sum of `|gradient . normal|` for pixels near the line,
/// used to find the offset that maximizes edge evidence.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn project_gradients_optimized(
    img: &ImageView,
    nx: f64,
    ny: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    scan_d: f64,
) -> (f64, usize) {
    let mut sum_g = 0.0;
    let mut count = 0;

    for py in y0..=y1 {
        let mut px = x0;
        let y = py as f64 + 0.5;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
            // SAFETY: AVX2 intrinsics guarded by cfg and runtime dispatch.
            unsafe {
                use std::arch::x86_64::*;
                let v_nx = _mm256_set1_pd(nx);
                let v_ny = _mm256_set1_pd(ny);
                let v_scan_d = _mm256_set1_pd(scan_d);
                let v_y = _mm256_set1_pd(y);
                let v_abs_mask = _mm256_set1_pd(-0.0);

                while px + 4 <= x1 {
                    let v_x = _mm256_set_pd(
                        (px + 3) as f64 + 0.5,
                        (px + 2) as f64 + 0.5,
                        (px + 1) as f64 + 0.5,
                        px as f64 + 0.5,
                    );

                    let v_dist = _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                        v_scan_d,
                    );

                    let v_abs_dist = _mm256_andnot_pd(v_abs_mask, v_dist);
                    let v_cmp = _mm256_cmp_pd(v_abs_dist, _mm256_set1_pd(1.0), _CMP_LT_OQ);
                    let mask = _mm256_movemask_pd(v_cmp);

                    if mask != 0 {
                        for j in 0..4 {
                            if (mask >> j) & 1 != 0 {
                                let g = img.sample_gradient_bilinear((px + j) as f64, y);
                                sum_g += (g[0] * nx + g[1] * ny).abs();
                                count += 1;
                            }
                        }
                    }
                    px += 4;
                }
            }
        }

        while px <= x1 {
            let x = px as f64 + 0.5;
            let dist = nx * x + ny * y + scan_d;
            if dist.abs() < 1.0 {
                let g = img.sample_gradient_bilinear(x, y);
                sum_g += (g[0] * nx + g[1] * ny).abs();
                count += 1;
            }
            px += 1;
        }
    }
    (sum_g, count)
}

/// SIMD-accelerated sample collection (stride = 1).
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn collect_samples_optimized<'a>(
    img: &ImageView,
    nx: f64,
    ny: f64,
    d: f64,
    p1: [f64; 2],
    dx: f64,
    dy: f64,
    len: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    window: f64,
    t_range: (f64, f64),
    arena: &'a Bump,
) -> BumpVec<'a, (f64, f64, f64)> {
    let mut samples = BumpVec::with_capacity_in(128, arena);
    let inv_len_sq = 1.0 / (len * len);

    for py in y0..=y1 {
        let mut px = x0;
        let y = py as f64 + 0.5;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
            // SAFETY: AVX2 intrinsics guarded by cfg and runtime dispatch.
            unsafe {
                use std::arch::x86_64::*;
                let v_nx = _mm256_set1_pd(nx);
                let v_ny = _mm256_set1_pd(ny);
                let v_d = _mm256_set1_pd(d);
                let v_y = _mm256_set1_pd(y);
                let v_p1x = _mm256_set1_pd(p1[0]);
                let v_p1y = _mm256_set1_pd(p1[1]);
                let v_dx = _mm256_set1_pd(dx);
                let v_dy = _mm256_set1_pd(dy);
                let v_inv_len_sq = _mm256_set1_pd(inv_len_sq);
                let v_abs_mask = _mm256_set1_pd(-0.0);

                while px + 4 <= x1 {
                    let v_x = _mm256_set_pd(
                        (px + 3) as f64 + 0.5,
                        (px + 2) as f64 + 0.5,
                        (px + 1) as f64 + 0.5,
                        px as f64 + 0.5,
                    );

                    let v_dist = _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                        v_d,
                    );
                    let v_abs_dist = _mm256_andnot_pd(v_abs_mask, v_dist);
                    let v_dist_mask = _mm256_cmp_pd(v_abs_dist, _mm256_set1_pd(window), _CMP_LE_OQ);

                    let v_t = _mm256_mul_pd(
                        _mm256_add_pd(
                            _mm256_mul_pd(_mm256_sub_pd(v_x, v_p1x), v_dx),
                            _mm256_mul_pd(_mm256_sub_pd(v_y, v_p1y), v_dy),
                        ),
                        v_inv_len_sq,
                    );

                    let v_t_mask_low = _mm256_cmp_pd(v_t, _mm256_set1_pd(t_range.0), _CMP_GE_OQ);
                    let v_t_mask_high = _mm256_cmp_pd(v_t, _mm256_set1_pd(t_range.1), _CMP_LE_OQ);

                    let final_mask = _mm256_movemask_pd(_mm256_and_pd(
                        v_dist_mask,
                        _mm256_and_pd(v_t_mask_low, v_t_mask_high),
                    ));

                    if final_mask != 0 {
                        for j in 0..4 {
                            if (final_mask >> j) & 1 != 0 {
                                let val = f64::from(img.get_pixel(px + j, py));
                                samples.push(((px + j) as f64 + 0.5, y, val));
                            }
                        }
                    }
                    px += 4;
                }
            }
        }

        while px <= x1 {
            let x = px as f64 + 0.5;
            let dist = nx * x + ny * y + d;
            if dist.abs() <= window {
                let t = ((x - p1[0]) * dx + (y - p1[1]) * dy) * inv_len_sq;
                if t >= t_range.0 && t <= t_range.1 {
                    let val = f64::from(img.get_pixel(px, py));
                    samples.push((x, y, val));
                }
            }
            px += 1;
        }
    }
    samples
}

/// Strided sample collection for decimated images.
///
/// Identical logic to `collect_samples_optimized` but with pixel stride > 1.
/// Used by the quad extraction stage for large edges in upscaled images.
#[allow(clippy::too_many_arguments)]
fn collect_samples_strided<'a>(
    img: &ImageView,
    nx: f64,
    ny: f64,
    d: f64,
    p1: [f64; 2],
    dx: f64,
    dy: f64,
    len: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    window: f64,
    t_range: (f64, f64),
    stride: usize,
    arena: &'a Bump,
) -> BumpVec<'a, (f64, f64, f64)> {
    let mut samples = BumpVec::with_capacity_in(128, arena);
    let inv_len_sq = 1.0 / (len * len);

    let mut py = y0;
    while py <= y1 {
        let mut px = x0;
        let y = py as f64 + 0.5;
        while px <= x1 {
            let x = px as f64 + 0.5;
            let dist = nx * x + ny * y + d;
            if dist.abs() <= window {
                let t = ((x - p1[0]) * dx + (y - p1[1]) * dy) * inv_len_sq;
                if t >= t_range.0 && t <= t_range.1 {
                    let val = f64::from(img.get_pixel(px, py));
                    samples.push((x, y, val));
                }
            }
            px += stride;
        }
        py += stride;
    }
    samples
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_utils::subpixel::{Line, SubpixelEdgeRenderer};

    #[test]
    fn quad_style_recovers_axis_aligned_subpixel_edge() {
        let width = 100;
        let height = 100;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(0.0, 255.0)
            .with_sigma(sigma);

        for x_gt in [50.0, 50.25, 50.5, 50.75] {
            let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 90.0]);
            let data = renderer.render_edge_u8(&line_gt);
            let img = ImageView::new(&data, width, height, width).expect("invalid image view");
            let arena = Bump::new();

            let mut fitter = ErfEdgeFitter::new(&img, [50.0, 10.0], [50.0, 90.0], true)
                .expect("edge length too short");
            let sample_cfg = SampleConfig::for_quad(fitter.edge_len(), 1);
            let refine_cfg = RefineConfig::quad_style(sigma);
            assert!(fitter.fit(&arena, &sample_cfg, &refine_cfg));

            // LHN: nx = -dy/len = 0, ny = dx/len = 0 for vertical edge p1→p2.
            // With p1=(50,10)→p2=(50,90): dy=80, dx=0 ⇒ nx=-1, ny=0.
            let (nx, _ny, d) = fitter.line_params();
            assert!((nx + 1.0).abs() < 1e-7, "nx = {nx}");
            let x_recovered = d; // -x + d = 0
            let error = (x_recovered - x_gt).abs();
            assert!(
                error < 0.02,
                "x_gt={x_gt} recovered={x_recovered} error={error}"
            );
        }
    }

    #[test]
    fn quad_style_recovers_angled_edge() {
        let width = 120;
        let height = 120;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(20.0, 230.0)
            .with_sigma(sigma);

        // 30-degree edge from (40, 20) to (80, 20 + 40*tan30)
        let angle = 30f64.to_radians();
        let p1 = [40.0, 20.0];
        let p2 = [80.0, 20.0 + 40.0 * angle.tan()];
        let line_gt = Line::from_points_cw(p1, p2);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");
        let arena = Bump::new();

        let mut fitter = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");
        let sample_cfg = SampleConfig::for_quad(fitter.edge_len(), 1);
        let refine_cfg = RefineConfig::quad_style(sigma);
        assert!(fitter.fit(&arena, &sample_cfg, &refine_cfg));

        // The ground-truth line is `line_gt.a*x + line_gt.b*y + line_gt.c = 0` (RHN),
        // while the fitter uses LHN — the triple is negated.
        let (nx, ny, d) = fitter.line_params();
        let err_a = (nx + line_gt.a).abs().min((nx - line_gt.a).abs());
        let err_b = (ny + line_gt.b).abs().min((ny - line_gt.b).abs());
        let err_c = (d + line_gt.c).abs().min((d - line_gt.c).abs());
        assert!(
            err_a < 1e-2 && err_b < 1e-2 && err_c < 5e-2,
            "angled-edge line-param error: a={err_a} b={err_b} c={err_c}"
        );
    }

    #[test]
    fn decoder_style_scan_initial_d_recovers_from_perturbed_seed() {
        let width = 120;
        let height = 120;
        let sigma = 0.6;
        let x_gt = 60.25;

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(30.0, 220.0)
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 110.0]);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");

        // Seed the fitter 1.5 px off the true edge — outside the GN capture radius
        // but within scan_initial_d's ±2.4 px search window.
        let p1 = [61.75, 10.0];
        let p2 = [61.75, 110.0];
        let arena = Bump::new();
        let mut fitter = ErfEdgeFitter::new(&img, p1, p2, false).expect("edge length too short");
        let sample_cfg = SampleConfig::for_decoder();
        let refine_cfg = RefineConfig::decoder_style(sigma);
        assert!(fitter.fit(&arena, &sample_cfg, &refine_cfg));

        let (nx, _ny, d) = fitter.line_params();
        assert!((nx + 1.0).abs() < 1e-7);
        let x_recovered = d;
        let error = (x_recovered - x_gt).abs();
        assert!(
            error < 0.05,
            "perturbed seed recovered={x_recovered} error={error}"
        );
    }

    #[test]
    fn decoder_style_early_exits_on_low_contrast() {
        let width = 80;
        let height = 80;
        let sigma = 0.6;
        let x_gt = 40.25;

        // |B - A| = 3 < min_contrast (5.0) — refinement must break out early and
        // leave the line params essentially at their seed values.
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(127.0, 130.0)
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 5.0], [x_gt, 75.0]);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");

        let p1 = [40.0, 5.0];
        let p2 = [40.0, 75.0];
        let arena = Bump::new();
        let mut fitter = ErfEdgeFitter::new(&img, p1, p2, false).expect("edge length too short");
        let (_, _, d_before_scan) = fitter.line_params();
        let sample_cfg = SampleConfig::for_decoder();
        let refine_cfg = RefineConfig::decoder_style(sigma);
        // `fit` may return true (samples collected) but the GN loop must bail on
        // min_contrast — so d stays within scan_initial_d's ±2.4 px adjustment.
        fitter.fit(&arena, &sample_cfg, &refine_cfg);
        let (_nx, _ny, d) = fitter.line_params();
        assert!(
            (d - d_before_scan).abs() <= 2.5,
            "low-contrast: d drifted beyond scan window (d_before={d_before_scan}, d_after={d})"
        );
    }
}
