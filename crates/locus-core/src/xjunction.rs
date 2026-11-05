//! Model fit of an X-junction: the corner where a marker's black border touches a black
//! connector square diagonally (AprilGrid boards).
//!
//! Near the junction the pattern is two straight lines crossing at `c`, dark where the signed
//! distances to both lines have the same sign. The image is modelled as
//!
//! ```text
//! μ(p) = A + B · erf(n₁·(p − c) / s) · erf(n₂·(p − c) / s)
//! ```
//!
//! with line normals `nᵢ = (−sin tᵢ, cos tᵢ)`, a blur scale `s` (`√2·σ` of a Gaussian edge) and
//! the two levels `A ± B`. The seven parameters are fitted to the pixels of a window around the
//! junction by weighted Levenberg–Marquardt, each pixel compared at its centre. The pixel values
//! are used as they are, so no interpolation phase enters the estimate.
//!
//! **Why it is unbiased without the true blur.** The product of two odd functions is point
//! symmetric about `c`, and so is the image of any X-junction under any point-symmetric PSF and
//! any per-pixel tone curve. The least-squares cost is then even in the corner offset, so its
//! minimum stays at the junction however wrong the edge profile is. The product form ignores the
//! correlation the blur introduces between the two lines when they are not orthogonal; it is
//! point symmetric too, and measured equal to the exact bivariate-normal model on rendered
//! AprilGrids at up to 70° incidence. A window the frame clips is not point symmetric, so there
//! a wrong edge profile can bias the fit; it is still consistent, and beats the seed.

use crate::image::ImageView;
use crate::simd::math::{F64x4, erf_and_gauss_x4, wrap_angle};
use nalgebra::{SMatrix, SVector};

/// Number of fitted parameters: `c` (2), `t₁`, `t₂`, `s`, `A`, `B`.
const PARAMS: usize = 7;
/// Largest half-extent (px) of the window's bounding box: bounds the stack buffers.
const MAX_EXTENT_PX: f64 = 10.0;
/// Side of the largest pixel window: the bounding box plus a pixel.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const MAX_SIDE: usize = 2 * (MAX_EXTENT_PX as usize + 1) + 1;
/// Smallest sine of the angle between the two edges: nearly parallel edges have no junction
/// to locate.
const MIN_SKEW_SINE: f64 = 0.2;
/// Window capacity, a lane multiple: the loop reads whole lanes and the zeroed tail has weight 0.
const MAX_SAMPLES: usize = (MAX_SIDE * MAX_SIDE).next_multiple_of(LANES);
/// Distance (px) from both lines past which a pixel is left out of the window: there the
/// pattern is flat and informs only the two levels. Measured on rendered AprilGrids: 2 px is
/// as accurate as the whole window (median error 0.0226 px either way) at 15-30 % less fit
/// time; 1.5 px is no better and 1.0 px is worse (0.040), since the edge profiles reach that
/// far.
const FLAT_PX: f64 = 2.0;
/// Fewest window pixels per fitted parameter. A 12 px Kalibr tag's window holds 21-30 pixels;
/// at 4 per parameter its corners kept their seeds (rendered 12 px tags: 6.8x the bound, 1.5 %
/// of tags 0.1 px inward); at 3 they are fitted (4.6x, 0.8 %), and 2 adds nothing.
const MIN_SAMPLES_PER_PARAM: usize = 3;
/// Width (px) of the window's soft rim: weights fall from 1 to 0 over it, so a pixel entering or
/// leaving the window does not make the cost jump.
const RIM_PX: f64 = 1.0;
/// Initial blur scale: `√2 · 0.6 px`, a sharp camera.
const INITIAL_SCALE: f64 = 0.85;
/// Iteration cap, and the relative cost decrease under which an accepted step ends the fit.
const MAX_ITER: usize = 30;
const COST_TOLERANCE: f64 = 1e-4;
/// Initial Levenberg–Marquardt damping, relative to the normal matrix's diagonal.
const INITIAL_DAMPING: f64 = 1e-3;
/// Rejections of a fit: largest move from the start (px; or half the window's shorter reach,
/// if larger: a seed under the cell floor sits up to ~1.1 px off, and the window still holds
/// the junction), largest turn of either line (rad), the blur scale's range (px), and the
/// largest weighted residual RMS over the half contrast `|B|` (a model that does not describe
/// the window, such as an L-corner).
const MAX_MOVE_PX: f64 = 1.0;
const MAX_TURN_RAD: f64 = 0.35;
const MIN_SCALE_PX: f64 = 0.2;
const MAX_RELATIVE_RMS: f64 = 0.35;

/// A refused fit that landed farther than this (px) from its window's centre gets one refit
/// over a window centred where it landed (`fit_x_junction`).
const RECENTRE_PX: f64 = 0.5;

/// The window: each pixel's centre relative to the window centre, its value and its weight.
struct Window {
    dx: [f64; MAX_SAMPLES],
    dy: [f64; MAX_SAMPLES],
    value: [f64; MAX_SAMPLES],
    weight: [f64; MAX_SAMPLES],
    len: usize,
    /// Shorter reach (px) after any shrink to the buffers: what the window actually holds.
    min_reach: f64,
}

impl Window {
    /// Pixels whose centres lie in the ellipse `{c + α·u + β·v : (α/a)² + (β/b)² ≤ 1}` for unit
    /// edge directions `axes = [u, v]` and `reach = [a, b]` px: the image of a disk on the
    /// marker's plane, so a foreshortened border gets a window as narrow as the border. Pixels
    /// farther than `FLAT_PX` from both lines are left out. `None` when the edges are nearly
    /// parallel or the window lies outside the image.
    fn gather(
        img: &ImageView,
        centre: [f64; 2],
        axes: [[f64; 2]; 2],
        reach: [f64; 2],
    ) -> Option<Self> {
        let [along_u, along_v] = axes;
        let skew = along_u[0] * along_v[1] - along_u[1] * along_v[0];
        if skew.abs() < MIN_SKEW_SINE {
            return None;
        }
        // Bounding-box half-extents; a window wider than the buffers shrinks, keeping its shape.
        let half_x = (reach[0] * along_u[0]).hypot(reach[1] * along_v[0]);
        let half_y = (reach[0] * along_u[1]).hypot(reach[1] * along_v[1]);
        let shrink = (MAX_EXTENT_PX / half_x.max(half_y)).min(1.0);
        // Distances to the lines over these give the normalised elliptic radius (α/a, β/b).
        let inv_reach = reach.map(|r| 1.0 / (r * shrink * skew));
        let (x0, x1) = (
            (centre[0] - half_x * shrink).floor(),
            (centre[0] + half_x * shrink).ceil(),
        );
        let (y0, y1) = (
            (centre[1] - half_y * shrink).floor(),
            (centre[1] + half_y * shrink).ceil(),
        );
        // A window the image clips keeps the pixels it has. The fit needs no symmetric window
        // to be consistent, only for its robustness to a wrong edge profile, and the corners
        // it serves (within a few pixels of the frame, at the fisheye rim) otherwise keep
        // seeds 0.4-2.8 px off.
        // `clamp`, unlike `max`/`min`, keeps a NaN bound (a degenerate seed or reach), which the
        // ordering test then refuses instead of scanning the whole image.
        let (x0, y0) = (x0.clamp(0.0, f64::INFINITY), y0.clamp(0.0, f64::INFINITY));
        let (x1, y1) = (
            x1.clamp(f64::NEG_INFINITY, img.width as f64 - 1.0),
            y1.clamp(f64::NEG_INFINITY, img.height as f64 - 1.0),
        );
        if !(x0 <= x1 && y0 <= y1) {
            return None;
        }
        // Non-negative, inside the image and ordered: clamped and checked just above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);
        let min_reach = reach[0].min(reach[1]) * shrink;
        // Rim width in units of the normalised radius, along the shorter axis.
        let rim = min_reach / RIM_PX;
        let mut win = Self {
            dx: [0.0; MAX_SAMPLES],
            dy: [0.0; MAX_SAMPLES],
            value: [0.0; MAX_SAMPLES],
            weight: [0.0; MAX_SAMPLES],
            len: 0,
            min_reach,
        };
        for y in y0..=y1 {
            let row = &img.data[y * img.stride + x0..=y * img.stride + x1];
            let dy = y as f64 + 0.5 - centre[1];
            for (x, &value) in (x0..).zip(row) {
                let dx = x as f64 + 0.5 - centre[0];
                // Signed distances to the lines along v and along u.
                let from_v = dx * along_v[1] - dy * along_v[0];
                let from_u = along_u[0] * dy - along_u[1] * dx;
                // Flat there: such a pixel informs only the levels, which the rest already fix.
                if from_u.abs().min(from_v.abs()) > FLAT_PX || win.len == MAX_SAMPLES {
                    continue;
                }
                let (along_a, along_b) = (from_v * inv_reach[0], from_u * inv_reach[1]);
                let radius2 = along_a * along_a + along_b * along_b;
                if radius2 >= 1.0 {
                    continue;
                }
                // sin² taper over the rim: the cost stays smooth as pixels cross it.
                let taper = (1.0 - radius2.sqrt()) * rim;
                let weight = if taper < 1.0 {
                    let rise = (0.5 * core::f64::consts::PI * taper).sin();
                    rise * rise
                } else {
                    1.0
                };
                win.dx[win.len] = dx;
                win.dy[win.len] = dy;
                win.value[win.len] = f64::from(value);
                win.weight[win.len] = weight;
                win.len += 1;
            }
        }
        Some(win)
    }
}

/// Lanes of the window loop.
const LANES: usize = F64x4::LANES;

/// Parameter vector `[cx, cy, t₁, t₂, s, A, B]`; `c` is relative to the window centre.
type Params = SVector<f64, PARAMS>;
const CX: usize = 0;
const CY: usize = 1;
const T1: usize = 2;
const T2: usize = 3;
const SCALE: usize = 4;
const LEVEL: usize = 5;
const CONTRAST: usize = 6;

/// `JᵀWJ`, `JᵀWr` and the cost `Σ w·r²` at `params`.
///
/// The AVX2+FMA kernel when the CPU has it, else the portable one.
fn normal_equations(win: &Window, params: &Params) -> NormalEquations {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        #[allow(
            unsafe_code,
            reason = "a target_feature function needs its features, detected just above"
        )]
        // SAFETY: AVX2 and FMA were detected on this CPU just above, which is the only
        // requirement of the `target_feature` function.
        let fast = unsafe { avx2::normal_equations(win, params) };
        return fast;
    }
    normal_equations_portable(win, params)
}

/// `JᵀWJ`, `JᵀWr` and the cost.
type NormalEquations = (SMatrix<f64, PARAMS, PARAMS>, SVector<f64, PARAMS>, f64);

/// Portable kernel: branch-free over `LANES` pixels at a time ([`F64x4`]); the window is
/// zero-padded to a lane multiple with weight 0. The reference for the AVX2 kernel.
fn normal_equations_portable(win: &Window, params: &Params) -> NormalEquations {
    const TWO_OVER_SQRT_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI;
    let (n1, n2) = (normal(params[T1]), normal(params[T2]));
    // d nᵢ/d tᵢ = −uᵢ.
    let (dn1, dn2) = ([-n1[1], n1[0]], [-n2[1], n2[0]]);
    let inv_scale = 1.0 / params[SCALE];
    let (level, contrast) = (params[LEVEL], params[CONTRAST]);
    let slope = contrast * TWO_OVER_SQRT_PI * inv_scale;
    let (cx, cy) = (params[CX], params[CY]);

    let padded = win.len.next_multiple_of(LANES);
    let (dx, dy, value, weight) = (
        &win.dx[..padded],
        &win.dy[..padded],
        &win.value[..padded],
        &win.weight[..padded],
    );
    let splat = F64x4::splat;
    // One 4-lane partial sum per entry: the lanes vectorize, and no single sum is reassociated.
    let mut tri = [splat(0.0); PARAMS * (PARAMS + 1) / 2];
    let mut grad = [splat(0.0); PARAMS];
    let mut cost = splat(0.0);
    for base in (0..padded).step_by(LANES) {
        let lanes = base..base + LANES;
        let ex = F64x4::load(&dx[lanes.clone()]) - splat(cx);
        let ey = F64x4::load(&dy[lanes.clone()]) - splat(cy);
        let h1 = (splat(n1[0]) * ex + splat(n1[1]) * ey) * splat(inv_scale);
        let h2 = (splat(n2[0]) * ex + splat(n2[1]) * ey) * splat(inv_scale);
        let (erf1, gauss1) = erf_and_gauss_x4(h1);
        let (erf2, gauss2) = erf_and_gauss_x4(h2);
        let product = erf1 * erf2;
        let res = F64x4::load(&value[lanes.clone()]) - (splat(level) + splat(contrast) * product);
        let w = F64x4::load(&weight[lanes]);
        // ∂μ/∂(distance to each line).
        let d1 = splat(slope) * gauss1 * erf2;
        let d2 = splat(slope) * gauss2 * erf1;
        let jac = [
            splat(0.0) - (d1 * splat(n1[0]) + d2 * splat(n2[0])),
            splat(0.0) - (d1 * splat(n1[1]) + d2 * splat(n2[1])),
            d1 * (splat(dn1[0]) * ex + splat(dn1[1]) * ey),
            d2 * (splat(dn2[0]) * ex + splat(dn2[1]) * ey),
            splat(0.0) - (d1 * h1 + d2 * h2),
            splat(1.0),
            product,
        ];
        let w_res = w * res;
        cost = cost + w_res * res;
        let mut at = 0;
        for row in 0..PARAMS {
            grad[row] = grad[row] + jac[row] * w_res;
            let w_row = w * jac[row];
            for &col in &jac[..=row] {
                tri[at] = tri[at] + w_row * col;
                at += 1;
            }
        }
    }
    assemble(tri.map(F64x4::sum), grad.map(F64x4::sum), cost.sum())
}

/// Entries of the packed lower triangle of `JᵀWJ`, row by row.
const TRIANGLE: usize = PARAMS * (PARAMS + 1) / 2;

/// The symmetric `JᵀWJ` from its packed lower triangle, with `JᵀWr` and the cost.
fn assemble(tri: [f64; TRIANGLE], grad: [f64; PARAMS], cost: f64) -> NormalEquations {
    let mut jtj = SMatrix::<f64, PARAMS, PARAMS>::zeros();
    let mut at = 0;
    for row in 0..PARAMS {
        for col in 0..=row {
            jtj[(row, col)] = tri[at];
            jtj[(col, row)] = tri[at];
            at += 1;
        }
    }
    (jtj, SVector::from(grad), cost)
}

/// The kernel on AVX2+FMA: [`normal_equations_portable`] with each [`F64x4`] in one register.
/// LLVM leaves the portable version scalar (measured: no `ymm` arithmetic in its AVX2 clone),
/// and this kernel is most of the fit's time.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{
        CONTRAST, CX, CY, LANES, LEVEL, NormalEquations, PARAMS, Params, SCALE, T1, T2, TRIANGLE,
        Window, assemble, normal,
    };
    use crate::simd::math::avx2::{erf_and_gauss, sum};
    use core::arch::x86_64::{
        _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd,
        _mm256_sub_pd,
    };

    /// [`super::normal_equations_portable`] on AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) fn normal_equations(win: &Window, params: &Params) -> NormalEquations {
        const TWO_OVER_SQRT_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI;
        let (n1, n2) = (normal(params[T1]), normal(params[T2]));
        let inv_scale = 1.0 / params[SCALE];
        let splat = _mm256_set1_pd;
        let (n1x, n1y, n2x, n2y) = (splat(n1[0]), splat(n1[1]), splat(n2[0]), splat(n2[1]));
        // −nᵢ, by line then axis.
        let minus = [
            [splat(-n1[0]), splat(-n1[1])],
            [splat(-n2[0]), splat(-n2[1])],
        ];
        let (level, contrast) = (splat(params[LEVEL]), splat(params[CONTRAST]));
        let slope = splat(params[CONTRAST] * TWO_OVER_SQRT_PI * inv_scale);
        let (cx, cy, inv_s) = (splat(params[CX]), splat(params[CY]), splat(inv_scale));
        let mut tri = [_mm256_setzero_pd(); TRIANGLE];
        let mut grad = [_mm256_setzero_pd(); PARAMS];
        let mut cost = _mm256_setzero_pd();
        let padded = win.len.next_multiple_of(LANES);
        for base in (0..padded).step_by(LANES) {
            let load = |s: &[f64; super::MAX_SAMPLES]| {
                let lanes = &s[base..base + LANES];
                #[allow(unsafe_code, reason = "std::arch loads take a raw pointer")]
                // SAFETY: `lanes` is a bounds-checked slice of exactly four `f64`, which is
                // what the unaligned load reads.
                unsafe {
                    _mm256_loadu_pd(lanes.as_ptr())
                }
            };
            let ex = _mm256_sub_pd(load(&win.dx), cx);
            let ey = _mm256_sub_pd(load(&win.dy), cy);
            let h1 = _mm256_mul_pd(_mm256_fmadd_pd(n1x, ex, _mm256_mul_pd(n1y, ey)), inv_s);
            let h2 = _mm256_mul_pd(_mm256_fmadd_pd(n2x, ex, _mm256_mul_pd(n2y, ey)), inv_s);
            let (erf1, gauss1) = erf_and_gauss(h1);
            let (erf2, gauss2) = erf_and_gauss(h2);
            let product = _mm256_mul_pd(erf1, erf2);
            let res = _mm256_sub_pd(load(&win.value), _mm256_fmadd_pd(contrast, product, level));
            let w = load(&win.weight);
            // ∂μ/∂(distance to each line).
            let d1 = _mm256_mul_pd(_mm256_mul_pd(slope, gauss1), erf2);
            let d2 = _mm256_mul_pd(_mm256_mul_pd(slope, gauss2), erf1);
            // d nᵢ/d tᵢ = −uᵢ = (−nᵢ.y, nᵢ.x).
            let along1 = _mm256_fmadd_pd(n1x, ey, _mm256_mul_pd(minus[0][1], ex));
            let along2 = _mm256_fmadd_pd(n2x, ey, _mm256_mul_pd(minus[1][1], ex));
            let jac = [
                _mm256_fmadd_pd(d1, minus[0][0], _mm256_mul_pd(d2, minus[1][0])),
                _mm256_fmadd_pd(d1, minus[0][1], _mm256_mul_pd(d2, minus[1][1])),
                _mm256_mul_pd(d1, along1),
                _mm256_mul_pd(d2, along2),
                _mm256_sub_pd(
                    _mm256_setzero_pd(),
                    _mm256_fmadd_pd(d1, h1, _mm256_mul_pd(d2, h2)),
                ),
                splat(1.0),
                product,
            ];
            let w_res = _mm256_mul_pd(w, res);
            cost = _mm256_fmadd_pd(w_res, res, cost);
            let mut at = 0;
            for row in 0..PARAMS {
                grad[row] = _mm256_fmadd_pd(jac[row], w_res, grad[row]);
                let w_row = _mm256_mul_pd(w, jac[row]);
                for col in jac.iter().take(row + 1) {
                    tri[at] = _mm256_fmadd_pd(w_row, *col, tri[at]);
                    at += 1;
                }
            }
        }
        assemble(tri.map(|v| sum(v)), grad.map(|v| sum(v)), sum(cost))
    }
}

/// Unit normal `(−sin t, cos t)` of a line with direction angle `t`.
fn normal(t: f64) -> [f64; 2] {
    let (sin, cos) = t.sin_cos();
    [-sin, cos]
}

/// The X-junction near `start` (Locus convention, pixel centres at +0.5) whose two lines run
/// along `toward[0]` and `toward[1]` (the marker's two edges from the corner, any length),
/// fitted over the window reaching `reach[i]` px along each (`Window::gather`).
///
/// A refused fit that landed more than `RECENTRE_PX` from its start is refitted once over a
/// window centred where it landed, and only that refit is judged. A window off-centre by a
/// pixel or more holds the junction but describes it badly: its far side is mostly one wedge,
/// so the fit fails the residual gate (a split piece's seed pushed 1 px into the neck), or the
/// move limit (foreshortened tags at 50° incidence start 1.5-2 px off).
///
/// `None` when the fit, or its refit, is refused.
pub(crate) fn fit_x_junction(
    img: &ImageView,
    start: [f64; 2],
    toward: [[f64; 2]; 2],
    reach: [f64; 2],
) -> Option<[f64; 2]> {
    let (fit, accepted) = fit_once(img, start, toward, reach)?;
    if accepted {
        return Some(fit);
    }
    if (fit[0] - start[0]).hypot(fit[1] - start[1]) <= RECENTRE_PX {
        return None;
    }
    let (refit, accepted) = fit_once(img, fit, toward, reach)?;
    accepted.then_some(refit)
}

/// One fit over the window at `start`: where it landed and whether it passed every gate.
///
/// The gates refuse a result that is not this marker's X-junction: it moved more than
/// `MAX_MOVE_PX` (or half the window's shorter reach, after any shrink to the buffers), a line
/// turned more than `MAX_TURN_RAD`, the blur scale left `[MIN_SCALE_PX, max(reach)]`, the
/// wedge between the edges is not the dark one, or the residual RMS exceeds
/// `MAX_RELATIVE_RMS` of `|B|`.
///
/// `None`, with no landing point to refit from, when the window lies outside the image or
/// holds too few pixels, or the fit does not converge, leaves the image or leaves the window.
fn fit_once(
    img: &ImageView,
    start: [f64; 2],
    toward: [[f64; 2]; 2],
    reach: [f64; 2],
) -> Option<([f64; 2], bool)> {
    let unit = |d: [f64; 2]| {
        let len = d[0].hypot(d[1]);
        (len > 0.0).then(|| [d[0] / len, d[1] / len])
    };
    let axes = [unit(toward[0])?, unit(toward[1])?];
    let win = Window::gather(img, start, axes, reach)?;
    if win.len < MIN_SAMPLES_PER_PARAM * PARAMS {
        return None;
    }
    let angles = axes.map(|d| d[1].atan2(d[0]));
    // Levels by linear least squares at the initial geometry: with `A = B = 0` the residual is
    // the pixel value, so the kernel's (A, B) block is exactly those normal equations.
    let mut params = Params::from([0.0, 0.0, angles[0], angles[1], INITIAL_SCALE, 0.0, 0.0]);
    let (jtj, grad, _) = normal_equations(&win, &params);
    let (sw, se, see) = (
        jtj[(LEVEL, LEVEL)],
        jtj[(LEVEL, CONTRAST)],
        jtj[(CONTRAST, CONTRAST)],
    );
    let det = sw * see - se * se;
    if det <= f64::EPSILON * sw * sw {
        return None;
    }
    let contrast = (sw * grad[CONTRAST] - se * grad[LEVEL]) / det;
    params[LEVEL] = (grad[LEVEL] - contrast * se) / sw;
    params[CONTRAST] = contrast;
    // The wedge between the edges is the marker's border, so it must be dark: `B` must have
    // the sign opposite to `erf·erf` there.
    let inward = [axes[0][0] + axes[1][0], axes[0][1] + axes[1][1]];
    let side = |n: [f64; 2]| (n[0] * inward[0] + n[1] * inward[1]).signum();
    let dark_sign = -side(normal(angles[0])) * side(normal(angles[1]));
    if contrast * dark_sign <= 0.0 {
        return None;
    }

    let (mut jtj, mut grad, mut cost) = normal_equations(&win, &params);
    let mut lambda = INITIAL_DAMPING;
    let mut converged = false;
    for _ in 0..MAX_ITER {
        let mut damped = jtj;
        for k in 0..PARAMS {
            damped[(k, k)] += lambda * jtj[(k, k)].max(f64::MIN_POSITIVE);
        }
        let mut trial = params + damped.cholesky()?.solve(&grad);
        trial[SCALE] = trial[SCALE].max(MIN_SCALE_PX);
        // The trial's normal equations carry its cost, so an accepted step (the usual case)
        // costs one pass over the window, not two.
        let (trial_jtj, trial_grad, trial_cost) = normal_equations(&win, &trial);
        if trial_cost <= cost {
            let decrease = cost - trial_cost;
            (params, jtj, grad, cost) = (trial, trial_jtj, trial_grad, trial_cost);
            lambda = (lambda / 3.0).max(1e-12);
            if decrease <= COST_TOLERANCE * cost {
                converged = true;
                break;
            }
        } else {
            lambda *= 10.0;
            // No step lowers the cost: a minimum to working precision.
            if lambda > 1e8 {
                converged = true;
                break;
            }
        }
    }
    let sum_w: f64 = win.weight[..win.len].iter().sum();
    let rms = (cost / (sum_w - PARAMS as f64).max(1.0)).sqrt();
    let accepted = wrap_angle(params[T1] - angles[0]).abs() <= MAX_TURN_RAD
        && wrap_angle(params[T2] - angles[1]).abs() <= MAX_TURN_RAD
        && (MIN_SCALE_PX..=reach[0].max(reach[1])).contains(&params[SCALE])
        && params[CONTRAST] * dark_sign > 0.0
        && rms <= MAX_RELATIVE_RMS * params[CONTRAST].abs();
    let fit = [start[0] + params[CX], start[1] + params[CY]];
    // A clipped window may let the fit run off the frame, where no pixel holds the junction
    // and the downstream probes read clamped edge pixels.
    let inside =
        (0.0..img.width as f64).contains(&fit[0]) && (0.0..img.height as f64).contains(&fit[1]);
    // The window's own reach, not the requested one: a shrunk window cannot vouch for a
    // junction it does not hold.
    let moved = params[CX].hypot(params[CY]);
    let held = moved <= MAX_MOVE_PX.max(0.5 * win.min_reach);
    (converged && inside && moved <= win.min_reach).then_some((fit, accepted && held))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;

    /// An X-junction at `c` with line directions `t1`, `t2`, dark (40) where both signed
    /// distances share a sign and bright (200) elsewhere, box-integrated over each pixel
    /// (8 × 8 samples), blurred by a 3-tap kernel.
    fn render(c: [f64; 2], t1: f64, t2: f64, side: usize) -> Vec<u8> {
        let (n1, n2) = (normal(t1), normal(t2));
        let mut irr = vec![0.0f64; side * side];
        for y in 0..side {
            for x in 0..side {
                let mut acc = 0.0;
                for sy in 0..8u8 {
                    for sx in 0..8u8 {
                        let px = x as f64 + (f64::from(sx) + 0.5) / 8.0 - c[0];
                        let py = y as f64 + (f64::from(sy) + 0.5) / 8.0 - c[1];
                        let a = n1[0] * px + n1[1] * py;
                        let b = n2[0] * px + n2[1] * py;
                        acc += if (a > 0.0) == (b > 0.0) { 40.0 } else { 200.0 };
                    }
                }
                irr[y * side + x] = acc / 64.0;
            }
        }
        let k = [0.2, 0.6, 0.2];
        let mut tmp = irr.clone();
        for y in 0..side {
            for x in 1..side - 1 {
                tmp[y * side + x] = (0..3).map(|d| k[d] * irr[y * side + x + d - 1]).sum();
            }
        }
        let mut out = vec![0u8; side * side];
        for y in 1..side - 1 {
            for x in 0..side {
                let v: f64 = (0..3).map(|d| k[d] * tmp[(y + d - 1) * side + x]).sum();
                out[y * side + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    #[test]
    fn recovers_a_skewed_junction_from_an_offset_start() {
        let side = 32;
        let c = [16.27, 15.61];
        let (t1, t2) = (0.3, 0.3 + 1.9);
        let data = render(c, t1, t2, side);
        let img = ImageView::new(&data, side, side, side).unwrap();
        // The dark wedges are where both signed distances agree: the one between `d1` and
        // `−d2` is a marker border's corner.
        let (d1, d2) = ([t1.cos(), t1.sin()], [t2.cos(), t2.sin()]);
        let fit = fit_x_junction(
            &img,
            [c[0] + 0.4, c[1] - 0.3],
            [d1, [-d2[0], -d2[1]]],
            [4.5; 2],
        )
        .expect("fit");
        let err = (fit[0] - c[0]).hypot(fit[1] - c[1]);
        assert!(err < 0.02, "error {err}");
    }

    /// A junction 2 px from the frame: the window is clipped, and the fit still places it.
    #[test]
    fn fits_a_junction_the_frame_clips() {
        let side = 32;
        let c = [2.37, 16.21];
        let (t1, t2) = (0.2, 0.2 + 1.7);
        let data = render(c, t1, t2, side);
        let img = ImageView::new(&data, side, side, side).unwrap();
        let (d1, d2) = ([t1.cos(), t1.sin()], [t2.cos(), t2.sin()]);
        let (start, reach) = ([c[0] + 0.3, c[1] - 0.2], 4.5);
        // The premise: the window's bounding box crosses the left frame edge.
        assert!(start[0] - reach * d1[0].hypot(d2[0]) < 0.0);
        let fit = fit_x_junction(&img, start, [d1, [-d2[0], -d2[1]]], [reach; 2]).expect("fit");
        let err = (fit[0] - c[0]).hypot(fit[1] - c[1]);
        assert!(err < 0.05, "error {err}");
    }

    /// A seed past one window's move limit: the first fit lands at the junction but cannot
    /// vouch for it; the refit centred there can.
    #[test]
    fn refits_a_seed_past_the_move_limit() {
        let side = 32;
        let c = [16.27, 15.61];
        let (t1, t2) = (0.3, 0.3 + 1.9);
        let data = render(c, t1, t2, side);
        let img = ImageView::new(&data, side, side, side).unwrap();
        let (d1, d2) = ([t1.cos(), t1.sin()], [t2.cos(), t2.sin()]);
        let toward = [d1, [-d2[0], -d2[1]]];
        let (start, reach) = ([c[0] + 1.3, c[1] - 1.0], [3.0; 2]);
        // The premise: one fit converges but moves past its limit (1.5 px here).
        assert!(matches!(
            fit_once(&img, start, toward, reach),
            Some((_, false))
        ));
        let fit = fit_x_junction(&img, start, toward, reach).expect("fit");
        let err = (fit[0] - c[0]).hypot(fit[1] - c[1]);
        assert!(err < 0.05, "error {err}");
    }

    #[test]
    fn rejects_an_l_corner() {
        // A dark square's corner: one dark quadrant only.
        let side = 32;
        let mut data = vec![200u8; side * side];
        for y in 16..side {
            for x in 16..side {
                data[y * side + x] = 40;
            }
        }
        let img = ImageView::new(&data, side, side, side).unwrap();
        assert!(fit_x_junction(&img, [16.2, 15.8], [[1.0, 0.0], [0.0, 1.0]], [4.5; 2]).is_none());
    }

    /// The AVX2 kernel agrees with the portable one on a real window.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernel_matches_the_portable_one() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let side = 32;
        let data = render([16.27, 15.61], 0.3, 2.2, side);
        let img = ImageView::new(&data, side, side, side).unwrap();
        let axes = [
            [0.3f64.cos(), 0.3f64.sin()],
            [-(2.2f64.cos()), -(2.2f64.sin())],
        ];
        let win = Window::gather(&img, [16.5, 15.4], axes, [4.5; 2]).unwrap();
        let params = Params::from([0.1, -0.2, 0.32, 2.21, 0.9, 120.0, -80.0]);
        let (a, ga, ca) = normal_equations_portable(&win, &params);
        #[allow(unsafe_code, reason = "test of the target_feature kernel")]
        // SAFETY: AVX2 and FMA were detected above.
        let (b, gb, cb) = unsafe { avx2::normal_equations(&win, &params) };
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(1.0);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| close(*x, *y)));
        assert!(ga.iter().zip(gb.iter()).all(|(x, y)| close(*x, *y)));
        assert!(close(ca, cb));
    }
}
