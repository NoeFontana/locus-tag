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
//! AprilGrids at up to 70° incidence.

use crate::image::ImageView;
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
/// Rejections of a fit: largest move from the start (px), largest turn of either line (rad),
/// the blur scale's range (px), and the largest weighted residual RMS over the half contrast
/// `|B|` (a model that does not describe the window, such as an L-corner).
const MAX_MOVE_PX: f64 = 1.0;
const MAX_TURN_RAD: f64 = 0.35;
const MIN_SCALE_PX: f64 = 0.2;
const MAX_RELATIVE_RMS: f64 = 0.35;

/// The window: each pixel's centre relative to the window centre, its value and its weight.
struct Window {
    dx: [f64; MAX_SAMPLES],
    dy: [f64; MAX_SAMPLES],
    value: [f64; MAX_SAMPLES],
    weight: [f64; MAX_SAMPLES],
    len: usize,
}

impl Window {
    /// Pixels whose centres lie in the ellipse `{c + α·u + β·v : (α/a)² + (β/b)² ≤ 1}` for unit
    /// edge directions `axes = [u, v]` and `reach = [a, b]` px: the image of a disk on the
    /// marker's plane, so a foreshortened border gets a window as narrow as the border. Pixels
    /// farther than `FLAT_PX` from both lines are left out. `None` when the edges are nearly
    /// parallel or the window leaves the image.
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
        if !(x0 >= 0.0 && y0 >= 0.0) || x1 >= img.width as f64 || y1 >= img.height as f64 {
            return None;
        }
        // Non-negative and inside the image: checked just above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);
        // Rim width in units of the normalised radius, along the shorter axis.
        let rim = reach[0].min(reach[1]) * shrink / RIM_PX;
        let mut win = Self {
            dx: [0.0; MAX_SAMPLES],
            dy: [0.0; MAX_SAMPLES],
            value: [0.0; MAX_SAMPLES],
            weight: [0.0; MAX_SAMPLES],
            len: 0,
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

/// Lanes of the window loop: four `f64` fill an AVX2 register.
const LANES: usize = 4;

/// Four `f64` lanes with element-wise arithmetic: straight-line code on it maps onto one
/// vector register per value, which loop auto-vectorization did not achieve here.
#[derive(Clone, Copy)]
struct F4([f64; LANES]);

impl F4 {
    #[inline]
    fn splat(v: f64) -> Self {
        Self([v; LANES])
    }
    #[inline]
    fn load(s: &[f64]) -> Self {
        Self([s[0], s[1], s[2], s[3]])
    }
    #[inline]
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self(self.0.map(f))
    }
    #[inline]
    fn zip(self, o: Self, f: impl Fn(f64, f64) -> f64) -> Self {
        Self([
            f(self.0[0], o.0[0]),
            f(self.0[1], o.0[1]),
            f(self.0[2], o.0[2]),
            f(self.0[3], o.0[3]),
        ])
    }
    #[inline]
    fn sum(self) -> f64 {
        (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
    }
}

macro_rules! f4_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl core::ops::$trait for F4 {
            type Output = F4;
            #[inline]
            fn $method(self, o: F4) -> F4 {
                self.zip(o, |a, b| a $op b)
            }
        }
    };
}
f4_op!(Add, add, +);
f4_op!(Sub, sub, -);
f4_op!(Mul, mul, *);
f4_op!(Div, div, /);

/// `e^(−x)` for `x ≥ 0` without a libm call or a branch, so it vectorizes: `2^(−k)·e^(−r)` with
/// `k = round(x/ln 2)`, `|r| ≤ ln2/2` and a degree-11 Taylor polynomial (under 10⁻¹⁴
/// relative). `2^(−k)` is written into the exponent bits; `x` is clamped where the result
/// underflows anyway.
#[inline]
fn exp_neg(x: F4) -> F4 {
    // 1.5·2⁵²: adding it rounds to an integer held in the low mantissa bits.
    const ROUND: f64 = 6_755_399_441_055_744.0;
    let x = x.map(|v| v.min(700.0));
    let shifted = x * F4::splat(core::f64::consts::LOG2_E) + F4::splat(ROUND);
    let k = shifted - F4::splat(ROUND);
    // r = k·ln2 − x, so e^r · 2^(−k) = e^(−x).
    let r = (k * F4::splat(LN2_HI) - x) + k * F4::splat(LN2_LO);
    let mut poly = F4::splat(INV_FACTORIALS[11]);
    for &c in INV_FACTORIALS[..11].iter().rev() {
        poly = poly * r + F4::splat(c);
    }
    let scale = shifted.map(|v| {
        let k = v.to_bits().wrapping_sub(ROUND.to_bits());
        f64::from_bits(1023u64.wrapping_sub(k) << 52)
    });
    poly * scale
}

/// `ln 2` split so that `k·LN2_HI` is exact for the `k` of [`exp_neg`].
const LN2_HI: f64 = 0.693_147_180_369_123_8;
const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
/// `1/n!` for the Taylor polynomial of [`exp_neg`].
const INV_FACTORIALS: [f64; 12] = [
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5_040.0,
    1.0 / 40_320.0,
    1.0 / 362_880.0,
    1.0 / 3_628_800.0,
    1.0 / 39_916_800.0,
];

/// Abramowitz & Stegun 7.1.26 constants of [`erf_and_gauss`].
const ERF_P: f64 = 0.327_591_1;
const ERF_COEFFS: [f64; 5] = [
    0.254_829_592,
    -0.284_496_736,
    1.421_413_741,
    -1.453_152_027,
    1.061_405_429,
];

/// `erf(x)` and `exp(−x²)`: Abramowitz & Stegun 7.1.26 (absolute error under 1.5·10⁻⁷), as
/// `simd::math::erf_approx` but branch-free. The Gaussian is also the Jacobian's, so one
/// exponential serves both.
#[inline]
fn erf_and_gauss(x: F4) -> (F4, F4) {
    const P: f64 = ERF_P;
    const COEFFS: [f64; 5] = ERF_COEFFS;
    let abs = x.map(f64::abs);
    let tau = F4::splat(1.0) / (F4::splat(1.0) + F4::splat(P) * abs);
    let gauss = exp_neg(abs * abs);
    let mut poly = F4::splat(COEFFS[4]);
    for &c in COEFFS[..4].iter().rev() {
        poly = poly * tau + F4::splat(c);
    }
    let erf = (F4::splat(1.0) - poly * tau * gauss).zip(x, f64::copysign);
    (erf, gauss)
}

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

/// Portable kernel: branch-free over `LANES` pixels at a time ([`F4`]); the window is
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
    let splat = F4::splat;
    // One 4-lane partial sum per entry: the lanes vectorize, and no single sum is reassociated.
    let mut tri = [splat(0.0); PARAMS * (PARAMS + 1) / 2];
    let mut grad = [splat(0.0); PARAMS];
    let mut cost = splat(0.0);
    for base in (0..padded).step_by(LANES) {
        let lanes = base..base + LANES;
        let ex = F4::load(&dx[lanes.clone()]) - splat(cx);
        let ey = F4::load(&dy[lanes.clone()]) - splat(cy);
        let h1 = (splat(n1[0]) * ex + splat(n1[1]) * ey) * splat(inv_scale);
        let h2 = (splat(n2[0]) * ex + splat(n2[1]) * ey) * splat(inv_scale);
        let (erf1, gauss1) = erf_and_gauss(h1);
        let (erf2, gauss2) = erf_and_gauss(h2);
        let product = erf1 * erf2;
        let res = F4::load(&value[lanes.clone()]) - (splat(level) + splat(contrast) * product);
        let w = F4::load(&weight[lanes]);
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
    assemble(tri.map(F4::sum), grad.map(F4::sum), cost.sum())
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

/// The kernel on AVX2+FMA: [`normal_equations_portable`] with each [`F4`] in one register.
/// LLVM leaves the portable version scalar (measured: no `ymm` arithmetic in its AVX2 clone),
/// and this kernel is most of the fit's time.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{
        CONTRAST, CX, CY, LANES, LEVEL, NormalEquations, PARAMS, Params, SCALE, T1, T2, TRIANGLE,
        Window, assemble, normal,
    };
    use core::arch::x86_64::{
        __m256d, _mm_add_pd, _mm_add_sd, _mm_cvtsd_f64, _mm_unpackhi_pd, _mm256_and_pd,
        _mm256_andnot_pd, _mm256_castpd_si256, _mm256_castpd256_pd128, _mm256_castsi256_pd,
        _mm256_div_pd, _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_loadu_pd,
        _mm256_min_pd, _mm256_mul_pd, _mm256_or_pd, _mm256_set1_epi64x, _mm256_set1_pd,
        _mm256_setzero_pd, _mm256_slli_epi64, _mm256_sub_epi64, _mm256_sub_pd,
    };

    /// `e^(−x)` for `x ≥ 0`, as [`super::exp_neg`].
    #[target_feature(enable = "avx2,fma")]
    fn exp_neg(x: __m256d) -> __m256d {
        const ROUND: f64 = 6_755_399_441_055_744.0;
        let x = _mm256_min_pd(x, _mm256_set1_pd(700.0));
        let round = _mm256_set1_pd(ROUND);
        let shifted = _mm256_fmadd_pd(x, _mm256_set1_pd(core::f64::consts::LOG2_E), round);
        let k = _mm256_sub_pd(shifted, round);
        let r = _mm256_fmadd_pd(
            k,
            _mm256_set1_pd(super::LN2_LO),
            _mm256_fmadd_pd(
                k,
                _mm256_set1_pd(super::LN2_HI),
                _mm256_sub_pd(_mm256_setzero_pd(), x),
            ),
        );
        let mut poly = _mm256_set1_pd(super::INV_FACTORIALS[11]);
        for &c in super::INV_FACTORIALS[..11].iter().rev() {
            poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(c));
        }
        // 2^(−k): exponent bits (1023 − k) << 52, k read from the rounded mantissa.
        let k_bits = _mm256_sub_epi64(_mm256_castpd_si256(shifted), _mm256_castpd_si256(round));
        let biased = _mm256_sub_epi64(_mm256_set1_epi64x(1023), k_bits);
        _mm256_mul_pd(poly, _mm256_castsi256_pd(_mm256_slli_epi64::<52>(biased)))
    }

    /// `erf(x)` and `exp(−x²)`, as [`super::erf_and_gauss`].
    #[target_feature(enable = "avx2,fma")]
    fn erf_and_gauss(x: __m256d) -> (__m256d, __m256d) {
        let sign = _mm256_set1_pd(-0.0);
        let abs = _mm256_andnot_pd(sign, x);
        let one = _mm256_set1_pd(1.0);
        let tau = _mm256_div_pd(one, _mm256_fmadd_pd(_mm256_set1_pd(super::ERF_P), abs, one));
        let gauss = exp_neg(_mm256_mul_pd(abs, abs));
        let mut poly = _mm256_set1_pd(super::ERF_COEFFS[4]);
        for &c in super::ERF_COEFFS[..4].iter().rev() {
            poly = _mm256_fmadd_pd(poly, tau, _mm256_set1_pd(c));
        }
        let magnitude = _mm256_fnmadd_pd(_mm256_mul_pd(poly, tau), gauss, one);
        (_mm256_or_pd(magnitude, _mm256_and_pd(sign, x)), gauss)
    }

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
        // (l0 + l2) + (l1 + l3): the lanes' partial sums.
        let sum = |v: __m256d| {
            let pair = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd::<1>(v));
            _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)))
        };
        assemble(tri.map(sum), grad.map(sum), sum(cost))
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
/// `None` when the window leaves the image, the fit does not converge, or the result is not
/// this marker's X-junction: it moved more than `MAX_MOVE_PX`, a line turned more than
/// `MAX_TURN_RAD`, the blur scale left `[MIN_SCALE_PX, max(reach)]`, the wedge between the
/// edges is not the dark one, or the residual RMS exceeds `MAX_RELATIVE_RMS` of `|B|`.
pub(crate) fn fit_x_junction(
    img: &ImageView,
    start: [f64; 2],
    toward: [[f64; 2]; 2],
    reach: [f64; 2],
) -> Option<[f64; 2]> {
    let unit = |d: [f64; 2]| {
        let len = d[0].hypot(d[1]);
        (len > 0.0).then(|| [d[0] / len, d[1] / len])
    };
    let axes = [unit(toward[0])?, unit(toward[1])?];
    let win = Window::gather(img, start, axes, reach)?;
    if win.len < 4 * PARAMS {
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
    let accepted = converged
        && params[CX].hypot(params[CY]) <= MAX_MOVE_PX
        && angle_diff(params[T1], angles[0]).abs() <= MAX_TURN_RAD
        && angle_diff(params[T2], angles[1]).abs() <= MAX_TURN_RAD
        && (MIN_SCALE_PX..=reach[0].max(reach[1])).contains(&params[SCALE])
        && params[CONTRAST] * dark_sign > 0.0
        && rms <= MAX_RELATIVE_RMS * params[CONTRAST].abs();
    accepted.then(|| [start[0] + params[CX], start[1] + params[CY]])
}

/// `a − b` wrapped to `(−π, π]`.
fn angle_diff(a: f64, b: f64) -> f64 {
    use core::f64::consts::{PI, TAU};
    let d = (a - b).rem_euclid(TAU);
    if d > PI { d - TAU } else { d }
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

    #[test]
    fn exp_neg_matches_std() {
        // Up to the clamp: past it the result is below 10⁻³⁰⁴ either way.
        for k in 0..18_000 {
            let x = f64::from(k) * 0.0371;
            let (got, want) = (exp_neg(F4::splat(x)).0[0], (-x).exp());
            assert!((got - want).abs() <= 1e-14 * want, "{x}: {got} vs {want}");
        }
    }

    #[test]
    fn erf_matches_reference_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (-2.0, -0.995_322_265),
        ] {
            assert!((erf_and_gauss(F4::splat(x)).0.0[0] - want).abs() < 2e-7);
        }
    }
}
