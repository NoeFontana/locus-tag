//! SIMD optimized mathematical kernels for Fast-Math.

use multiversion::multiversion;

/// Compute 1.0 / w using SIMD reciprocal estimation with Newton-Raphson refinement.
///
/// This is significantly faster than standard floating-point division.
/// $w_{inv} = w_{inv} \cdot (2.0 - w \cdot w_{inv})$
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[must_use]
pub(crate) fn rcp_nr(w: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: SSE/AVX intrinsics are safe on x86_64 with avx2.
    unsafe {
        use std::arch::x86_64::*;
        let w_vec = _mm_set_ss(w);
        let rcp = _mm_rcp_ss(w_vec);

        // Newton-Raphson: r1 = r0 * (2.0 - w * r0)
        let two = _mm_set_ss(2.0);
        let prod = _mm_mul_ss(w_vec, rcp);
        let diff = _mm_sub_ss(two, prod);
        let res = _mm_mul_ss(rcp, diff);

        return _mm_cvtss_f32(res);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(unsafe_code)]
    // SAFETY: NEON intrinsics are safe on aarch64 with neon feature.
    unsafe {
        use std::arch::aarch64::*;
        // Load f32 into a D-register (float32x2_t)
        let v_w = vdupq_n_f32(w);
        let res_vec = vrecpeq_f32(v_w);
        let res_vec = vmulq_f32(res_vec, vrecpsq_f32(v_w, res_vec));
        return vgetq_lane_f32(res_vec, 0);
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        1.0 / w
    }
}

/// Perform bilinear interpolation using 16.16 fixed-point arithmetic.
///
/// This is faster than floating-point bilinear interpolation on most CPUs.
/// Coordinates (x, y) should be sub-pixel floats.
/// Pixels (p00, p10, p01, p11) are the 4 surrounding pixels.
#[must_use]
#[allow(clippy::cast_sign_loss, dead_code)]
pub(crate) fn bilinear_interpolate_fixed(x: f32, y: f32, p00: u8, p10: u8, p01: u8, p11: u8) -> u8 {
    // Convert to 16.16 fixed point (using only fractional part for weights)
    let fx = ((x.fract() * 65536.0) as u32) & 0xFFFF;
    let fy = ((y.fract() * 65536.0) as u32) & 0xFFFF;

    let inv_x = 0x10000 - fx;
    let inv_y = 0x10000 - fy;

    // Weights: w00 = (1-fx)(1-fy), w10 = fx(1-fy), w01 = (1-fx)fy, w11 = fxfy
    // Use u64 for intermediate product to avoid overflow (u32 * u32 can be up to 2^32)
    let w00 = (u64::from(inv_x) * u64::from(inv_y)) >> 16;
    let w10 = (u64::from(fx) * u64::from(inv_y)) >> 16;
    let w01 = (u64::from(inv_x) * u64::from(fy)) >> 16;
    let w11 = (u64::from(fx) * u64::from(fy)) >> 16;

    let res =
        (u64::from(p00) * w00 + u64::from(p10) * w10 + u64::from(p01) * w01 + u64::from(p11) * w11)
            >> 16;
    res as u8
}

/// Abramowitz & Stegun 7.1.26: `erf(x) ≈ 1 − (a₁t + … + a₅t⁵)·e^(−x²)` with
/// `t = 1/(1 + p·x)`, `x ≥ 0`. Shared by every erf kernel so they cannot drift apart.
pub(crate) const ERF_P: f64 = 0.327_591_1;
pub(crate) const ERF_COEFFS: [f64; 5] = [
    0.254_829_592,
    -0.284_496_736,
    1.421_413_741,
    -1.453_152_027,
    1.061_405_429,
];

/// Approximate error function (erf) using the Abramowitz and Stegun approximation.
///
/// Maximum error: 1.5e-7 over the entire domain.
/// This is a pure, stateless mathematical function extracted from the quad module
/// to serve as a foundational leaf dependency for both quad refinement and decoder stages.
#[must_use]
pub(crate) fn erf_approx(x: f64) -> f64 {
    let a = x.abs();
    erf_approx_with_gauss(x, (-a * a).exp())
}

/// [`erf_approx`] given `gauss = exp(-x²)`, for callers that also need the Gaussian (the ERF
/// edge model's Jacobian): the result is bit-identical to `erf_approx(x)` when `gauss` is
/// computed as `(-|x| * |x|).exp()` (equivalently `(-x * x).exp()`).
#[inline]
pub(crate) fn erf_approx_with_gauss(x: f64, gauss: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let [a1, a2, a3, a4, a5] = ERF_COEFFS;
    let t = 1.0 / (1.0 + ERF_P * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * gauss;

    sign * y
}

/// Vectorized error function approximation over 4 lanes (AVX2 `__m256d`).
///
/// Computes `erf_approx` for 4 `f64` values simultaneously using FMA instructions,
/// eliminating the register-spill penalty of unpacking to scalar in the Gauss-Newton loop.
///
/// On non-AVX2 targets, falls back to 4 scalar evaluations.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[must_use]
pub(crate) unsafe fn erf_approx_v4(x: std::arch::x86_64::__m256d) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;

    // sign = copysign(1.0, x)
    let sign_mask = _mm256_set1_pd(-0.0);
    let sign_bits = _mm256_and_pd(x, sign_mask);
    let abs_x = _mm256_andnot_pd(sign_mask, x);

    // Abramowitz and Stegun constants
    let a1 = _mm256_set1_pd(0.254_829_592);
    let a2 = _mm256_set1_pd(-0.284_496_736);
    let a3 = _mm256_set1_pd(1.421_413_741);
    let a4 = _mm256_set1_pd(-1.453_152_027);
    let a5 = _mm256_set1_pd(1.061_405_429);
    let p = _mm256_set1_pd(0.327_591_1);
    let one = _mm256_set1_pd(1.0);

    // t = 1.0 / (1.0 + p * |x|)
    let t = _mm256_div_pd(one, _mm256_fmadd_pd(p, abs_x, one));

    // Horner's method: poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)
    let poly = _mm256_fmadd_pd(a5, t, a4);
    let poly = _mm256_fmadd_pd(poly, t, a3);
    let poly = _mm256_fmadd_pd(poly, t, a2);
    let poly = _mm256_fmadd_pd(poly, t, a1);

    // exp(-x^2): compute using scalar fallback since there's no fast _mm256_exp_pd.
    // We extract, compute exp, and re-pack. This is still faster than full scalar erf
    // because the polynomial chain above is fully vectorized.
    let neg_x2 = _mm256_mul_pd(abs_x, abs_x);
    let neg_x2 = _mm256_xor_pd(neg_x2, sign_mask); // negate

    // SAFETY: transmute is safe for same-size SIMD ↔ array conversions.
    let neg_x2_arr: [f64; 4] = std::mem::transmute(neg_x2);
    let exp_vals = _mm256_set_pd(
        neg_x2_arr[3].exp(),
        neg_x2_arr[2].exp(),
        neg_x2_arr[1].exp(),
        neg_x2_arr[0].exp(),
    );

    // y = 1.0 - poly * t * exp(-x^2)
    let y = _mm256_fnmadd_pd(_mm256_mul_pd(poly, t), exp_vals, one);

    // Apply sign: result = y XOR sign_bits
    _mm256_or_pd(y, sign_bits)
}

/// Scalar fallback for `erf_approx_v4` on non-AVX2 targets.
///
/// Evaluates 4 `f64` values independently using the scalar `erf_approx`.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
#[must_use]
#[allow(dead_code)]
pub(crate) fn erf_approx_v4(x: [f64; 4]) -> [f64; 4] {
    [
        erf_approx(x[0]),
        erf_approx(x[1]),
        erf_approx(x[2]),
        erf_approx(x[3]),
    ]
}

// ── 4-lane f64 kernels ───────────────────────────────────────────────────────

/// Four `f64` lanes with element-wise arithmetic: the portable form of the 4-wide kernels
/// (the AVX2 ones in [`avx2`] mirror it op for op). LLVM does not reliably turn it into vector
/// code, so hot loops dispatch to [`avx2`] where the CPU has it.
#[derive(Clone, Copy)]
pub(crate) struct F64x4(pub(crate) [f64; 4]);

impl F64x4 {
    /// Lane count: four `f64` fill one AVX2 register.
    pub(crate) const LANES: usize = 4;

    #[inline]
    pub(crate) fn splat(v: f64) -> Self {
        Self([v; Self::LANES])
    }
    #[inline]
    pub(crate) fn load(s: &[f64]) -> Self {
        Self([s[0], s[1], s[2], s[3]])
    }
    #[inline]
    pub(crate) fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self(self.0.map(f))
    }
    #[inline]
    pub(crate) fn zip(self, o: Self, f: impl Fn(f64, f64) -> f64) -> Self {
        Self([
            f(self.0[0], o.0[0]),
            f(self.0[1], o.0[1]),
            f(self.0[2], o.0[2]),
            f(self.0[3], o.0[3]),
        ])
    }
    /// `(l0 + l2) + (l1 + l3)`: [`avx2::sum`]'s order, so the portable and AVX2 kernels differ
    /// only by FMA rounding.
    #[inline]
    pub(crate) fn sum(self) -> f64 {
        (self.0[0] + self.0[2]) + (self.0[1] + self.0[3])
    }
}

macro_rules! f64x4_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl core::ops::$trait for F64x4 {
            type Output = F64x4;
            #[inline]
            fn $method(self, o: F64x4) -> F64x4 {
                self.zip(o, |a, b| a $op b)
            }
        }
    };
}
f64x4_op!(Add, add, +);
f64x4_op!(Sub, sub, -);
f64x4_op!(Mul, mul, *);
f64x4_op!(Div, div, /);

/// `e^(−x)` for `x ≥ 0` without a libm call or a branch, so it vectorizes: `2^(−k)·e^(−r)` with
/// `k = round(x/ln 2)`, `|r| ≤ ln2/2` and a degree-11 Taylor polynomial (under 10⁻¹⁴
/// relative). `2^(−k)` is written into the exponent bits; `x` is clamped where the result
/// underflows anyway.
#[inline]
pub(crate) fn exp_neg_x4(x: F64x4) -> F64x4 {
    // 1.5·2⁵²: adding it rounds to an integer held in the low mantissa bits.
    const ROUND: f64 = 6_755_399_441_055_744.0;
    let x = x.map(|v| v.min(700.0));
    let shifted = x * F64x4::splat(core::f64::consts::LOG2_E) + F64x4::splat(ROUND);
    let k = shifted - F64x4::splat(ROUND);
    // r = k·ln2 − x, so e^r · 2^(−k) = e^(−x).
    let r = (k * F64x4::splat(LN2_HI) - x) + k * F64x4::splat(LN2_LO);
    let mut poly = F64x4::splat(INV_FACTORIALS[11]);
    for &c in INV_FACTORIALS[..11].iter().rev() {
        poly = poly * r + F64x4::splat(c);
    }
    let scale = shifted.map(|v| {
        let k = v.to_bits().wrapping_sub(ROUND.to_bits());
        f64::from_bits(1023u64.wrapping_sub(k) << 52)
    });
    poly * scale
}

/// `ln 2` split so that `k·LN2_HI` is exact for the `k` of [`exp_neg_x4`].
const LN2_HI: f64 = 0.693_147_180_369_123_8;
const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
/// `1/n!` for the Taylor polynomial of [`exp_neg_x4`].
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

/// `erf(x)` and `exp(−x²)`: Abramowitz & Stegun 7.1.26 (absolute error under 1.5·10⁻⁷), as
/// [`erf_approx`] but branch-free and with a vectorizable exponential. Both are returned
/// because an erf model's Jacobian needs the Gaussian, and one exponential serves both.
#[inline]
pub(crate) fn erf_and_gauss_x4(x: F64x4) -> (F64x4, F64x4) {
    let abs = x.map(f64::abs);
    let tau = F64x4::splat(1.0) / (F64x4::splat(1.0) + F64x4::splat(ERF_P) * abs);
    let gauss = exp_neg_x4(abs * abs);
    let mut poly = F64x4::splat(ERF_COEFFS[4]);
    for &c in ERF_COEFFS[..4].iter().rev() {
        poly = poly * tau + F64x4::splat(c);
    }
    let erf = (F64x4::splat(1.0) - poly * tau * gauss).zip(x, f64::copysign);
    (erf, gauss)
}

/// AVX2+FMA forms of the [`F64x4`] kernels, op for op. Callable without `unsafe` only from
/// functions with the same features enabled.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use core::arch::x86_64::{
        __m256d, _mm_add_pd, _mm_add_sd, _mm_cvtsd_f64, _mm_unpackhi_pd, _mm256_and_pd,
        _mm256_andnot_pd, _mm256_castpd_si256, _mm256_castpd256_pd128, _mm256_castsi256_pd,
        _mm256_div_pd, _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_min_pd,
        _mm256_mul_pd, _mm256_or_pd, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_setzero_pd,
        _mm256_slli_epi64, _mm256_sub_epi64, _mm256_sub_pd,
    };

    /// [`super::exp_neg_x4`] in one register.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn exp_neg(x: __m256d) -> __m256d {
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

    /// [`super::erf_and_gauss_x4`] in one register.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn erf_and_gauss(x: __m256d) -> (__m256d, __m256d) {
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

    /// `(l0 + l2) + (l1 + l3)`, as [`super::F64x4::sum`].
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn sum(v: __m256d) -> f64 {
        let pair = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd::<1>(v));
        _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)))
    }
}

/// `a` wrapped to `(−π, π]`: the signed difference of two angles is `wrap_angle(a − b)`.
#[must_use]
pub(crate) fn wrap_angle(a: f64) -> f64 {
    use core::f64::consts::{PI, TAU};
    let d = a.rem_euclid(TAU);
    if d > PI { d - TAU } else { d }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn exp_neg_x4_matches_std() {
        // Up to the clamp: past it the result is below 10⁻³⁰⁴ either way.
        for k in 0..18_000 {
            let x = f64::from(k) * 0.0371;
            let (got, want) = (exp_neg_x4(F64x4::splat(x)).0[0], (-x).exp());
            assert!((got - want).abs() <= 1e-14 * want, "{x}: {got} vs {want}");
        }
    }

    #[test]
    fn erf_and_gauss_x4_matches_reference_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (-2.0, -0.995_322_265),
        ] {
            assert!((erf_and_gauss_x4(F64x4::splat(x)).0.0[0] - want).abs() < 2e-7);
        }
    }

    use proptest::prelude::*;

    proptest! {
        /// Sharing exp(-s²) between erf and the ERF Jacobian changes no bit.
        #[test]
        fn erf_with_shared_gauss_is_bit_identical(x in -6.0..6.0f64) {
            prop_assert_eq!(
                erf_approx_with_gauss(x, (-x * x).exp()).to_bits(),
                erf_approx(x).to_bits()
            );
        }
    }

    #[test]
    fn test_rcp_nr_precision() {
        let values = [1.0, 2.0, 10.0, 0.5, 123.456];
        for &w in &values {
            let expected = 1.0 / w;
            let actual = rcp_nr(w);
            let diff = (expected - actual).abs();
            // Newton-Raphson iteration should get us very close to 1.0/w
            assert!(
                diff < 1e-4,
                "rcp_nr({w}) failed: expected {expected}, got {actual}, diff {diff}"
            );
        }
    }

    #[test]
    fn test_erf_approx_properties() {
        // Zero crossing
        assert_eq!(erf_approx(0.0), 0.0);

        // Symmetry: erf(-x) == -erf(x)
        for x in [0.1, 0.5, 1.0, 2.0, 5.0] {
            assert!((erf_approx(-x) + erf_approx(x)).abs() < 1e-15);
        }

        // Asymptotic bounds
        assert!((erf_approx(10.0) - 1.0).abs() < 1e-7);
        assert!((erf_approx(-10.0) + 1.0).abs() < 1e-7);
        assert!((erf_approx(100.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_erf_approx_accuracy() {
        let cases = [
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_8),
            (2.0, 0.995_322_265_018_952_7),
        ];

        for (x, expected) in cases {
            let actual = erf_approx(x);
            let diff = (actual - expected).abs();
            assert!(
                diff < 1.5e-7,
                "erf_approx({x}) error {diff} exceeds tolerance 1.5e-7"
            );
        }
    }

    #[test]
    fn test_erf_approx_v4_matches_scalar() {
        let inputs = [0.5, -1.0, 2.0, -0.3];

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            use std::arch::x86_64::*;
            // SAFETY: AVX2+FMA checked by cfg.
            unsafe {
                let v = _mm256_set_pd(inputs[3], inputs[2], inputs[1], inputs[0]);
                let result = erf_approx_v4(v);
                let result_arr: [f64; 4] = std::mem::transmute(result);
                for i in 0..4 {
                    let scalar = erf_approx(inputs[i]);
                    let diff = (result_arr[i] - scalar).abs();
                    assert!(
                        diff < 1e-15,
                        "erf_approx_v4 lane {i}: expected {scalar}, got {}, diff {diff}",
                        result_arr[i]
                    );
                }
            }
        }

        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )))]
        {
            let result = erf_approx_v4(inputs);
            for i in 0..4 {
                let scalar = erf_approx(inputs[i]);
                let diff = (result[i] - scalar).abs();
                assert!(
                    diff < 1e-15,
                    "erf_approx_v4 lane {i}: expected {scalar}, got {}, diff {diff}",
                    result[i]
                );
            }
        }
    }

    #[test]
    fn test_bilinear_fixed() {
        // Center of 4 pixels: average
        assert_eq!(
            bilinear_interpolate_fixed(0.5, 0.5, 100, 200, 100, 200),
            150
        );
        // Top-left: p00
        assert_eq!(bilinear_interpolate_fixed(0.0, 0.0, 100, 200, 50, 250), 100);
        // Bottom-right: p11
        assert_eq!(
            bilinear_interpolate_fixed(0.999, 0.999, 100, 200, 50, 250),
            249
        ); // Rounding
    }
}
