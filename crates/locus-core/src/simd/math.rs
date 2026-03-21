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

/// Approximate error function (erf) using the Abramowitz and Stegun approximation.
///
/// Maximum error: 1.5e-7 over the entire domain.
/// This is a pure, stateless mathematical function extracted from the quad module
/// to serve as a foundational leaf dependency for both quad refinement and decoder stages.
#[must_use]
pub(crate) fn erf_approx(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Abramowitz and Stegun constants (formula 7.1.26)
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

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

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

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
