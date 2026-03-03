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
pub fn rcp_nr(w: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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
    unsafe {
        use std::arch::aarch64::*;
        // vrecpes_f32 provides the estimate
        let rcp = vrecpes_f32(w);
        // vrecps_f32 provides the Newton-Raphson step (2.0 - w * r0)
        let step = vrecps_f32(w, rcp);
        // Final refinement
        return vmul_n_f32(rcp, step);
    }

    // Scalar fallback
    1.0 / w
}

/// Perform bilinear interpolation using 16.16 fixed-point arithmetic.
/// 
/// This is faster than floating-point bilinear interpolation on most CPUs.
/// Coordinates (x, y) should be sub-pixel floats.
/// Pixels (p00, p10, p01, p11) are the 4 surrounding pixels.
#[must_use]
#[allow(clippy::cast_sign_loss)]
pub fn bilinear_interpolate_fixed(x: f32, y: f32, p00: u8, p10: u8, p01: u8, p11: u8) -> u8 {
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
    
    let res = (u64::from(p00) * w00 + u64::from(p10) * w10 + u64::from(p01) * w01 + u64::from(p11) * w11) >> 16;
    res as u8
}

#[cfg(test)]
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
    fn test_bilinear_fixed() {
        // Center of 4 pixels: average
        assert_eq!(bilinear_interpolate_fixed(0.5, 0.5, 100, 200, 100, 200), 150);
        // Top-left: p00
        assert_eq!(bilinear_interpolate_fixed(0.0, 0.0, 100, 200, 50, 250), 100);
        // Bottom-right: p11
        assert_eq!(bilinear_interpolate_fixed(0.999, 0.999, 100, 200, 50, 250), 249); // Rounding
    }
}
