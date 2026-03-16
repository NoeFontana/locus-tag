use crate::image::ImageView;
use multiversion::multiversion;

/// Vectorized bilinear interpolation for 8 points simultaneously.
#[multiversion(targets(
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
pub fn sample_bilinear_v8(img: &ImageView, x: &[f32; 8], y: &[f32; 8], out: &mut [f32; 8]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use std::arch::x86_64::*;
        
        let vx = _mm256_loadu_ps(x.as_ptr());
        let vy = _mm256_loadu_ps(y.as_ptr());
        
        // Offset by -0.5 to match ImageView::sample_bilinear logic (pixel centers)
        let half = _mm256_set1_ps(0.5);
        let vx = _mm256_sub_ps(vx, half);
        let vy = _mm256_sub_ps(vy, half);
        
        // Integer parts (clamped to 0..width-2, 0..height-2 to ensure 2x2 neighborhood is safe)
        let zero = _mm256_setzero_ps();
        let max_x = _mm256_set1_ps((img.width as f32 - 2.0).max(0.0));
        let max_y = _mm256_set1_ps((img.height as f32 - 2.0).max(0.0));
        
        let vx_clamped = _mm256_max_ps(zero, _mm256_min_ps(vx, max_x));
        let vy_clamped = _mm256_max_ps(zero, _mm256_min_ps(vy, max_y));
        
        let vx_floor = _mm256_floor_ps(vx_clamped);
        let vy_floor = _mm256_floor_ps(vy_clamped);
        
        let vix = _mm256_cvtps_epi32(vx_floor);
        let viy = _mm256_cvtps_epi32(vy_floor);
        
        // Fractional parts (weights)
        let fx = _mm256_sub_ps(vx_clamped, vx_floor);
        let fy = _mm256_sub_ps(vy_clamped, vy_floor);
        
        let one = _mm256_set1_ps(1.0);
        let inv_fx = _mm256_sub_ps(one, fx);
        let inv_fy = _mm256_sub_ps(one, fy);
        
        // 1D memory offsets: idx = y_int * stride + x_int
        let stride = _mm256_set1_epi32(img.stride as i32);
        let idx_tl = _mm256_add_epi32(_mm256_mullo_epi32(viy, stride), vix);
        let idx_tr = _mm256_add_epi32(idx_tl, _mm256_set1_epi32(1));
        let idx_bl = _mm256_add_epi32(idx_tl, stride);
        let idx_br = _mm256_add_epi32(idx_bl, _mm256_set1_epi32(1));
        
        // Gather 4 surrounding pixels (as 32-bit ints, then convert to f32)
        let base_ptr = img.data.as_ptr() as *const i32;
        
        let gather_to_f32 = |offsets: __m256i| -> __m256 {
            let gathered = _mm256_i32gather_epi32(base_ptr, offsets, 1);
            let masked = _mm256_and_si256(gathered, _mm256_set1_epi32(0xFF));
            _mm256_cvtepi32_ps(masked)
        };
        
        let v_tl = gather_to_f32(idx_tl);
        let v_tr = gather_to_f32(idx_tr);
        let v_bl = gather_to_f32(idx_bl);
        let v_br = gather_to_f32(idx_br);
        
        // Bilinear interpolation using FMA
        // I = (1-fx)(1-fy)TL + fx(1-fy)TR + (1-fx)fy*BL + fx*fy*BR
        
        let w_tl = _mm256_mul_ps(inv_fx, inv_fy);
        let w_tr = _mm256_mul_ps(fx, inv_fy);
        let w_bl = _mm256_mul_ps(inv_fx, fy);
        let w_br = _mm256_mul_ps(fx, fy);
        
        let mut res = _mm256_mul_ps(w_tl, v_tl);
        res = _mm256_fmadd_ps(w_tr, v_tr, res);
        res = _mm256_fmadd_ps(w_bl, v_bl, res);
        res = _mm256_fmadd_ps(w_br, v_br, res);
        
        _mm256_storeu_ps(out.as_mut_ptr(), res);
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // NEON implementation (will implement in next task)
    }

    // Fallback: Scalar
    for i in 0..8 {
        out[i] = img.sample_bilinear(f64::from(x[i]), f64::from(y[i])) as f32;
    }
}
