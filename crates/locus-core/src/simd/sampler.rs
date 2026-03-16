//! SIMD-vectorized image sampling kernels.
use crate::image::ImageView;
use multiversion::multiversion;

/// Vectorized bilinear interpolation for 8 points simultaneously.
///
/// # Safety
/// This function uses `_mm256_i32gather_epi32` to fetch 8-bit pixels by performing
/// 32-bit unaligned loads. This requires the input image buffer to have at least
/// **3 bytes of padding** at the end to avoid out-of-bounds reads when sampling
/// pixels near the bottom-right corner.
#[multiversion(targets(
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
pub fn sample_bilinear_v8(img: &ImageView, x: &[f32; 8], y: &[f32; 8], out: &mut [f32; 8]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    if img.has_simd_padding() {
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
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        use std::arch::aarch64::*;

        // NEON processes 4 floats at a time (float32x4_t)
        // We need to do it twice for 8 points.
        for chunk in 0..2 {
            let offset = chunk * 4;
            let vx = vld1q_f32(x.as_ptr().add(offset));
            let vy = vld1q_f32(y.as_ptr().add(offset));

            // Offset by -0.5
            let half = vdupq_n_f32(0.5);
            let vx = vsubq_f32(vx, half);
            let vy = vsubq_f32(vy, half);

            // Clamp to bounds
            let zero = vdupq_n_f32(0.0);
            let max_x = vdupq_n_f32((img.width as f32 - 2.0).max(0.0));
            let max_y = vdupq_n_f32((img.height as f32 - 2.0).max(0.0));

            let vx_clamped = vmaxq_f32(zero, vminq_f32(vx, max_x));
            let vy_clamped = vmaxq_f32(zero, vminq_f32(vy, max_y));

            // Integer parts (floor)
            // NEON floor: vrndmq_f32
            let vx_floor = vrndmq_f32(vx_clamped);
            let vy_floor = vrndmq_f32(vy_clamped);

            let vix = vcvtq_s32_f32(vx_floor);
            let viy = vcvtq_s32_f32(vy_floor);

            // Fractional parts
            let fx = vsubq_f32(vx_clamped, vx_floor);
            let fy = vsubq_f32(vy_clamped, vy_floor);

            let one = vdupq_n_f32(1.0);
            let inv_fx = vsubq_f32(one, fx);
            let inv_fy = vsubq_f32(one, fy);

            // Fetch pixels (No gather, use scalar loads)
            let mut pix_tl = [0.0f32; 4];
            let mut pix_tr = [0.0f32; 4];
            let mut pix_bl = [0.0f32; 4];
            let mut pix_br = [0.0f32; 4];

            let mut vix_arr = [0i32; 4];
            let mut viy_arr = [0i32; 4];
            vst1q_s32(vix_arr.as_mut_ptr(), vix);
            vst1q_s32(viy_arr.as_mut_ptr(), viy);

            for i in 0..4 {
                let px = vix_arr[i] as usize;
                let py = viy_arr[i] as usize;
                let stride = img.stride;
                let base = py * stride + px;
                pix_tl[i] = f32::from(img.data[base]);
                pix_tr[i] = f32::from(img.data[base + 1]);
                pix_bl[i] = f32::from(img.data[base + stride]);
                pix_br[i] = f32::from(img.data[base + stride + 1]);
            }

            let v_tl = vld1q_f32(pix_tl.as_ptr());
            let v_tr = vld1q_f32(pix_tr.as_ptr());
            let v_bl = vld1q_f32(pix_bl.as_ptr());
            let v_br = vld1q_f32(pix_br.as_ptr());

            // Bilinear interpolation
            let w_tl = vmulq_f32(inv_fx, inv_fy);
            let w_tr = vmulq_f32(fx, inv_fy);
            let w_bl = vmulq_f32(inv_fx, fy);
            let w_br = vmulq_f32(fx, fy);

            let mut res = vmulq_f32(w_tl, v_tl);
            res = vfmaq_f32(res, w_tr, v_tr);
            res = vfmaq_f32(res, w_bl, v_bl);
            res = vfmaq_f32(res, w_br, v_br);

            vst1q_f32(out.as_mut_ptr().add(offset), res);
        }
        return;
    }

    // Fallback: Scalar
    for i in 0..8 {
        out[i] = img.sample_bilinear(f64::from(x[i]), f64::from(y[i])) as f32;
    }
}
