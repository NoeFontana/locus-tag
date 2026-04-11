#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

use super::RleSegment;
use crate::image::ImageView;

/// SIMD-Accelerated Fused Threshold + RLE extraction.
#[must_use]
pub fn extract_rle_segments(img: &ImageView, threshold_map: &[u8]) -> Vec<RleSegment> {
    let mut segments = Vec::new();
    let width = img.width;
    let height = img.height;

    for y in 0..height {
        let src_row = img.get_row(y);
        let thresh_row = &threshold_map[y * width..(y + 1) * width];

        process_row_simd(src_row, thresh_row, y as u16, &mut segments);
    }

    segments
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
fn process_row_simd(src: &[u8], thresh: &[u8], y: u16, segments: &mut Vec<RleSegment>) {
    super::process_row_scalar(src, thresh, y, segments);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn process_row_simd(src: &[u8], thresh: &[u8], y: u16, segments: &mut Vec<RleSegment>) {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        // SAFETY: Feature check above ensures AVX-512 is available.
        unsafe { process_row_avx512(src, thresh, y, segments) }
    } else if std::is_x86_feature_detected!("avx2") {
        // SAFETY: Feature check above ensures AVX2 is available.
        unsafe { process_row_avx2(src, thresh, y, segments) }
    } else {
        super::process_row_scalar(src, thresh, y, segments);
    }
}

#[cfg(target_arch = "aarch64")]
fn process_row_simd(src: &[u8], thresh: &[u8], y: u16, segments: &mut Vec<RleSegment>) {
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { process_row_neon(src, thresh, y, segments) }
    } else {
        super::process_row_scalar(src, thresh, y, segments);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn process_row_avx2(src: &[u8], thresh: &[u8], y: u16, segments: &mut Vec<RleSegment>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_andnot_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_min_epu8,
        _mm256_movemask_epi8,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_andnot_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_min_epu8,
        _mm256_movemask_epi8,
    };

    let width = src.len();
    let mut x = 0;
    let mut in_segment = false;
    let mut start_x = 0;

    let chunks = width / 32;
    for i in 0..chunks {
        let offset = i * 32;
        let s_ptr = src.as_ptr().add(offset).cast();
        let t_ptr = thresh.as_ptr().add(offset).cast();

        let s_vec = _mm256_loadu_si256(s_ptr);
        let t_vec = _mm256_loadu_si256(t_ptr);

        let min_val = _mm256_min_epu8(s_vec, t_vec);
        let is_le = _mm256_cmpeq_epi8(min_val, s_vec); // 0xFF where s <= t
        let is_eq = _mm256_cmpeq_epi8(s_vec, t_vec); // 0xFF where s == t
        let cmp = _mm256_andnot_si256(is_eq, is_le); // is_le & ~is_eq -> s < t

        let mask = _mm256_movemask_epi8(cmp).cast_unsigned();
        let inverted = !mask;
        let mut bit_idx = 0;

        while bit_idx < 32 {
            if in_segment {
                let m = inverted >> bit_idx;
                if m == 0 {
                    break;
                }
                bit_idx += m.trailing_zeros();
                segments.push(RleSegment::new(
                    y,
                    start_x,
                    (offset as u32 + bit_idx) as u16,
                ));
                in_segment = false;
            } else {
                let m = mask >> bit_idx;
                if m == 0 {
                    break;
                }
                bit_idx += m.trailing_zeros();
                start_x = (offset as u32 + bit_idx) as u16;
                in_segment = true;
            }
        }

        x += 32;
    }

    for (ix, (&s, &t)) in src.iter().zip(thresh.iter()).enumerate().skip(x) {
        let is_foreground = s < t;
        if is_foreground && !in_segment {
            in_segment = true;
            start_x = ix as u16;
        } else if !is_foreground && in_segment {
            in_segment = false;
            segments.push(RleSegment::new(y, start_x, ix as u16));
        }
    }
    if in_segment {
        segments.push(RleSegment::new(y, start_x, width as u16));
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn process_row_avx512(src: &[u8], thresh: &[u8], y: u16, segments: &mut Vec<RleSegment>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm512_cmp_epu8_mask, _mm512_loadu_si512};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm512_cmp_epu8_mask, _mm512_loadu_si512};

    let width = src.len();
    let mut x = 0;
    let mut in_segment = false;
    let mut start_x = 0;

    let chunks = width / 64;
    for i in 0..chunks {
        let offset = i * 64;
        let s_ptr = src.as_ptr().add(offset).cast();
        let t_ptr = thresh.as_ptr().add(offset).cast();

        let s_vec = _mm512_loadu_si512(s_ptr);
        let t_vec = _mm512_loadu_si512(t_ptr);

        // _MM_CMPINT_LT = 1
        let mask = _mm512_cmp_epu8_mask(s_vec, t_vec, 1);
        let inverted = !mask;
        let mut bit_idx = 0;

        while bit_idx < 64 {
            if in_segment {
                let m = inverted >> bit_idx;
                if m == 0 {
                    break;
                }
                bit_idx += m.trailing_zeros();
                segments.push(RleSegment::new(
                    y,
                    start_x,
                    (offset as u32 + bit_idx) as u16,
                ));
                in_segment = false;
            } else {
                let m = mask >> bit_idx;
                if m == 0 {
                    break;
                }
                bit_idx += m.trailing_zeros();
                start_x = (offset as u32 + bit_idx) as u16;
                in_segment = true;
            }
        }

        x += 64;
    }

    for (ix, (&s, &t)) in src.iter().zip(thresh.iter()).enumerate().skip(x) {
        let is_foreground = s < t;
        if is_foreground && !in_segment {
            in_segment = true;
            start_x = ix as u16;
        } else if !is_foreground && in_segment {
            in_segment = false;
            segments.push(RleSegment::new(y, start_x, ix as u16));
        }
    }
    if in_segment {
        segments.push(RleSegment::new(y, start_x, width as u16));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn process_row_neon(src: &[u8], thresh: &[u8], y: u16, segments: &mut Vec<RleSegment>) {
    use std::arch::aarch64::*;

    let width = src.len();
    let mut x = 0;
    let mut in_segment = false;
    let mut start_x = 0;

    let chunks = width / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let s_ptr = src.as_ptr().add(offset);
        let t_ptr = thresh.as_ptr().add(offset);

        let s_vec = vld1q_u8(s_ptr);
        let t_vec = vld1q_u8(t_ptr);

        let cmp = vcgtq_u8(t_vec, s_vec); // t > s <==> s < t
        let m_half = vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4);
        let mask_u64 = vget_lane_u64(vreinterpret_u64_u8(m_half), 0);

        let inverted = !mask_u64;
        let mut bit_idx = 0;

        while bit_idx < 16 {
            let shift = bit_idx * 4;
            if in_segment {
                let m = inverted >> shift;
                if m == 0 {
                    break;
                }
                bit_idx += m.trailing_zeros() / 4;
                segments.push(RleSegment::new(
                    y,
                    start_x,
                    (offset as u32 + bit_idx) as u16,
                ));
                in_segment = false;
            } else {
                let m = mask_u64 >> shift;
                if m == 0 {
                    break;
                }
                bit_idx += m.trailing_zeros() / 4;
                start_x = (offset as u32 + bit_idx) as u16;
                in_segment = true;
            }
        }

        x += 16;
    }

    for (ix, (&s, &t)) in src.iter().zip(thresh.iter()).enumerate().skip(x) {
        let is_foreground = s < t;
        if is_foreground && !in_segment {
            in_segment = true;
            start_x = ix as u16;
        } else if !is_foreground && in_segment {
            in_segment = false;
            segments.push(RleSegment::new(y, start_x, ix as u16));
        }
    }
    if in_segment {
        segments.push(RleSegment::new(y, start_x, width as u16));
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::simd_ccl_fusion::extract_rle_segments_scalar;
    use proptest::prelude::*;

    #[test]
    fn test_extract_rle_segments_simd_basic() {
        let width = 64;
        let height = 2;
        let mut pixels = vec![200u8; width * height];
        let threshold_map = vec![128u8; width * height];

        // Add some runs
        for p in pixels.iter_mut().take(15).skip(5) {
            *p = 50;
        }
        for p in pixels.iter_mut().take(40).skip(30) {
            *p = 50;
        }
        for p in pixels.iter_mut().take(64).skip(60) {
            *p = 50;
        }
        for p in pixels.iter_mut().take(75).skip(65) {
            *p = 50;
        }

        let img = ImageView::new(&pixels, width, height, width).expect("Valid test image");

        let scalar_segments = extract_rle_segments_scalar(&img, &threshold_map);
        let simd_segments = extract_rle_segments(&img, &threshold_map);

        assert_eq!(simd_segments, scalar_segments);
    }

    proptest! {
        #[test]
        fn test_extract_rle_segments_simd_vs_scalar(
            pixels in prop::collection::vec(0..255u8, 128..1024),
            thresholds in prop::collection::vec(0..255u8, 128..1024)
        ) {
            let width = std::cmp::min(pixels.len(), thresholds.len());
            let height = 1;

            let mut pixels_map = pixels;
            pixels_map.truncate(width);

            let mut threshold_map = thresholds;
            threshold_map.truncate(width);

            let img = ImageView::new(&pixels_map, width, height, width).expect("Valid test image");

            let scalar_segments = extract_rle_segments_scalar(&img, &threshold_map);
            let simd_segments = extract_rle_segments(&img, &threshold_map);

            prop_assert_eq!(simd_segments, scalar_segments);
        }
    }
}
