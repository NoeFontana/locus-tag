#![allow(unsafe_code, clippy::cast_sign_loss)]
//! Pre-processing filters for the detection pipeline.
//!
//! This module provides gradient computation and sharpening
//! to enhance detection of small tags (<32px) by accentuating edges.

use crate::config::SharpeningMode;
use crate::image::ImageView;
use multiversion::multiversion;
use rayon::prelude::*;

/// Compute gradient magnitude map using Scharr operator.
///
/// The Scharr operator provides better rotational symmetry than Sobel,
/// making it more suitable for gradient-based adaptive window sizing.
///
/// # Parameters
/// - `img`: Input grayscale image
/// - `output`: Output buffer for gradient magnitude (normalized to [0, 255])
///
/// # Implementation Notes
/// - Uses 3x3 Scharr kernels for x and y directions
/// - Returns normalized gradient magnitude: sqrt(gx² + gy²)
/// - SIMD-optimized for modern architectures
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[expect(
    clippy::cast_sign_loss,
    reason = "gradient magnitude is a sum of absolute values, right-shifted and clamped to 0..=255, so it is provably non-negative before the as u8 cast"
)]
pub fn compute_gradient_map(img: &ImageView, output: &mut [u8]) {
    let w = img.width;
    let h = img.height;

    // Scharr kernels (better rotational symmetry than Sobel)
    // Gx kernel:          Gy kernel:
    //  -3   0   3         -3  -10  -3
    // -10   0  10          0    0   0
    //  -3   0   3          3   10   3

    // use rayon::prelude::*;

    // Process rows in parallel
    (0..h).into_par_iter().for_each(|y| {
        // SAFETY: `into_par_iter()` over `0..h` yields each `y` exactly once
        // across rayon workers, so the `y * w .. y * w + w` slice for each
        // `y` is disjoint from every other worker's slice. `output` has
        // length `h * w`, so `ptr.add(y * w)` and the resulting `w`-element
        // slice are in-bounds. The original `output` slice is borrowed
        // mutably for the duration of `par_iter`, so no other reader exists.
        let dst_row = unsafe {
            let ptr = output.as_ptr().cast_mut();
            std::slice::from_raw_parts_mut(ptr.add(y * w), w)
        };

        let y0 = y.saturating_sub(1);
        let y1 = y;
        let y2 = (y + 1).min(h - 1);

        let r0 = img.get_row(y0);
        let r1 = img.get_row(y1);
        let r2 = img.get_row(y2);

        // 1. Left border
        {
            let x = 0;
            let x0 = 0;
            let x1 = 0;
            let x2 = 1.min(w - 1);
            let p00 = i32::from(r0[x0]);
            let p01 = i32::from(r0[x1]);
            let p02 = i32::from(r0[x2]);
            let p10 = i32::from(r1[x0]);
            let p12 = i32::from(r1[x2]);
            let p20 = i32::from(r2[x0]);
            let p21 = i32::from(r2[x1]);
            let p22 = i32::from(r2[x2]);
            let gx = -3 * p00 + 3 * p02 - 10 * p10 + 10 * p12 - 3 * p20 + 3 * p22;
            let gy = -3 * p00 - 10 * p01 - 3 * p02 + 3 * p20 + 10 * p21 + 3 * p22;
            dst_row[x] = ((gx.abs() + gy.abs()) >> 4).min(255) as u8;
        }

        // 2. Interior (Vectorized with 4x unroll for SIMD autovectorization)
        if w > 2 {
            let interior_end = w - 1;
            let mut x = 1;

            // Process 4 pixels at a time
            while x + 3 < interior_end {
                // Prefetch rows for cache efficiency
                let x1 = x;
                let x2 = x + 1;
                let x3 = x + 2;
                let x4 = x + 3;

                // Load values once
                let r0_m1 = [
                    i32::from(r0[x1 - 1]),
                    i32::from(r0[x2 - 1]),
                    i32::from(r0[x3 - 1]),
                    i32::from(r0[x4 - 1]),
                ];
                let r0_p1 = [
                    i32::from(r0[x1 + 1]),
                    i32::from(r0[x2 + 1]),
                    i32::from(r0[x3 + 1]),
                    i32::from(r0[x4 + 1]),
                ];
                let r1_m1 = [
                    i32::from(r1[x1 - 1]),
                    i32::from(r1[x2 - 1]),
                    i32::from(r1[x3 - 1]),
                    i32::from(r1[x4 - 1]),
                ];
                let r1_p1 = [
                    i32::from(r1[x1 + 1]),
                    i32::from(r1[x2 + 1]),
                    i32::from(r1[x3 + 1]),
                    i32::from(r1[x4 + 1]),
                ];
                let r2_c = [
                    i32::from(r2[x1]),
                    i32::from(r2[x2]),
                    i32::from(r2[x3]),
                    i32::from(r2[x4]),
                ];
                let r0_c = [
                    i32::from(r0[x1]),
                    i32::from(r0[x2]),
                    i32::from(r0[x3]),
                    i32::from(r0[x4]),
                ];
                let r2_m1 = [
                    i32::from(r2[x1 - 1]),
                    i32::from(r2[x2 - 1]),
                    i32::from(r2[x3 - 1]),
                    i32::from(r2[x4 - 1]),
                ];
                let r2_p1 = [
                    i32::from(r2[x1 + 1]),
                    i32::from(r2[x2 + 1]),
                    i32::from(r2[x3 + 1]),
                    i32::from(r2[x4 + 1]),
                ];

                // Compute 4 gradients in parallel
                for i in 0..4 {
                    let gx = 3 * (r0_p1[i] - r0_m1[i])
                        + 10 * (r1_p1[i] - r1_m1[i])
                        + 3 * (r2_p1[i] - r2_m1[i]);
                    let gy = 3 * (r2_m1[i] - r0_m1[i])
                        + 10 * (r2_c[i] - r0_c[i])
                        + 3 * (r2_p1[i] - r0_p1[i]);
                    dst_row[x + i] = ((gx.abs() + gy.abs()) >> 4).min(255) as u8;
                }
                x += 4;
            }

            // Handle remaining pixels
            while x < interior_end {
                let gx = 3 * (i32::from(r0[x + 1]) - i32::from(r0[x - 1]))
                    + 10 * (i32::from(r1[x + 1]) - i32::from(r1[x - 1]))
                    + 3 * (i32::from(r2[x + 1]) - i32::from(r2[x - 1]));
                let gy = 3 * (i32::from(r2[x - 1]) - i32::from(r0[x - 1]))
                    + 10 * (i32::from(r2[x]) - i32::from(r0[x]))
                    + 3 * (i32::from(r2[x + 1]) - i32::from(r0[x + 1]));
                dst_row[x] = ((gx.abs() + gy.abs()) >> 4).min(255) as u8;
                x += 1;
            }
        }

        // 3. Right border
        if w > 1 {
            let x = w - 1;
            let x0 = w - 2;
            let x1 = w - 1;
            let x2 = w - 1;
            let p00 = i32::from(r0[x0]);
            let p01 = i32::from(r0[x1]);
            let p02 = i32::from(r0[x2]);
            let p10 = i32::from(r1[x0]);
            let p12 = i32::from(r1[x2]);
            let p20 = i32::from(r2[x0]);
            let p21 = i32::from(r2[x1]);
            let p22 = i32::from(r2[x2]);
            let gx = -3 * p00 + 3 * p02 - 10 * p10 + 10 * p12 - 3 * p20 + 3 * p22;
            let gy = -3 * p00 - 10 * p01 - 3 * p02 + 3 * p20 + 10 * p21 + 3 * p22;
            dst_row[x] = ((gx.abs() + gy.abs()) >> 4).min(255) as u8;
        }
    });
}

/// Sharpen one image row into `dst_row`, reading the three pre-fetched source
/// rows with edge replication at the horizontal borders.
///
/// `SHOOT_LIMITED` selects the output limiter, and is a const parameter so the
/// choice is resolved at compile time and the branch never enters the pixel
/// loop (both instantiations stay auto-vectorizable). `#[inline(always)]` is
/// load-bearing: it makes the body part of the caller's `multiversion` clone,
/// so it is compiled with that clone's target features.
///
/// - `false` — stock behaviour: clamp the Laplacian result to `0..=255`.
///   Over/undershoot (halos) around high-contrast edges is preserved.
/// - `true` — "shoot-limited": clamp to the min/max of the five samples that
///   produced it. Mathematically this is the stock filter followed by an
///   overshoot limiter; the result can never leave the local intensity range,
///   so sharpening cannot raise a tile's maximum (or lower its minimum) above
///   what the unsharpened image already contained.
#[inline(always)]
#[expect(
    clippy::needless_range_loop,
    clippy::cast_sign_loss,
    reason = "the x loop indexes several parallel rows (r0/r1/r2 and dst_row) by offset, not a single iterated slice; the sharpened value is clamped into 0..=255 before the as u8 cast"
)]
fn sharpen_row<const SHOOT_LIMITED: bool>(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    dst_row: &mut [u8],
    w: usize,
) {
    for x in 0..w {
        let x0 = x.saturating_sub(1);
        let x1 = x;
        let x2 = (x + 1).min(w - 1);

        // Sample from pre-fetched rows
        let p11 = i32::from(r1[x1]);
        let p01 = i32::from(r0[x1]);
        let p10 = i32::from(r1[x0]);
        let p12 = i32::from(r1[x2]);
        let p21 = i32::from(r2[x1]);

        // Apply sharpening: 5*center - (sum of 4 neighbors)
        let sharpened = 5 * p11 - (p01 + p10 + p12 + p21);
        let limited = if SHOOT_LIMITED {
            // Branchless min/max reduction over the 5-point neighbourhood.
            // Every sample is in 0..=255, so the clamp bounds are too and the
            // result needs no further 0..=255 clamp.
            let lo = p11.min(p01).min(p10).min(p12).min(p21);
            let hi = p11.max(p01).max(p10).max(p12).max(p21);
            sharpened.clamp(lo, hi)
        } else {
            sharpened.clamp(0, 255)
        };
        dst_row[x] = limited as u8;
    }
}

/// Apply a 3x3 Laplacian sharpening filter to enhance edges.
///
/// This filter uses the kernel:
/// ```text
/// [ 0  -1   0 ]
/// [-1   5  -1 ]
/// [ 0  -1   0 ]
/// ```
///
/// # Parameters
/// - `img`: Input grayscale image
/// - `output`: Output buffer (must be img.width * img.height)
/// - `mode`: output limiter — see [`SharpeningMode`].
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
pub(crate) fn laplacian_sharpen(img: &ImageView, output: &mut [u8], mode: SharpeningMode) {
    let w = img.width;
    let h = img.height;

    // use rayon::prelude::*;

    (0..h).into_par_iter().for_each(|y| {
        let y0 = y.saturating_sub(1);
        let y1 = y;
        let y2 = (y + 1).min(h - 1);

        let r0 = img.get_row(y0);
        let r1 = img.get_row(y1);
        let r2 = img.get_row(y2);

        // SAFETY: `into_par_iter()` over `0..h` yields each `y` exactly once
        // across rayon workers, so the `y * w .. y * w + w` slice for each
        // `y` is disjoint from every other worker's slice. `output` has
        // length `h * w`, so `ptr.add(y * w)` and the resulting `w`-element
        // slice are in-bounds. The original `output` slice is borrowed
        // mutably for the duration of `par_iter`, so no other reader exists.
        let dst_row = unsafe {
            let ptr = output.as_ptr().cast_mut();
            std::slice::from_raw_parts_mut(ptr.add(y * w), w)
        };

        // Dispatch once per row, not once per pixel.
        match mode {
            SharpeningMode::Standard => sharpen_row::<false>(r0, r1, r2, dst_row, w),
            SharpeningMode::ShootLimited => sharpen_row::<true>(r0, r1, r2, dst_row, w),
        }
    });
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::image::ImageView;

    #[test]
    fn test_gradient_map_detects_edges() {
        // Create test image with vertical and horizontal edges
        let width = 32;
        let height = 32;
        let mut data = vec![50u8; width * height];

        // Vertical edge at x=16
        for y in 0..height {
            for x in 16..width {
                data[y * width + x] = 200;
            }
        }

        let img = ImageView::new(&data, width, height, width).unwrap();
        let mut gradient = vec![0u8; width * height];

        compute_gradient_map(&img, &mut gradient);

        // Gradient should be high at edge (x=15,16)
        let edge_grad = gradient[16 * width + 15];
        assert!(edge_grad > 100, "Gradient too low at edge: {edge_grad}");

        // Gradient should be low in uniform regions
        let uniform_grad = gradient[16 * width + 8];
        assert!(
            uniform_grad < 20,
            "Gradient too high in uniform region: {uniform_grad}"
        );
    }

    #[test]
    fn test_gradient_map_border_handling() {
        // Test that corners and edges don't panic
        let width = 8;
        let height = 8;
        let data = vec![128u8; width * height];

        let img = ImageView::new(&data, width, height, width).unwrap();
        let mut gradient = vec![0u8; width * height];

        // Should not panic
        compute_gradient_map(&img, &mut gradient);

        // All gradients should be low for uniform image
        assert!(gradient.iter().all(|&g| g < 10));
    }

    /// Straight-line scalar reference for [`laplacian_sharpen`]. Deliberately
    /// written without `multiversion`, rayon or the const-generic row kernel so
    /// it can act as the parity oracle for the auto-vectorized clones.
    fn sharpen_reference(data: &[u8], w: usize, h: usize, mode: SharpeningMode) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let px = |xx: usize, yy: usize| i32::from(data[yy * w + xx]);
                let (x0, x2) = (x.saturating_sub(1), (x + 1).min(w - 1));
                let (y0, y2) = (y.saturating_sub(1), (y + 1).min(h - 1));
                let s = [px(x, y), px(x, y0), px(x0, y), px(x2, y), px(x, y2)];
                let sharp = 5 * s[0] - (s[1] + s[2] + s[3] + s[4]);
                let v = match mode {
                    SharpeningMode::Standard => sharp.clamp(0, 255),
                    SharpeningMode::ShootLimited => {
                        let lo = s.iter().copied().min().unwrap();
                        let hi = s.iter().copied().max().unwrap();
                        sharp.clamp(lo, hi)
                    },
                };
                out[y * w + x] = u8::try_from(v).unwrap();
            }
        }
        out
    }

    /// Deterministic pseudo-random image (xorshift) — no `rand` dependency,
    /// which `docs/engineering/constraints.md` §5 forbids outside dev-deps.
    fn pseudo_random_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..w * h)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s & 0xFF) as u8
            })
            .collect()
    }

    /// The dispatched (SIMD-cloned, rayon-parallel) kernel must be bit-exact
    /// against the scalar reference for both modes, including the replicated
    /// borders and a width that is not a multiple of any vector lane count.
    #[test]
    fn test_sharpen_simd_matches_scalar_reference() {
        for (w, h) in [(67usize, 33usize), (1, 1), (2, 5), (64, 8)] {
            let data = pseudo_random_image(w, h, 0x9E37_79B9);
            let img = ImageView::new(&data, w, h, w).unwrap();
            for mode in [SharpeningMode::Standard, SharpeningMode::ShootLimited] {
                let mut got = vec![0u8; w * h];
                laplacian_sharpen(&img, &mut got, mode);
                assert_eq!(
                    got,
                    sharpen_reference(&data, w, h, mode),
                    "SIMD/scalar mismatch at {w}x{h} in {mode:?}"
                );
            }
        }
    }

    /// The whole point of `ShootLimited`: no output pixel may leave the
    /// intensity range of the five samples that produced it, so sharpening can
    /// never raise a neighbourhood's maximum or lower its minimum.
    #[test]
    fn test_shoot_limited_never_overshoots() {
        let (w, h) = (48usize, 48usize);
        let data = pseudo_random_image(w, h, 0x1234_5678);
        let img = ImageView::new(&data, w, h, w).unwrap();
        let mut out = vec![0u8; w * h];
        laplacian_sharpen(&img, &mut out, SharpeningMode::ShootLimited);

        for y in 0..h {
            for x in 0..w {
                let px = |xx: usize, yy: usize| data[yy * w + xx];
                let (x0, x2) = (x.saturating_sub(1), (x + 1).min(w - 1));
                let (y0, y2) = (y.saturating_sub(1), (y + 1).min(h - 1));
                let s = [px(x, y), px(x, y0), px(x0, y), px(x2, y), px(x, y2)];
                let lo = s.iter().copied().min().unwrap();
                let hi = s.iter().copied().max().unwrap();
                let v = out[y * w + x];
                assert!(v >= lo && v <= hi, "({x},{y}) = {v} outside {lo}..={hi}");
            }
        }
    }

    /// `ShootLimited` still sharpens: on a blurred edge the transition must get
    /// steeper than the input, it just saturates at the local extremes.
    #[test]
    fn test_shoot_limited_still_steepens_edges() {
        let (w, h) = (16usize, 16usize);
        // Blurred (non-linear) edge: a linear ramp has a zero Laplacian and
        // would be left untouched, so the transition curves at both knees.
        let mut data = vec![60u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = match x {
                    0..=5 => 60,
                    6 => 90,
                    7 => 150,
                    _ => 180,
                };
            }
        }
        let img = ImageView::new(&data, w, h, w).unwrap();
        let mut out = vec![0u8; w * h];
        laplacian_sharpen(&img, &mut out, SharpeningMode::ShootLimited);

        let row = h / 2;
        // Dark side of the ramp is pulled down, bright side pulled up.
        assert!(out[row * w + 6] < data[row * w + 6]);
        assert!(out[row * w + 7] > data[row * w + 7]);
        // …but neither escapes the flat plateaus that bound the ramp.
        assert!(out.iter().all(|&v| (60..=180).contains(&v)));
    }

    #[test]
    fn test_laplacian_sharpen_enhances_edges() {
        let width = 8;
        let height = 8;
        let mut data = vec![100u8; width * height];
        // Create a horizontal line (edge)
        for x in 0..width {
            data[4 * width + x] = 200;
        }

        let img = ImageView::new(&data, width, height, width).unwrap();
        let mut output = vec![0u8; width * height];

        laplacian_sharpen(&img, &mut output, SharpeningMode::Standard);

        // The value at 4,4 (center of edge) should be higher than 200 due to sharpening
        // Center is 200, neighbors are 100 (top), 200 (left), 200 (right), 100 (bottom)
        // 5*200 - (100 + 200 + 200 + 100) = 1000 - 600 = 400 -> 255
        assert!(output[4 * width + 4] > 200);

        // The value at 3,4 (just above edge) should be lower than 100
        // Center is 100, neighbors are 100, 100, 100, 200
        // 5*100 - (100 + 100 + 100 + 200) = 500 - 500 = 0
        assert_eq!(output[3 * width + 4], 0);
    }
}
