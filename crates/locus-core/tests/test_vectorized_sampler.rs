#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Tests for the SIMD-vectorized image sampler.
use locus_core::bench_api::*;
use locus_core::image::ImageView;

#[test]
fn test_vectorized_bilinear_interpolation() {
    let width = 32;
    let height = 32;
    let mut data = vec![0u8; width * height];
    // Fill with a gradient
    for y in 0..height {
        for x in 0..width {
            data[y * width + x] = (x + y) as u8;
        }
    }
    let img = ImageView::new(&data, width, height, width).expect("valid image");

    // Sample points (some integer, some sub-pixel)
    let x = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5];
    let y = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5];

    // Expected values using scalar bilinear
    let mut expected = [0.0f32; 8];
    for i in 0..8 {
        expected[i] = img.sample_bilinear(f64::from(x[i]), f64::from(y[i])) as f32;
    }

    let mut actual = [0.0f32; 8];
    sample_bilinear_v8(&img, &x, &y, &mut actual);

    for i in 0..8 {
        // We expect high precision agreement between f32 SIMD and f64 scalar math.
        // Tolerating 1e-5 for floating point precision differences.
        assert!(
            (actual[i] - expected[i]).abs() < 1e-5,
            "Mismatch at index {}: actual={}, expected={}",
            i,
            actual[i],
            expected[i]
        );
    }
}
