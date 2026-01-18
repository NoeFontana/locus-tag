//! Robustness tests for edge cases and hostile production environments.
//!
//! These tests verify the detector handles:
//! - Strided/padded camera buffers (64-byte aligned rows)
//! - Malformed inputs
//! - Boundary conditions

use locus_core::{Detector, config::{DetectOptions, TagFamily}};
use locus_core::image::ImageView;
use rand::prelude::*;

/// Test that the detection pipeline correctly handles strided buffers.
///
/// Camera drivers often return buffers with padding for alignment (e.g., 64-byte rows).
/// If kernels use `width` instead of `stride`, they will read garbage padding bytes
/// as the start of the next row, causing detection failures.
#[test]
fn test_strided_buffer_detection() {
    const WIDTH: usize = 200;
    const HEIGHT: usize = 200;
    const PADDING: usize = 13; // Non-power-of-2 to catch alignment assumptions
    const STRIDE: usize = WIDTH + PADDING;
    
    let mut rng = thread_rng();
    
    // Create buffer with random garbage (simulating uninitialized padding)
    let mut buffer = vec![0u8; HEIGHT * STRIDE];
    rng.fill_bytes(&mut buffer);
    
    // Generate a synthetic tag and copy it into the strided buffer
    let (tag_data, _gt_corners) = locus_core::test_utils::generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        42, // tag ID
        80, // tag size
        WIDTH, // canvas size = width (for easy copying)
        0.0, // no noise
    );
    
    // Copy tag data row-by-row respecting stride
    for y in 0..HEIGHT {
        let src_start = y * WIDTH;
        let dst_start = y * STRIDE;
        buffer[dst_start..dst_start + WIDTH].copy_from_slice(&tag_data[src_start..src_start + WIDTH]);
        // Padding bytes remain random garbage - this is intentional
    }
    
    // Create ImageView with explicit stride
    let img = ImageView::new(&buffer, WIDTH, HEIGHT, STRIDE)
        .expect("Failed to create strided ImageView");
    
    // Run detection
    let mut detector = Detector::new();
    let options = DetectOptions::with_families(&[TagFamily::AprilTag36h11]);
    let detections = detector.detect_with_options(&img, &options);
    
    // Assert: Tag must be detected
    assert!(
        !detections.is_empty(),
        "Strided buffer test FAILED: No tags detected. \
         This likely means a kernel is using width instead of stride."
    );
    
    // Assert: Correct tag ID
    assert!(
        detections.iter().any(|d| d.id == 42),
        "Strided buffer test FAILED: Expected tag ID 42, got {:?}",
        detections.iter().map(|d| d.id).collect::<Vec<_>>()
    );
}

/// Test 64-byte aligned stride (common GPU/camera buffer alignment).
#[test]
fn test_64_byte_aligned_stride() {
    const WIDTH: usize = 150;
    const HEIGHT: usize = 150;
    // Round up to next 64-byte boundary
    const STRIDE: usize = ((WIDTH + 63) / 64) * 64; // 192
    
    let mut rng = thread_rng();
    let mut buffer = vec![0u8; HEIGHT * STRIDE];
    rng.fill_bytes(&mut buffer);
    
    let (tag_data, _) = locus_core::test_utils::generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        7,
        60,
        WIDTH,
        0.0,
    );
    
    // Copy with stride
    for y in 0..HEIGHT {
        buffer[y * STRIDE..y * STRIDE + WIDTH].copy_from_slice(&tag_data[y * WIDTH..(y + 1) * WIDTH]);
    }
    
    let img = ImageView::new(&buffer, WIDTH, HEIGHT, STRIDE).unwrap();
    let mut detector = Detector::new();
    let detections = detector.detect(&img);
    
    assert!(
        detections.iter().any(|d| d.id == 7),
        "64-byte aligned stride test FAILED: Tag ID 7 not detected"
    );
}

/// Test minimum viable image size (edge condition).
#[test]
fn test_minimum_image_size() {
    // Very small image - should not panic
    let data = vec![128u8; 16 * 16];
    let img = ImageView::new(&data, 16, 16, 16).unwrap();
    
    let mut detector = Detector::new();
    // Should not panic, even if no tags found
    let _detections = detector.detect(&img);
}

/// Test that ImageView rejects invalid stride.
#[test]
fn test_invalid_stride_rejected() {
    let data = vec![0u8; 100];
    
    // Stride less than width should fail
    let result = ImageView::new(&data, 10, 10, 9);
    assert!(result.is_err(), "ImageView should reject stride < width");
}
