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

// =============================================================================
// 1-PIXEL BORDER TESTS (Segmentation Boundary Conditions)
// =============================================================================

/// Test that the detector does not panic when a tag's outer border touches
/// the image edge (pixel 0 or pixel width-1).
///
/// This tests the Union-Find segmentation and bilinear sampling at boundaries.
/// Expected behavior: Either detect the tag (robust) or gracefully reject it,
/// but NEVER panic due to underflow or out-of-bounds access.
#[test]
fn test_tag_touching_top_left_edge() {
    // Create a small canvas where the tag starts at pixel (0, 0)
    const SIZE: usize = 100;
    let mut data = vec![255u8; SIZE * SIZE];
    
    // Draw a simple tag pattern starting at (0, 0)
    // Black border on top and left edges
    let tag_dim = 50;
    for y in 0..tag_dim {
        for x in 0..tag_dim {
            // Black border
            if x < 5 || y < 5 || x >= tag_dim - 5 || y >= tag_dim - 5 {
                data[y * SIZE + x] = 0;
            }
        }
    }
    
    let img = ImageView::new(&data, SIZE, SIZE, SIZE).unwrap();
    let mut detector = Detector::new();
    
    // This should NOT panic - that's the main assertion
    let _detections = detector.detect(&img);
    // We don't require detection (no quiet zone), just no crash
}

/// Test tag touching bottom-right edge.
#[test]
fn test_tag_touching_bottom_right_edge() {
    const SIZE: usize = 100;
    let mut data = vec![255u8; SIZE * SIZE];
    
    let tag_dim = 50;
    let offset = SIZE - tag_dim;
    
    for y in offset..SIZE {
        for x in offset..SIZE {
            let rel_x = x - offset;
            let rel_y = y - offset;
            if rel_x < 5 || rel_y < 5 || rel_x >= tag_dim - 5 || rel_y >= tag_dim - 5 {
                data[y * SIZE + x] = 0;
            }
        }
    }
    
    let img = ImageView::new(&data, SIZE, SIZE, SIZE).unwrap();
    let mut detector = Detector::new();
    
    // Should NOT panic
    let _detections = detector.detect(&img);
}

/// Test with a real tag placed at the very edge of the image.
/// Uses synthetic tag generation, then crops to remove quiet zone.
#[test]
fn test_real_tag_no_quiet_zone() {
    // Generate a larger image with tag centered
    let (full_data, _) = locus_core::test_utils::generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        99,
        80,
        160,
        0.0,
    );
    
    // Crop to remove most of the quiet zone (aggressive crop)
    const CROP_SIZE: usize = 90;
    const CROP_OFFSET: usize = 35; // Start cropping at offset to catch tag edge
    
    let mut cropped = vec![255u8; CROP_SIZE * CROP_SIZE];
    for y in 0..CROP_SIZE {
        for x in 0..CROP_SIZE {
            let src_y = y + CROP_OFFSET;
            let src_x = x + CROP_OFFSET;
            if src_y < 160 && src_x < 160 {
                cropped[y * CROP_SIZE + x] = full_data[src_y * 160 + src_x];
            }
        }
    }
    
    let img = ImageView::new(&cropped, CROP_SIZE, CROP_SIZE, CROP_SIZE).unwrap();
    let mut detector = Detector::new();
    let options = DetectOptions::with_families(&[TagFamily::AprilTag36h11]);
    
    // Should NOT panic - crash check is the primary assertion
    let detections = detector.detect_with_options(&img, &options);
    
    // Note: Detection may or may not succeed depending on quiet zone requirements
    // The key is: NO PANIC from bilinear sampling or segmentation underflow
    println!("Edge tag detection result: {} tags found", detections.len());
}

/// Test bilinear sampling at exact image corners (x=0, y=0).
#[test]
fn test_bilinear_at_image_corners() {
    let data = vec![128u8; 10 * 10];
    let img = ImageView::new(&data, 10, 10, 10).unwrap();
    
    // These should not panic due to underflow
    let _ = img.sample_bilinear(0.0, 0.0);
    let _ = img.sample_bilinear(0.5, 0.5);
    let _ = img.sample_bilinear(8.9, 8.9); // Near bottom-right
    
    // With clamping, these edge cases should also work
    let _ = img.get_pixel(0, 0);
    let _ = img.get_pixel(9, 9);
    let _ = img.get_pixel(100, 100); // Should clamp, not panic
}
