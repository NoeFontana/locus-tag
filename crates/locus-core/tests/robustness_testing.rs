//! Robustness tests for edge cases and hostile production environments.
//!
//! These tests verify the detector handles:
//! - Strided/padded camera buffers (64-byte aligned rows)
//! - Malformed inputs
//! - Boundary conditions

use locus_core::image::ImageView;
use locus_core::{
    Detector,
    config::{DetectOptions, TagFamily},
};
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
        42,    // tag ID
        80,    // tag size
        WIDTH, // canvas size = width (for easy copying)
        0.0,   // no noise
    );

    // Copy tag data row-by-row respecting stride
    for y in 0..HEIGHT {
        let src_start = y * WIDTH;
        let dst_start = y * STRIDE;
        buffer[dst_start..dst_start + WIDTH]
            .copy_from_slice(&tag_data[src_start..src_start + WIDTH]);
        // Padding bytes remain random garbage - this is intentional
    }

    // Create ImageView with explicit stride
    let img =
        ImageView::new(&buffer, WIDTH, HEIGHT, STRIDE).expect("Failed to create strided ImageView");

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
        buffer[y * STRIDE..y * STRIDE + WIDTH]
            .copy_from_slice(&tag_data[y * WIDTH..(y + 1) * WIDTH]);
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
    let _detections = detector.detect_with_options(&img, &options);
    // Note: Detection may or may not succeed depending on quiet zone requirements
    // The key is: NO PANIC from bilinear sampling or segmentation underflow
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

// =============================================================================
// DEGENERATE QUAD TESTS (Geometric Stability)
// =============================================================================

/// Test that the homography solver gracefully rejects collinear corners.
///
/// When 3+ corners are collinear, the DLT matrix becomes singular.
/// This must return None, not panic or produce NaN.
#[test]
fn test_homography_collinear_corners() {
    use locus_core::decoder::Homography;

    // All points on a line (collinear)
    let collinear_quad: [[f64; 2]; 4] = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]];

    let result = Homography::square_to_quad(&collinear_quad);
    assert!(
        result.is_none(),
        "Homography should return None for collinear corners, not panic"
    );
}

/// Test that the homography solver handles zero-area "line" polygons.
#[test]
fn test_homography_zero_area_line() {
    use locus_core::decoder::Homography;

    // Two pairs of identical points (degenerate line)
    let zero_area: [[f64; 2]; 4] = [[0.0, 0.0], [100.0, 0.0], [100.0, 0.0], [0.0, 0.0]];

    let result = Homography::square_to_quad(&zero_area);
    assert!(
        result.is_none(),
        "Homography should return None for zero-area polygon"
    );
}

/// Test that the homography solver handles self-intersecting "bowtie" polygons.
#[test]
fn test_homography_bowtie_self_intersecting() {
    use locus_core::decoder::Homography;

    // Bowtie: corners cross over each other
    let bowtie: [[f64; 2]; 4] = [
        [0.0, 0.0],     // TL
        [100.0, 100.0], // BR (crossed!)
        [100.0, 0.0],   // TR
        [0.0, 100.0],   // BL (crossed!)
    ];

    let result = Homography::square_to_quad(&bowtie);
    // May return Some (mathematically valid homography) or None (rejected)
    // The key is: NO PANIC, and if Some, no NaN in matrix
    if let Some(h) = result {
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    !h.h[(i, j)].is_nan(),
                    "Homography matrix contains NaN at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}

/// Test that homography handles very small quads (near-zero area).
#[test]
fn test_homography_tiny_quad() {
    use locus_core::decoder::Homography;

    // Extremely small quad (sub-pixel)
    let tiny: [[f64; 2]; 4] = [
        [50.0, 50.0],
        [50.001, 50.0],
        [50.001, 50.001],
        [50.0, 50.001],
    ];

    // Should not panic - may return None or unstable homography
    let result = Homography::square_to_quad(&tiny);
    if let Some(h) = result {
        // Check no NaN
        for i in 0..3 {
            for j in 0..3 {
                assert!(!h.h[(i, j)].is_nan(), "Homography contains NaN");
                assert!(!h.h[(i, j)].is_infinite(), "Homography contains Inf");
            }
        }
    }
}

/// Test that duplicate corners are handled gracefully.
#[test]
fn test_homography_duplicate_corners() {
    use locus_core::decoder::Homography;

    // All four corners are the same point
    let degenerate: [[f64; 2]; 4] = [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0], [50.0, 50.0]];

    let result = Homography::square_to_quad(&degenerate);
    assert!(
        result.is_none(),
        "Homography should return None for degenerate point"
    );
}

// =============================================================================
// WHITE NOISE STRESS TEST (False Positive Rejection)
// =============================================================================

/// Test that the detector does not hallucinate tags in pure random noise.
///
/// In low-light conditions, camera gain creates grain. A "high recall" detector
/// that is too sensitive will find false positives in this noise.
///
/// This test generates 100 random noise images and asserts 0 total detections.
#[test]
fn test_no_false_positives_in_white_noise() {
    use rand::prelude::*;

    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const NUM_IMAGES: usize = 100;

    let mut rng = thread_rng();
    let mut total_false_positives = 0;
    let mut false_positive_images: Vec<usize> = Vec::new();

    for image_idx in 0..NUM_IMAGES {
        // Generate pure uniform random noise
        let mut noise_data = vec![0u8; WIDTH * HEIGHT];
        rng.fill_bytes(&mut noise_data);

        let img = ImageView::new(&noise_data, WIDTH, HEIGHT, WIDTH).unwrap();
        let mut detector = Detector::new();
        let options = DetectOptions::with_families(&[TagFamily::AprilTag36h11]);

        let detections = detector.detect_with_options(&img, &options);

        if !detections.is_empty() {
            total_false_positives += detections.len();
            false_positive_images.push(image_idx);
        }
    }

    assert_eq!(
        total_false_positives, 0,
        "Detector found {} false positive(s) in {} random noise images. \
         Images with false positives: {:?}. \
         Consider tightening quad_min_edge_score or decoder Hamming threshold.",
        total_false_positives, NUM_IMAGES, false_positive_images
    );
}

/// Test false positive rejection in structured noise (gradient patterns).
#[test]
fn test_no_false_positives_in_gradient_noise() {
    const WIDTH: usize = 256;
    const HEIGHT: usize = 256;

    // Horizontal gradient
    let mut gradient = vec![0u8; WIDTH * HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            gradient[y * WIDTH + x] = x as u8;
        }
    }

    let img = ImageView::new(&gradient, WIDTH, HEIGHT, WIDTH).unwrap();
    let mut detector = Detector::new();
    let detections = detector.detect(&img);

    assert!(
        detections.is_empty(),
        "Detector found {} false positive(s) in gradient image",
        detections.len()
    );

    // Diagonal gradient
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            gradient[y * WIDTH + x] = ((x + y) / 2) as u8;
        }
    }

    let img = ImageView::new(&gradient, WIDTH, HEIGHT, WIDTH).unwrap();
    let detections = detector.detect(&img);

    assert!(
        detections.is_empty(),
        "Detector found {} false positive(s) in diagonal gradient",
        detections.len()
    );
}

// =============================================================================
// GRADIENT SHADOW TEST (Adaptive Threshold Verification)
// =============================================================================

/// Test that adaptive thresholding handles extreme dynamic range within a frame.
///
/// This verifies the local window logic hasn't regressed to global mean.
/// A global threshold would fail because the dark-zone tag and bright-zone tag
/// have completely different intensity ranges.
///
/// Setup:
/// - Background: Linear gradient 0→255 across x-axis
/// - Dark zone (x≈80): Tag with black=0, white=50 (good local contrast)
/// - Bright zone (x≈width-80): Tag with black=200, white=255 (good local contrast)
///
/// Assertion: BOTH tags must be detected.
#[test]
fn test_adaptive_threshold_gradient_shadow() {
    const WIDTH: usize = 640;
    const HEIGHT: usize = 200;

    // Create gradient background (0→255 linearly across x-axis)
    let mut data = vec![0u8; WIDTH * HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            data[y * WIDTH + x] = ((x * 255) / (WIDTH - 1)) as u8;
        }
    }

    // Generate two real tags with different IDs
    let (dark_tag, _) = locus_core::test_utils::generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        10, // ID 10 for dark region
        50, // tag size
        60, // canvas size
        0.0,
    );

    let (bright_tag, _) = locus_core::test_utils::generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        20, // ID 20 for bright region
        50,
        60,
        0.0,
    );

    // Paste dark tag in dark region (left side, x≈80)
    // Remap intensities: 0→0, 255→50 (local contrast of 50 levels)
    let dark_offset_x = 50;
    let dark_offset_y = 70;
    for ty in 0..60 {
        for tx in 0..60 {
            let src_val = dark_tag[ty * 60 + tx];
            // Dark zone: black stays 0, white becomes 50
            let remapped = (src_val as u32 * 50 / 255) as u8;
            let x = dark_offset_x + tx;
            let y = dark_offset_y + ty;
            if x < WIDTH && y < HEIGHT {
                data[y * WIDTH + x] = remapped;
            }
        }
    }

    // Paste bright tag in bright region (right side, x≈width-80)
    // Remap intensities: 0→200, 255→255 (local contrast of 55 levels)
    let bright_offset_x = WIDTH - 110;
    let bright_offset_y = 70;
    for ty in 0..60 {
        for tx in 0..60 {
            let src_val = bright_tag[ty * 60 + tx];
            // Bright zone: black becomes 200, white stays 255
            let remapped = 200 + (src_val as u32 * 55 / 255) as u8;
            let x = bright_offset_x + tx;
            let y = bright_offset_y + ty;
            if x < WIDTH && y < HEIGHT {
                data[y * WIDTH + x] = remapped;
            }
        }
    }

    let img = ImageView::new(&data, WIDTH, HEIGHT, WIDTH).unwrap();
    let mut detector = Detector::new();
    let options = DetectOptions::with_families(&[TagFamily::AprilTag36h11]);
    let detections = detector.detect_with_options(&img, &options);

    let detected_ids: Vec<u32> = detections.iter().map(|d| d.id).collect();

    // STRICT ASSERTION: Both tags must be detected
    assert!(
        detected_ids.contains(&10),
        "Adaptive threshold FAILED: Dark-zone tag (ID 10) not detected. \
         Detected IDs: {:?}. This indicates the local window is too large or \
         threshold is regressing to global mean.",
        detected_ids
    );

    assert!(
        detected_ids.contains(&20),
        "Adaptive threshold FAILED: Bright-zone tag (ID 20) not detected. \
         Detected IDs: {:?}. This indicates the local window is too large or \
         threshold is regressing to global mean.",
        detected_ids
    );
}
