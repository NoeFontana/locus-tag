//! Integration test for ArUco 4x4_50 tag detection.
//!
//! This test validates the complete detection pipeline using synthetic ArUco tags,
//! matching the conditions used in Python benchmarks (`tests/test_config.py`).

use locus_core::config::TagFamily;
use locus_core::image::ImageView;
use locus_core::{DetectOptions, Detector};

/// Generate a synthetic ArUco 4x4_50 tag image.
///
/// Produces a tag image with white quiet zone border, matching the structure
/// of `cv2.aruco.generateImageMarker`.
///
/// # Arguments
/// * `id` - Tag ID (0-49 for ArUco 4x4_50)
/// * `tag_size` - Size of the tag in pixels (excluding quiet zone)
///
/// # Returns
/// Tuple of (image data, ground truth corners)
fn generate_aruco_tag(id: u16, tag_size: usize) -> (Vec<u8>, usize) {
    // Total size includes quiet zone (1 cell on each side)
    let grid_size = 6; // 1 border + 4 data + 1 border
    let cell_size = tag_size / grid_size;
    let total_size = cell_size * grid_size;

    let mut data = vec![255u8; total_size * total_size]; // White background

    let dict = &*locus_core::dictionaries::ARUCO_4X4_50;
    let code = dict.get_code(id).expect("Invalid ID for ArUco 4x4_50");

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let is_border = gx == 0 || gx == 5 || gy == 0 || gy == 5;

            let color = if is_border {
                0u8 // Black border
            } else {
                // Interior 4x4 data grid
                let ix = gx - 1;
                let iy = gy - 1;
                let bit_idx = iy * 4 + ix;
                if (code >> bit_idx) & 1 != 0 {
                    255u8
                } else {
                    0u8
                }
            };

            // Fill cell
            for py in 0..cell_size {
                for px in 0..cell_size {
                    let y = gy * cell_size + py;
                    let x = gx * cell_size + px;
                    data[y * total_size + x] = color;
                }
            }
        }
    }

    (data, total_size)
}

/// Apply a simple 3x3 box blur to simulate camera blur.
fn apply_box_blur(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut blurred = data.to_vec();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0u32;
            for dy in 0..3 {
                for dx in 0..3 {
                    sum += u32::from(data[(y + dy - 1) * width + (x + dx - 1)]);
                }
            }
            blurred[y * width + x] = (sum / 9) as u8;
        }
    }

    blurred
}

#[test]
fn test_aruco_4x4_50_detection() {
    const TAG_ID: u16 = 0;
    const TAG_SIZE: usize = 96; // Must be divisible by 6 for clean cells
    const CANVAS_SIZE: usize = 400;

    // Generate tag
    let (tag_data, tag_total_size) = generate_aruco_tag(TAG_ID, TAG_SIZE);

    // Create canvas with gray background
    let mut canvas = vec![128u8; CANVAS_SIZE * CANVAS_SIZE];

    // Center the tag on canvas
    let offset = (CANVAS_SIZE - tag_total_size) / 2;
    for y in 0..tag_total_size {
        for x in 0..tag_total_size {
            canvas[(offset + y) * CANVAS_SIZE + (offset + x)] = tag_data[y * tag_total_size + x];
        }
    }

    // Apply blur to match Python test conditions
    let blurred = apply_box_blur(&canvas, CANVAS_SIZE, CANVAS_SIZE);

    // Detect
    let img = ImageView::new(&blurred, CANVAS_SIZE, CANVAS_SIZE, CANVAS_SIZE)
        .expect("Failed to create ImageView");

    let mut detector = Detector::new();
    let options = DetectOptions {
        families: vec![TagFamily::ArUco4x4_50],
        ..Default::default()
    };

    let results = detector.detect_with_options(&img, &options);

    // Assertions
    assert_eq!(results.len(), 1, "Should detect exactly 1 ArUco tag");
    assert_eq!(results[0].id, u32::from(TAG_ID), "Detected ID should match");
    assert_eq!(
        results[0].hamming, 0,
        "Perfect detection should have 0 hamming distance"
    );
}

#[test]
fn test_aruco_multiple_ids() {
    const CANVAS_SIZE: usize = 400;
    const TAG_SIZE: usize = 96;

    let mut detector = Detector::new();
    let options = DetectOptions {
        families: vec![TagFamily::ArUco4x4_50],
        ..Default::default()
    };

    // Test several tag IDs
    for tag_id in [0u16, 1, 5, 10, 25, 49] {
        let (tag_data, tag_total_size) = generate_aruco_tag(tag_id, TAG_SIZE);

        let mut canvas = vec![128u8; CANVAS_SIZE * CANVAS_SIZE];
        let offset = (CANVAS_SIZE - tag_total_size) / 2;

        for y in 0..tag_total_size {
            for x in 0..tag_total_size {
                canvas[(offset + y) * CANVAS_SIZE + (offset + x)] =
                    tag_data[y * tag_total_size + x];
            }
        }

        let blurred = apply_box_blur(&canvas, CANVAS_SIZE, CANVAS_SIZE);
        let img = ImageView::new(&blurred, CANVAS_SIZE, CANVAS_SIZE, CANVAS_SIZE).unwrap();

        let results = detector.detect_with_options(&img, &options);

        assert_eq!(results.len(), 1, "Should detect tag ID {tag_id}");
        assert_eq!(
            results[0].id,
            u32::from(tag_id),
            "ID mismatch for tag {tag_id}"
        );
    }
}
