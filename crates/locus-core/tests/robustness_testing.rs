#![allow(
    missing_docs,
    clippy::panic,
    clippy::needless_range_loop,
    clippy::unwrap_used,
    clippy::unreadable_literal
)]
//! Robustness tests for edge cases and hostile production environments.
//!
//! These tests verify the detector handles:
//! - Strided/padded camera buffers (64-byte aligned rows)
//! - Malformed inputs
//! - Boundary conditions

use locus_core::decoder::Homography;
use locus_core::image::ImageView;
use locus_core::{
    Detector,
    config::{DetectOptions, TagFamily},
};
use proptest::prelude::*;
use rand::RngCore;
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
    const STRIDE: usize = WIDTH.div_ceil(64) * 64; // 192

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
    // Crop to remove most of the quiet zone (aggressive crop)
    const CROP_SIZE: usize = 90;
    const CROP_OFFSET: usize = 35; // Start cropping at offset to catch tag edge

    // Generate a larger image with tag centered
    let (full_data, _) = locus_core::test_utils::generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        99,
        80,
        160,
        0.0,
    );

    // Crop to remove most of the quiet zone (aggressive crop)

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
                    "Homography matrix contains NaN at ({i}, {j})"
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
        "Detector found {total_false_positives} false positive(s) in {NUM_IMAGES} random noise images. \
         Images with false positives: {false_positive_images:?}. \
         Consider tightening quad_min_edge_score or decoder Hamming threshold."
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
            gradient[y * WIDTH + x] = usize::midpoint(x, y) as u8;
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
            let remapped = (u32::from(src_val) * 50 / 255) as u8;
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
            let remapped = 200 + (u32::from(src_val) * 55 / 255) as u8;
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
         Detected IDs: {detected_ids:?}. This indicates the local window is too large or \
         threshold is regressing to global mean."
    );

    assert!(
        detected_ids.contains(&20),
        "Adaptive threshold FAILED: Bright-zone tag (ID 20) not detected. \
         Detected IDs: {detected_ids:?}. This indicates the local window is too large or \
         threshold is regressing to global mean."
    );
}

// =============================================================================
// HOMOGRAPHY ROBUSTNESS (Proptests)
// =============================================================================

/// Strategy for a single 2D point [x, y] in a reasonable range.
fn point_strategy() -> impl Strategy<Value = [f64; 2]> {
    prop_oneof![
        // Standard coordinates (e.g. 640x480 or 4k)
        [(-2000.0..2000.0), (-2000.0..2000.0)],
        // Coordinates near 0
        [(-1.0..1.0), (-1.0..1.0)],
    ]
}

/// Strategy for invalid numerical values (NaN, Inf).
fn invalid_point_strategy() -> impl Strategy<Value = [[f64; 2]; 4]> {
    prop::collection::vec(
        prop_oneof![
            Just(f64::NAN),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
            (-1e10..1e10)
        ],
        8,
    )
    .prop_map(|v| [[v[0], v[1]], [v[2], v[3]], [v[4], v[5]], [v[6], v[7]]])
}

/// Helper to check if a quad is convex and CCW.
/// A quad (p0, p1, p2, p3) is convex if all internal angles are < 180 deg
/// and it's CCW if cross products of consecutive edges are positive.
fn is_convex_ccw(pts: &[[f64; 2]; 4]) -> bool {
    let mut cross_products = Vec::with_capacity(4);
    for i in 0..4 {
        let p0 = pts[i];
        let p1 = pts[(i + 1) % 4];
        let p2 = pts[(i + 2) % 4];

        let dx1 = p1[0] - p0[0];
        let dy1 = p1[1] - p0[1];
        let dx2 = p2[0] - p1[0];
        let dy2 = p2[1] - p1[1];

        let cp = dx1 * dy2 - dy1 * dx2;
        cross_products.push(cp);
    }

    // All cross products must have the same sign (convex)
    // and must be positive for CCW.
    cross_products.iter().all(|&cp| cp > 1e-9)
}

/// Strategy for a valid (convex, CCW) quadrilateral.
fn valid_quad_strategy() -> impl Strategy<Value = [[f64; 2]; 4]> {
    prop::collection::vec(point_strategy(), 4)
        .prop_map(|v| {
            let mut pts = [[0.0; 2]; 4];
            pts.copy_from_slice(&v);
            pts
        })
        .prop_filter("Must be convex and CCW", is_convex_ccw)
}

/// Strategy for degenerate quads (strictly collinear or coincident).
fn collinear_quad_strategy() -> impl Strategy<Value = [[f64; 2]; 4]> {
    prop_oneof![
        // Coincident points: p0 == p1
        point_strategy().prop_map(|p| [p, p, [p[0] + 10.0, p[1]], [p[0], p[1] + 10.0]]),
        // Collinear points: p0, p1, p2 on the same line (y = p[1])
        point_strategy().prop_map(|p| {
            [
                p,
                [p[0] + 10.0, p[1]],
                [p[0] + 20.0, p[1]],
                [p[0] + 30.0, p[1]],
            ]
        }),
        // All points same
        point_strategy().prop_map(|p| [p, p, p, p]),
    ]
}

proptest! {
    /// Property: Homography correctly projects source points to destination points.
    #[test]
    fn test_homography_reprojection(
        src in valid_quad_strategy(),
        dst in valid_quad_strategy()
    ) {
        if let Some(h) = Homography::from_pairs(&src, &dst) {
            for i in 0..4 {
                let p_proj = h.project(src[i]);
                let err_x = (p_proj[0] - dst[i][0]).abs();
                let err_y = (p_proj[1] - dst[i][1]).abs();

                // Tolerance must scale with the magnitude of the coordinates.
                // For points ~2000, 1e-4 is reasonably tight for unnormalized DLT.
                let max_coord = dst[i][0].abs().max(dst[i][1].abs()).max(1.0);
                let tol = 1e-7 * max_coord;

                assert!(err_x < tol, "Reprojection error X too large: {err_x} at point {i} (tol: {tol})");
                assert!(err_y < tol, "Reprojection error Y too large: {err_y} at point {i} (tol: {tol})");
            }
        }
    }

    /// Property: square_to_quad is consistent with from_pairs using the unit square.
    #[test]
    fn test_square_to_quad_consistency(
        dst in valid_quad_strategy()
    ) {
        let src = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
        ];

        let h1 = Homography::from_pairs(&src, &dst);
        let h2 = Homography::square_to_quad(&dst);

        match (h1.as_ref(), h2.as_ref()) {
            (Some(h1_ref), Some(h2_ref)) => {
                // Test a sample point
                let p = [0.5, -0.2];
                let p1 = h1_ref.project(p);
                let p2 = h2_ref.project(p);

                assert!((p1[0] - p2[0]).abs() < 1e-9);
                assert!((p1[1] - p2[1]).abs() < 1e-9);
            },
            (None, None) => {},
            _ => panic!("Consistency mismatch between from_pairs and square_to_quad"),
        }
    }

    /// Property: Collinear points (Phase 2: Line Test) MUST return None.
    #[test]
    fn test_homography_line_test_singularity(
        src in collinear_quad_strategy(),
        dst in valid_quad_strategy()
    ) {
        let h = Homography::from_pairs(&src, &dst);

        // Rigorous check: Collinear source points should NOT result in a valid homography.
        // If it returns Some, it's a "ghost" solution due to numerical noise in LU.
        if let Some(_h_val) = h {
            // Check reprojection error - it should be high or it's a miracle
            // Actually, we expect it to be None.
            panic!("DLT returned a solution for collinear points: {src:?}");
        }
    }

    /// Property: Invalid numbers (NaN, Inf) never panic (Phase 2: NaN Check).
    #[test]
    fn test_homography_nan_safety(
        src in invalid_point_strategy(),
        dst in invalid_point_strategy()
    ) {
        // This should never panic
        let _ = Homography::from_pairs(&src, &dst);
        let _ = Homography::square_to_quad(&src);
    }

    /// Property: Extreme slant (Phase 3: Horizon Effect).
    /// Simulates a 1x1m tag at 10m distance, rotated by 80-85 degrees.
    #[test]
    fn test_homography_extreme_slant_precision(
        slant_deg in 80.0..85.0f64,
        z_dist in 5.0..15.0f64
    ) {
        // Intrinsics: 640x480 camera
        let fx = 600.0;
        let fy = 600.0;
        let cx = 320.0;
        let cy = 240.0;

        // Tag corners in 3D (1x1m centered at origin)
        let world_corners = [
            [-0.5, -0.5, 0.0],
            [ 0.5, -0.5, 0.0],
            [ 0.5,  0.5, 0.0],
            [-0.5,  0.5, 0.0],
        ];

        // Rotation: slant around Y axis
        let theta = slant_deg.to_radians();
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Project corners to image
        let mut img_corners = [[0.0; 2]; 4];
        for i in 0..4 {
            let wx = world_corners[i][0];
            let wy = world_corners[i][1];

            // Rotate around Y: x' = x*cos + z*sin, z' = -x*sin + z*cos
            // Translation: z' += z_dist
            let rx = wx * cos_t;
            let ry = wy;
            let rz = -wx * sin_t + z_dist;

            img_corners[i][0] = fx * (rx / rz) + cx;
            img_corners[i][1] = fy * (ry / rz) + cy;
        }

        // Target: canonical square
        let src_sq = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
        ];

        if let Some(h) = Homography::from_pairs(&src_sq, &img_corners) {
            // Project center (0, 0)
            let p_center = h.project([0.0, 0.0]);

            // Ground truth center projection
            let g_cx = fx * (0.0 / z_dist) + cx;
            let g_cy = fy * (0.0 / z_dist) + cy;

            let err_x = (p_center[0] - g_cx).abs();
            let err_y = (p_center[1] - g_cy).abs();
            let err_total = (err_x * err_x + err_y * err_y).sqrt();

            // Phase 3: Precision Gate - Error must be < 0.5 pixels
            assert!(err_total < 0.5, "Extreme slant ({slant_deg:.1} deg) center error too large: {err_total:.4}px");
        }
    }

    /// Property: Optimizer Cross-Check (Phase 3).
    /// square_to_quad must be identical to from_pairs(unit_square, dst).
    #[test]
    fn test_homography_optimizer_cross_check(
        dst in valid_quad_strategy()
    ) {
        let src_sq = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
        ];

        let h_gen = Homography::from_pairs(&src_sq, &dst);
        let h_opt = Homography::square_to_quad(&dst);

        match (h_gen, h_opt) {
            (Some(g), Some(o)) => {
                for i in 0..3 {
                    for j in 0..3 {
                        let diff = (g.h[(i, j)] - o.h[(i, j)]).abs();
                        assert!(diff < 1e-6, "Optimizer divergence at ({i}, {j}): diff={diff:.2e}");
                    }
                }
            },
            (None, None) => {},
            _ => panic!("Solver consensus failed between DLT and optimized solver"),
        }
    }

    /// Property: Monte Carlo Noise Injection (Phase 4: Noise Gate).
    /// Verifies that small input noise doesn't cause disproportionate output error.
    #[test]
    fn test_homography_noise_robustness(
        // Noise sigma in pixels
        noise_sigma in 0.01..0.2f64,
        // Canonical points scale (to simulate realistic pixel coordinates)
        scale in 100.0..1000.0f64
    ) {
        let src_sq = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
        ];

        // Ground truth destination: a slightly rotated and translated quad
        let center_x = 500.0;
        let center_y = 500.0;
        let angle: f64 = 0.2; // some rotation
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let mut dst_gt = [[0.0; 2]; 4];
        for i in 0..4 {
            let x = src_sq[i][0] * scale;
            let y = src_sq[i][1] * scale;
            dst_gt[i][0] = center_x + x * cos_a - y * sin_a;
            dst_gt[i][1] = center_y + x * sin_a + y * cos_a;
        }

        let h_gt = Homography::from_pairs(&src_sq, &dst_gt).expect("GT must be valid");

        // Add Gaussian noise (approximated here for the test)
        // Since proptest doesn't have a built-in normal distribution strategist easily available here,
        // we'll use 8 uniform samples to approximate a normal distribution (Central Limit Theorem).
        let mut noisy_dst = dst_gt;
        let mut seed = 42u64;
        let mut pseudo_rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed as f64) / (u64::MAX as f64)
        };

        for i in 0..4 {
            // Box-Muller or CLT for noise
            let mut noise_x = 0.0;
            let mut noise_y = 0.0;
            for _ in 0..4 {
                noise_x += pseudo_rand() - 0.5;
                noise_y += pseudo_rand() - 0.5;
            }
            // Variance of sum of 4 U(-0.5, 0.5) is 4 * (1/12) = 1/3.
            // So scale by sqrt(3) * sigma to get N(0, sigma^2)
            noisy_dst[i][0] += noise_x * 1.732 * noise_sigma;
            noisy_dst[i][1] += noise_y * 1.732 * noise_sigma;
        }

        if let Some(h_noisy) = Homography::from_pairs(&src_sq, &noisy_dst) {
            // Check internal grid of points
            let k = 3.5; // Error amplification factor threshold
            let max_err_allowed = k * noise_sigma;

            for gx in -5..=5 {
                for gy in -5..=5 {
                    let p = [f64::from(gx) * 0.2, f64::from(gy) * 0.2];
                    let p_gt = h_gt.project(p);
                    let p_noisy = h_noisy.project(p);

                    let err_x = (p_gt[0] - p_noisy[0]).abs();
                    let err_y = (p_gt[1] - p_noisy[1]).abs();
                    let err = (err_x * err_x + err_y * err_y).sqrt();

                    assert!(err < max_err_allowed,
                        "Noise amplification too high: err={err:.4}px > {max_err_allowed:.4}px (k*sigma) at point {p:?}");
                }
            }
        }
    }
}
