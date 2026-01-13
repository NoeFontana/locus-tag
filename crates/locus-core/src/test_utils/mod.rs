use rand::prelude::*;
use rand_distr::{Distribution, Normal};

/// Generate a synthetic image containing a single AprilTag or ArUco tag.
///
/// This generates a tag with a white quiet zone, placed on a white background,
/// matching the setup used in Python benchmarks.
/// rotation (rad), translation (x,y), scaling (pixels).
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc
)]
pub fn generate_synthetic_test_image(
    family: crate::config::TagFamily,
    id: u16,
    tag_size: usize,
    canvas_size: usize,
    noise_sigma: f32,
) -> (Vec<u8>, [[f64; 2]; 4]) {
    let mut data = vec![255u8; canvas_size * canvas_size];

    // Calculate tag position (centered)
    let margin = (canvas_size - tag_size) / 2;
    let quiet_zone = tag_size / 5;

    // Draw white quiet zone (optional since background is 255, but adds robustness)
    for y in margin.saturating_sub(quiet_zone)..(margin + tag_size + quiet_zone).min(canvas_size) {
        for x in
            margin.saturating_sub(quiet_zone)..(margin + tag_size + quiet_zone).min(canvas_size)
        {
            data[y * canvas_size + x] = 255;
        }
    }

    // Generate tag pattern (bits)
    let decoder = crate::decoder::family_to_decoder(family);
    let code = decoder.get_code(id).expect("Invalid tag ID for family");
    let dim = decoder.dimension();

    let cell_size = tag_size / (dim + 2); // dim + 2 for black border
    let actual_tag_size = cell_size * (dim + 2);
    let start_x = margin + (tag_size - actual_tag_size) / 2;
    let start_y = margin + (tag_size - actual_tag_size) / 2;

    // Draw black border
    for y in 0..(dim + 2) {
        for x in 0..(dim + 2) {
            if x == 0 || x == dim + 1 || y == 0 || y == dim + 1 {
                draw_cell(&mut data, canvas_size, start_x, start_y, x, y, cell_size, 0);
            } else {
                let row = y - 1;
                let col = x - 1;
                let bit = (code >> (row * dim + col)) & 1;
                let val = if bit != 0 { 255 } else { 0 };
                draw_cell(
                    &mut data,
                    canvas_size,
                    start_x,
                    start_y,
                    x,
                    y,
                    cell_size,
                    val,
                );
            }
        }
    }

    if noise_sigma > 0.0 {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, f64::from(noise_sigma)).expect("Invalid noise params");

        for pixel in &mut data {
            let noise = normal.sample(&mut rng) as i32;
            let val = (i32::from(*pixel) + noise).clamp(0, 255);
            *pixel = val as u8;
        }
    }

    let gt_corners = [
        [start_x as f64, start_y as f64],
        [(start_x + actual_tag_size) as f64, start_y as f64],
        [
            (start_x + actual_tag_size) as f64,
            (start_y + actual_tag_size) as f64,
        ],
        [start_x as f64, (start_y + actual_tag_size) as f64],
    ];

    (data, gt_corners)
}

#[allow(clippy::too_many_arguments)]
fn draw_cell(
    data: &mut [u8],
    stride: usize,
    start_x: usize,
    start_y: usize,
    cx: usize,
    cy: usize,
    size: usize,
    val: u8,
) {
    let px = start_x + cx * size;
    let py = start_y + cy * size;
    for y in py..(py + size) {
        for x in px..(px + size) {
            data[y * stride + x] = val;
        }
    }
}

/// Compute mean Euclidean distance between detected and ground truth corners.
/// Handles 4 rotations to find the minimum error.
#[must_use]
pub fn compute_corner_error(detected: &[[f64; 2]; 4], ground_truth: &[[f64; 2]; 4]) -> f64 {
    let mut min_error = f64::MAX;

    // Try all 4 rotations
    for rot in 0..4 {
        let mut sum_dist = 0.0;
        for i in 0..4 {
            let d = &detected[(i + rot) % 4];
            let g = &ground_truth[i];
            let dx = d[0] - g[0];
            let dy = d[1] - g[1];
            sum_dist += (dx * dx + dy * dy).sqrt();
        }
        let avg_dist = sum_dist / 4.0;
        if avg_dist < min_error {
            min_error = avg_dist;
        }
    }

    min_error
}

// ============================================================================
// ROBUSTNESS TEST UTILITIES
// ============================================================================

/// Parameters for generating test images with photometric variations.
#[derive(Clone, Debug)]
pub struct TestImageParams {
    /// Tag family to generate.
    pub family: crate::config::TagFamily,
    /// Tag ID to generate.
    pub id: u16,
    /// Tag size in pixels.
    pub tag_size: usize,
    /// Canvas size in pixels.
    pub canvas_size: usize,
    /// Gaussian noise standard deviation (0.0 = no noise).
    pub noise_sigma: f32,
    /// Brightness offset (-255 to +255).
    pub brightness_offset: i16,
    /// Contrast scale (1.0 = no change, 0.5 = reduce, 1.5 = increase).
    pub contrast_scale: f32,
}

impl Default for TestImageParams {
    fn default() -> Self {
        Self {
            family: crate::config::TagFamily::AprilTag36h11,
            id: 0,
            tag_size: 100,
            canvas_size: 320,
            noise_sigma: 0.0,
            brightness_offset: 0,
            contrast_scale: 1.0,
        }
    }
}

/// Generate a test image based on the provided parameters.
/// Includes tag generation, placement, and photometric adjustments.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn generate_test_image_with_params(params: &TestImageParams) -> (Vec<u8>, [[f64; 2]; 4]) {
    // First generate base image
    let (mut data, corners) = generate_synthetic_test_image(
        params.family,
        params.id,
        params.tag_size,
        params.canvas_size,
        params.noise_sigma,
    );

    // Apply brightness and contrast adjustments
    if params.brightness_offset != 0 || (params.contrast_scale - 1.0).abs() > 0.001 {
        apply_brightness_contrast(
            &mut data,
            i32::from(params.brightness_offset),
            params.contrast_scale,
        );
    }

    (data, corners)
}

/// Apply brightness and contrast to an image.
/// `brightness`: -255 to +255
/// `contrast`: 0.0 to 127.0 (1.0 = no change)
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub fn apply_brightness_contrast(image: &mut [u8], brightness: i32, contrast: f32) {
    for pixel in image.iter_mut() {
        let b = f32::from(*pixel);
        let with_contrast = (b - 128.0) * contrast + 128.0;
        let with_brightness = with_contrast as i32 + brightness;
        *pixel = with_brightness.clamp(0, 255) as u8;
    }
}

/// Count black pixels in binary data.
#[must_use]
#[allow(clippy::naive_bytecount)]
pub fn count_black_pixels(data: &[u8]) -> usize {
    data.iter().filter(|&&p| p == 0).count()
}

/// Check if the tag's outer black border is correctly binarized.
/// Returns the ratio of correctly black pixels in the 1-cell-wide border (0.0 to 1.0).
#[must_use]
#[allow(clippy::cast_sign_loss)]
pub fn measure_border_integrity(binary: &[u8], width: usize, corners: &[[f64; 2]; 4]) -> f64 {
    let min_x = corners
        .iter()
        .map(|c| c[0])
        .fold(f64::MAX, f64::min)
        .max(0.0) as usize;
    let max_x = corners
        .iter()
        .map(|c| c[0])
        .fold(f64::MIN, f64::max)
        .max(0.0) as usize;
    let min_y = corners
        .iter()
        .map(|c| c[1])
        .fold(f64::MAX, f64::min)
        .max(0.0) as usize;
    let max_y = corners
        .iter()
        .map(|c| c[1])
        .fold(f64::MIN, f64::max)
        .max(0.0) as usize;

    let height = binary.len() / width;
    let min_x = min_x.min(width.saturating_sub(1));
    let max_x = max_x.min(width.saturating_sub(1));
    let min_y = min_y.min(height.saturating_sub(1));
    let max_y = max_y.min(height.saturating_sub(1));

    if max_x <= min_x || max_y <= min_y {
        return 0.0;
    }

    let tag_width = max_x - min_x;
    let tag_height = max_y - min_y;

    // For AprilTag 36h11: 8 cells total, border is 1 cell = 1/8 of tag
    let cell_size_x = tag_width / 8;
    let cell_size_y = tag_height / 8;

    if cell_size_x == 0 || cell_size_y == 0 {
        return 0.0;
    }

    let mut black_count = 0usize;
    let mut total_count = 0usize;

    // Top border row
    for y in min_y..(min_y + cell_size_y).min(max_y) {
        for x in min_x..=max_x {
            if y < height && x < width {
                total_count += 1;
                if binary[y * width + x] == 0 {
                    black_count += 1;
                }
            }
        }
    }

    // Bottom border row
    let bottom_start = max_y.saturating_sub(cell_size_y);
    for y in bottom_start..=max_y {
        for x in min_x..=max_x {
            if y < height && x < width {
                total_count += 1;
                if binary[y * width + x] == 0 {
                    black_count += 1;
                }
            }
        }
    }

    // Left border column (excluding corners)
    for y in (min_y + cell_size_y)..(max_y.saturating_sub(cell_size_y)) {
        for x in min_x..(min_x + cell_size_x).min(max_x) {
            if y < height && x < width {
                total_count += 1;
                if binary[y * width + x] == 0 {
                    black_count += 1;
                }
            }
        }
    }

    // Right border column (excluding corners)
    let right_start = max_x.saturating_sub(cell_size_x);
    for y in (min_y + cell_size_y)..(max_y.saturating_sub(cell_size_y)) {
        for x in right_start..=max_x {
            if y < height && x < width {
                total_count += 1;
                if binary[y * width + x] == 0 {
                    black_count += 1;
                }
            }
        }
    }

    if total_count == 0 {
        0.0
    } else {
        black_count as f64 / total_count as f64
    }
}

/// Complex multi-tag scene generation for integration testing.
#[cfg(any(feature = "extended-tests", feature = "extended-bench"))]
pub mod scene;
#[cfg(any(feature = "extended-tests", feature = "extended-bench"))]
pub use scene::{SceneBuilder, TagPlacement};
