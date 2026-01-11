use rand::prelude::*;
use rand_distr::{Distribution, Normal};

/// Generate a synthetic image containing a single AprilTag or ArUco tag.
///
/// This generates a tag with a white quiet zone, placed on a white background,
/// matching the setup used in Python benchmarks.
pub fn generate_test_image(
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
        let normal = Normal::new(0.0, f64::from(noise_sigma)).unwrap();

        for pixel in &mut data {
            let noise = normal.sample(&mut rng) as i32;
            let val = (*pixel as i32 + noise).clamp(0, 255);
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
/// Complex multi-tag scene generation for integration testing.
#[cfg(any(feature = "extended-tests", feature = "extended-bench"))]
pub mod scene;
#[cfg(any(feature = "extended-tests", feature = "extended-bench"))]
pub use scene::{SceneBuilder, TagPlacement};
