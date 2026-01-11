/// Generate a synthetic AprilTag-like image for testing.
/// This is a simplified version of the OpenCV generator.
#[must_use]
pub fn generate_synthetic_tag(
    width: usize,
    height: usize,
    _tag_id: u32,
    x: usize,
    y: usize,
    tag_size: usize,
) -> (Vec<u8>, [[f64; 2]; 4]) {
    let mut data = vec![0u8; width * height];
    let padding = 20;

    // White quiet zone
    for py in (y.saturating_sub(padding))..((y + tag_size + padding).min(height)) {
        for px in (x.saturating_sub(padding))..((x + tag_size + padding).min(width)) {
            data[py * width + px] = 255;
        }
    }

    // Black border
    for py in y..(y + tag_size) {
        for px in x..(x + tag_size) {
            if py < height && px < width {
                data[py * width + px] = 0;
            }
        }
    }

    // Interior (simplified tag pattern based on tag_id)
    let border_width = tag_size / 8;
    for py in (y + border_width)..(y + tag_size - border_width) {
        for px in (x + border_width)..(x + tag_size - border_width) {
            if py < height && px < width {
                // Alternating pattern with ~50% white bits to ensure fill ratio passes (approx 0.6 total)
                let tx = (px - x - border_width) / border_width;
                let ty = (py - y - border_width) / border_width;

                let bit = if (tx + ty).is_multiple_of(2) { 255 } else { 0 };
                data[py * width + px] = bit;
            }
        }
    }

    let ts = (tag_size - 1) as f64;
    let ground_truth = [
        [x as f64, y as f64],
        [(x as f64 + ts), y as f64],
        [(x as f64 + ts), (y as f64 + ts)],
        [x as f64, (y as f64 + ts)],
    ];

    (data, ground_truth)
}

/// Compute mean Euclidean distance between detected and ground truth corners.
/// Handles 4 rotations and 2 winding orders to find the minimum error.
#[must_use]
pub fn compute_corner_error(detected: [[f64; 2]; 4], ground_truth: [[f64; 2]; 4]) -> f64 {
    let mut min_error = f64::MAX;

    // Try both winding orders
    let windings = [
        detected,
        [detected[3], detected[2], detected[1], detected[0]], // Flipped
    ];

    for points in windings {
        // Try all 4 rotations
        for rot in 0..4 {
            let mut sum_dist = 0.0;
            for i in 0..4 {
                let d = &points[(i + rot) % 4];
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
    }

    min_error
}
