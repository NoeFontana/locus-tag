//! Gradient computation for edge-based quad detection.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_panics_doc)]

use crate::image::ImageView;

/// Gradient data for a single pixel.
#[derive(Clone, Copy, Default)]
pub struct Gradient {
    /// Gradient in x-direction.
    pub gx: i16,
    /// Gradient in y-direction.
    pub gy: i16,
    /// Gradient magnitude.
    pub mag: u16,
}

/// Compute Sobel gradients for the entire image.
/// Returns a flat array of Gradient structs.
#[must_use]
pub fn compute_sobel(img: &ImageView) -> Vec<Gradient> {
    let w = img.width;
    let h = img.height;
    let mut grads = vec![Gradient::default(); w * h];

    // Sobel kernels:
    // Gx: [-1 0 1; -2 0 2; -1 0 1]
    // Gy: [-1 -2 -1; 0 0 0; 1 2 1]

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p00 = i16::from(img.get_pixel(x - 1, y - 1));
            let p10 = i16::from(img.get_pixel(x, y - 1));
            let p20 = i16::from(img.get_pixel(x + 1, y - 1));
            let p01 = i16::from(img.get_pixel(x - 1, y));
            let p21 = i16::from(img.get_pixel(x + 1, y));
            let p02 = i16::from(img.get_pixel(x - 1, y + 1));
            let p12 = i16::from(img.get_pixel(x, y + 1));
            let p22 = i16::from(img.get_pixel(x + 1, y + 1));

            let gx = -p00 + p20 - 2 * p01 + 2 * p21 - p02 + p22;
            let gy = -p00 - 2 * p10 - p20 + p02 + 2 * p12 + p22;

            let mag = ((gx.abs() + gy.abs()) as u16).min(1000);

            grads[y * w + x] = Gradient { gx, gy, mag };
        }
    }

    grads
}

/// A detected line segment.
#[derive(Clone, Copy, Debug)]
pub struct LineSegment {
    /// Start x coordinate.
    pub x0: f32,
    /// Start y coordinate.
    pub y0: f32,
    /// End x coordinate.
    pub x1: f32,
    /// End y coordinate.
    pub y1: f32,
    /// Angle of the gradient.
    pub angle: f32,
}

/// Extract line segments from gradient image using a simplified LSD approach.
/// This is a greedy region-growing algorithm on gradient direction.
#[must_use]
pub fn extract_line_segments(
    grads: &[Gradient],
    width: usize,
    height: usize,
    mag_thresh: u16,
) -> Vec<LineSegment> {
    let mut used = vec![false; width * height];
    let mut segments = Vec::new();

    for y in 2..height - 2 {
        for x in 2..width - 2 {
            let idx = y * width + x;
            if used[idx] || grads[idx].mag < mag_thresh {
                continue;
            }

            // Seed point found - grow a line segment
            let seed_angle = f32::from(grads[idx].gy).atan2(f32::from(grads[idx].gx));

            let mut points: Vec<(usize, usize)> = vec![(x, y)];
            used[idx] = true;

            // Simple 8-connected region growing with angle constraint
            let mut changed = true;
            while changed && points.len() < 500 {
                changed = false;
                let (lx, ly) = *points
                    .last()
                    .expect("Points list should not be empty during refinement");

                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = (lx as i32 + dx) as usize;
                        let ny = (ly as i32 + dy) as usize;
                        if nx >= width || ny >= height {
                            continue;
                        }

                        let nidx = ny * width + nx;
                        if used[nidx] || grads[nidx].mag < mag_thresh {
                            continue;
                        }

                        let angle = f32::from(grads[nidx].gy).atan2(f32::from(grads[nidx].gx));
                        let angle_diff = (angle - seed_angle).abs();
                        let angle_diff = angle_diff.min(std::f32::consts::PI - angle_diff);

                        if angle_diff < 0.3 {
                            // ~17 degrees tolerance
                            points.push((nx, ny));
                            used[nidx] = true;
                            changed = true;
                            break;
                        }
                    }
                    if changed {
                        break;
                    }
                }
            }

            if points.len() >= 10 {
                // Fit line to points (simple endpoints for now)
                let (x0, y0) = points[0];
                let (x1, y1) = points[points.len() - 1];

                segments.push(LineSegment {
                    x0: x0 as f32,
                    y0: y0 as f32,
                    x1: x1 as f32,
                    y1: y1 as f32,
                    angle: seed_angle,
                });
            }
        }
    }

    segments
}

/// Find quads by grouping 4 line segments that form a closed quadrilateral.
#[must_use]
pub fn find_quads_from_segments(segments: &[LineSegment]) -> Vec<[[f32; 2]; 4]> {
    let mut quads = Vec::new();

    if segments.len() < 4 {
        return quads;
    }

    // For each segment, find 3 others that could form a quad
    // This is O(n^4) in worst case but typically n is small (<100 segments)
    for i in 0..segments.len() {
        for j in i + 1..segments.len() {
            for k in j + 1..segments.len() {
                for l in k + 1..segments.len() {
                    if let Some(corners) =
                        try_form_quad(&segments[i], &segments[j], &segments[k], &segments[l])
                    {
                        quads.push(corners);
                    }
                }
            }
        }
    }

    quads
}

fn try_form_quad(
    s0: &LineSegment,
    s1: &LineSegment,
    s2: &LineSegment,
    s3: &LineSegment,
) -> Option<[[f32; 2]; 4]> {
    // Cluster segments into two groups by their gradient direction (roughly parallel pairs)
    // We sort segments by angle to easily find two pairs.
    let mut segs = [*s0, *s1, *s2, *s3];
    segs.sort_by(|a, b| a.angle.total_cmp(&b.angle));
    
    // Check if we have two pairs of roughly parallel segments.
    // Parallel segments have angles ~PI apart or very similar (depending on sign/direction).
    // A better way is to check that we have two groups of 2 segments each,
    // where within each group segments are roughly parallel, and between groups they are roughly perpendicular.
    
    // Group 1: segs[0] and segs[1]? No, angles might be [0, 0.1, 1.5, 1.6]
    // Let's try to find a gap.
    let mut group1 = Vec::with_capacity(2);
    let mut group2 = Vec::with_capacity(2);
    group1.push(segs[0]);
    
    for i in 1..4 {
        let diff = angle_diff(segs[0].angle, segs[i].angle);
        if diff < 0.6 { // Within ~35 degrees
            group1.push(segs[i]);
        } else {
            group2.push(segs[i]);
        }
    }

    if group1.len() != 2 || group2.len() != 2 {
        return None;
    }

    // Ensure the two groups are roughly perpendicular
    let angle_between_groups = angle_diff(group1[0].angle, group2[0].angle);
    if (angle_between_groups - std::f32::consts::FRAC_PI_2).abs() > 0.6 {
        return None;
    }

    // Compute intersections
    let c0 = line_intersection(&group1[0], &group2[0])?;
    let c1 = line_intersection(&group1[0], &group2[1])?;
    let c2 = line_intersection(&group1[1], &group2[1])?;
    let c3 = line_intersection(&group1[1], &group2[0])?;

    // Validate quad: check area and convexity
    let area = quad_area(&[c0, c1, c2, c3]);
    if area.abs() < 16.0 || area.abs() > 1_000_000.0 {
        return None;
    }

    // In image coordinates (Y-down), positive shoelace sum means Clockwise.
    if area > 0.0 {
        Some([c0, c1, c2, c3])
    } else {
        Some([c0, c3, c2, c1]) // Reverse CCW to CW
    }
}

fn line_intersection(s1: &LineSegment, s2: &LineSegment) -> Option<[f32; 2]> {
    let dx1 = s1.x1 - s1.x0;
    let dy1 = s1.y1 - s1.y0;
    let dx2 = s2.x1 - s2.x0;
    let dy2 = s2.y1 - s2.y0;

    let denom = dx1 * dy2 - dy1 * dx2;
    if denom.abs() < 1e-6 {
        return None; // Parallel
    }

    let t = ((s2.x0 - s1.x0) * dy2 - (s2.y0 - s1.y0) * dx2) / denom;

    Some([s1.x0 + t * dx1, s1.y0 + t * dy1])
}

fn quad_area(corners: &[[f32; 2]; 4]) -> f32 {
    let mut area = 0.0;
    for i in 0..4 {
        let j = (i + 1) % 4;
        area += corners[i][0] * corners[j][1];
        area -= corners[j][0] * corners[i][1];
    }
    area * 0.5
}

/// Fit a quad from a small component using on-demand gradient computation.
///
/// This is the optimized version that computes Sobel gradients only within
/// the component's bounding box, avoiding full-image gradient computation.
///
/// # Arguments
/// * `img` - The grayscale image
/// * `labels` - Component labels from CCL
/// * `label` - The specific component label to fit
/// * `min_x`, `min_y`, `max_x`, `max_y` - Bounding box of the component
///
/// # Returns
/// Quad corners if successfully fitted, None otherwise
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fit_quad_from_component(
    img: &ImageView,
    labels: &[u32],
    label: u32,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
) -> Option<[[f32; 2]; 4]> {
    let width = img.width;
    let height = img.height;

    // Expand bbox by 1 for gradient computation
    let x0 = min_x.saturating_sub(1);
    let y0 = min_y.saturating_sub(1);
    let x1 = (max_x + 2).min(width);
    let y1 = (max_y + 2).min(height);

    // Collect boundary pixels with inline Sobel gradient
    let mut boundary_points: Vec<(usize, usize, f32, f32)> = Vec::new();

    for y in y0.max(1)..y1.min(height - 1) {
        for x in x0.max(1)..x1.min(width - 1) {
            let idx = y * width + x;
            if labels[idx] != label {
                continue;
            }

            // Check if boundary
            if labels[idx - 1] == label
                && labels[idx + 1] == label
                && labels[idx - width] == label
                && labels[idx + width] == label
            {
                continue; // Interior pixel, skip
            }

            // Compute Sobel gradient inline for this pixel only
            let p00 = i16::from(img.get_pixel(x - 1, y - 1));
            let p10 = i16::from(img.get_pixel(x, y - 1));
            let p20 = i16::from(img.get_pixel(x + 1, y - 1));
            let p01 = i16::from(img.get_pixel(x - 1, y));
            let p21 = i16::from(img.get_pixel(x + 1, y));
            let p02 = i16::from(img.get_pixel(x - 1, y + 1));
            let p12 = i16::from(img.get_pixel(x, y + 1));
            let p22 = i16::from(img.get_pixel(x + 1, y + 1));

            let gx = -p00 + p20 - 2 * p01 + 2 * p21 - p02 + p22;
            let gy = -p00 - 2 * p10 - p20 + p02 + 2 * p12 + p22;
            let mag = (gx.abs() + gy.abs()) as u16;

            if mag > 20 {
                let angle = f32::from(gy).atan2(f32::from(gx));
                boundary_points.push((x, y, angle, f32::from(mag)));
            }
        }
    }

    if boundary_points.len() < 12 {
        // At least 3 per edge
        return None;
    }

    // PCA-based centroid initialization for rotation-invariance
    // Compute covariance matrix of gradient vectors to find dominant direction
    // Weight by magnitude to ignore noise
    let mut weight_sum = 0.0f32;
    let mut gx_sum = 0.0f32;
    let mut gy_sum = 0.0f32;
    let mut gxx_sum = 0.0f32;
    let mut gyy_sum = 0.0f32;
    let mut gxy_sum = 0.0f32;

    for (_x, _y, angle, mag) in &boundary_points {
        let w = *mag;
        let gx = angle.cos();
        let gy = angle.sin();
        weight_sum += w;
        gx_sum += gx * w;
        gy_sum += gy * w;
        gxx_sum += gx * gx * w;
        gyy_sum += gy * gy * w;
        gxy_sum += gx * gy * w;
    }

    // Covariance matrix elements (centered)
    let mean_gx = gx_sum / weight_sum;
    let mean_gy = gy_sum / weight_sum;
    let cov_xx = gxx_sum / weight_sum - mean_gx * mean_gx;
    let cov_yy = gyy_sum / weight_sum - mean_gy * mean_gy;
    let cov_xy = gxy_sum / weight_sum - mean_gx * mean_gy;

    // Find principal eigenvector
    let trace = cov_xx + cov_yy;
    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    let discriminant = (trace * trace / 4.0 - det).max(0.0);
    let lambda1 = trace / 2.0 + discriminant.sqrt();

    let theta = if cov_xy.abs() > 1e-6 {
        (lambda1 - cov_xx).atan2(cov_xy)
    } else if cov_xx >= cov_yy {
        0.0
    } else {
        std::f32::consts::FRAC_PI_2
    };

    let mut centroids = [
        theta,
        theta + std::f32::consts::FRAC_PI_2,
        theta + std::f32::consts::PI,
        theta - std::f32::consts::FRAC_PI_2,
    ];
    for c in &mut centroids {
        while *c > std::f32::consts::PI {
            *c -= 2.0 * std::f32::consts::PI;
        }
        while *c < -std::f32::consts::PI {
            *c += 2.0 * std::f32::consts::PI;
        }
    }

    let mut assignments = vec![0usize; boundary_points.len()];

    for _ in 0..5 {
        // Assignment step
        for (i, (_x, _y, angle, _mag)) in boundary_points.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f32::MAX;
            for (c, &centroid) in centroids.iter().enumerate() {
                let diff = angle_diff(*angle, centroid);
                if diff < best_dist {
                    best_dist = diff;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update step (Weighted by magnitude)
        for c in 0..4 {
            let mut sum_sin = 0.0f32;
            let mut sum_cos = 0.0f32;
            for (i, (_x, _y, angle, mag)) in boundary_points.iter().enumerate() {
                if assignments[i] == c {
                    sum_sin += angle.sin() * mag;
                    sum_cos += angle.cos() * mag;
                }
            }
            if sum_sin.abs() > 1e-6 || sum_cos.abs() > 1e-6 {
                centroids[c] = sum_sin.atan2(sum_cos);
            }
        }
    }

    // Fit line to each cluster using Weighted Least Squares (WLS)
    let mut lines: Vec<LineSegment> = Vec::new();
    for c in 0..4 {
        let cluster_points: Vec<(f32, f32, f32)> = boundary_points
            .iter()
            .enumerate()
            .filter(|(i, _)| assignments[*i] == c)
            .map(|(_, &(x, y, _, mag))| (x as f32, y as f32, mag))
            .collect();

        if cluster_points.len() < 3 {
            continue;
        }

        // Weighted Least Squares Line Fitting
        let mut sum_w = 0.0f32;
        let mut sum_wx = 0.0f32;
        let mut sum_wy = 0.0f32;
        for &(x, y, mag) in &cluster_points {
            let w = mag * mag; // QUADRATIC weighting for maximum precision on sharp edges
            sum_w += w;
            sum_wx += x * w;
            sum_wy += y * w;
        }
        let mean_x = sum_wx / sum_w;
        let mean_y = sum_wy / sum_w;

        let mut cov_xx = 0.0f32;
        let mut cov_yy = 0.0f32;
        let mut cov_xy = 0.0f32;
        for &(x, y, mag) in &cluster_points {
            let w = mag * mag;
            let dx = x - mean_x;
            let dy = y - mean_y;
            cov_xx += dx * dx * w;
            cov_yy += dy * dy * w;
            cov_xy += dx * dy * w;
        }

        // Robust principal direction using atan2(2b, a-c)
        let direction = 0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy);
        let nx = direction.cos();
        let ny = direction.sin();

        // Line segment for intersection logic
        let mut min_t = f32::MAX;
        let mut max_t = f32::MIN;
        for &(x, y, _) in &cluster_points {
            let t = (x - mean_x) * nx + (y - mean_y) * ny;
            min_t = min_t.min(t);
            max_t = max_t.max(t);
        }

        // Correct angle for grouping: quad extractor expects GRADIENT direction
        let mut grad_angle = direction + std::f32::consts::FRAC_PI_2;
        if grad_angle > std::f32::consts::PI {
            grad_angle -= 2.0 * std::f32::consts::PI;
        }

        lines.push(LineSegment {
            x0: mean_x + nx * min_t,
            y0: mean_y + ny * min_t,
            x1: mean_x + nx * max_t,
            y1: mean_y + ny * max_t,
            angle: grad_angle,
        });
    }

    if lines.len() < 4 {
        return None;
    }

    find_quads_from_segments(&lines).into_iter().next()
}

/// Fit a quad from boundary pixels using gradient direction clustering.
///
/// For small tags (~9px), there are essentially 4 gradient directions
/// (one per edge). This function:
/// 1. Extracts boundary pixels from the component
/// 2. Clusters them by gradient direction into 4 groups
/// 3. Fits a line to each group
/// 4. Intersects lines to find quad corners
///
/// Returns None if a valid quad cannot be formed.
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::similar_names)]
#[must_use]
pub fn fit_quad_from_gradients(
    grads: &[Gradient],
    labels: &[u32],
    label: u32,
    width: usize,
    height: usize,
    min_edge_pixels: usize,
) -> Option<[[f32; 2]; 4]> {
    // Collect boundary pixels: pixels in this component adjacent to different component
    let mut boundary_points: Vec<(usize, usize, f32)> = Vec::new(); // (x, y, angle)

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            if labels[idx] != label {
                continue;
            }

            // Check if boundary (any neighbor is different)
            let is_boundary = labels[idx - 1] != label
                || labels[idx + 1] != label
                || labels[idx - width] != label
                || labels[idx + width] != label;

            if is_boundary && grads[idx].mag > 20 {
                let angle = f32::from(grads[idx].gy).atan2(f32::from(grads[idx].gx));
                boundary_points.push((x, y, angle));
            }
        }
    }

    if boundary_points.len() < min_edge_pixels * 4 {
        return None; // Not enough boundary points
    }

    // Cluster into 4 directions using simple k-means on angles
    // Initialize with 4 orthogonal directions
    let mut centroids = [
        0.0f32,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        -std::f32::consts::FRAC_PI_2,
    ];
    let mut assignments = vec![0usize; boundary_points.len()];

    for _ in 0..5 {
        // 5 iterations of k-means
        // Assignment step
        for (i, (_x, _y, angle)) in boundary_points.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f32::MAX;
            for (c, &centroid) in centroids.iter().enumerate() {
                let diff = angle_diff(*angle, centroid);
                if diff < best_dist {
                    best_dist = diff;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update step
        for c in 0..4 {
            let mut sum_sin = 0.0f32;
            let mut sum_cos = 0.0f32;
            for (i, (_x, _y, angle)) in boundary_points.iter().enumerate() {
                if assignments[i] == c {
                    sum_sin += angle.sin();
                    sum_cos += angle.cos();
                }
            }
            if sum_sin.abs() > 1e-6 || sum_cos.abs() > 1e-6 {
                centroids[c] = sum_sin.atan2(sum_cos);
            }
        }
    }

    // Fit line to each cluster
    let mut lines: Vec<LineSegment> = Vec::new();
    for c in 0..4 {
        let cluster_points: Vec<(f32, f32)> = boundary_points
            .iter()
            .enumerate()
            .filter(|(i, _)| assignments[*i] == c)
            .map(|(_, (x, y, _))| (*x as f32, *y as f32))
            .collect();

        if cluster_points.len() < min_edge_pixels {
            continue;
        }

        // Simple line fit: use endpoints (could use least squares)
        let (x0, y0) = cluster_points[0];
        let (x1, y1) = cluster_points[cluster_points.len() - 1];

        lines.push(LineSegment {
            x0,
            y0,
            x1,
            y1,
            angle: centroids[c],
        });
    }

    if lines.len() < 4 {
        return None;
    }

    // Find 4 line segments that form a quad
    find_quads_from_segments(&lines).into_iter().next()
}

/// Compute angle difference in range [0, π/2] (perpendicular equivalence)
fn angle_diff(a: f32, b: f32) -> f32 {
    let diff = (a - b).abs();
    let diff = diff % std::f32::consts::PI;
    diff.min(std::f32::consts::PI - diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageView;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_sobel_magnitude_bounds(
            width in 3..64usize,
            height in 3..64usize,
            data in prop::collection::vec(0..=255u8, 64*64)
        ) {
            let slice = &data[..width * height];
            let view = ImageView::new(slice, width, height, width).unwrap();
            let grads = compute_sobel(&view);

            for g in grads {
                assert!(g.mag <= 1000);
            }
        }

        #[test]
        fn prop_sobel_orientation_ramp(
            width in 3..10usize,
            height in 3..10usize,
            is_horizontal in prop::bool::ANY
        ) {
            let mut data = vec![0u8; width * height];
            for y in 0..height {
                for x in 0..width {
                    data[y * width + x] = if is_horizontal { x as u8 * 10 } else { y as u8 * 10 };
                }
            }

            let view = ImageView::new(&data, width, height, width).unwrap();
            let grads = compute_sobel(&view);

            // Checking interior pixels
            for y in 1..height-1 {
                for x in 1..width-1 {
                    let g = grads[y * width + x];
                    if is_horizontal {
                        assert!(g.gx > 0);
                        assert_eq!(g.gy, 0);
                    } else {
                        assert_eq!(g.gx, 0);
                        assert!(g.gy > 0);
                    }
                }
            }
        }

        #[test]
        fn prop_quad_area_invariants(
            c in prop::collection::vec((0.0..100.0f32, 0.0..100.0f32), 4)
        ) {
            let corners = [
                [c[0].0, c[0].1],
                [c[1].0, c[1].1],
                [c[2].0, c[2].1],
                [c[3].0, c[3].1],
            ];
            let area = quad_area(&corners);
            assert!(area >= 0.0);

            // Area should be zero if points are identical
            let identical_corners = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
            assert_eq!(quad_area(&identical_corners), 0.0);
        }
    }
}
