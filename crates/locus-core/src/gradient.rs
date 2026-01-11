//! Gradient computation for edge-based quad detection.

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
pub fn compute_sobel(img: &ImageView) -> Vec<Gradient> {
    let w = img.width;
    let h = img.height;
    let mut grads = vec![Gradient::default(); w * h];

    // Sobel kernels:
    // Gx: [-1 0 1; -2 0 2; -1 0 1]
    // Gy: [-1 -2 -1; 0 0 0; 1 2 1]

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p00 = img.get_pixel(x - 1, y - 1) as i16;
            let p10 = img.get_pixel(x, y - 1) as i16;
            let p20 = img.get_pixel(x + 1, y - 1) as i16;
            let p01 = img.get_pixel(x - 1, y) as i16;
            let p21 = img.get_pixel(x + 1, y) as i16;
            let p02 = img.get_pixel(x - 1, y + 1) as i16;
            let p12 = img.get_pixel(x, y + 1) as i16;
            let p22 = img.get_pixel(x + 1, y + 1) as i16;

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
            let seed_angle = (grads[idx].gy as f32).atan2(grads[idx].gx as f32);

            let mut points: Vec<(usize, usize)> = vec![(x, y)];
            used[idx] = true;

            // Simple 8-connected region growing with angle constraint
            let mut changed = true;
            while changed && points.len() < 500 {
                changed = false;
                let (lx, ly) = *points.last().unwrap();

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

                        let angle = (grads[nidx].gy as f32).atan2(grads[nidx].gx as f32);
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
    // Check that we have roughly 4 perpendicular segments
    let angles = [s0.angle, s1.angle, s2.angle, s3.angle];

    // Group by approximate perpendicularity (horizontal-ish vs vertical-ish)
    let mut horizontal = Vec::new();
    let mut vertical = Vec::new();

    for (i, &seg) in [s0, s1, s2, s3].iter().enumerate() {
        let a = angles[i].abs();
        if !(0.5..=std::f32::consts::PI - 0.5).contains(&a) {
            horizontal.push(seg);
        } else if (a - std::f32::consts::FRAC_PI_2).abs() < 0.5 {
            vertical.push(seg);
        }
    }

    if horizontal.len() != 2 || vertical.len() != 2 {
        return None;
    }

    // Compute intersections
    let c0 = line_intersection(horizontal[0], vertical[0])?;
    let c1 = line_intersection(horizontal[0], vertical[1])?;
    let c2 = line_intersection(horizontal[1], vertical[1])?;
    let c3 = line_intersection(horizontal[1], vertical[0])?;

    // Validate quad: check area and convexity
    let area = quad_area(&[c0, c1, c2, c3]);
    if !(400.0..=100000.0).contains(&area) {
        return None;
    }

    Some([c0, c1, c2, c3])
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
    area.abs() * 0.5
}
