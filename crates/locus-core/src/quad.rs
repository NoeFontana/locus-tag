#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(unsafe_code)]

use crate::Detection;
use crate::config::DetectorConfig;
use crate::image::ImageView;
use crate::segmentation::LabelResult;
use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use multiversion::multiversion;

/// A 2D point with subpixel precision.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

/// Fast quad extraction using bounding box stats from CCL.
/// Only traces contours for components that pass geometric filters.
/// Uses default configuration.
pub fn extract_quads_fast(
    arena: &Bump,
    img: &ImageView,
    label_result: &LabelResult,
) -> Vec<Detection> {
    extract_quads_with_config(arena, img, label_result, &DetectorConfig::default())
}

/// Quad extraction with custom configuration.
///
/// This is the main entry point for quad detection with custom parameters.
/// Components are processed in parallel for maximum throughput.
pub fn extract_quads_with_config(
    _arena: &Bump,
    img: &ImageView,
    label_result: &LabelResult,
    config: &DetectorConfig,
) -> Vec<Detection> {
    use rayon::prelude::*;

    let labels = label_result.labels;
    let stats = &label_result.component_stats;
    let min_edge_len_sq = config.quad_min_edge_length * config.quad_min_edge_length;

    // Process components in parallel, each with its own thread-local arena
    stats
        .par_iter()
        .enumerate()
        .filter_map(|(label_idx, stat)| {
            // Thread-local arena for this component
            let arena = Bump::new();
            
            let label = (label_idx + 1) as u32;

            // Fast geometric filtering using bounding box
            let bbox_w = u32::from(stat.max_x - stat.min_x) + 1;
            let bbox_h = u32::from(stat.max_y - stat.min_y) + 1;
            let bbox_area = bbox_w * bbox_h;

            // Filter: too small or too large
            if bbox_area < config.quad_min_area || bbox_area > (img.width * img.height * 9 / 10) as u32 {
                return None;
            }

            // Filter: not roughly square (aspect ratio)
            let aspect = bbox_w.max(bbox_h) as f32 / bbox_w.min(bbox_h).max(1) as f32;
            if aspect > config.quad_max_aspect_ratio {
                return None;
            }

            // Filter: fill ratio (should be ~50-80% for a tag with inner pattern)
            let fill = stat.pixel_count as f32 / bbox_area as f32;
            if fill < config.quad_min_fill_ratio || fill > config.quad_max_fill_ratio {
                return None;
            }

            // Passed filters - trace boundary and fit quad
            let sx = stat.first_pixel_x as usize;
            let sy = stat.first_pixel_y as usize;

            // For small components, try the 9-pixel foundation (gradient-based fitting)
            if bbox_area < 1200 {
                if let Some(grad_corners) = crate::gradient::fit_quad_from_component(
                    img, labels, label,
                    stat.min_x as usize, stat.min_y as usize,
                    stat.max_x as usize, stat.max_y as usize,
                ) {
                    let corners = [
                        refine_corner(&arena, img, Point { x: grad_corners[0][0] as f64, y: grad_corners[0][1] as f64 },
                                     Point { x: grad_corners[3][0] as f64, y: grad_corners[3][1] as f64 },
                                     Point { x: grad_corners[1][0] as f64, y: grad_corners[1][1] as f64 },
                                     config.subpixel_refinement_sigma),
                        refine_corner(&arena, img, Point { x: grad_corners[1][0] as f64, y: grad_corners[1][1] as f64 },
                                     Point { x: grad_corners[0][0] as f64, y: grad_corners[0][1] as f64 },
                                     Point { x: grad_corners[2][0] as f64, y: grad_corners[2][1] as f64 },
                                     config.subpixel_refinement_sigma),
                        refine_corner(&arena, img, Point { x: grad_corners[2][0] as f64, y: grad_corners[2][1] as f64 },
                                     Point { x: grad_corners[1][0] as f64, y: grad_corners[1][1] as f64 },
                                     Point { x: grad_corners[3][0] as f64, y: grad_corners[3][1] as f64 },
                                     config.subpixel_refinement_sigma),
                        refine_corner(&arena, img, Point { x: grad_corners[3][0] as f64, y: grad_corners[3][1] as f64 },
                                     Point { x: grad_corners[2][0] as f64, y: grad_corners[2][1] as f64 },
                                     Point { x: grad_corners[0][0] as f64, y: grad_corners[0][1] as f64 },
                                     config.subpixel_refinement_sigma),
                    ];

                    let edge_score = calculate_edge_score(img, corners);
                    if edge_score > config.quad_min_edge_score {
                        return Some(Detection {
                            id: label,
                            center: [
                                (grad_corners[0][0] + grad_corners[2][0]) as f64 / 2.0,
                                (grad_corners[0][1] + grad_corners[2][1]) as f64 / 2.0,
                            ],
                            corners: [
                                [corners[0].x, corners[0].y],
                                [corners[1].x, corners[1].y],
                                [corners[2].x, corners[2].y],
                                [corners[3].x, corners[3].y],
                            ],
                            hamming: 0,
                            decision_margin: bbox_area as f64,
                            pose: None,
                        });
                    }
                }
            }

            let contour = trace_boundary(&arena, labels, img.width, img.height, sx, sy, label);

            if contour.len() >= 12 {
                let simple_contour = chain_approximation(&arena, &contour);
                let perimeter = contour.len() as f64;
                let epsilon = (perimeter * 0.02).max(1.0);
                let simplified = douglas_peucker(&arena, &simple_contour, epsilon);

                if simplified.len() >= 4 && simplified.len() <= 11 {
                    let simpl_len = simplified.len();
                    let reduced = if simpl_len == 5 {
                        simplified
                    } else if simpl_len == 4 {
                        let mut closed = BumpVec::new_in(&arena);
                        for p in simplified.iter() { closed.push(*p); }
                        closed.push(simplified[0]);
                        closed
                    } else {
                        reduce_to_quad(&arena, &simplified)
                    };

                    if reduced.len() == 5 {
                        let area = polygon_area(&reduced);
                        let compactness = (12.566 * area) / (perimeter * perimeter);

                        if area > config.quad_min_area as f64 && compactness > 0.1 {
                            let mut ok = true;
                            for i in 0..4 {
                                let d2 = (reduced[i].x - reduced[i + 1].x).powi(2)
                                    + (reduced[i].y - reduced[i + 1].y).powi(2);
                                if d2 < min_edge_len_sq { ok = false; break; }
                            }

                            if ok {
                                let center = polygon_center(&reduced);
                                let corners = [
                                    refine_corner(&arena, img, reduced[0], reduced[3], reduced[1], config.subpixel_refinement_sigma),
                                    refine_corner(&arena, img, reduced[1], reduced[0], reduced[2], config.subpixel_refinement_sigma),
                                    refine_corner(&arena, img, reduced[2], reduced[1], reduced[3], config.subpixel_refinement_sigma),
                                    refine_corner(&arena, img, reduced[3], reduced[2], reduced[0], config.subpixel_refinement_sigma),
                                ];

                                let edge_score = calculate_edge_score(img, corners);
                                if edge_score > config.quad_min_edge_score {
                                    return Some(Detection {
                                        id: label,
                                        center,
                                        corners: [
                                            [corners[0].x, corners[0].y],
                                            [corners[1].x, corners[1].y],
                                            [corners[2].x, corners[2].y],
                                            [corners[3].x, corners[3].y],
                                        ],
                                        hamming: 0,
                                        decision_margin: area,
                                        pose: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            None
        })
        .collect()
}

/// Legacy extract_quads for backward compatibility.
pub fn extract_quads(arena: &Bump, img: &ImageView, labels: &[u32]) -> Vec<Detection> {
    // Create a fake LabelResult with stats computed on-the-fly
    let mut detections = Vec::new();
    let num_labels = (labels.len() / 32) + 1;
    let processed_labels = arena.alloc_slice_fill_copy(num_labels, 0u32);

    let width = img.width;
    let height = img.height;

    for y in 1..height - 1 {
        let row_off = y * width;
        let prev_row_off = (y - 1) * width;

        for x in 1..width - 1 {
            let idx = row_off + x;
            let label = labels[idx];

            if label == 0 {
                continue;
            }

            if labels[idx - 1] == label || labels[prev_row_off + x] == label {
                continue;
            }

            let bit_idx = (label as usize) / 32;
            let bit_mask = 1 << (label % 32);
            if processed_labels[bit_idx] & bit_mask != 0 {
                continue;
            }

            processed_labels[bit_idx] |= bit_mask;
            let contour = trace_boundary(arena, labels, width, height, x, y, label);

            if contour.len() >= 12 {
                // Lowered from 30 to support 8px+ tags
                let simplified = douglas_peucker(arena, &contour, 4.0);
                if simplified.len() == 5 {
                    let area = polygon_area(&simplified);
                    let perimeter = contour.len() as f64;
                    let compactness = (12.566 * area) / (perimeter * perimeter);

                    if area > 400.0 && compactness > 0.5 {
                        let mut ok = true;
                        for i in 0..4 {
                            let d2 = (simplified[i].x - simplified[i + 1].x).powi(2)
                                + (simplified[i].y - simplified[i + 1].y).powi(2);
                            if d2 < 100.0 {
                                ok = false;
                                break;
                            }
                        }

                        if ok {
                            detections.push(Detection {
                                id: label,
                                center: polygon_center(&simplified),
                                corners: [
                                    [simplified[0].x, simplified[0].y],
                                    [simplified[1].x, simplified[1].y],
                                    [simplified[2].x, simplified[2].y],
                                    [simplified[3].x, simplified[3].y],
                                ],
                                hamming: 0,
                                decision_margin: area,
                                pose: None,
                            });
                        }
                    }
                }
            }
        }
    }
    detections
}

/// Simplify a contour using the Douglas-Peucker algorithm.
/// 
/// Leverages an iterative implementation with a manual stack to avoid
/// the overhead of recursive function calls and multiple temporary allocations.
pub fn douglas_peucker<'a>(arena: &'a Bump, points: &[Point], epsilon: f64) -> BumpVec<'a, Point> {
    if points.len() < 3 {
        let mut v = BumpVec::new_in(arena);
        v.extend_from_slice(points);
        return v;
    }

    let n = points.len();
    let mut keep = BumpVec::from_iter_in((0..n).map(|_| false), arena);
    keep[0] = true;
    keep[n - 1] = true;

    let mut stack = BumpVec::new_in(arena);
    stack.push((0, n - 1));

    while let Some((start, end)) = stack.pop() {
        if end - start < 2 {
            continue;
        }

        let mut dmax = 0.0;
        let mut index = start;

        for i in start + 1..end {
            let d = perpendicular_distance(points[i], points[start], points[end]);
            if d > dmax {
                index = i;
                dmax = d;
            }
        }

        if dmax > epsilon {
            keep[index] = true;
            stack.push((start, index));
            stack.push((index, end));
        }
    }

    let mut simplified = BumpVec::new_in(arena);
    for (i, &k) in keep.iter().enumerate() {
        if k {
            simplified.push(points[i]);
        }
    }
    simplified
}

fn perpendicular_distance(p: Point, a: Point, b: Point) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let mag = (dx * dx + dy * dy).sqrt();
    if mag < 1e-9 {
        return ((p.x - a.x).powi(2) + (p.y - a.y).powi(2)).sqrt();
    }
    ((dy * p.x - dx * p.y + b.x * a.y - b.y * a.x).abs()) / mag
}

fn polygon_area(points: &[Point]) -> f64 {
    let mut area = 0.0;
    for i in 0..points.len() - 1 {
        area += (points[i].x * points[i + 1].y) - (points[i + 1].x * points[i].y);
    }
    area.abs() * 0.5
}

fn polygon_center(points: &[Point]) -> [f64; 2] {
    let mut cx = 0.0;
    let mut cy = 0.0;
    let n = points.len() - 1;
    for p in points.iter().take(n) {
        cx += p.x;
        cy += p.y;
    }
    [cx / n as f64, cy / n as f64]
}

/// Refine corners to sub-pixel accuracy using intensity-based optimization.
///
/// Fits lines to the two edges meeting at a corner using PSF-blurred step function
/// model and Gauss-Newton optimization, then computes their intersection.
/// Achieves ~0.02px accuracy vs ~0.2px for gradient-peak methods.
#[must_use]
pub fn refine_corner(arena: &Bump, img: &ImageView, p: Point, p_prev: Point, p_next: Point, sigma: f64) -> Point {
    // Try intensity-based refinement first (higher accuracy)
    let line1 = refine_edge_intensity(arena, img, p_prev, p, sigma).or_else(|| fit_edge_line(img, p_prev, p));
    let line2 = refine_edge_intensity(arena, img, p, p_next, sigma).or_else(|| fit_edge_line(img, p, p_next));

    if let (Some(l1), Some(l2)) = (line1, line2) {
        // Intersect lines: a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0
        let det = l1.0 * l2.1 - l2.0 * l1.1;
        if det.abs() > 1e-6 {
            let x = (l1.1 * l2.2 - l2.1 * l1.2) / det;
            let y = (l2.0 * l1.2 - l1.0 * l2.2) / det;

            // Sanity check: intersection must be near original point
            let dist_sq = (x - p.x).powi(2) + (y - p.y).powi(2);
            if dist_sq < 4.0 {
                return Point { x, y };
            }
        }
    }

    p
}

/// Fit a line (a*x + b*y + c = 0) to an edge by sampling gradient peaks.
fn fit_edge_line(img: &ImageView, p1: Point, p2: Point) -> Option<(f64, f64, f64)> {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 4.0 {
        return None;
    }

    let nx = -dy / len;
    let ny = dx / len;

    let mut sum_d = 0.0;
    let mut count = 0;

    let n_samples = (len as usize).clamp(5, 15);
    for i in 1..=n_samples {
        let t = i as f64 / (n_samples + 1) as f64;
        let px = p1.x + dx * t;
        let py = p1.y + dy * t;

        let mut best_px = px;
        let mut best_py = py;
        let mut best_mag = 0.0;

        for step in -3..=3 {
            let sx = px + nx * f64::from(step);
            let sy = py + ny * f64::from(step);

            let g = img.sample_gradient_bilinear(sx, sy);
            let mag = g[0] * g[0] + g[1] * g[1];
            if mag > best_mag {
                best_mag = mag;
                best_px = sx;
                best_py = sy;
            }
        }

        if best_mag > 10.0 {
            let mut mags = [0.0f64; 3];
            for (j, offset) in [-1.0, 0.0, 1.0].iter().enumerate() {
                let sx = best_px + nx * offset;
                let sy = best_py + ny * offset;
                let g = img.sample_gradient_bilinear(sx, sy);
                mags[j] = g[0] * g[0] + g[1] * g[1];
            }

            let num = mags[2] - mags[0];
            let den = 2.0 * (mags[0] + mags[2] - 2.0 * mags[1]);
            let sub_offset = if den.abs() > 1e-6 {
                (-num / den).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            let refined_x = best_px + nx * sub_offset;
            let refined_y = best_py + ny * sub_offset;

            sum_d += -(nx * refined_x + ny * refined_y);
            count += 1;
        }
    }

    if count < 3 {
        return None;
    }

    Some((nx, ny, sum_d / count as f64))
}

/// Refine edge position using intensity-based optimization (Kallwies method).
///
/// Instead of finding gradient peaks, this minimizes the difference between
/// observed pixel intensities and a PSF-blurred step function model:
///
/// Model(x,y) = (A+B)/2 + (A-B)/2 * erf(dist(x,y,line) / σ)
///
/// where A,B are intensities on either side, dist is perpendicular distance
/// to the edge line, and σ is the blur factor (~0.6 pixels).
///
/// This achieves ~0.02px accuracy vs ~0.2px for gradient-based methods.
#[allow(clippy::similar_names)]
// Collected in the quad extraction process
fn refine_edge_intensity(arena: &Bump, img: &ImageView, p1: Point, p2: Point, sigma: f64) -> Option<(f64, f64, f64)> {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 4.0 {
        return None;
    }

    // Initial line parameters: normal (nx, ny) and distance d from origin
    let nx = -dy / len;
    let ny = dx / len;
    let mid_x = (p1.x + p2.x) / 2.0;
    let mid_y = (p1.y + p2.y) / 2.0;
    let mut d = -(nx * mid_x + ny * mid_y);

    // Collect pixels within 2 pixels of the edge
    let x0 = (p1.x.min(p2.x) - 3.0).max(1.0) as usize;
    let x1 = (p1.x.max(p2.x) + 3.0).min((img.width - 2) as f64) as usize;
    let y0 = (p1.y.min(p2.y) - 3.0).max(1.0) as usize;
    let y1 = (p1.y.max(p2.y) + 3.0).min((img.height - 2) as f64) as usize;

    // Use arena for samples to avoid heap allocation in hot loop
    // (x, y, intensity, projection)
    let mut samples = BumpVec::new_in(arena);

    for py in y0..=y1 {
        for px in x0..=x1 {
            let x = px as f64;
            let y = py as f64;

            // Check if near the edge segment (not infinite line)
            let t = ((x - p1.x) * dx + (y - p1.y) * dy) / (len * len);
            if t < -0.1 || t > 1.1 {
                continue;
            }

            let dist_signed = nx * x + ny * y + d;
            if dist_signed.abs() < 2.5 {
                let intensity = f64::from(img.get_pixel(px, py));
                // Pre-calculate nx*x + ny*y
                let projection = nx * x + ny * y;
                samples.push((x, y, intensity, projection));
            }
        }
    }

    if samples.len() < 10 {
        return Some((nx, ny, d)); // Fall back to initial estimate
    }

    // Estimate A (dark side) and B (light side) from samples
    let mut dark_sum = 0.0;
    let mut dark_count = 0;
    let mut light_sum = 0.0;
    let mut light_count = 0;

    for &(_x, _y, intensity, projection) in &samples {
        let signed_dist = projection + d;
        if signed_dist < -0.5 {
            dark_sum += intensity;
            dark_count += 1;
        } else if signed_dist > 0.5 {
            light_sum += intensity;
            light_count += 1;
        }
    }

    if dark_count == 0 || light_count == 0 {
        return Some((nx, ny, d));
    }

    let a = dark_sum / dark_count as f64;
    let b = light_sum / light_count as f64;
    let inv_sigma = 1.0 / sigma;

    // Gauss-Newton optimization: refine d (perpendicular offset)
    // We fix the line direction (nx, ny) and only optimize the offset d
    // This is a 1D optimization which is fast and stable
    for _iter in 0..5 {
        let mut jtj = 0.0; // J^T * J (scalar for 1D)
        let mut jtr = 0.0; // J^T * residual

        for &(_x, _y, intensity, projection) in &samples {
            let signed_dist = (projection + d) * inv_sigma;
            
            // Fast approximation of exp(-x^2): 1.0 / (1 + x^2 + 0.5*x^4)
            let x2 = signed_dist * signed_dist;
            let exp_term = 1.0 / (1.0 + x2 + 0.5 * x2 * x2);
            
            let model = (a + b) * 0.5 + (b - a) * 0.5 * signed_dist.tanh(); // erf approx
            let residual = intensity - model;
            
            let jacobian = (b - a) * 0.5 * exp_term * inv_sigma;
            jtj += jacobian * jacobian;
            jtr += jacobian * residual;
        }

        if jtj.abs() > 1e-10 {
            let delta = jtr / jtj;
            d += delta.clamp(-0.5, 0.5); // Conservative step

            if delta.abs() < 0.001 {
                break; // Converged
            }
        }
    }

    Some((nx, ny, d))
}

/// Reducing a polygon to a quad (4 vertices + 1 closing) by iteratively removing
/// the vertex that forms the smallest area triangle with its neighbors.
/// This is robust for noisy/jagged shapes that are approximately quadrilateral.
fn reduce_to_quad<'a>(arena: &'a Bump, poly: &[Point]) -> BumpVec<'a, Point> {
    if poly.len() <= 5 {
        return BumpVec::from_iter_in(poly.iter().copied(), arena);
    }

    // Work on a mutable copy
    let mut current = BumpVec::from_iter_in(poly.iter().copied(), arena);
    // Remove closing point for processing
    current.pop();

    while current.len() > 4 {
        let n = current.len();
        let mut min_area = f64::MAX;
        let mut min_idx = 0;

        for i in 0..n {
            let p_prev = current[(i + n - 1) % n];
            let p_curr = current[i];
            let p_next = current[(i + 1) % n];

            // Triangle area: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
            let area = (p_prev.x * (p_curr.y - p_next.y)
                + p_curr.x * (p_next.y - p_prev.y)
                + p_next.x * (p_prev.y - p_curr.y))
                .abs()
                * 0.5;

            if area < min_area {
                min_area = area;
                min_idx = i;
            }
        }

        // Remove the vertex contributing least to the shape
        current.remove(min_idx);
    }

    // Re-close the loop
    if !current.is_empty() {
        let first = current[0];
        current.push(first);
    }

    current
}

/// Calculate the minimum average gradient magnitude along the 4 edges of the quad.
///
/// Returns the lowest score among the 4 edges. If any edge is very weak,
/// the return value will be low, indicating a likely false positive.
fn calculate_edge_score(img: &ImageView, corners: [Point; 4]) -> f64 {
    let mut min_score = f64::MAX;

    for i in 0..4 {
        let p1 = corners[i];
        let p2 = corners[(i + 1) % 4];

        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let len = (dx * dx + dy * dy).sqrt();

        if len < 4.0 {
            // Tiny edge, likely noise or degenerate
            return 0.0;
        }

        // Sample points along the edge (excluding corners to avoid corner effects)
        let n_samples = (len as usize).clamp(3, 10); // At least 3, at most 10
        let mut edge_mag_sum = 0.0;

        for k in 1..=n_samples {
            // t goes from roughly 0.1 to 0.9 to avoid corners
            let t = k as f64 / (n_samples + 1) as f64;
            let x = p1.x + dx * t;
            let y = p1.y + dy * t;

            let g = img.sample_gradient_bilinear(x, y);
            edge_mag_sum += (g[0] * g[0] + g[1] * g[1]).sqrt();
        }

        let avg_mag = edge_mag_sum / n_samples as f64;
        if avg_mag < min_score {
            min_score = avg_mag;
        }
    }

    min_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;
    use proptest::prelude::*;

    #[test]
    fn test_edge_score_rejection() {
        // Create a 20x20 image mostly gray
        let width = 20;
        let height = 20;
        let stride = 20;
        let mut data = vec![128u8; width * height];

        // Draw a weak quad (contrast 10)
        // Center 10,10. Size 8x8.
        // Inside 128, Outside 138. Gradient ~5.
        // 5 < 10, should be rejected.
        for y in 6..14 {
            for x in 6..14 {
                data[y * width + x] = 138;
            }
        }

        let img = ImageView::new(&data, width, height, stride).unwrap();

        let corners = [
            Point { x: 6.0, y: 6.0 },
            Point { x: 14.0, y: 6.0 },
            Point { x: 14.0, y: 14.0 },
            Point { x: 6.0, y: 14.0 },
        ];

        let score = calculate_edge_score(&img, corners);
        // Gradient should be roughly (138-128)/2 = 5 per pixel boundary?
        // Sobel-like (p(x+1)-p(x-1))/2.
        // At edge x=6: left=128, right=138. (138-128)/2 = 5.
        // Magnitude 5.0.
        // Threshold is 10.0.
        assert!(score < 10.0, "Score {} should be < 10.0", score);

        // Draw a strong quad (contrast 50)
        // Inside 200, Outside 50. Gradient ~75.
        // 75 > 10, should pass.
        for y in 6..14 {
            for x in 6..14 {
                data[y * width + x] = 200;
            }
        }
        // Restore background
        for y in 0..height {
            for x in 0..width {
                if x < 6 || x >= 14 || y < 6 || y >= 14 {
                    data[y * width + x] = 50;
                }
            }
        }
        let img = ImageView::new(&data, width, height, stride).unwrap();
        let score = calculate_edge_score(&img, corners);
        assert!(score > 40.0, "Score {} should be > 40.0", score);
    }

    proptest! {
        #[test]
        fn prop_douglas_peucker_invariants(
            points in prop::collection::vec((0.0..1000.0, 0.0..1000.0), 3..100),
            epsilon in 0.1..10.0f64
        ) {
            let arena = Bump::new();
            let contour: Vec<Point> = points.iter().map(|&(x, y)| Point { x, y }).collect();
            let simplified = douglas_peucker(&arena, &contour, epsilon);

            // 1. Simplified points are a subset of original points (by coordinates)
            for p in &simplified {
                assert!(contour.iter().any(|&op| (op.x - p.x).abs() < 1e-9 && (op.y - p.y).abs() < 1e-9));
            }

            // 2. End points are preserved
            assert_eq!(simplified[0].x, contour[0].x);
            assert_eq!(simplified[0].y, contour[0].y);
            assert_eq!(simplified.last().unwrap().x, contour.last().unwrap().x);
            assert_eq!(simplified.last().unwrap().y, contour.last().unwrap().y);

            // 3. Simplified contour has fewer or equal points
            assert!(simplified.len() <= contour.len());

            // 4. All original points are at most epsilon away from the simplified segment
            for i in 1..simplified.len() {
                let a = simplified[i-1];
                let b = simplified[i];

                // Find indices in original contour matching simplified points
                let mut start_idx = None;
                let mut end_idx = None;
                for (j, op) in contour.iter().enumerate() {
                    if (op.x - a.x).abs() < 1e-9 && (op.y - a.y).abs() < 1e-9 {
                        start_idx = Some(j);
                    }
                    if (op.x - b.x).abs() < 1e-9 && (op.y - b.y).abs() < 1e-9 {
                        end_idx = Some(j);
                    }
                }

                if let (Some(s), Some(e)) = (start_idx, end_idx) {
                    for j in s..=e {
                        let d = perpendicular_distance(contour[j], a, b);
                        assert!(d <= epsilon + 1e-7, "Distance {} > epsilon {} at point index {}", d, epsilon, j);
                    }
                }
            }
        }
    }

    // ========================================================================
    // QUAD EXTRACTION ROBUSTNESS TESTS
    // ========================================================================

    use crate::config::TagFamily;
    use crate::segmentation::label_components_with_stats;
    use crate::test_utils::{
        TestImageParams, compute_corner_error, generate_test_image_with_params,
    };
    use crate::threshold::ThresholdEngine;

    /// Helper: Generate a tag image and run through threshold + segmentation + quad extraction.
    fn run_quad_extraction(tag_size: usize, canvas_size: usize) -> (Vec<Detection>, [[f64; 2]; 4]) {
        let params = TestImageParams {
            family: TagFamily::AprilTag36h11,
            id: 0,
            tag_size,
            canvas_size,
            ..Default::default()
        };

        let (data, corners) = generate_test_image_with_params(&params);
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

        let engine = ThresholdEngine::new();
        let stats = engine.compute_tile_stats(&img);
        let mut binary = vec![0u8; canvas_size * canvas_size];
        engine.apply_threshold(&img, &stats, &mut binary);

        let arena = Bump::new();
        let label_result = label_components_with_stats(&arena, &binary, canvas_size, canvas_size, true);
        let detections = extract_quads_fast(&arena, &img, &label_result);

        (detections, corners)
    }

    /// Test quad extraction at varying tag sizes.
    #[test]
    fn test_quad_extraction_at_varying_sizes() {
        let canvas_size = 640;
        let tag_sizes = [32, 48, 64, 100, 150, 200, 300];

        for tag_size in tag_sizes {
            let (detections, _corners) = run_quad_extraction(tag_size, canvas_size);
            let detected = !detections.is_empty();

            if tag_size >= 48 {
                assert!(detected, "Tag size {}: No quad detected", tag_size);
            }

            if detected {
                println!(
                    "Tag size {:>3}px: {} quads, center=[{:.1},{:.1}]",
                    tag_size,
                    detections.len(),
                    detections[0].center[0],
                    detections[0].center[1]
                );
            } else {
                println!("Tag size {:>3}px: No quad detected", tag_size);
            }
        }
    }

    /// Test corner detection accuracy vs ground truth.
    #[test]
    fn test_quad_corner_accuracy() {
        let canvas_size = 640;
        let tag_sizes = [100, 150, 200, 300];

        for tag_size in tag_sizes {
            let (detections, gt_corners) = run_quad_extraction(tag_size, canvas_size);

            assert!(
                !detections.is_empty(),
                "Tag size {}: No detection",
                tag_size
            );

            let det_corners = detections[0].corners;
            let error = compute_corner_error(&det_corners, &gt_corners);

            let max_error = 5.0;
            assert!(
                error < max_error,
                "Tag size {}: Corner error {:.2}px exceeds max",
                tag_size,
                error
            );

            println!("Tag size {:>3}px: Corner error = {:.2}px", tag_size, error);
        }
    }

    /// Test that quad center is approximately correct.
    #[test]
    fn test_quad_center_accuracy() {
        let canvas_size = 640;
        let tag_size = 150;

        let (detections, gt_corners) = run_quad_extraction(tag_size, canvas_size);
        assert!(!detections.is_empty(), "No detection");

        let expected_cx =
            (gt_corners[0][0] + gt_corners[1][0] + gt_corners[2][0] + gt_corners[3][0]) / 4.0;
        let expected_cy =
            (gt_corners[0][1] + gt_corners[1][1] + gt_corners[2][1] + gt_corners[3][1]) / 4.0;

        let det_center = detections[0].center;
        let dx = det_center[0] - expected_cx;
        let dy = det_center[1] - expected_cy;
        let center_error = (dx * dx + dy * dy).sqrt();

        assert!(
            center_error < 2.0,
            "Center error {:.2}px exceeds 2px",
            center_error
        );

        println!(
            "Quad center: detected=[{:.1},{:.1}], expected=[{:.1},{:.1}], error={:.2}px",
            det_center[0], det_center[1], expected_cx, expected_cy, center_error
        );
    }
}

#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
/// SOTA Boundary Tracing using robust border following.
///
/// This implementation uses a state-machine based approach to follow the border
/// of a connected component. It is designed to be robust and efficient,
/// avoiding expensive operations in the inner loop.
fn trace_boundary<'a>(
    arena: &'a Bump,
    labels: &[u32],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    target_label: u32,
) -> BumpVec<'a, Point> {
    let mut points = BumpVec::new_in(arena);

    // Moore Neighborhood directions (CW order starting from Top)
    // index: 0, 1, 2, 3, 4, 5, 6, 7
    // dir:   T, TR, R, BR, B, BL, L, TL
    let dx = [0, 1, 1, 1, 0, -1, -1, -1];
    let dy = [-1, -1, 0, 1, 1, 1, 0, -1];

    let mut curr_x = start_x;
    let mut curr_y = start_y;
    let mut walk_dir = 2; // Initial guess: we found the leftmost-topmost, so move Right

    for _ in 0..10000 {
        points.push(Point {
            x: curr_x as f64,
            y: curr_y as f64,
        });

        let mut found = false;
        // Search neighbors CCW starting from "relative left" of movement
        // If we moved in walk_dir, we start searching from (walk_dir + 6) % 8
        for i in 0..8 {
            let dir = (walk_dir + 6 + i) % 8;
            let nx = curr_x as isize + dx[dir];
            let ny = curr_y as isize + dy[dir];

            if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                if labels[ny as usize * width + nx as usize] == target_label {
                    curr_x = nx as usize;
                    curr_y = ny as usize;
                    walk_dir = dir;
                    found = true;
                    break;
                }
            }
        }

        if !found || (curr_x == start_x && curr_y == start_y) {
            break;
        }
    }

    points
}

/// Simplified version of CHAIN_APPROX_SIMPLE:
/// Removes all redundant points on straight lines.
pub fn chain_approximation<'a>(arena: &'a Bump, points: &[Point]) -> BumpVec<'a, Point> {
    if points.len() < 3 {
        let mut v = BumpVec::new_in(arena);
        v.extend_from_slice(points);
        return v;
    }

    let mut result = BumpVec::new_in(arena);
    result.push(points[0]);

    for i in 1..points.len() - 1 {
        let p_prev = points[i - 1];
        let p_curr = points[i];
        let p_next = points[i + 1];

        let dx1 = p_curr.x - p_prev.x;
        let dy1 = p_curr.y - p_prev.y;
        let dx2 = p_next.x - p_curr.x;
        let dy2 = p_next.y - p_curr.y;

        // If directions are strictly different, it's a corner
        // Using exact float comparison is safe here because these are pixel coordinates (integers)
        if (dx1 * dy2 - dx2 * dy1).abs() > 1e-6 {
            result.push(p_curr);
        }
    }

    result.push(*points.last().unwrap_or(&points[0]));
    result
}
