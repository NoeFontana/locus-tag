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
pub fn extract_quads_with_config(
    arena: &Bump,
    img: &ImageView,
    label_result: &LabelResult,
    config: &DetectorConfig,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    let labels = label_result.labels;
    let stats = &label_result.component_stats;

    // Compute minimum edge length squared once
    let min_edge_len_sq = config.quad_min_edge_length * config.quad_min_edge_length;

    for (label_idx, stat) in stats.iter().enumerate() {
        let label = (label_idx + 1) as u32;

        // Fast geometric filtering using bounding box
        let bbox_w = u32::from(stat.max_x - stat.min_x) + 1;
        let bbox_h = u32::from(stat.max_y - stat.min_y) + 1;
        let bbox_area = bbox_w * bbox_h;

        // Filter: too small or too large
        if bbox_area < config.quad_min_area || bbox_area > (img.width * img.height * 9 / 10) as u32
        {
            continue;
        }

        // Filter: not roughly square (aspect ratio)
        let aspect = bbox_w.max(bbox_h) as f32 / bbox_w.min(bbox_h).max(1) as f32;
        if aspect > config.quad_max_aspect_ratio {
            continue;
        }

        // Filter: fill ratio (should be ~50-80% for a tag with inner pattern)
        let fill = stat.pixel_count as f32 / bbox_area as f32;
        if fill < config.quad_min_fill_ratio || fill > config.quad_max_fill_ratio {
            continue;
        }

        // Passed filters - trace boundary and fit quad
        // Use pre-computed first pixel from CCL stats (O(1) instead of O(w*h))
        let sx = stat.first_pixel_x as usize;
        let sy = stat.first_pixel_y as usize;

        let contour = trace_boundary(arena, labels, img.width, img.height, sx, sy, label);

        if contour.len() >= 30 {
            // Adaptive epsilon: 2% of perimeter (like OpenCV approxPolyDP)
            let perimeter = contour.len() as f64;
            let epsilon = (perimeter * 0.02).max(2.0);
            let simplified = douglas_peucker(arena, &contour, epsilon);
            // Allow somewhat jagged quads (up to 10 points) and reduce them
            if simplified.len() >= 5 && simplified.len() <= 11 {
                let simpl_len = simplified.len();
                let reduced = if simpl_len == 5 {
                    simplified
                } else {
                    reduce_to_quad(arena, &simplified)
                };

                // Must have successfully reduced to 4 corners (+1 closing)
                if reduced.len() == 5 {
                    let area = polygon_area(&reduced);
                    let perimeter = contour.len() as f64;
                    // Relax compactness check slightly for small jagged tags (was 0.5)
                    let compactness = (12.566 * area) / (perimeter * perimeter);

                    let min_area_f64 = f64::from(config.quad_min_area);
                    if area > min_area_f64 && compactness > 0.4 {
                        let mut ok = true;
                        for i in 0..4 {
                            let d2 = (reduced[i].x - reduced[i + 1].x).powi(2)
                                + (reduced[i].y - reduced[i + 1].y).powi(2);
                            if d2 < min_edge_len_sq {
                                ok = false;
                                break;
                            }
                        }

                        if ok {
                            // Compute center first, then refine corners
                            let center = polygon_center(&reduced);
                            let corners = [
                                refine_corner(img, reduced[0], center),
                                refine_corner(img, reduced[1], center),
                                refine_corner(img, reduced[2], center),
                                refine_corner(img, reduced[3], center),
                            ];

                            // Filter: weak edge alignment
                            let edge_score = calculate_edge_score(img, corners);
                            if edge_score > config.quad_min_edge_score {
                                detections.push(Detection {
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
    }
    detections
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

            if contour.len() >= 30 {
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
pub fn douglas_peucker<'a>(arena: &'a Bump, points: &[Point], epsilon: f64) -> BumpVec<'a, Point> {
    if points.len() < 3 {
        let mut v = BumpVec::new_in(arena);
        v.extend_from_slice(points);
        return v;
    }

    let mut dmax = 0.0;
    let mut index = 0;
    let end = points.len() - 1;

    for i in 1..end {
        let d = perpendicular_distance(points[i], points[0], points[end]);
        if d > dmax {
            index = i;
            dmax = d;
        }
    }

    if dmax > epsilon {
        let mut rec_results1 = douglas_peucker(arena, &points[0..=index], epsilon);
        let rec_results2 = douglas_peucker(arena, &points[index..=end], epsilon);

        rec_results1.pop();
        rec_results1.extend(rec_results2);
        rec_results1
    } else {
        let mut v = BumpVec::new_in(arena);
        v.push(points[0]);
        v.push(points[end]);
        v
    }
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

/// Refine corners to sub-pixel accuracy using gradient-based edge search.
///
/// Searches along the direction from tag center to corner to find the true
/// outer edge of the tag (bright-to-dark transition).
#[must_use]
pub fn refine_corner(img: &ImageView, p: Point, center: [f64; 2]) -> Point {
    // Direction from center to corner (points outward)
    let dir_x = p.x - center[0];
    let dir_y = p.y - center[1];
    let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();

    if dir_len < 1.0 {
        return p;
    }

    // Normalize direction
    let dx = dir_x / dir_len;
    let dy = dir_y / dir_len;

    // Search along the ray from corner outward (and slightly inward)
    let mut best_x = p.x;
    let mut best_y = p.y;
    let mut best_mag = 0.0f64;

    // Sample 11 points: 5 inward, current, 5 outward
    for step in -3..=7 {
        let x = p.x + dx * f64::from(step);
        let y = p.y + dy * f64::from(step);

        let ix = x as isize;
        let iy = y as isize;

        // Use a safe margin to allow accessing neighbors without checks
        if ix > 1 && ix < (img.width - 2) as isize && iy > 1 && iy < (img.height - 2) as isize {
            unsafe {
                let ix_u = ix as usize;
                let iy_u = iy as usize;
                let gx = (f64::from(img.get_pixel_unchecked(ix_u + 1, iy_u))
                    - f64::from(img.get_pixel_unchecked(ix_u - 1, iy_u)))
                    * 0.5;
                let gy = (f64::from(img.get_pixel_unchecked(ix_u, iy_u + 1))
                    - f64::from(img.get_pixel_unchecked(ix_u, iy_u - 1)))
                    * 0.5;
                let mag = gx * gx + gy * gy;

                if mag > best_mag {
                    best_mag = mag;
                    best_x = x;
                    best_y = y;
                }
            }
        }
    }

    // Quadratic sub-pixel refinement along the search direction
    let best_ix = best_x as isize;
    let best_iy = best_y as isize;

    if best_ix > 1
        && best_ix < (img.width - 2) as isize
        && best_iy > 1
        && best_iy < (img.height - 2) as isize
    {
        // Sample gradient magnitudes at best-1, best, best+1 along direction
        let mut mags = [0.0f64; 3];
        for (i, offset) in [-1.0, 0.0, 1.0].iter().enumerate() {
            let x = best_x + dx * offset;
            let y = best_y + dy * offset;
            let ix = x as isize;
            let iy = y as isize;

            if ix > 0 && ix < (img.width - 1) as isize && iy > 0 && iy < (img.height - 1) as isize {
                unsafe {
                    let ix_u = ix as usize;
                    let iy_u = iy as usize;
                    let gx = (f64::from(img.get_pixel_unchecked(ix_u + 1, iy_u))
                        - f64::from(img.get_pixel_unchecked(ix_u - 1, iy_u)))
                        * 0.5;
                    let gy = (f64::from(img.get_pixel_unchecked(ix_u, iy_u + 1))
                        - f64::from(img.get_pixel_unchecked(ix_u, iy_u - 1)))
                        * 0.5;
                    mags[i] = gx * gx + gy * gy;
                }
            }
        }

        // Quadratic interpolation for sub-pixel peak
        let num = mags[2] - mags[0];
        let den = 2.0 * (mags[0] + mags[2] - 2.0 * mags[1]);
        let sub_offset = if den.abs() > 1e-6 {
            (-num / den).clamp(-0.5, 0.5)
        } else {
            0.0
        };

        return Point {
            x: best_x + dx * sub_offset,
            y: best_y + dy * sub_offset,
        };
    }

    Point {
        x: best_x,
        y: best_y,
    }
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

            let ix = x as isize;
            let iy = y as isize;

            if ix > 0 && ix < (img.width - 1) as isize && iy > 0 && iy < (img.height - 1) as isize {
                // SAFETY: We checked bounds above. ix, iy >= 1 and <= width/height - 2.
                // So (ix +/- 1) are safely within [0, width-1].
                unsafe {
                    let gx = (f64::from(img.get_pixel_unchecked((ix + 1) as usize, iy as usize))
                        - f64::from(img.get_pixel_unchecked((ix - 1) as usize, iy as usize)))
                        * 0.5;
                    let gy = (f64::from(img.get_pixel_unchecked(ix as usize, (iy + 1) as usize))
                        - f64::from(img.get_pixel_unchecked(ix as usize, (iy - 1) as usize)))
                        * 0.5;
                    edge_mag_sum += (gx * gx + gy * gy).sqrt();
                }
            }
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
        let label_result = label_components_with_stats(&arena, &binary, canvas_size, canvas_size);
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
    let dx = [0, 1, 1, 1, 0, -1, -1, -1];
    let dy = [-1, -1, 0, 1, 1, 1, 0, -1];

    let mut curr_x = start_x;
    let mut curr_y = start_y;
    let mut enter_dir = 0;

    loop {
        points.push(Point {
            x: curr_x as f64,
            y: curr_y as f64,
        });

        let mut found = false;
        for i in 0..8 {
            let dir = (enter_dir + i) % 8;
            let nx = curr_x as isize + dx[dir];
            let ny = curr_y as isize + dy[dir];

            if nx >= 0
                && nx < width as isize
                && ny >= 0
                && ny < height as isize
                && labels[ny as usize * width + nx as usize] == target_label
            {
                curr_x = nx as usize;
                curr_y = ny as usize;
                enter_dir = (dir + 5) % 8;
                found = true;
                break;
            }
        }

        if !found || (curr_x == start_x && curr_y == start_y) || points.len() > 10000 {
            break;
        }
    }
    points
}
