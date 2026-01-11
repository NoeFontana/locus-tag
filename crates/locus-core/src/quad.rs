use crate::Detection;
use crate::image::ImageView;
use crate::segmentation::{ComponentStats, LabelResult};
use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use multiversion::multiversion;
use nalgebra::SMatrix;

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Fast quad extraction using bounding box stats from CCL.
/// Only traces contours for components that pass geometric filters.
pub fn extract_quads_fast(
    arena: &Bump,
    img: &ImageView,
    label_result: &LabelResult,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    let labels = label_result.labels;
    let stats = &label_result.component_stats;

    for (label_idx, stat) in stats.iter().enumerate() {
        let label = (label_idx + 1) as u32;

        // Fast geometric filtering using bounding box
        let bbox_w = (stat.max_x - stat.min_x) as u32 + 1;
        let bbox_h = (stat.max_y - stat.min_y) as u32 + 1;
        let bbox_area = bbox_w * bbox_h;

        // Filter: too small or too large
        if bbox_area < 400 || bbox_area > (img.width * img.height / 4) as u32 {
            continue;
        }

        // Filter: not roughly square (aspect ratio)
        let aspect = bbox_w.max(bbox_h) as f32 / bbox_w.min(bbox_h).max(1) as f32;
        if aspect > 3.0 {
            continue;
        }

        // Filter: fill ratio (should be ~50-80% for a tag with inner pattern)
        let fill = stat.pixel_count as f32 / bbox_area as f32;
        if fill < 0.3 || fill > 0.95 {
            continue;
        }

        // Passed filters - trace boundary and fit quad
        let start_x = stat.min_x as usize;
        let start_y = stat.min_y as usize;

        // Find actual boundary start point
        let mut found = false;
        let mut sx = start_x;
        let mut sy = start_y;
        for y in stat.min_y..=stat.max_y {
            for x in stat.min_x..=stat.max_x {
                if labels[y as usize * img.width + x as usize] == label {
                    sx = x as usize;
                    sy = y as usize;
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }

        if !found {
            continue;
        }

        let contour = trace_boundary(arena, labels, img.width, img.height, sx, sy, label);

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
                        // Compute center first, then refine corners
                        let center = polygon_center(&simplified);
                        let corners = [
                            refine_corner(img, simplified[0], center),
                            refine_corner(img, simplified[1], center),
                            refine_corner(img, simplified[2], center),
                            refine_corner(img, simplified[3], center),
                        ];

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
                        });
                    }
                }
            }
        }
    }
    detections
}

/// Legacy extract_quads for backward compatibility.
pub fn extract_quads<'a>(arena: &'a Bump, img: &ImageView, labels: &[u32]) -> Vec<Detection> {
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
    for i in 0..n {
        cx += points[i].x;
        cy += points[i].y;
    }
    [cx / n as f64, cy / n as f64]
}

/// Refine corners to sub-pixel accuracy using gradient-based edge search.
///
/// Searches along the direction from tag center to corner to find the true
/// outer edge of the tag (bright-to-dark transition).
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
        let x = p.x + dx * step as f64;
        let y = p.y + dy * step as f64;

        let ix = x as isize;
        let iy = y as isize;

        if ix > 1 && ix < (img.width - 2) as isize && iy > 1 && iy < (img.height - 2) as isize {
            let gx = (img.get_pixel((ix + 1) as usize, iy as usize) as f64
                - img.get_pixel((ix - 1) as usize, iy as usize) as f64)
                * 0.5;
            let gy = (img.get_pixel(ix as usize, (iy + 1) as usize) as f64
                - img.get_pixel(ix as usize, (iy - 1) as usize) as f64)
                * 0.5;
            let mag = gx * gx + gy * gy;

            if mag > best_mag {
                best_mag = mag;
                best_x = x;
                best_y = y;
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
                let gx = (img.get_pixel((ix + 1) as usize, iy as usize) as f64
                    - img.get_pixel((ix - 1) as usize, iy as usize) as f64)
                    * 0.5;
                let gy = (img.get_pixel(ix as usize, (iy + 1) as usize) as f64
                    - img.get_pixel(ix as usize, (iy - 1) as usize) as f64)
                    * 0.5;
                mags[i] = gx * gx + gy * gy;
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

#[multiversion(targets = "simd")]
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

            if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                if labels[ny as usize * width + nx as usize] == target_label {
                    curr_x = nx as usize;
                    curr_y = ny as usize;
                    enter_dir = (dir + 5) % 8;
                    found = true;
                    break;
                }
            }
        }

        if !found || (curr_x == start_x && curr_y == start_y) || points.len() > 10000 {
            break;
        }
    }
    points
}
