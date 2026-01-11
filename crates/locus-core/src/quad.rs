use crate::Detection;
use crate::image::ImageView;
use bumpalo::Bump;
use multiversion::multiversion;

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

use bumpalo::collections::Vec as BumpVec;

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

        rec_results1.pop(); // Remove duplicate point
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

/// Extract quads from labeled connected components.
pub fn extract_quads<'a>(arena: &'a Bump, img: &ImageView, labels: &[u32]) -> Vec<Detection> {
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

            // A label can only be a "top-left" corner of a component if
            // the pixel above or to the left is not the same label.
            // This avoids many redundant 'processed_labels' checks.
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

use nalgebra::SMatrix;

/// Refine corners to sub-pixel accuracy based on image gradients.
pub fn refine_corner(img: &ImageView, p: Point) -> Point {
    let win_size = 3;
    let mut ata = SMatrix::<f64, 2, 2>::zeros();
    let mut atb = SMatrix::<f64, 2, 1>::zeros();

    let ix = p.x as isize;
    let iy = p.y as isize;

    for dy in -win_size..=win_size {
        for dx in -win_size..=win_size {
            let x = ix + dx;
            let y = iy + dy;

            if x > 0 && x < (img.width - 1) as isize && y > 0 && y < (img.height - 1) as isize {
                let gx = (img.get_pixel(x as usize + 1, y as usize) as f64
                    - img.get_pixel(x as usize - 1, y as usize) as f64)
                    * 0.5;
                let gy = (img.get_pixel(x as usize, y as usize + 1) as f64
                    - img.get_pixel(x as usize, y as usize - 1) as f64)
                    * 0.5;

                ata[(0, 0)] += gx * gx;
                ata[(0, 1)] += gx * gy;
                ata[(1, 1)] += gy * gy;

                atb[(0, 0)] += gx * (gx * x as f64 + gy * y as f64);
                atb[(1, 0)] += gy * (gx * x as f64 + gy * y as f64);
            }
        }
    }
    ata[(1, 0)] = ata[(0, 1)];

    if let Some(inv) = ata.try_inverse() {
        let res = inv * atb;
        Point {
            x: res[(0, 0)],
            y: res[(1, 0)],
        }
    } else {
        p
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
    let mut enter_dir = 0; // Direction we entered from

    loop {
        points.push(Point {
            x: curr_x as f64,
            y: curr_y as f64,
        });

        let mut found = false;
        // Check 8 neighbors clockwise starting from (enter_dir + 1)
        for i in 0..8 {
            let dir = (enter_dir + i) % 8;
            let nx = curr_x as isize + dx[dir];
            let ny = curr_y as isize + dy[dir];

            if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                if labels[ny as usize * width + nx as usize] == target_label {
                    curr_x = nx as usize;
                    curr_y = ny as usize;
                    enter_dir = (dir + 5) % 8; // Backtrack direction for next step
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
