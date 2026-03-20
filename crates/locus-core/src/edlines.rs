//! EDLines quad extraction: localized Edge Drawing within a component bounding box.
//!
//! For each surviving component, this module:
//! 1. Computes Sobel gradients within the ROI.
//! 2. Identifies anchor pixels (local gradient maxima above threshold).
//! 3. Routes edge chains from anchors along the gradient ridge.
//! 4. Fits line segments to chains via PCA (reusing `MomentAccumulator`).
//! 5. Clusters segments into 4 angular groups and intersects adjacent lines to get corners.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::items_after_statements)]

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

use crate::gwlf::MomentAccumulator;
use crate::image::ImageView;
use crate::quad::Point;
use crate::segmentation::ComponentStats;

/// Configuration for the EDLines quad extractor.
pub(crate) struct EdLinesConfig {
    /// Gradient magnitude threshold to qualify as an anchor (default: 20).
    pub grad_threshold: u16,
    /// Maximum mean-squared residual for a pixel to remain in a line segment (default: 1.5).
    pub line_mse_threshold: f64,
    /// Minimum number of pixels in an accepted line segment (default: 8).
    pub min_segment_length: usize,
}

impl EdLinesConfig {
    /// Derive config from the detector config (uses hard-coded defaults for now).
    #[must_use]
    pub fn from_detector_config(_cfg: &crate::config::DetectorConfig) -> Self {
        Self {
            grad_threshold: 20,
            line_mse_threshold: 1.5,
            min_segment_length: 8,
        }
    }
}

/// Compact gradient cell for the ROI / full-image gradient buffer.
#[derive(Clone, Copy, Default)]
pub(crate) struct Grad {
    pub gx: i16,
    pub gy: i16,
    pub mag: u16,
}


/// Compute Sobel gradients into a pre-allocated flat ROI buffer.
/// `roi_w` and `roi_h` include a 1-pixel border on each side.
fn compute_roi_gradients(
    img: &ImageView,
    x0: usize, // ROI left  (inclusive, with border already added by caller)
    y0: usize, // ROI top
    roi_w: usize,
    roi_h: usize,
    buf: &mut [Grad],
) {
    for ry in 1..roi_h.saturating_sub(1) {
        let iy = y0 + ry;
        if iy == 0 || iy + 1 >= img.height {
            continue;
        }
        for rx in 1..roi_w.saturating_sub(1) {
            let ix = x0 + rx;
            if ix == 0 || ix + 1 >= img.width {
                continue;
            }
            let p00 = i16::from(img.get_pixel(ix - 1, iy - 1));
            let p10 = i16::from(img.get_pixel(ix, iy - 1));
            let p20 = i16::from(img.get_pixel(ix + 1, iy - 1));
            let p01 = i16::from(img.get_pixel(ix - 1, iy));
            let p21 = i16::from(img.get_pixel(ix + 1, iy));
            let p02 = i16::from(img.get_pixel(ix - 1, iy + 1));
            let p12 = i16::from(img.get_pixel(ix, iy + 1));
            let p22 = i16::from(img.get_pixel(ix + 1, iy + 1));

            let gx = -p00 + p20 - 2 * p01 + 2 * p21 - p02 + p22;
            let gy = -p00 - 2 * p10 - p20 + p02 + 2 * p12 + p22;
            let mag = ((gx.abs() + gy.abs()) as u16).min(1023);

            buf[ry * roi_w + rx] = Grad { gx, gy, mag };
        }
    }
}

/// Walk from an anchor along the gradient ridge in one direction, collecting pixels.
/// Marks visited pixels to prevent reuse by subsequent anchors.
#[allow(clippy::too_many_arguments)]
fn route_chain_one_dir<'bump>(
    arena: &'bump Bump,
    grads: &[Grad],
    roi_w: usize,
    roi_h: usize,
    start_rx: usize,
    start_ry: usize,
    visited: &mut [bool],
    mag_thresh: u16,
    forward: bool,
) -> BumpVec<'bump, (u16, u16)> {
    let mut chain: BumpVec<(u16, u16)> = BumpVec::new_in(arena);
    let mut rx = start_rx as i32;
    let mut ry = start_ry as i32;

    loop {
        if rx < 1 || ry < 1 || rx >= roi_w as i32 - 1 || ry >= roi_h as i32 - 1 {
            break;
        }
        let idx = ry as usize * roi_w + rx as usize;
        if visited[idx] {
            break;
        }
        visited[idx] = true;
        chain.push((rx as u16, ry as u16));

        // Step direction: perpendicular to gradient (along the edge).
        let cur = grads[idx];
        if cur.mag < mag_thresh {
            break;
        }
        let angle = f32::from(cur.gy).atan2(f32::from(cur.gx));
        let sign = if forward { 1.0f32 } else { -1.0 };
        let step_dx = (-angle.sin() * sign).round() as i32;
        let step_dy = (angle.cos() * sign).round() as i32;

        // Find the strongest unvisited neighbor in the forward half-plane
        // whose gradient direction is consistent with the current pixel (≤ 45° tolerance).
        // This stops routing at corners where the edge direction changes sharply.
        let angle_tolerance = std::f32::consts::PI / 6.0; // 30 degrees — strict enough to stop at corners
        let mut best_mag = mag_thresh.saturating_sub(1);
        let mut best_dx = step_dx;
        let mut best_dy = step_dy;
        let mut found = false;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if step_dx != 0 || step_dy != 0 {
                    // Only forward half-plane.
                    if dx * step_dx + dy * step_dy < 0 {
                        continue;
                    }
                }
                let nx = rx + dx;
                let ny = ry + dy;
                if nx < 1 || ny < 1 || nx >= roi_w as i32 - 1 || ny >= roi_h as i32 - 1 {
                    continue;
                }
                let nidx = ny as usize * roi_w + nx as usize;
                if visited[nidx] {
                    continue;
                }
                let g_next = grads[nidx];
                let m = g_next.mag;
                if m <= best_mag {
                    continue;
                }
                // Gradient direction consistency: reject sharp turns.
                let next_angle = f32::from(g_next.gy).atan2(f32::from(g_next.gx));
                let diff = {
                    let d = (next_angle - angle).rem_euclid(2.0 * std::f32::consts::PI);
                    if d > std::f32::consts::PI { 2.0 * std::f32::consts::PI - d } else { d }
                };
                if diff > angle_tolerance {
                    continue;
                }
                best_mag = m;
                best_dx = dx;
                best_dy = dy;
                found = true;
            }
        }

        if !found {
            break;
        }
        rx += best_dx;
        ry += best_dy;
    }

    chain
}

/// Walk both directions from an anchor and merge into a single chain.
#[allow(clippy::too_many_arguments)]
fn route_chain<'bump>(
    arena: &'bump Bump,
    grads: &[Grad],
    roi_w: usize,
    roi_h: usize,
    start_rx: usize,
    start_ry: usize,
    visited: &mut [bool],
    mag_thresh: u16,
) -> BumpVec<'bump, (u16, u16)> {
    // Forward direction: start -> end
    let fwd = route_chain_one_dir(
        arena, grads, roi_w, roi_h, start_rx, start_ry, visited, mag_thresh, true,
    );
    // Backward direction from start (start is already visited, so begin from first unvisited neighbor).
    // Re-find the backward neighbor manually.
    let mut bwd: BumpVec<(u16, u16)> = BumpVec::new_in(arena);
    {
        let cur = grads[start_ry * roi_w + start_rx];
        if cur.mag >= mag_thresh {
            let angle = f32::from(cur.gy).atan2(f32::from(cur.gx));
            let step_dx = (angle.sin()).round() as i32; // backward: opposite sign
            let step_dy = (-angle.cos()).round() as i32;
            let nx = start_rx as i32 + step_dx;
            let ny = start_ry as i32 + step_dy;
            if nx >= 1 && ny >= 1 && nx < roi_w as i32 - 1 && ny < roi_h as i32 - 1 {
                let nidx = ny as usize * roi_w + nx as usize;
                if !visited[nidx] {
                    bwd = route_chain_one_dir(
                        arena, grads, roi_w, roi_h, nx as usize, ny as usize,
                        visited, mag_thresh, false,
                    );
                }
            }
        }
    }

    // Merge: bwd (reversed) + fwd
    let mut chain: BumpVec<(u16, u16)> = BumpVec::with_capacity_in(fwd.len() + bwd.len(), arena);
    for &pt in bwd.iter().rev() {
        chain.push(pt);
    }
    for &pt in fwd.as_slice() {
        chain.push(pt);
    }
    chain
}

/// Fit a line segment to a pixel chain using PCA via `MomentAccumulator`.
/// Returns `(nx, ny, d)` (homogeneous line: nx*x + ny*y + d = 0) and MSE,
/// or `None` if the chain is too short or degenerate.
fn fit_line_pca(chain: &[(u16, u16)]) -> Option<(f64, f64, f64, f64)> {
    let mut acc = MomentAccumulator::new();
    for &(x, y) in chain {
        acc.add(f64::from(x), f64::from(y), 1.0);
    }
    let cov = acc.covariance()?;
    let centroid = acc.centroid()?;

    let a = cov[(0, 0)];
    let b_val = cov[(0, 1)];
    let c = cov[(1, 1)];

    // Eigendecomposition for the minimum-variance direction (line normal).
    let trace = a + c;
    let disc = ((a - c) * (a - c) + 4.0 * b_val * b_val).sqrt();
    let lambda_min = (trace - disc) / 2.0;

    let (nx, ny) = if b_val.abs() > 1e-9 {
        let vx = b_val;
        let vy = lambda_min - a;
        let len = (vx * vx + vy * vy).sqrt();
        (vx / len, vy / len)
    } else if a < c {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };

    let d = -(nx * centroid.x + ny * centroid.y);

    // Compute MSE = λ_min / n  (mean squared orthogonal distance).
    let mse = lambda_min.max(0.0) / acc.sum_w.max(1.0);

    Some((nx, ny, d, mse))
}

/// Intersect two homogeneous lines (n1x*x + n1y*y + d1 = 0) and (n2x*x + n2y*y + d2 = 0).
#[inline]
fn intersect_lines(
    n1x: f64, n1y: f64, d1: f64,
    n2x: f64, n2y: f64, d2: f64,
) -> Option<(f64, f64)> {
    // Cross product of homogeneous lines [n1x, n1y, d1] x [n2x, n2y, d2]
    let wx = n1y * d2 - d1 * n2y;
    let wy = d1 * n2x - n1x * d2;
    let ww = n1x * n2y - n1y * n2x;
    if ww.abs() < 1e-9 {
        return None; // Parallel
    }
    Some((wx / ww, wy / ww))
}

/// Angular difference between two DIRECTED gradient angles (period 2π), result in [0, π].
/// Opposite directions (differing by π) have diff = π, not 0.
#[inline]
fn grad_angle_diff(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(2.0 * std::f32::consts::PI);
    if d > std::f32::consts::PI {
        2.0 * std::f32::consts::PI - d
    } else {
        d
    }
}

/// Extract a quad using the EDLines algorithm within a component's bounding box.
///
/// Computes Sobel gradients within the ROI, identifies anchors, routes edge chains,
/// fits line segments via PCA, and returns 4 corners as intersections of dominant lines.
pub(crate) fn extract_quad_edlines(
    arena: &Bump,
    img: &ImageView,
    stat: &ComponentStats,
    cfg: &EdLinesConfig,
) -> Option<[Point; 4]> {
    // ROI with 1-pixel border on each side.
    let rx0 = (stat.min_x as usize).saturating_sub(1);
    let ry0 = (stat.min_y as usize).saturating_sub(1);
    let rx1 = (stat.max_x as usize + 2).min(img.width);
    let ry1 = (stat.max_y as usize + 2).min(img.height);

    let roi_w = rx1 - rx0;
    let roi_h = ry1 - ry0;

    if roi_w < 5 || roi_h < 5 {
        return None;
    }

    // Allocate gradient buffer in the arena.
    let grads: &mut [Grad] = arena.alloc_slice_fill_default(roi_w * roi_h);
    compute_roi_gradients(img, rx0, ry0, roi_w, roi_h, grads);

    let visited: &mut [bool] = arena.alloc_slice_fill_copy(roi_w * roi_h, false);

    // Collect anchors: scan every interior pixel; the local-maximum + magnitude
    // criteria do the filtering. Subsampling via anchor_interval would risk missing
    // edges that happen to fall on off-phase rows/columns.
    let mut anchors: BumpVec<(usize, usize)> = BumpVec::new_in(arena);

    for ry in 1..roi_h - 1 {
        for rx in 1..roi_w - 1 {
            let g = grads[ry * roi_w + rx];
            if g.mag < cfg.grad_threshold {
                continue;
            }
            // Local max along the gradient direction.
            let ax = f32::from(g.gx);
            let ay = f32::from(g.gy);
            let len = (ax * ax + ay * ay).sqrt().max(1.0);
            let gnx = (ax / len).round() as i32;
            let gny = (ay / len).round() as i32;

            let m_prev = grads[(ry as i32 - gny) as usize * roi_w + (rx as i32 - gnx) as usize].mag;
            let m_next = grads[(ry as i32 + gny) as usize * roi_w + (rx as i32 + gnx) as usize].mag;
            if g.mag >= m_prev && g.mag >= m_next {
                anchors.push((rx, ry));
            }
        }
    }

    if anchors.is_empty() {
        return None;
    }

    // A fitted line segment: PCA line params + mean gradient direction for clustering.
    struct LineFit {
        /// Homogeneous line normal (from PCA minimum-variance eigenvector).
        nx: f64,
        ny: f64,
        /// Homogeneous offset: nx*x + ny*y + d = 0 for all points on the line.
        d: f64,
        /// Mean GRADIENT direction of the chain pixels in [-π, π].
        /// This distinguishes the 4 edges of a quad (top/bottom/left/right have distinct
        /// gradient directions unlike their line normals which come in parallel pairs).
        grad_angle: f32,
    }

    let mut line_fits: BumpVec<LineFit> = BumpVec::new_in(arena);

    for &(rx, ry) in &anchors {
        let idx = ry * roi_w + rx;
        if visited[idx] {
            continue;
        }

        let chain = route_chain(arena, grads, roi_w, roi_h, rx, ry, visited, cfg.grad_threshold);

        if chain.len() < cfg.min_segment_length {
            continue;
        }

        if let Some((nx, ny, d, mse)) = fit_line_pca(&chain)
            && mse <= cfg.line_mse_threshold
        {
            // Compute mean gradient direction (directed, period 2π) for clustering.
            let mut sum_gx = 0.0f32;
            let mut sum_gy = 0.0f32;
            for &(prx, pry) in chain.as_slice() {
                let g = grads[pry as usize * roi_w + prx as usize];
                sum_gx += f32::from(g.gx);
                sum_gy += f32::from(g.gy);
            }
            let grad_angle = sum_gy.atan2(sum_gx);
            line_fits.push(LineFit { nx, ny, d, grad_angle });
        }
    }

    if line_fits.len() < 4 {
        return None;
    }

    // Cluster into 4 angular groups using the mean gradient DIRECTION (period 2π).
    // For a square: top ≈ π/2, right ≈ 0, bottom ≈ -π/2, left ≈ ±π.
    // These are 4 distinct directed angles separated by ~π/2.
    //
    // Seed initialization: use the 4*angle circular mean trick (4-fold symmetry at π/2).
    let mut sum_sin4 = 0.0f32;
    let mut sum_cos4 = 0.0f32;
    for lf in line_fits.as_slice() {
        sum_sin4 += (4.0 * lf.grad_angle).sin();
        sum_cos4 += (4.0 * lf.grad_angle).cos();
    }
    let mean_4theta = sum_sin4.atan2(sum_cos4);
    let theta0 = mean_4theta / 4.0;

    let seeds = [
        theta0,
        theta0 + std::f32::consts::FRAC_PI_2,
        theta0 + std::f32::consts::PI,
        theta0 - std::f32::consts::FRAC_PI_2,
    ];

    let mut groups: [BumpVec<usize>; 4] = [
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
        BumpVec::new_in(arena),
    ];

    for (i, lf) in line_fits.iter().enumerate() {
        let mut best_g = 0;
        let mut best_d = f32::MAX;
        for (g, &seed) in seeds.iter().enumerate() {
            let d = grad_angle_diff(lf.grad_angle, seed);
            if d < best_d {
                best_d = d;
                best_g = g;
            }
        }
        groups[best_g].push(i);
    }

    // All 4 groups must be populated.
    for grp in &groups {
        if grp.is_empty() {
            return None;
        }
    }

    // Pick the dominant line per group: the one with grad_angle closest to the group's mean.
    let mut dominant: [Option<(f64, f64, f64)>; 4] = [None; 4];
    for (g, grp) in groups.iter().enumerate() {
        let mut s = 0.0f32;
        let mut c_val = 0.0f32;
        for &idx in grp.as_slice() {
            s += line_fits[idx].grad_angle.sin();
            c_val += line_fits[idx].grad_angle.cos();
        }
        let mean_angle = s.atan2(c_val);

        let mut best_idx = grp[0];
        let mut best_diff = f32::MAX;
        for &idx in grp.as_slice() {
            let d = grad_angle_diff(line_fits[idx].grad_angle, mean_angle);
            if d < best_diff {
                best_diff = d;
                best_idx = idx;
            }
        }

        let lf = &line_fits[best_idx];
        dominant[g] = Some((lf.nx, lf.ny, lf.d));
    }

    // Compute 4 corners as intersections of adjacent dominant lines.
    // Groups in angular order (0 → π/2 → π → 3π/2): corners at group boundaries 0&1, 1&2, 2&3, 3&0.
    let mut corners_roi = [[0.0f64; 2]; 4];
    for i in 0..4 {
        let j = (i + 1) % 4;
        let (n1x, n1y, d1) = dominant[i]?;
        let (n2x, n2y, d2) = dominant[j]?;
        let (cx, cy) = intersect_lines(n1x, n1y, d1, n2x, n2y, d2)?;
        if cx < -2.0 || cy < -2.0 || cx > roi_w as f64 + 2.0 || cy > roi_h as f64 + 2.0 {
            return None;
        }
        corners_roi[i] = [cx, cy];
    }

    // Validate via signed area (shoelace).
    let mut area = 0.0f64;
    for i in 0..4 {
        let j = (i + 1) % 4;
        area += corners_roi[i][0] * corners_roi[j][1];
        area -= corners_roi[j][0] * corners_roi[i][1];
    }
    area *= 0.5;

    if area.abs() < f64::from(stat.pixel_count) * 0.1 {
        return None;
    }

    // Convert ROI-relative coords to full image coords, return in CW order.
    let ox = rx0 as f64;
    let oy = ry0 as f64;

    let pts = if area > 0.0 {
        [
            Point { x: corners_roi[0][0] + ox, y: corners_roi[0][1] + oy },
            Point { x: corners_roi[1][0] + ox, y: corners_roi[1][1] + oy },
            Point { x: corners_roi[2][0] + ox, y: corners_roi[2][1] + oy },
            Point { x: corners_roi[3][0] + ox, y: corners_roi[3][1] + oy },
        ]
    } else {
        [
            Point { x: corners_roi[0][0] + ox, y: corners_roi[0][1] + oy },
            Point { x: corners_roi[3][0] + ox, y: corners_roi[3][1] + oy },
            Point { x: corners_roi[2][0] + ox, y: corners_roi[2][1] + oy },
            Point { x: corners_roi[1][0] + ox, y: corners_roi[1][1] + oy },
        ]
    };

    Some(pts)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::image::ImageView;
    use crate::segmentation::ComponentStats;

    fn make_square_image(canvas: usize, sq_x0: usize, sq_y0: usize, sq_size: usize) -> Vec<u8> {
        let mut img = vec![200u8; canvas * canvas];
        for y in sq_y0..sq_y0 + sq_size {
            for x in sq_x0..sq_x0 + sq_size {
                img[y * canvas + x] = 30;
            }
        }
        img
    }

    fn make_stats(min_x: u16, min_y: u16, max_x: u16, max_y: u16, pixel_count: u32) -> ComponentStats {
        ComponentStats {
            min_x, min_y, max_x, max_y, pixel_count,
            first_pixel_x: min_x, first_pixel_y: min_y,
            m10: 0, m01: 0, m20: 0, m02: 0, m11: 0,
        }
    }

    #[test]
    fn test_edlines_square_returns_corners() {
        let canvas = 100usize;
        let sq_x0 = 30usize;
        let sq_y0 = 30usize;
        let sq_size = 40usize;

        let pixels = make_square_image(canvas, sq_x0, sq_y0, sq_size);
        let img = ImageView::new(&pixels, canvas, canvas, canvas).unwrap();
        let stats = make_stats(
            sq_x0 as u16, sq_y0 as u16,
            (sq_x0 + sq_size - 1) as u16,
            (sq_y0 + sq_size - 1) as u16,
            (sq_size * sq_size) as u32,
        );
        let cfg = EdLinesConfig {
            grad_threshold: 10,
            line_mse_threshold: 3.0,
            min_segment_length: 4,
        };
        let arena = Bump::new();
        let result = extract_quad_edlines(&arena, &img, &stats, &cfg);
        assert!(result.is_some(), "EDLines should find a quad in a clean square image");

        // All corners must be within 4px of the true square corners.
        let true_corners = [
            [sq_x0 as f64, sq_y0 as f64],
            [(sq_x0 + sq_size) as f64, sq_y0 as f64],
            [(sq_x0 + sq_size) as f64, (sq_y0 + sq_size) as f64],
            [sq_x0 as f64, (sq_y0 + sq_size) as f64],
        ];
        let corners = result.unwrap();
        let mut matched = [false; 4];
        for corner in &corners {
            let mut found = false;
            for (i, tc) in true_corners.iter().enumerate() {
                if !matched[i] {
                    let dist = ((corner.x - tc[0]).powi(2) + (corner.y - tc[1]).powi(2)).sqrt();
                    if dist < 4.0 {
                        matched[i] = true;
                        found = true;
                        break;
                    }
                }
            }
            assert!(found, "Corner ({:.1}, {:.1}) not near any true corner", corner.x, corner.y);
        }
    }

    #[test]
    fn test_edlines_tiny_roi_returns_none() {
        let pixels = vec![100u8; 10 * 10];
        let img = ImageView::new(&pixels, 10, 10, 10).unwrap();
        let stats = make_stats(1, 1, 3, 3, 9);
        let cfg = EdLinesConfig {
            grad_threshold: 20,
            line_mse_threshold: 1.5,
            min_segment_length: 8,
        };
        let arena = Bump::new();
        let result = extract_quad_edlines(&arena, &img, &stats, &cfg);
        assert!(result.is_none(), "Tiny ROI should return None");
    }
}
