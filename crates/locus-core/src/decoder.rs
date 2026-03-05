//! Tag decoding, homography computation, and bit sampling.
//!
//! This module handles the final stage of the pipeline:
//! 1. **Homography**: Computing the projection from canonical tag space to image pixels.
//! 2. **Bit Sampling**: Bilinear interpolation of intensities at grid points.
//! 3. **Error Correction**: Correcting bit flips using tag-family specific Hamming distances.

#![allow(unsafe_code, clippy::cast_sign_loss)]
use crate::batch::{Matrix3x3, Point2f};
use crate::config;
use crate::simd::math::{bilinear_interpolate_fixed, rcp_nr};
use crate::simd::roi::RoiCache;
use bumpalo::Bump;
use multiversion::multiversion;
use nalgebra::{SMatrix, SVector};
use std::cell::RefCell;

thread_local! {
    // Use a small initial capacity (e.g., 4KB) to avoid system allocation overhead for small workloads.
    // The arena will still grow if needed.
    pub(crate) static DECODE_ARENA: RefCell<Bump> = RefCell::new(Bump::with_capacity(4096));
}

/// A 3x3 Homography matrix.
pub struct Homography {
    /// The 3x3 homography matrix.
    pub h: SMatrix<f64, 3, 3>,
}

impl Homography {
    /// Compute homography from 4 source points to 4 destination points using DLT.
    /// Points are [x, y].
    /// Compute homography from 4 source points to 4 destination points using DLT.
    /// Points are [x, y].
    #[must_use]
    pub fn from_pairs(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<Self> {
        let mut a = SMatrix::<f64, 8, 9>::zeros();

        for i in 0..4 {
            let sx = src[i][0];
            let sy = src[i][1];
            let dx = dst[i][0];
            let dy = dst[i][1];

            a[(i * 2, 0)] = -sx;
            a[(i * 2, 1)] = -sy;
            a[(i * 2, 2)] = -1.0;
            a[(i * 2, 6)] = sx * dx;
            a[(i * 2, 7)] = sy * dx;
            a[(i * 2, 8)] = dx;

            a[(i * 2 + 1, 3)] = -sx;
            a[(i * 2 + 1, 4)] = -sy;
            a[(i * 2 + 1, 5)] = -1.0;
            a[(i * 2 + 1, 6)] = sx * dy;
            a[(i * 2 + 1, 7)] = sy * dy;
            a[(i * 2 + 1, 8)] = dy;
        }

        let mut b = SVector::<f64, 8>::zeros();
        let mut m = SMatrix::<f64, 8, 8>::zeros();
        for i in 0..8 {
            for j in 0..8 {
                m[(i, j)] = a[(i, j)];
            }
            b[i] = -a[(i, 8)];
        }

        m.lu().solve(&b).and_then(|h_vec| {
            let mut h = SMatrix::<f64, 3, 3>::identity();
            h[(0, 0)] = h_vec[0];
            h[(0, 1)] = h_vec[1];
            h[(0, 2)] = h_vec[2];
            h[(1, 0)] = h_vec[3];
            h[(1, 1)] = h_vec[4];
            h[(1, 2)] = h_vec[5];
            h[(2, 0)] = h_vec[6];
            h[(2, 1)] = h_vec[7];
            h[(2, 2)] = 1.0;
            let res = Self { h };
            for i in 0..4 {
                let p_proj = res.project(src[i]);
                let err_sq = (p_proj[0] - dst[i][0]).powi(2) + (p_proj[1] - dst[i][1]).powi(2);
                if !err_sq.is_finite() || err_sq > 1e-4 {
                    return None;
                }
            }
            Some(res)
        })
    }

    /// Optimized homography computation from canonical unit square to a quad.
    /// Source points are assumed to be: `[(-1,-1), (1,-1), (1,1), (-1,1)]`.
    #[must_use]
    pub fn square_to_quad(dst: &[[f64; 2]; 4]) -> Option<Self> {
        let mut b = SVector::<f64, 8>::zeros();
        let mut m = SMatrix::<f64, 8, 8>::zeros();

        // Hardcoded coefficients for src = [(-1,-1), (1,-1), (1,1), (-1,1)]
        // Point 0: (-1, -1) -> (x0, y0)
        let x0 = dst[0][0];
        let y0 = dst[0][1];
        // h0 + h1 - h2 - x0*h6 - x0*h7 = -x0  =>  1, 1, -1, ..., -x0, -x0
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 1.0;
        m[(0, 2)] = -1.0;
        m[(0, 6)] = -x0;
        m[(0, 7)] = -x0;
        b[0] = -x0;
        // h3 + h4 - h5 - y0*h6 - y0*h7 = -y0  =>  ..., 1, 1, -1, -y0, -y0
        m[(1, 3)] = 1.0;
        m[(1, 4)] = 1.0;
        m[(1, 5)] = -1.0;
        m[(1, 6)] = -y0;
        m[(1, 7)] = -y0;
        b[1] = -y0;

        // Point 1: (1, -1) -> (x1, y1)
        let x1 = dst[1][0];
        let y1 = dst[1][1];
        // -h0 + h1 + h2 + x1*h6 - x1*h7 = -x1
        m[(2, 0)] = -1.0;
        m[(2, 1)] = 1.0;
        m[(2, 2)] = -1.0;
        m[(2, 6)] = x1;
        m[(2, 7)] = -x1;
        b[2] = -x1;
        m[(3, 3)] = -1.0;
        m[(3, 4)] = 1.0;
        m[(3, 5)] = -1.0;
        m[(3, 6)] = y1;
        m[(3, 7)] = -y1;
        b[3] = -y1;

        // Point 2: (1, 1) -> (x2, y2)
        let x2 = dst[2][0];
        let y2 = dst[2][1];
        // -h0 - h1 + h2 + x2*h6 + x2*h7 = -x2
        m[(4, 0)] = -1.0;
        m[(4, 1)] = -1.0;
        m[(4, 2)] = -1.0;
        m[(4, 6)] = x2;
        m[(4, 7)] = x2;
        b[4] = -x2;
        m[(5, 3)] = -1.0;
        m[(5, 4)] = -1.0;
        m[(5, 5)] = -1.0;
        m[(5, 6)] = y2;
        m[(5, 7)] = y2;
        b[5] = -y2;

        // Point 3: (-1, 1) -> (x3, y3)
        let x3 = dst[3][0];
        let y3 = dst[3][1];
        // h0 - h1 + h2 - x3*h6 + x3*h7 = -x3
        m[(6, 0)] = 1.0;
        m[(6, 1)] = -1.0;
        m[(6, 2)] = -1.0;
        m[(6, 6)] = -x3;
        m[(6, 7)] = x3;
        b[6] = -x3;
        m[(7, 3)] = 1.0;
        m[(7, 4)] = -1.0;
        m[(7, 5)] = -1.0;
        m[(7, 6)] = -y3;
        m[(7, 7)] = y3;
        b[7] = -y3;

        m.lu().solve(&b).and_then(|h_vec| {
            let mut h = SMatrix::<f64, 3, 3>::identity();
            h[(0, 0)] = h_vec[0];
            h[(0, 1)] = h_vec[1];
            h[(0, 2)] = h_vec[2];
            h[(1, 0)] = h_vec[3];
            h[(1, 1)] = h_vec[4];
            h[(1, 2)] = h_vec[5];
            h[(2, 0)] = h_vec[6];
            h[(2, 1)] = h_vec[7];
            h[(2, 2)] = 1.0;
            let res = Self { h };
            let src_unit = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
            for i in 0..4 {
                let p_proj = res.project(src_unit[i]);
                let err_sq = (p_proj[0] - dst[i][0]).powi(2) + (p_proj[1] - dst[i][1]).powi(2);
                if err_sq > 1e-4 {
                    return None;
                }
            }
            Some(res)
        })
    }

    /// Project a point using the homography.
    #[must_use]
    pub fn project(&self, p: [f64; 2]) -> [f64; 2] {
        let res = self.h * SVector::<f64, 3>::new(p[0], p[1], 1.0);
        let w = res[2];
        [res[0] / w, res[1] / w]
    }
}

/// Compute homographies for all quads in the batch using a pure-function SoA approach.
///
/// This uses `rayon` for data-parallel computation of the square-to-quad homographies.
/// Quads are defined by 4 corners in `corners` for each candidate index.
pub fn compute_homographies_soa(corners: &[Point2f], homographies: &mut [Matrix3x3]) {
    use rayon::prelude::*;

    // Each homography maps from canonical square [(-1,-1), (1,-1), (1,1), (-1,1)] to image quads.
    homographies
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, h_out)| {
            let offset = i * 4;
            let dst = [
                [f64::from(corners[offset].x), f64::from(corners[offset].y)],
                [
                    f64::from(corners[offset + 1].x),
                    f64::from(corners[offset + 1].y),
                ],
                [
                    f64::from(corners[offset + 2].x),
                    f64::from(corners[offset + 2].y),
                ],
                [
                    f64::from(corners[offset + 3].x),
                    f64::from(corners[offset + 3].y),
                ],
            ];

            if let Some(h) = Homography::square_to_quad(&dst) {
                // Copy data to f32 batch. Nalgebra stores in column-major order.
                for (j, val) in h.h.iter().enumerate() {
                    h_out.data[j] = *val as f32;
                }
                h_out.padding = [0.0; 7];
            } else {
                // Failed to compute homography (e.g. degenerate quad).
                h_out.data = [0.0; 9];
                h_out.padding = [0.0; 7];
            }
        });
}

/// Refine corner positions using edge-based optimization with the homography.
///
/// After successful decoding, we fit lines to each edge using gradient-weighted
/// least squares, then compute corners as line intersections. This provides
/// more accurate corner localization than the initial detection.
#[must_use]
pub(crate) fn refine_corners_with_homography(
    img: &crate::image::ImageView,
    corners: &[[f64; 2]; 4],
    _homography: &Homography,
) -> [[f64; 2]; 4] {
    // Fit a line to each of the 4 edges
    let mut lines = [(0.0f64, 0.0f64, 0.0f64); 4]; // (a, b, c) where ax + by + c = 0
    let mut line_valid = [false; 4];

    for i in 0..4 {
        let next = (i + 1) % 4;
        let p1 = corners[i];
        let p2 = corners[next];

        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let len = (dx * dx + dy * dy).sqrt();

        if len < 4.0 {
            continue;
        }

        // Normal direction (perpendicular to edge)
        let nx = -dy / len;
        let ny = dx / len;

        // Collect weighted samples along the edge
        let mut sum_w = 0.0;
        let mut sum_d = 0.0;
        let n_samples = (len as usize).clamp(5, 20);

        for s in 1..=n_samples {
            let t = s as f64 / (n_samples + 1) as f64;
            let px = p1[0] + dx * t;
            let py = p1[1] + dy * t;

            // Search for gradient peak perpendicular to edge
            let mut best_pos = (px, py);
            let mut best_mag = 0.0;

            for step in -3..=3 {
                let sx = px + nx * f64::from(step);
                let sy = py + ny * f64::from(step);

                if sx < 1.0
                    || sx >= (img.width - 2) as f64
                    || sy < 1.0
                    || sy >= (img.height - 2) as f64
                {
                    continue;
                }

                let g = img.sample_gradient_bilinear(sx, sy);
                let mag = (g[0] * g[0] + g[1] * g[1]).sqrt();

                if mag > best_mag {
                    best_mag = mag;
                    best_pos = (sx, sy);
                }
            }

            if best_mag > 5.0 {
                // Distance from best_pos to original line
                let d = nx * best_pos.0 + ny * best_pos.1;
                sum_w += best_mag;
                sum_d += d * best_mag;
            }
        }

        if sum_w > 1.0 {
            // Line equation: nx * x + ny * y + c = 0
            let c = -sum_d / sum_w;
            lines[i] = (nx, ny, c);
            line_valid[i] = true;
        }
    }

    // If we don't have all 4 lines, return original corners
    if !line_valid.iter().all(|&v| v) {
        return *corners;
    }

    // Compute corner intersections
    let mut refined = *corners;
    for i in 0..4 {
        let prev = (i + 3) % 4;
        let (a1, b1, c1) = lines[prev];
        let (a2, b2, c2) = lines[i];

        let det = a1 * b2 - a2 * b1;
        if det.abs() > 1e-6 {
            let x = (b1 * c2 - b2 * c1) / det;
            let y = (a2 * c1 - a1 * c2) / det;

            // Sanity check: intersection should be near original corner
            let dist_sq = (x - corners[i][0]).powi(2) + (y - corners[i][1]).powi(2);
            if dist_sq < 4.0 {
                refined[i] = [x, y];
            }
        }
    }

    refined
}

/// Refine corner positions using GridFit optimization.
///
/// This method optimizes the homography by adjusting corners to maximize the
/// contrast between expected black and white cells in the decoded grid.
/// This minimizes photometric error in the tag's coordinate system.
pub(crate) fn refine_corners_gridfit(
    img: &crate::image::ImageView,
    corners: &[[f64; 2]; 4],
    decoder: &(impl TagDecoder + ?Sized),
    bits: u64,
) -> [[f64; 2]; 4] {
    let mut current_corners = *corners;
    let mut best_corners = *corners;
    let mut best_contrast = -1.0;

    // Optimization parameters
    let step_sizes = [0.5, 0.25, 0.125]; // Coarse to fine
    // let _window = 1;

    // Compute initial contrast
    if let Some(h) = Homography::square_to_quad(&current_corners)
        && let Some(contrast) = compute_grid_contrast(img, &h, decoder, bits)
    {
        best_contrast = contrast;
    }

    // We assume the initial decode was valid, so we should have a valid contrast.
    // If not, just return original corners.
    if best_contrast < 0.0 {
        return *corners;
    }

    // Coordinate Descent Optimization
    // Iteratively perturb each coordinate of each corner
    for &step in &step_sizes {
        let mut improved = true;
        let mut iters = 0;

        while improved && iters < 5 {
            improved = false;
            iters += 1;

            for i in 0..4 {
                for axis in 0..2 {
                    // Try moving -step, 0, +step
                    let original_val = current_corners[i][axis];

                    let candidate_vals = [original_val - step, original_val + step];

                    for &val in &candidate_vals {
                        current_corners[i][axis] = val;

                        // Check hypothesis
                        let mut valid = false;
                        if let Some(h) = Homography::square_to_quad(&current_corners)
                            && let Some(contrast) = compute_grid_contrast(img, &h, decoder, bits)
                            && contrast > best_contrast
                        {
                            best_contrast = contrast;
                            best_corners = current_corners;
                            improved = true;
                            valid = true;
                        }

                        if valid {
                            // Greedy update: accept improvement and continue.
                            break;
                        }
                        // Revert if worse or invalid
                        current_corners[i][axis] = original_val;
                    }

                    // Ensure current matches best before moving to next coordinate
                    current_corners = best_corners;
                }
            }
        }
    }

    best_corners
}

/// Compute the contrast of the grid given a homography and the expected bit pattern.
/// Returns (mean_white - mean_black).
fn compute_grid_contrast(
    img: &crate::image::ImageView,
    h: &Homography,
    decoder: &(impl TagDecoder + ?Sized),
    bits: u64,
) -> Option<f64> {
    let points = decoder.sample_points();
    let _n = points.len();

    // Pre-calculate homography terms for speed
    let h00 = h.h[(0, 0)];
    let h01 = h.h[(0, 1)];
    let h02 = h.h[(0, 2)];
    let h10 = h.h[(1, 0)];
    let h11 = h.h[(1, 1)];
    let h12 = h.h[(1, 2)];
    let h20 = h.h[(2, 0)];
    let h21 = h.h[(2, 1)];
    let h22 = h.h[(2, 2)];

    let mut sum_white = 0.0;
    let mut cnt_white = 0;
    let mut sum_black = 0.0;
    let mut cnt_black = 0;

    for (i, &p) in points.iter().enumerate() {
        // Project
        let wz = h20 * p.0 + h21 * p.1 + h22;
        if wz.abs() < 1e-6 {
            return None;
        }
        let img_x = (h00 * p.0 + h01 * p.1 + h02) / wz;
        let img_y = (h10 * p.0 + h11 * p.1 + h12) / wz;

        // Check bounds
        if img_x < 0.0
            || img_x >= (img.width - 1) as f64
            || img_y < 0.0
            || img_y >= (img.height - 1) as f64
        {
            return None;
        }

        // Bilinear sample
        let xf = img_x.floor();
        let yf = img_y.floor();
        let ix = xf as usize;
        let iy = yf as usize;
        let dx = img_x - xf;
        let dy = img_y - yf;

        let val = unsafe {
            let row0 = img.get_row_unchecked(iy);
            let row1 = img.get_row_unchecked(iy + 1);
            let v00 = f64::from(*row0.get_unchecked(ix));
            let v10 = f64::from(*row0.get_unchecked(ix + 1));
            let v01 = f64::from(*row1.get_unchecked(ix));
            let v11 = f64::from(*row1.get_unchecked(ix + 1));
            let top = v00 + dx * (v10 - v00);
            let bot = v01 + dx * (v11 - v01);
            top + dy * (bot - top)
        };

        let expected_bit = (bits >> i) & 1;
        if expected_bit == 1 {
            sum_white += val;
            cnt_white += 1;
        } else {
            sum_black += val;
            cnt_black += 1;
        }
    }

    if cnt_white == 0 || cnt_black == 0 {
        return None;
    }

    let mean_white = sum_white / f64::from(cnt_white);
    let mean_black = sum_black / f64::from(cnt_black);

    Some(mean_white - mean_black)
}

/// Refine corners using "Erf-Fit" (Gaussian fit to intensity profile).
///
/// This assumes the edge intensity profile is an Error Function (convolution of step edge with Gaussian PSF).
/// We minimize the photometric error between the image and the ERF model using Gauss-Newton.
pub(crate) fn refine_corners_erf(
    arena: &bumpalo::Bump,
    img: &crate::image::ImageView,
    corners: &[[f64; 2]; 4],
    sigma: f64,
) -> [[f64; 2]; 4] {
    let mut lines = [(0.0f64, 0.0f64, 0.0f64); 4];
    let mut line_valid = [false; 4];

    // Sub-pixel edge refinement for each of the 4 edges
    for i in 0..4 {
        let next = (i + 1) % 4;
        let p1 = corners[i];
        let p2 = corners[next];

        if let Some((nx, ny, d)) = fit_edge_erf(arena, img, p1, p2, sigma) {
            lines[i] = (nx, ny, d);
            line_valid[i] = true;
        }
    }

    if !line_valid.iter().all(|&v| v) {
        return *corners;
    }

    // Intersect lines to get refined corners
    let mut refined = *corners;
    for i in 0..4 {
        let prev = (i + 3) % 4;
        let (a1, b1, c1) = lines[prev];
        let (a2, b2, c2) = lines[i];
        let det = a1 * b2 - a2 * b1;
        if det.abs() > 1e-6 {
            let x = (b1 * c2 - b2 * c1) / det;
            let y = (a2 * c1 - a1 * c2) / det;

            // Sanity check
            let dist_sq = (x - corners[i][0]).powi(2) + (y - corners[i][1]).powi(2);
            if dist_sq < 4.0 {
                refined[i] = [x, y];
            }
        }
    }
    refined
}

/// Helper for ERF edge fitting
struct EdgeFitter<'a> {
    img: &'a crate::image::ImageView<'a>,
    p1: [f64; 2],
    dx: f64,
    dy: f64,
    len: f64,
    nx: f64,
    ny: f64,
    d: f64,
}

impl<'a> EdgeFitter<'a> {
    fn new(img: &'a crate::image::ImageView<'a>, p1: [f64; 2], p2: [f64; 2]) -> Option<Self> {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 4.0 {
            return None;
        }
        let nx = -dy / len;
        let ny = dx / len;
        // Initial d from input corners
        let d = -(nx * p1[0] + ny * p1[1]);

        Some(Self {
            img,
            p1,
            dx,
            dy,
            len,
            nx,
            ny,
            d,
        })
    }

    fn scan_initial_d(&mut self) {
        let window = 2.5;
        let (x0, x1, y0, y1) = self.get_scan_bounds(window);

        let mut best_offset = 0.0;
        let mut best_grad = 0.0;

        for k in -6..=6 {
            let offset = f64::from(k) * 0.4;
            let scan_d = self.d + offset;

            let (sum_g, count) =
                project_gradients_optimized(self.img, self.nx, self.ny, x0, x1, y0, y1, scan_d);

            if count > 0 && sum_g > best_grad {
                best_grad = sum_g;
                best_offset = offset;
            }
        }
        self.d += best_offset;
    }

    fn collect_samples(
        &self,
        arena: &'a bumpalo::Bump,
    ) -> bumpalo::collections::Vec<'a, (f64, f64, f64)> {
        let window = 2.5;
        let (x0, x1, y0, y1) = self.get_scan_bounds(window);

        collect_samples_optimized(
            self.img, self.nx, self.ny, self.d, self.p1, self.dx, self.dy, self.len, x0, x1, y0,
            y1, window, arena,
        )
    }

    fn refine(&mut self, samples: &[(f64, f64, f64)], sigma: f64) {
        if samples.len() < 10 {
            return;
        }
        let mut a = 128.0;
        let mut b = 128.0;
        let inv_sigma = 1.0 / sigma;
        let _sqrt_pi = std::f64::consts::PI.sqrt();

        for _ in 0..15 {
            let mut dark_sum = 0.0;
            let mut dark_weight = 0.0;
            let mut light_sum = 0.0;
            let mut light_weight = 0.0;

            for &(x, y, _) in samples {
                let dist = self.nx * x + self.ny * y + self.d;
                let val = self.img.sample_bilinear(x, y);
                if dist < -1.0 {
                    let w = (-dist - 0.5).clamp(0.1, 2.0);
                    dark_sum += val * w;
                    dark_weight += w;
                } else if dist > 1.0 {
                    let w = (dist - 0.5).clamp(0.1, 2.0);
                    light_sum += val * w;
                    light_weight += w;
                }
            }

            if dark_weight > 0.0 && light_weight > 0.0 {
                a = dark_sum / dark_weight;
                b = light_sum / light_weight;
            }

            if (b - a).abs() < 5.0 {
                break;
            }

            let (sum_jtj, sum_jt_res) = refine_accumulate_optimized(
                samples, self.img, self.nx, self.ny, self.d, a, b, sigma, inv_sigma,
            );

            if sum_jtj < 1e-6 {
                break;
            }
            let step = sum_jt_res / sum_jtj;
            self.d += step.clamp(-0.5, 0.5);
            if step.abs() < 1e-4 {
                break;
            }
        }
    }

    #[allow(clippy::cast_sign_loss)]
    fn get_scan_bounds(&self, window: f64) -> (usize, usize, usize, usize) {
        let p2_0 = self.p1[0] + self.dx;
        let p2_1 = self.p1[1] + self.dy;

        // Clamp to valid image coordinates (padding of 1 pixel)
        let w_limit = (self.img.width - 2) as f64;
        let h_limit = (self.img.height - 2) as f64;

        let x0 = (self.p1[0].min(p2_0) - window - 0.5).clamp(1.0, w_limit) as usize;
        let x1 = (self.p1[0].max(p2_0) + window + 0.5).clamp(1.0, w_limit) as usize;
        let y0 = (self.p1[1].min(p2_1) - window - 0.5).clamp(1.0, h_limit) as usize;
        let y1 = (self.p1[1].max(p2_1) + window + 0.5).clamp(1.0, h_limit) as usize;
        (x0, x1, y0, y1)
    }
}

#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn project_gradients_optimized(
    img: &crate::image::ImageView,
    nx: f64,
    ny: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    scan_d: f64,
) -> (f64, usize) {
    let mut sum_g = 0.0;
    let mut count = 0;

    for py in y0..=y1 {
        let mut px = x0;
        let y = py as f64;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
            unsafe {
                use std::arch::x86_64::*;
                let v_nx = _mm256_set1_pd(nx);
                let v_ny = _mm256_set1_pd(ny);
                let v_scan_d = _mm256_set1_pd(scan_d);
                let v_y = _mm256_set1_pd(y);
                let v_abs_mask = _mm256_set1_pd(-0.0);

                while px + 4 <= x1 {
                    let v_x =
                        _mm256_set_pd((px + 3) as f64, (px + 2) as f64, (px + 1) as f64, px as f64);

                    let v_dist = _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                        v_scan_d,
                    );

                    let v_abs_dist = _mm256_andnot_pd(v_abs_mask, v_dist);
                    let v_cmp = _mm256_cmp_pd(v_abs_dist, _mm256_set1_pd(1.0), _CMP_LT_OQ);
                    let mask = _mm256_movemask_pd(v_cmp);

                    if mask != 0 {
                        for j in 0..4 {
                            if (mask >> j) & 1 != 0 {
                                let g = img.sample_gradient_bilinear((px + j) as f64, y);
                                sum_g += (g[0] * nx + g[1] * ny).abs();
                                count += 1;
                            }
                        }
                    }
                    px += 4;
                }
            }
        }

        while px <= x1 {
            let x = px as f64;
            let dist = nx * x + ny * y + scan_d;
            if dist.abs() < 1.0 {
                let g = img.sample_gradient_bilinear(x, y);
                sum_g += (g[0] * nx + g[1] * ny).abs();
                count += 1;
            }
            px += 1;
        }
    }
    (sum_g, count)
}

#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn collect_samples_optimized<'a>(
    img: &crate::image::ImageView,
    nx: f64,
    ny: f64,
    d: f64,
    p1: [f64; 2],
    dx: f64,
    dy: f64,
    len: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    window: f64,
    arena: &'a bumpalo::Bump,
) -> bumpalo::collections::Vec<'a, (f64, f64, f64)> {
    let mut samples = bumpalo::collections::Vec::with_capacity_in(128, arena);

    for py in y0..=y1 {
        let mut px = x0;
        let y = py as f64;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
            unsafe {
                use std::arch::x86_64::*;
                let v_nx = _mm256_set1_pd(nx);
                let v_ny = _mm256_set1_pd(ny);
                let v_d = _mm256_set1_pd(d);
                let v_y = _mm256_set1_pd(y);
                let v_p1x = _mm256_set1_pd(p1[0]);
                let v_p1y = _mm256_set1_pd(p1[1]);
                let v_dx = _mm256_set1_pd(dx);
                let v_dy = _mm256_set1_pd(dy);
                let v_inv_len_sq = _mm256_set1_pd(1.0 / (len * len));
                let v_abs_mask = _mm256_set1_pd(-0.0);

                while px + 4 <= x1 {
                    let v_x =
                        _mm256_set_pd((px + 3) as f64, (px + 2) as f64, (px + 1) as f64, px as f64);

                    let v_dist = _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                        v_d,
                    );
                    let v_abs_dist = _mm256_andnot_pd(v_abs_mask, v_dist);
                    let v_dist_mask = _mm256_cmp_pd(v_abs_dist, _mm256_set1_pd(window), _CMP_LE_OQ);

                    let v_t = _mm256_mul_pd(
                        _mm256_add_pd(
                            _mm256_mul_pd(_mm256_sub_pd(v_x, v_p1x), v_dx),
                            _mm256_mul_pd(_mm256_sub_pd(v_y, v_p1y), v_dy),
                        ),
                        v_inv_len_sq,
                    );

                    let v_t_mask_low = _mm256_cmp_pd(v_t, _mm256_set1_pd(-0.1), _CMP_GE_OQ);
                    let v_t_mask_high = _mm256_cmp_pd(v_t, _mm256_set1_pd(1.1), _CMP_LE_OQ);

                    let final_mask = _mm256_movemask_pd(_mm256_and_pd(
                        v_dist_mask,
                        _mm256_and_pd(v_t_mask_low, v_t_mask_high),
                    ));

                    if final_mask != 0 {
                        for j in 0..4 {
                            if (final_mask >> j) & 1 != 0 {
                                let val = f64::from(img.get_pixel(px + j, py));
                                samples.push(((px + j) as f64, y, val));
                            }
                        }
                    }
                    px += 4;
                }
            }
        }

        while px <= x1 {
            let x = px as f64;
            let dist = nx * x + ny * y + d;
            if dist.abs() <= window {
                let t = ((x - p1[0]) * dx + (y - p1[1]) * dy) / (len * len);
                if (-0.1..=1.1).contains(&t) {
                    let val = f64::from(img.get_pixel(px, py));
                    samples.push((x, y, val));
                }
            }
            px += 1;
        }
    }
    samples
}

#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn refine_accumulate_optimized(
    samples: &[(f64, f64, f64)],
    #[allow(unused_variables)] img: &crate::image::ImageView,
    nx: f64,
    ny: f64,
    d: f64,
    a: f64,
    b: f64,
    sigma: f64,
    inv_sigma: f64,
) -> (f64, f64) {
    let mut sum_jtj = 0.0;
    let mut sum_jt_res = 0.0;
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let k = (b - a) / (sqrt_pi * sigma);

    let mut i = 0;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
        unsafe {
            use std::arch::x86_64::*;
            let v_nx = _mm256_set1_pd(nx);
            let v_ny = _mm256_set1_pd(ny);
            let v_d = _mm256_set1_pd(d);
            let v_a = _mm256_set1_pd(a);
            let v_b = _mm256_set1_pd(b);
            let v_inv_sigma = _mm256_set1_pd(inv_sigma);
            let v_k = _mm256_set1_pd(k);
            let v_half = _mm256_set1_pd(0.5);
            let v_one = _mm256_set1_pd(1.0);
            let v_abs_mask = _mm256_set1_pd(-0.0);

            let mut v_sum_jtj = _mm256_setzero_pd();
            let mut v_sum_jt_res = _mm256_setzero_pd();

            while i + 4 <= samples.len() {
                let s0 = samples[i];
                let s1 = samples[i + 1];
                let s2 = samples[i + 2];
                let s3 = samples[i + 3];

                let v_x = _mm256_set_pd(s3.0, s2.0, s1.0, s0.0);
                let v_y = _mm256_set_pd(s3.1, s2.1, s1.1, s0.1);
                let v_img_val = _mm256_set_pd(s3.2, s2.2, s1.2, s0.2);

                let v_dist = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                    v_d,
                );
                let v_s = _mm256_mul_pd(v_dist, v_inv_sigma);

                let v_abs_s = _mm256_andnot_pd(v_abs_mask, v_s);
                let v_range_mask = _mm256_cmp_pd(v_abs_s, _mm256_set1_pd(3.0), _CMP_LE_OQ);

                if _mm256_movemask_pd(v_range_mask) != 0 {
                    // Simple ERF approx: erf(x) ~= sign(x) * sqrt(1 - exp(-x^2 * 4/pi))
                    // For now, just use scalar fallback for the math inside to keep it sound,
                    // but we could use a SIMD exp/erf here.
                    let ss: [f64; 4] = std::mem::transmute(v_s);
                    let iv: [f64; 4] = std::mem::transmute(v_img_val);
                    let mm: [f64; 4] = std::mem::transmute(v_range_mask);

                    for j in 0..4 {
                        if mm[j] != 0.0 {
                            let s_val = ss[j];
                            let model =
                                (a + b) * 0.5 + (b - a) * 0.5 * crate::quad::erf_approx(s_val);
                            let residual = iv[j] - model;
                            let jac = k * (-s_val * s_val).exp();
                            sum_jtj += jac * jac;
                            sum_jt_res += jac * residual;
                        }
                    }
                }
                i += 4;
            }
        }
    }

    // Scalar tail
    while i < samples.len() {
        let (x, y, img_val) = samples[i];
        let dist = nx * x + ny * y + d;
        let s = dist * inv_sigma;
        if s.abs() <= 3.0 {
            let model = (a + b) * 0.5 + (b - a) * 0.5 * crate::quad::erf_approx(s);
            let residual = img_val - model;
            let jac = k * (-s * s).exp();
            sum_jtj += jac * jac;
            sum_jt_res += jac * residual;
        }
        i += 1;
    }

    (sum_jtj, sum_jt_res)
}

/// Fit a line (nx*x + ny*y + d = 0) to an edge using the ERF intensity model.
fn fit_edge_erf(
    arena: &bumpalo::Bump,
    img: &crate::image::ImageView,
    p1: [f64; 2],
    p2: [f64; 2],
    sigma: f64,
) -> Option<(f64, f64, f64)> {
    let mut fitter = EdgeFitter::new(img, p1, p2)?;
    fitter.scan_initial_d();
    let samples = fitter.collect_samples(arena);
    if samples.len() < 10 {
        return None;
    }
    fitter.refine(&samples, sigma);
    Some((fitter.nx, fitter.ny, fitter.d))
}

/// Returns the threshold that maximizes inter-class variance.
pub(crate) fn compute_otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 128.0;
    }

    let n = values.len() as f64;
    let total_sum: f64 = values.iter().sum();

    // Find min/max to define search range
    let min_val = values.iter().copied().fold(f64::MAX, f64::min);
    let max_val = values.iter().copied().fold(f64::MIN, f64::max);

    if (max_val - min_val) < 1.0 {
        return f64::midpoint(min_val, max_val);
    }

    // Search for optimal threshold
    let mut best_threshold = f64::midpoint(min_val, max_val);
    let mut best_variance = 0.0;

    // Use 16 candidate thresholds between min and max
    for i in 1..16 {
        let t = min_val + (max_val - min_val) * (f64::from(i) / 16.0);

        let mut w0 = 0.0;
        let mut sum0 = 0.0;

        for &v in values {
            if v <= t {
                w0 += 1.0;
                sum0 += v;
            }
        }

        let w1 = n - w0;
        if w0 < 1.0 || w1 < 1.0 {
            continue;
        }

        let mean0 = sum0 / w0;
        let mean1 = (total_sum - sum0) / w1;

        // Inter-class variance
        let variance = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);

        if variance > best_variance {
            best_variance = variance;
            best_threshold = t;
        }
    }

    best_threshold
}

/// Sample values from the image using SIMD-optimized Fast-Math and ROI caching.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn sample_grid_values_optimized(
    img: &crate::image::ImageView,
    h: &Homography,
    roi: &RoiCache,
    points: &[(f64, f64)],
    intensities: &mut [f64],
    n: usize,
) -> bool {
    let h00 = h.h[(0, 0)] as f32;
    let h01 = h.h[(0, 1)] as f32;
    let h02 = h.h[(0, 2)] as f32;
    let h10 = h.h[(1, 0)] as f32;
    let h11 = h.h[(1, 1)] as f32;
    let h12 = h.h[(1, 2)] as f32;
    let h20 = h.h[(2, 0)] as f32;
    let h21 = h.h[(2, 1)] as f32;
    let h22 = h.h[(2, 2)] as f32;

    let w_limit = (img.width - 1) as f32;
    let h_limit = (img.height - 1) as f32;

    for (i, &p) in points.iter().take(n).enumerate() {
        let px = p.0 as f32;
        let py = p.1 as f32;

        // Fast-Math Reciprocal
        let wz = h20 * px + h21 * py + h22;
        let winv = rcp_nr(wz);

        let img_x = (h00 * px + h01 * py + h02) * winv;
        let img_y = (h10 * px + h11 * py + h12) * winv;

        if img_x < 0.0 || img_x >= w_limit || img_y < 0.0 || img_y >= h_limit {
            return false;
        }

        let ix = img_x.floor() as usize;
        let iy = img_y.floor() as usize;

        // Sample from ROI cache using fixed-point bilinear
        let v00 = roi.get(ix, iy);
        let v10 = roi.get(ix + 1, iy);
        let v01 = roi.get(ix, iy + 1);
        let v11 = roi.get(ix + 1, iy + 1);

        intensities[i] = f64::from(bilinear_interpolate_fixed(img_x, img_y, v00, v10, v01, v11));
    }
    true
}

/// Sample the bit grid from the image using the homography and decoder points.
///
/// Uses bilinear interpolation for sampling and a spatially adaptive threshold
/// (based on min/max stats of the grid) to determine bit values.
///
/// # Parameters
/// - `min_contrast`: Minimum contrast range for Otsu-based classification.
///   Default is 20.0. Lower values (e.g., 10.0) improve recall on small/blurry tags.
///
/// This computes the intensities at sample points and the adaptive thresholds,
/// then delegates to the strategy to produce the code.
#[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
pub fn sample_grid_generic<S: crate::strategy::DecodingStrategy>(
    img: &crate::image::ImageView,
    arena: &Bump,
    detection: &crate::Detection,
    decoder: &(impl TagDecoder + ?Sized),
) -> Option<S::Code> {
    let (min_x, min_y, max_x, max_y) = detection.aabb();
    let roi = RoiCache::new(img, arena, min_x, min_y, max_x, max_y);

    let homography = Homography::square_to_quad(&detection.corners)?;

    let points = decoder.sample_points();
    // Stack-allocated buffer for up to 64 sample points (covers all standard tag families)
    let mut intensities = [0.0f64; 64];
    let n = points.len().min(64);

    if !sample_grid_values_optimized(img, &homography, &roi, points, &mut intensities, n) {
        return None;
    }

    Some(S::from_intensities(
        &intensities[..n],
        &compute_adaptive_thresholds(&intensities[..n], points),
    ))
}

/// Sample the bit grid using Structure of Arrays (SoA) data.
pub fn sample_grid_soa<S: crate::strategy::DecodingStrategy>(
    img: &crate::image::ImageView,
    arena: &Bump,
    corners: &[Point2f],
    homography: &Matrix3x3,
    decoder: &(impl TagDecoder + ?Sized),
) -> Option<S::Code> {
    // Compute AABB for RoiCache
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for p in corners {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }

    let roi = RoiCache::new(
        img,
        arena,
        (min_x.floor() as i32).max(0) as usize,
        (min_y.floor() as i32).max(0) as usize,
        (max_x.ceil() as i32).max(0) as usize,
        (max_y.ceil() as i32).max(0) as usize,
    );

    // Convert Matrix3x3 to Homography (internal use).
    // Nalgebra stores in column-major order, which matches our data layout.
    let mut h_mat = SMatrix::<f64, 3, 3>::identity();
    for (i, val) in homography.data.iter().enumerate() {
        h_mat.as_mut_slice()[i] = f64::from(*val);
    }
    let homography_obj = Homography { h: h_mat };

    let points = decoder.sample_points();
    let mut intensities = [0.0f64; 64];
    let n = points.len().min(64);

    if !sample_grid_values_optimized(img, &homography_obj, &roi, points, &mut intensities, n) {
        return None;
    }

    Some(S::from_intensities(
        &intensities[..n],
        &compute_adaptive_thresholds(&intensities[..n], points),
    ))
}

/// Sample the bit grid using Structure of Arrays (SoA) data and a precomputed ROI cache.
pub fn sample_grid_soa_precomputed<S: crate::strategy::DecodingStrategy>(
    img: &crate::image::ImageView,
    roi: &RoiCache,
    homography: &Matrix3x3,
    decoder: &(impl TagDecoder + ?Sized),
) -> Option<S::Code> {
    // Convert Matrix3x3 to Homography (internal use).
    let mut h_mat = SMatrix::<f64, 3, 3>::identity();
    for (i, val) in homography.data.iter().enumerate() {
        h_mat.as_mut_slice()[i] = f64::from(*val);
    }
    let homography_obj = Homography { h: h_mat };

    let points = decoder.sample_points();
    let mut intensities = [0.0f64; 64];
    let n = points.len().min(64);

    if !sample_grid_values_optimized(img, &homography_obj, roi, points, &mut intensities, n) {
        return None;
    }

    Some(S::from_intensities(
        &intensities[..n],
        &compute_adaptive_thresholds(&intensities[..n], points),
    ))
}

/// Internal helper to compute adaptive thresholds for a grid of intensities.
fn compute_adaptive_thresholds(intensities: &[f64], points: &[(f64, f64)]) -> [f64; 64] {
    let n = intensities.len();
    let global_threshold = compute_otsu_threshold(intensities);

    let mut quad_sums = [0.0; 4];
    let mut quad_counts = [0; 4];
    for (i, p) in points.iter().take(n).enumerate() {
        let qi = if p.0 < 0.0 {
            usize::from(p.1 >= 0.0)
        } else {
            2 + usize::from(p.1 >= 0.0)
        };
        quad_sums[qi] += intensities[i];
        quad_counts[qi] += 1;
    }

    let mut thresholds = [0.0f64; 64];
    for (i, p) in points.iter().take(n).enumerate() {
        let qi = if p.0 < 0.0 {
            usize::from(p.1 >= 0.0)
        } else {
            2 + usize::from(p.1 >= 0.0)
        };
        let quad_avg = if quad_counts[qi] > 0 {
            quad_sums[qi] / f64::from(quad_counts[qi])
        } else {
            global_threshold
        };

        // Blend global Otsu and local mean (0.7 / 0.3 weighting is common for fiducials)
        thresholds[i] = 0.7 * global_threshold + 0.3 * quad_avg;
    }
    thresholds
}

/// Sample the bit grid from the image (Legacy/Hard wrapper).
#[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
pub fn sample_grid(
    img: &crate::image::ImageView,
    arena: &Bump,
    detection: &crate::Detection,
    decoder: &(impl TagDecoder + ?Sized),
    _min_contrast: f64,
) -> Option<u64> {
    sample_grid_generic::<crate::strategy::HardStrategy>(img, arena, detection, decoder)
}

/// A trait for decoding binary payloads from extracted tags.
/// Rotate a square bit grid 90 degrees clockwise.
/// This is an O(1) bitwise operation but conceptually represents rotating the N x N pixel grid.
#[must_use]
pub(crate) fn rotate90(bits: u64, dim: usize) -> u64 {
    let mut res = 0u64;
    for y in 0..dim {
        for x in 0..dim {
            if (bits >> (y * dim + x)) & 1 != 0 {
                let nx = dim - 1 - y;
                let ny = x;
                res |= 1 << (ny * dim + nx);
            }
        }
    }
    res
}

/// Decode all active candidates in the batch using the Structure of Arrays (SoA) layout.
///
/// This phase executes SIMD bilinear interpolation and Hamming error correction.
/// If a candidate fails decoding, its `status_mask` is flipped to `FailedDecode`.
pub fn decode_batch_soa(
    batch: &mut crate::batch::DetectionBatch,
    n: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
) {
    match config.decode_mode {
        crate::config::DecodeMode::Hard => {
            decode_batch_soa_generic::<crate::strategy::HardStrategy>(
                batch, n, img, decoders, config,
            );
        },
        crate::config::DecodeMode::Soft => {
            decode_batch_soa_generic::<crate::strategy::SoftStrategy>(
                batch, n, img, decoders, config,
            );
        },
    }
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::collapsible_if,
    unused_assignments
)]
fn decode_batch_soa_generic<S: crate::strategy::DecodingStrategy>(
    batch: &mut crate::batch::DetectionBatch,
    n: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
) {
    use crate::batch::CandidateState;
    use rayon::prelude::*;

    // We collect results into a temporary Vec to avoid unsafe parallel writes to the batch.
    let results: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            DECODE_ARENA.with_borrow_mut(|arena| {
                    arena.reset();

                    let corners = &batch.corners[i * 4..i * 4 + 4];
                    let homography = &batch.homographies[i];

                    // Compute AABB for RoiCache ONCE per candidate.
                    // We expand it slightly (10%) to ensure scaled versions (0.9, 1.1) still fit.
                    let mut min_x = f32::MAX;
                    let mut min_y = f32::MAX;
                    let mut max_x = f32::MIN;
                    let mut max_y = f32::MIN;
                    for p in corners {
                        min_x = min_x.min(p.x);
                        min_y = min_y.min(p.y);
                        max_x = max_x.max(p.x);
                        max_y = max_y.max(p.y);
                    }
                    let w_aabb = max_x - min_x;
                    let h_aabb = max_y - min_y;
                    let roi = RoiCache::new(
                        img,
                        arena,
                        ((min_x - w_aabb * 0.1).floor() as i32).max(0) as usize,
                        ((min_y - h_aabb * 0.1).floor() as i32).max(0) as usize,
                        (((max_x + w_aabb * 0.1).ceil() as i32).min(img.width as i32 - 1)).max(0)
                            as usize,
                        (((max_y + h_aabb * 0.1).ceil() as i32).min(img.height as i32 - 1)).max(0)
                            as usize,
                    );


                    let mut best_h = u32::MAX;
                    let mut best_code = None;
                    let mut best_id = 0;
                    let mut best_rot = 0;
                    let mut best_overall_code = None;

                    let scales = [1.0, 0.9, 1.1];
                    let center = [
                        (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4.0,
                        (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4.0,
                    ];

                    for scale in scales {
                        let mut scaled_corners = [Point2f::default(); 4];
                        let mut scaled_h_mat = Matrix3x3 {
                            data: [0.0; 9],
                            padding: [0.0; 7],
                        };

                        let current_homography: &Matrix3x3;

                        let mut best_h_in_scale = u32::MAX;
                        let mut best_match_in_scale: Option<(u32, u32, u8, S::Code, usize)> = None;

                        if (scale - 1.0f32).abs() > 1e-4 {
                            for j in 0..4 {
                                scaled_corners[j].x =
                                    center[0] + (corners[j].x - center[0]) * scale;
                                scaled_corners[j].y =
                                    center[1] + (corners[j].y - center[1]) * scale;
                            }

                            // Must recompute homography for scaled corners
                            let dst = [
                                [
                                    f64::from(scaled_corners[0].x),
                                    f64::from(scaled_corners[0].y),
                                ],
                                [
                                    f64::from(scaled_corners[1].x),
                                    f64::from(scaled_corners[1].y),
                                ],
                                [
                                    f64::from(scaled_corners[2].x),
                                    f64::from(scaled_corners[2].y),
                                ],
                                [
                                    f64::from(scaled_corners[3].x),
                                    f64::from(scaled_corners[3].y),
                                ],
                            ];

                            if let Some(h_new) = Homography::square_to_quad(&dst) {
                                for (j, val) in h_new.h.iter().enumerate() {
                                    scaled_h_mat.data[j] = *val as f32;
                                }
                                current_homography = &scaled_h_mat;
                            } else {
                                // Degenerate scale, skip
                                continue;
                            }
                        } else {
                            scaled_corners.copy_from_slice(&corners[..4]);
                            current_homography = homography;
                        }

                        for (decoder_idx, decoder) in decoders.iter().enumerate() {
                            if let Some(code) = sample_grid_soa_precomputed::<S>(
                                img,
                                &roi,
                                current_homography,
                                decoder.as_ref(),
                            ) {
                                if let Some((id, hamming, rot)) =
                                    S::decode(&code, decoder.as_ref(), 255)
                                {
                                    if hamming < best_h {
                                        best_h = hamming;
                                        let _ = code.clone();
                                        best_overall_code = Some(code.clone());
                                    }

                                    if hamming <= config.max_hamming_error
                                        && (best_code.is_none() || hamming < best_h_in_scale)
                                    {
                                        best_h_in_scale = hamming;
                                        best_match_in_scale =
                                            Some((id, hamming, rot, code, decoder_idx));
                                    }
                                }
                            }
                        }
                        if let Some((id, hamming, rot, code, decoder_idx)) = best_match_in_scale {
                            best_id = id;
                            best_rot = rot;
                            best_code = Some(code.clone());
                            let decoder = decoders[decoder_idx].as_ref();

                            // Always perform ERF refinement for finalists if requested
                            if config.refinement_mode == crate::config::CornerRefinementMode::Erf {
                                // Reassemble corners for ERF (it uses [f64; 2])
                                let mut current_corners = [[0.0f64; 2]; 4];
                                for j in 0..4 {
                                    current_corners[j] =
                                        [f64::from(corners[j].x), f64::from(corners[j].y)];
                                }

                                let refined_corners = refine_corners_erf(
                                    arena,
                                    img,
                                    &current_corners,
                                    config.subpixel_refinement_sigma,
                                );

                                // Verify that refined corners still yield a valid decode
                                let mut refined_corners_f32 = [Point2f::default(); 4];
                                for j in 0..4 {
                                    refined_corners_f32[j] = Point2f {
                                        x: refined_corners[j][0] as f32,
                                        y: refined_corners[j][1] as f32,
                                    };
                                }

                                // Must recompute homography for refined corners
                                let mut ref_h_mat = Matrix3x3 {
                                    data: [0.0; 9],
                                    padding: [0.0; 7],
                                };
                                if let Some(h_new) = Homography::square_to_quad(&refined_corners) {
                                    for (j, val) in h_new.h.iter().enumerate() {
                                        ref_h_mat.data[j] = *val as f32;
                                    }
                                } else {
                                    // Degenerate refinement, reject
                                    continue;
                                }

                                if let Some(code_ref) =
                                    sample_grid_soa_precomputed::<S>(img, &roi, &ref_h_mat, decoder)
                                {
                                    if let Some((id_ref, hamming_ref, _)) =
                                        S::decode(&code_ref, decoder, 255)
                                    {
                                        // Only keep if it's the same tag and hamming is not worse
                                        if id_ref == id && hamming_ref <= hamming {
                                            best_h = hamming_ref;
                                            best_code = Some(code_ref);
                                            // Update the actual corners in the batch!
                                            if let Some(code_inner) = &best_code {
                                                return (
                                                    CandidateState::Valid,
                                                    best_id,
                                                    best_rot,
                                                    S::to_debug_bits(code_inner),
                                                    best_h as f32,
                                                    Some(refined_corners_f32),
                                                );
                                            }
                                        }
                                    }
                                }
                            }

                            return (
                                CandidateState::Valid,
                                best_id,
                                best_rot,
                                S::to_debug_bits(&code),
                                hamming as f32,
                                None,
                            );
                        }

                        if best_h == 0 {
                            break;
                        }
                    }

                    // Stage 2: Configurable Corner Refinement (Recovery for near-misses)
                    let max_h_for_refine = if decoders.iter().any(|d| d.name() == "36h11") {
                        10
                    } else {
                        4
                    };

                    if best_h > config.max_hamming_error
                        && best_h <= max_h_for_refine
                        && best_overall_code.is_some()
                    {
                        match config.refinement_mode {
                            crate::config::CornerRefinementMode::None
                            | crate::config::CornerRefinementMode::GridFit => {
                                // GridFit not ported to SoA yet to save complexity.
                            }
                            crate::config::CornerRefinementMode::Edge
                            | crate::config::CornerRefinementMode::Erf => {
                                let nudge = 0.2;
                                let mut current_corners = [Point2f::default(); 4];
                                current_corners.copy_from_slice(corners);

                                for _pass in 0..2 {
                                    let mut pass_improved = false;
                                    for c_idx in 0..4 {
                                        for (dx, dy) in [
                                            (nudge, 0.0),
                                            (-nudge, 0.0),
                                            (0.0, nudge),
                                            (0.0, -nudge),
                                        ] {
                                            let mut test_corners = current_corners;
                                            test_corners[c_idx].x += dx;
                                            test_corners[c_idx].y += dy;

                                            // Must recompute homography for the nudged corners
                                            let dst = [
                                                [
                                                    f64::from(test_corners[0].x),
                                                    f64::from(test_corners[0].y),
                                                ],
                                                [
                                                    f64::from(test_corners[1].x),
                                                    f64::from(test_corners[1].y),
                                                ],
                                                [
                                                    f64::from(test_corners[2].x),
                                                    f64::from(test_corners[2].y),
                                                ],
                                                [
                                                    f64::from(test_corners[3].x),
                                                    f64::from(test_corners[3].y),
                                                ],
                                            ];

                                            if let Some(h_new) = Homography::square_to_quad(&dst) {
                                                let mut h_mat = Matrix3x3 {
                                                    data: [0.0; 9],
                                                    padding: [0.0; 7],
                                                };
                                                for (j, val) in h_new.h.iter().enumerate() {
                                                    h_mat.data[j] = *val as f32;
                                                }

                                                for decoder in decoders {
                                                    if let Some(code) =
                                                        sample_grid_soa_precomputed::<S>(
                                                            img,
                                                            &roi,
                                                            &h_mat,
                                                            decoder.as_ref(),
                                                        )
                                                    {
                                                        if let Some((id, hamming, rot)) =
                                                            S::decode(&code, decoder.as_ref(), 255)
                                                        {
                                                            if hamming < best_h {
                                                                best_h = hamming;
                                                                best_overall_code =
                                                                    Some(code.clone());
                                                                current_corners = test_corners;
                                                                pass_improved = true;

                                                                if hamming
                                                                    <= config.max_hamming_error
                                                                {
                                                                    best_id = id;
                                                                    best_rot = rot;
                                                                    best_code = Some(code.clone());

                                                                    return (
                                                                        CandidateState::Valid,
                                                                        best_id,
                                                                        best_rot,
                                                                        S::to_debug_bits(&code),
                                                                        best_h as f32,
                                                                        Some(current_corners),
                                                                    );
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if !pass_improved {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if let Some(code) = best_code {
                        (
                            CandidateState::Valid,
                            best_id,
                            best_rot,
                            S::to_debug_bits(&code),
                            best_h as f32,
                            None,
                        )
                    } else {
                        (CandidateState::FailedDecode, 0, 0, 0, 0.0, None)
                    }
                })
        })
        .collect();

    for (i, (state, id, rot, payload, error_rate, refined_corners)) in
        results.into_iter().enumerate()
    {
        batch.status_mask[i] = state;
        batch.ids[i] = id;
        batch.payloads[i] = payload;
        batch.error_rates[i] = error_rate;

        if let Some(refined) = refined_corners {
            for (j, corner) in refined.iter().enumerate() {
                batch.corners[i * 4 + j] = *corner;
            }
        }

        if state == CandidateState::Valid && rot > 0 {
            // Reorder corners based on rotation
            let mut temp_corners = [Point2f::default(); 4];
            for (j, item) in temp_corners.iter_mut().enumerate() {
                let src_idx = (j + usize::from(rot)) % 4;
                *item = batch.corners[i * 4 + src_idx];
            }
            for (j, item) in temp_corners.iter().enumerate() {
                batch.corners[i * 4 + j] = *item;
            }
        }
    }
}

/// A trait for decoding binary payloads from extracted tags.
pub trait TagDecoder: Send + Sync {
    /// Returns the name of the decoder family (e.g., "AprilTag36h11").
    fn name(&self) -> &str;
    /// Returns the dimension of the tag grid (e.g., 6 for 36h11).
    fn dimension(&self) -> usize;
    /// Returns the active number of bits in the tag (e.g., 41 for 41h12).
    fn bit_count(&self) -> usize;
    /// Returns the ideal sample points in canonical coordinates [-1, 1].
    fn sample_points(&self) -> &[(f64, f64)];
    /// Decodes the extracted bits into a tag ID, hamming distance, and rotation count.
    ///
    /// Returns `Some((id, hamming, rotation))` if decoding is successful, `None` otherwise.
    /// `rotation` is 0-3, representing 90-degree CW increments.
    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)>; // (id, hamming, rotation)
    /// Decodes with custom maximum hamming distance.
    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)>;
    /// Get the original code for a given ID (useful for testing/simulation).
    fn get_code(&self, id: u16) -> Option<u64>;
    /// Returns the total number of codes in the dictionary.
    fn num_codes(&self) -> usize;
    /// Returns all rotated versions of all codes in the dictionary: (bits, id, rotation)
    fn rotated_codes(&self) -> &[(u64, u16, u8)];
    /// Executes a callback for each candidate in the dictionary within a given Hamming distance.
    /// This uses Multi-Index Hashing if available for efficiency.
    fn for_each_candidate_within_hamming(
        &self,
        bits: u64,
        max_hamming: u32,
        callback: &mut dyn FnMut(u64, u16, u8),
    );
}

/// Decoder for the AprilTag 36h11 family.
pub struct AprilTag36h11;

impl TagDecoder for AprilTag36h11 {
    fn name(&self) -> &'static str {
        "36h11"
    }
    fn dimension(&self) -> usize {
        6
    } // 6x6 grid of bits (excluding border)
    fn bit_count(&self) -> usize {
        36
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_APRILTAG36H11
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        // Use the pre-calculated dictionary with O(1) exact match + cached rotations.
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11)
            .decode(bits, 4) // Allow up to 4 bit errors for maximum recall
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[] // Removed from runtime struct, only used by testing/simulation which we will adjust later.
    }
    fn for_each_candidate_within_hamming(
        &self,
        bits: u64,
        max_hamming: u32,
        callback: &mut dyn FnMut(u64, u16, u8),
    ) {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11)
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Decoder for the AprilTag 41h12 family.
pub struct AprilTag41h12;

impl TagDecoder for AprilTag41h12 {
    fn name(&self) -> &'static str {
        "41h12"
    }
    fn dimension(&self) -> usize {
        9
    }
    fn bit_count(&self) -> usize {
        41
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_APRILTAG41H12
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag41h12)
            .decode(bits, 4)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag41h12)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag41h12).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag41h12).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }
    fn for_each_candidate_within_hamming(
        &self,
        bits: u64,
        max_hamming: u32,
        callback: &mut dyn FnMut(u64, u16, u8),
    ) {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag41h12)
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Decoder for the ArUco 4x4_50 family.
pub struct ArUco4x4_50;

impl TagDecoder for ArUco4x4_50 {
    fn name(&self) -> &'static str {
        "4X4_50"
    }
    fn dimension(&self) -> usize {
        4
    }
    fn bit_count(&self) -> usize {
        16
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_ARUCO4X4_50
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50)
            .decode(bits, 2)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }
    fn for_each_candidate_within_hamming(
        &self,
        bits: u64,
        max_hamming: u32,
        callback: &mut dyn FnMut(u64, u16, u8),
    ) {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_50)
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Decoder for the ArUco 4x4_100 family.
pub struct ArUco4x4_100;

impl TagDecoder for ArUco4x4_100 {
    fn name(&self) -> &'static str {
        "4X4_100"
    }
    fn dimension(&self) -> usize {
        4
    }
    fn bit_count(&self) -> usize {
        16
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_ARUCO4X4_100
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100)
            .decode(bits, 2)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100).len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }
    fn for_each_candidate_within_hamming(
        &self,
        bits: u64,
        max_hamming: u32,
        callback: &mut dyn FnMut(u64, u16, u8),
    ) {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco4x4_100)
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Generic decoder for any TagDictionary (static or custom).
pub(crate) struct GenericDecoder {
    dict: std::sync::Arc<crate::dictionaries::TagDictionary>,
}

impl GenericDecoder {
    /// Create a new generic decoder from a dictionary.
    #[must_use]
    pub fn new(dict: crate::dictionaries::TagDictionary) -> Self {
        Self {
            dict: std::sync::Arc::new(dict),
        }
    }
}

impl TagDecoder for GenericDecoder {
    fn name(&self) -> &'static str {
        "Generic"
    }

    fn dimension(&self) -> usize {
        f64::from(self.dict.payload_length).sqrt() as usize
    }

    fn bit_count(&self) -> usize {
        self.dict.payload_length as usize
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        &[] // We can add this back via dictionary lookup if generic decoder needs it fully typed, for now unused internally.
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        self.dict
            .decode(bits, self.dict.min_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        self.dict
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        self.dict.get_code(id)
    }

    fn num_codes(&self) -> usize {
        self.dict.len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        &[]
    }

    fn for_each_candidate_within_hamming(
        &self,
        bits: u64,
        max_hamming: u32,
        callback: &mut dyn FnMut(u64, u16, u8),
    ) {
        self.dict
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Convert a TagFamily enum to a boxed decoder instance.
#[must_use]
pub fn family_to_decoder(family: config::TagFamily) -> Box<dyn TagDecoder + Send + Sync> {
    match family {
        config::TagFamily::AprilTag36h11 => Box::new(AprilTag36h11),
        config::TagFamily::AprilTag41h12 => Box::new(AprilTag41h12),
        config::TagFamily::ArUco4x4_50 => Box::new(ArUco4x4_50),
        config::TagFamily::ArUco4x4_100 => Box::new(ArUco4x4_100),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_rotation_invariants(bits in 0..u64::MAX) {
            let dim = 6;
            let r1 = rotate90(bits, dim);
            let r2 = rotate90(r1, dim);
            let r3 = rotate90(r2, dim);
            let r4 = rotate90(r3, dim);

            // Mask to dim*dim bits to avoid noise in upper bits
            let mask = (1u64 << (dim * dim)) - 1;
            prop_assert_eq!(bits & mask, r4 & mask);
        }

        #[test]
        fn test_hamming_robustness(
            id_idx in 0usize..10,
            rotation in 0..4usize,
            flip1 in 0..36usize,
            flip2 in 0..36usize
        ) {
            let decoder = AprilTag36h11;
            let orig_id = id_idx as u16;
            let dict = crate::dictionaries::get_dictionary(config::TagFamily::AprilTag36h11);

            // Get the correctly geometrically rotated code directly from our generated dictionaries
            let mut test_bits = dict.codes[(id_idx * 4) + rotation];

            // Flip bits
            test_bits ^= 1 << flip1;
            test_bits ^= 1 << flip2;

            let result = decoder.decode(test_bits);
            prop_assert!(result.is_some());
            let (decoded_id, _, _) = result.expect("Should decode valid pattern");
            prop_assert_eq!(decoded_id, u32::from(orig_id));
        }

        #[test]
        fn test_false_positive_resistance(bits in 0..u64::MAX) {
            let decoder = AprilTag36h11;
            // Random bitstreams should rarely match any of the 587 codes
            if let Some((_id, hamming, _rot)) = decoder.decode(bits) {
                // If it decodes, it must have low hamming distance
                prop_assert!(hamming <= 4);
            }
        }

        #[test]
        fn prop_homography_projection(
            src in prop::collection::vec((-100.0..100.0, -100.0..100.0), 4),
            dst in prop::collection::vec((0.0..1000.0, 0.0..1000.0), 4)
        ) {
            let src_pts = [
                [src[0].0, src[0].1],
                [src[1].0, src[1].1],
                [src[2].0, src[2].1],
                [src[3].0, src[3].1],
            ];
            let dst_pts = [
                [dst[0].0, dst[0].1],
                [dst[1].0, dst[1].1],
                [dst[2].0, dst[2].1],
                [dst[3].0, dst[3].1],
            ];

            if let Some(h) = Homography::from_pairs(&src_pts, &dst_pts) {
                for i in 0..4 {
                    let p = h.project(src_pts[i]);
                    // Check for reasonable accuracy. 1e-4 is conservative for float precision
                    // issues in near-singular cases where from_pairs still returns Some.
                    prop_assert!((p[0] - dst_pts[i][0]).abs() < 1e-3,
                        "Point {}: project({:?}) -> {:?}, expected {:?}", i, src_pts[i], p, dst_pts[i]);
                    prop_assert!((p[1] - dst_pts[i][1]).abs() < 1e-3);
                }
            }
        }
    }

    #[test]
    fn test_all_codes_decode() {
        let decoder = AprilTag36h11;
        for id in 0..587u16 {
            let code = crate::dictionaries::DICT_APRILTAG36H11
                .get_code(id)
                .expect("valid ID");
            let result = decoder.decode(code);
            assert!(result.is_some());
            let (id_out, hamming_out, rot_out) = result.unwrap();
            assert_eq!(id_out, u32::from(id));
            assert_eq!(hamming_out, 0);
            assert_eq!(rot_out, 0);
        }
    }

    #[test]
    fn test_all_codes_decode_41h12() {
        let decoder = AprilTag41h12;
        let dict = crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag41h12);
        for id in 0..dict.len() as u16 {
            let code = dict.get_code(id).expect("valid ID");
            let result = decoder.decode(code);
            assert!(result.is_some(), "Failed to decode 41h12 ID {id}");
            let (id_out, hamming_out, rot_out) = result.unwrap();
            assert_eq!(id_out, u32::from(id));
            assert_eq!(hamming_out, 0);
            assert_eq!(rot_out, 0);
        }
    }

    #[test]
    fn test_grid_sampling() {
        let width = 64;
        let height = 64;
        let stride = 64;
        let mut data = vec![0u8; width * height];

        // Draw a simulated 36h6 tag (6x6 bits)
        // Center at 32, 32. Tag size 36x36 pixels.
        // Each bit is 6x6 pixels.
        // We will set bit 0 (top-left) to 255 (white), others 0 (black).
        // Plus a white border for contrast calculation.
        // We need min/max to be different by > 20.

        // Fill background with gray = 100
        data.fill(100);

        // Set bit 0 region to 200 (White)
        // Canonical (-0.625, -0.625) -> Image coords.
        // Assume identity homography mapping:
        // -1 -> 14, +1 -> 50 (width 36 centered at 32)
        // scale = 18. offset = 32.

        // Bit 0 center is -0.625.
        // Image x = 32 + 18 * -0.625 = 32 - 11.25 = 20.75
        // Let's paint a blob at 21, 21.
        for y in 18..24 {
            for x in 18..24 {
                data[y * width + x] = 200;
            }
        }

        // Set another region (last bit) to 50 (Black) to ensure dynamic range
        // Last bit 35 is at (0.625, 0.625).
        // Image x = 32 + 18 * 0.625 = 43.25
        for y in 40..46 {
            for x in 40..46 {
                data[y * width + x] = 50;
            }
        }

        let img = crate::image::ImageView::new(&data, width, height, stride).unwrap();

        // Construct Homography that maps canonical [-1, 1] to image [14, 50]
        // This is a simple scaling matrix:
        // [ sx  0  tx ]
        // [ 0  sy  ty ]
        // [ 0   0   1 ]
        // sx = 18, tx = 32.
        let mut h = SMatrix::<f64, 3, 3>::identity();
        h[(0, 0)] = 18.0;
        h[(0, 2)] = 32.0;
        h[(1, 1)] = 18.0;
        h[(1, 2)] = 32.0;

        let decoder = AprilTag36h11;
        let arena = Bump::new();
        let cand = crate::Detection {
            corners: [
                [32.0, 32.0],
                [32.0 + 18.0, 32.0],
                [32.0 + 18.0, 32.0 + 18.0],
                [32.0, 32.0 + 18.0],
            ],
            ..Default::default()
        };
        let bits =
            sample_grid(&img, &arena, &cand, &decoder, 20.0).expect("Should sample successfully");

        // bit 0 should be 1 (high intensity)
        assert_eq!(bits & 1, 1, "Bit 0 should be 1");
        // bit 35 should be 0 (low intensity)
        assert_eq!((bits >> 35) & 1, 0, "Bit 35 should be 0");
    }

    #[test]
    fn test_homography_dlt() {
        let src = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let dst = [[10.0, 10.0], [20.0, 11.0], [19.0, 21.0], [9.0, 20.0]];

        let h = Homography::from_pairs(&src, &dst).expect("DLT should succeed");
        for i in 0..4 {
            let p = h.project(src[i]);
            assert!((p[0] - dst[i][0]).abs() < 1e-6);
            assert!((p[1] - dst[i][1]).abs() < 1e-6);
        }
    }

    // ========================================================================
    // END-TO-END DECODER ROBUSTNESS TESTS
    // ========================================================================

    use crate::config::TagFamily;
    use crate::image::ImageView;
    use crate::quad::extract_quads_fast;
    use crate::segmentation::label_components_with_stats;
    use crate::test_utils::{TestImageParams, generate_test_image_with_params};
    use crate::threshold::ThresholdEngine;
    use bumpalo::Bump;

    /// Run full pipeline from image to decoded tags.
    fn run_full_pipeline(tag_size: usize, canvas_size: usize, tag_id: u16) -> Vec<(u32, u32)> {
        let params = TestImageParams {
            family: TagFamily::AprilTag36h11,
            id: tag_id,
            tag_size,
            canvas_size,
            ..Default::default()
        };

        let (data, _corners) = generate_test_image_with_params(&params);
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

        let arena = Bump::new();
        let engine = ThresholdEngine::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let mut binary = vec![0u8; canvas_size * canvas_size];
        engine.apply_threshold(&arena, &img, &stats, &mut binary);
        let label_result =
            label_components_with_stats(&arena, &binary, canvas_size, canvas_size, true);
        let detections = extract_quads_fast(&arena, &img, &label_result);

        let decoder = AprilTag36h11;
        let mut results = Vec::new();

        for quad in &detections {
            if let Some(bits) = sample_grid(&img, &arena, quad, &decoder, 20.0)
                && let Some((id, hamming, _rot)) = decoder.decode(bits)
            {
                results.push((id, hamming));
            }
        }

        results
    }

    /// Test E2E pipeline decodes correctly at varying sizes.
    #[test]
    fn test_e2e_decoding_at_varying_sizes() {
        let canvas_size = 640;
        let tag_sizes = [64, 100, 150, 200, 300];
        let test_id: u16 = 42;

        for tag_size in tag_sizes {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            let found = decoded.iter().any(|(id, _)| *id == u32::from(test_id));

            if tag_size >= 64 {
                assert!(found, "Tag size {tag_size}: ID {test_id} not found");
            }

            if found {
                let (_, hamming) = decoded
                    .iter()
                    .find(|(id, _)| *id == u32::from(test_id))
                    .unwrap();
                println!("Tag size {tag_size:>3}px: ID {test_id} with hamming {hamming}");
            }
        }
    }

    /// Test that multiple tag IDs decode correctly.
    #[test]
    fn test_e2e_multiple_ids() {
        let canvas_size = 400;
        let tag_size = 150;
        let test_ids: [u16; 5] = [0, 42, 100, 200, 500];

        for &test_id in &test_ids {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            let found = decoded.iter().any(|(id, _)| *id == u32::from(test_id));
            assert!(found, "ID {test_id} not decoded");

            let (_, hamming) = decoded
                .iter()
                .find(|(id, _)| *id == u32::from(test_id))
                .unwrap();
            assert_eq!(*hamming, 0, "ID {test_id} should have 0 hamming");
            println!("ID {test_id:>3}: Decoded with hamming {hamming}");
        }
    }

    /// Test decoding with edge ID values.
    #[test]
    fn test_e2e_edge_ids() {
        let canvas_size = 400;
        let tag_size = 150;
        let edge_ids: [u16; 2] = [0, 586];

        for &test_id in &edge_ids {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            let found = decoded.iter().any(|(id, _)| *id == u32::from(test_id));
            assert!(found, "Edge ID {test_id} not decoded");
            println!("Edge ID {test_id}: Decoded");
        }
    }
}
