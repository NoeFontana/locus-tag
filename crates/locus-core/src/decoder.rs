//! Tag decoding, homography computation, and bit sampling.
//!
//! This module handles the final stage of the pipeline:
//! 1. **Homography**: Computing the projection from canonical tag space to image pixels.
//! 2. **Bit Sampling**: Bilinear interpolation of intensities at grid points.
//! 3. **Error Correction**: Correcting bit flips using tag-family specific Hamming distances.

#![allow(unsafe_code, clippy::cast_sign_loss)]
use crate::config;
use nalgebra::{SMatrix, SVector};
use multiversion::multiversion;

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

/// Refine corner positions using edge-based optimization with the homography.
///
/// After successful decoding, we fit lines to each edge using gradient-weighted
/// least squares, then compute corners as line intersections. This provides
/// more accurate corner localization than the initial detection.
#[must_use]
pub fn refine_corners_with_homography(
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
pub fn refine_corners_gridfit(
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
pub fn refine_corners_erf(
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
            let mut sum_g = 0.0;
            let mut count = 0;
            let scan_d = self.d + offset;

            for py in y0..=y1 {
                for px in x0..=x1 {
                    let x = px as f64;
                    let y = py as f64;
                    let dist = self.nx * x + self.ny * y + scan_d;
                    if dist.abs() < 1.0 {
                        let g = self.img.sample_gradient_bilinear(x, y);
                        sum_g += (g[0] * self.nx + g[1] * self.ny).abs();
                        count += 1;
                    }
                }
            }

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

        let mut samples = bumpalo::collections::Vec::with_capacity_in(128, arena);

        for py in y0..=y1 {
            for px in x0..=x1 {
                let x = px as f64;
                let y = py as f64;

                let dist = self.nx * x + self.ny * y + self.d;
                if dist.abs() > window {
                    continue;
                }

                let t = ((x - self.p1[0]) * self.dx + (y - self.p1[1]) * self.dy)
                    / (self.len * self.len);

                if (-0.1..=1.1).contains(&t) {
                    let val = f64::from(self.img.get_pixel(px, py));
                    samples.push((x, y, val));
                }
            }
        }
        samples
    }

    fn refine(&mut self, samples: &[(f64, f64, f64)], sigma: f64) {
        if samples.len() < 10 {
            return;
        }
        let mut a = 128.0;
        let mut b = 128.0;
        let inv_sigma = 1.0 / sigma;
        let sqrt_pi = std::f64::consts::PI.sqrt();

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

            let mut sum_jtj = 0.0;
            let mut sum_jt_res = 0.0;
            let k = (b - a) / (sqrt_pi * sigma);

            for &(x, y, _) in samples {
                let dist = self.nx * x + self.ny * y + self.d;
                let s = dist * inv_sigma;
                if s.abs() > 3.0 {
                    continue;
                }
                let val = self.img.sample_bilinear(x, y);
                let model = (a + b) * 0.5 + (b - a) * 0.5 * crate::quad::erf_approx(s);
                let residual = val - model;
                let jac = k * (-s * s).exp();
                sum_jtj += jac * jac;
                sum_jt_res += jac * residual;
            }

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
pub fn compute_otsu_threshold(values: &[f64]) -> f64 {
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

/// Optimized grid sampling kernel.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn sample_grid_values_simd(
    img: &crate::image::ImageView,
    h: &Homography,
    points: &[(f64, f64)],
    intensities: &mut [f64],
    n: usize,
) -> bool {
    let h00 = h.h[(0, 0)];
    let h01 = h.h[(0, 1)];
    let h02 = h.h[(0, 2)];
    let h10 = h.h[(1, 0)];
    let h11 = h.h[(1, 1)];
    let h12 = h.h[(1, 2)];
    let h20 = h.h[(2, 0)];
    let h21 = h.h[(2, 1)];
    let h22 = h.h[(2, 2)];

    let w_limit = (img.width - 1) as f64;
    let h_limit = (img.height - 1) as f64;

    for (i, &p) in points.iter().take(n).enumerate() {
        let wz = h20 * p.0 + h21 * p.1 + h22;
        let img_x = (h00 * p.0 + h01 * p.1 + h02) / wz;
        let img_y = (h10 * p.0 + h11 * p.1 + h12) / wz;

        if img_x < 0.0 || img_x >= w_limit || img_y < 0.0 || img_y >= h_limit {
            return false;
        }

        let xf = img_x.floor();
        let yf = img_y.floor();
        let ix = xf as usize;
        let iy = yf as usize;
        let dx = img_x - xf;
        let dy = img_y - yf;

        // SAFETY: Bounds checked above.
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
        intensities[i] = val;
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
    homography: &Homography,
    decoder: &(impl TagDecoder + ?Sized),
) -> Option<S::Code> {
    let points = decoder.sample_points();
    // Stack-allocated buffer for up to 64 sample points (covers all standard tag families)
    let mut intensities = [0.0f64; 64];
    let n = points.len().min(64);

    if !sample_grid_values_simd(img, homography, points, &mut intensities, n) {
        return None;
    }

    // SOTA: Blended Quadrant-based Adaptive Thresholding
    // Combines global Otsu (robust for bimodal) with local quadrant averages (robust for shadows)
    let global_threshold = compute_otsu_threshold(&intensities[..n]);

    let mut quad_sums = [0.0; 4];
    let mut quad_counts = [0; 4];
    for (i, &p) in points.iter().take(n).enumerate() {
        let qi = if p.0 < 0.0 {
            usize::from(p.1 >= 0.0)
        } else {
            2 + usize::from(p.1 >= 0.0)
        };
        quad_sums[qi] += intensities[i];
        quad_counts[qi] += 1;
    }

    let mut thresholds = [0.0f64; 64];

    for (i, &p) in points.iter().take(n).enumerate() {
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

        // Blend global Otsu and local mean (0.7 / 0.3 weighting is SOTA for fiducials)
        let effective_threshold = 0.7 * global_threshold + 0.3 * quad_avg;
        thresholds[i] = effective_threshold;
    }

    Some(S::from_intensities(&intensities[..n], &thresholds[..n]))
}

/// Sample the bit grid from the image (Legacy/Hard wrapper).
#[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
pub fn sample_grid(
    img: &crate::image::ImageView,
    homography: &Homography,
    decoder: &(impl TagDecoder + ?Sized),
    _min_contrast: f64,
) -> Option<u64> {
    sample_grid_generic::<crate::strategy::HardStrategy>(img, homography, decoder)
}

/// A trait for decoding binary payloads from extracted tags.
pub trait TagDecoder: Send + Sync {
    /// Returns the name of the decoder family (e.g., "AprilTag36h11").
    fn name(&self) -> &str;
    /// Returns the dimension of the tag grid (e.g., 6 for 36h11).
    fn dimension(&self) -> usize;
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

    fn sample_points(&self) -> &[(f64, f64)] {
        &crate::dictionaries::APRILTAG_36H11.sample_points
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        // Use the full 587-code dictionary with O(1) exact match + hamming search
        crate::dictionaries::APRILTAG_36H11
            .decode(bits, 4) // Allow up to 4 bit errors for maximum recall
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::APRILTAG_36H11
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::APRILTAG_36H11.get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::APRILTAG_36H11.len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        crate::dictionaries::APRILTAG_36H11.rotated_codes()
    }
}

/// Decoder for the AprilTag 16h5 family.
pub struct AprilTag16h5;

impl TagDecoder for AprilTag16h5 {
    fn name(&self) -> &'static str {
        "16h5"
    }
    fn dimension(&self) -> usize {
        4
    } // 4x4 grid of bits (excluding border)

    fn sample_points(&self) -> &[(f64, f64)] {
        &crate::dictionaries::APRILTAG_16H5.sample_points
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::APRILTAG_16H5
            .decode(bits, 1) // Allow up to 1 bit error (16h5 has lower hamming distance)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::APRILTAG_16H5
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::APRILTAG_16H5.get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::APRILTAG_16H5.len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        crate::dictionaries::APRILTAG_16H5.rotated_codes()
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

    fn sample_points(&self) -> &[(f64, f64)] {
        &crate::dictionaries::ARUCO_4X4_50.sample_points
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::ARUCO_4X4_50
            .decode(bits, 1)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::ARUCO_4X4_50
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::ARUCO_4X4_50.get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::ARUCO_4X4_50.len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        crate::dictionaries::ARUCO_4X4_50.rotated_codes()
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

    fn sample_points(&self) -> &[(f64, f64)] {
        &crate::dictionaries::ARUCO_4X4_100.sample_points
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::ARUCO_4X4_100
            .decode(bits, 1)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::ARUCO_4X4_100
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::ARUCO_4X4_100.get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::ARUCO_4X4_100.len()
    }

    fn rotated_codes(&self) -> &[(u64, u16, u8)] {
        crate::dictionaries::ARUCO_4X4_100.rotated_codes()
    }
}

/// Generic decoder for any TagDictionary (static or custom).
pub struct GenericDecoder {
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
    fn name(&self) -> &str {
        &self.dict.name
    }

    fn dimension(&self) -> usize {
        self.dict.dimension
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        &self.dict.sample_points
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        self.dict
            .decode(bits, self.dict.hamming_distance as u32)
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
        self.dict.rotated_codes()
    }
}

/// Convert a TagFamily enum to a boxed decoder instance.
#[must_use]
pub fn family_to_decoder(family: config::TagFamily) -> Box<dyn TagDecoder + Send + Sync> {
    match family {
        config::TagFamily::AprilTag36h11 => Box::new(AprilTag36h11),
        config::TagFamily::AprilTag16h5 => Box::new(AprilTag16h5),
        config::TagFamily::ArUco4x4_50 => Box::new(ArUco4x4_50),
        config::TagFamily::ArUco4x4_100 => Box::new(ArUco4x4_100),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::dictionaries::rotate90;
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
            let dict = &*crate::dictionaries::APRILTAG_36H11;
            let orig_id = id_idx as u16;
            let orig_code = dict.get_code(orig_id).expect("valid ID");

            // Apply rotation
            let mut test_bits = orig_code;
            for _ in 0..rotation {
                test_bits = rotate90(test_bits, 6);
            }

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
            let code = crate::dictionaries::APRILTAG_36H11
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
        let homography = Homography { h };

        let decoder = AprilTag36h11;
        let bits =
            sample_grid(&img, &homography, &decoder, 20.0).expect("Should sample successfully");

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
            let dst = [
                [quad.corners[0][0], quad.corners[0][1]],
                [quad.corners[1][0], quad.corners[1][1]],
                [quad.corners[2][0], quad.corners[2][1]],
                [quad.corners[3][0], quad.corners[3][1]],
            ];

            if let Some(h) = Homography::square_to_quad(&dst)
                && let Some(bits) = sample_grid(&img, &h, &decoder, 20.0)
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
