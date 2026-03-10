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
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 1.0;
        m[(0, 2)] = -1.0;
        m[(0, 6)] = -x0;
        m[(0, 7)] = -x0;
        b[0] = -x0;
        m[(1, 3)] = 1.0;
        m[(1, 4)] = 1.0;
        m[(1, 5)] = -1.0;
        m[(1, 6)] = -y0;
        m[(1, 7)] = -y0;
        b[1] = -y0;

        // Point 1: (1, -1) -> (x1, y1)
        let x1 = dst[1][0];
        let y1 = dst[1][1];
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
#[tracing::instrument(skip_all, name = "pipeline::homography_pass")]
pub fn compute_homographies_soa(corners: &[[Point2f; 4]], homographies: &mut [Matrix3x3]) {
    use rayon::prelude::*;

    homographies
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, h_out)| {
            let dst = [
                [f64::from(corners[i][0].x), f64::from(corners[i][0].y)],
                [f64::from(corners[i][1].x), f64::from(corners[i][1].y)],
                [f64::from(corners[i][2].x), f64::from(corners[i][2].y)],
                [f64::from(corners[i][3].x), f64::from(corners[i][3].y)],
            ];

            if let Some(h) = Homography::square_to_quad(&dst) {
                for (j, val) in h.h.iter().enumerate() {
                    h_out.data[j] = *val as f32;
                }
                h_out.padding = [0.0; 7];
            } else {
                h_out.data = [0.0; 9];
                h_out.padding = [0.0; 7];
            }
        });
}

/// Refine corner positions using edge-based optimization.
#[must_use]
#[allow(dead_code)]
pub(crate) fn refine_corners_with_homography(
    img: &crate::image::ImageView,
    corners: &[[f64; 2]; 4],
    _homography: &Homography,
) -> [[f64; 2]; 4] {
    let mut lines = [(0.0f64, 0.0f64, 0.0f64); 4];
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

        let nx = -dy / len;
        let ny = dx / len;

        let mut sum_w = 0.0;
        let mut sum_d = 0.0;
        let n_samples = (len as usize).clamp(5, 20);

        for s in 1..=n_samples {
            let t = s as f64 / (n_samples + 1) as f64;
            let px = p1[0] + dx * t;
            let py = p1[1] + dy * t;

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
                let d = nx * best_pos.0 + ny * best_pos.1;
                sum_w += best_mag;
                sum_d += d * best_mag;
            }
        }

        if sum_w > 1.0 {
            let c = -sum_d / sum_w;
            lines[i] = (nx, ny, c);
            line_valid[i] = true;
        }
    }

    if !line_valid.iter().all(|&v| v) {
        return *corners;
    }

    let mut refined = *corners;
    for i in 0..4 {
        let prev = (i + 3) % 4;
        let (a1, b1, c1) = lines[prev];
        let (a2, b2, c2) = lines[i];

        let det = a1 * b2 - a2 * b1;
        if det.abs() > 1e-6 {
            let x = (b1 * c2 - b2 * c1) / det;
            let y = (a2 * c1 - a1 * c2) / det;

            let dist_sq = (x - corners[i][0]).powi(2) + (y - corners[i][1]).powi(2);
            if dist_sq < 4.0 {
                refined[i] = [x, y];
            }
        }
    }

    refined
}

/// Refine corners using Erf-Fit.
pub(crate) fn refine_corners_erf(
    img: &crate::image::ImageView,
    corners: &[[f64; 2]; 4],
    sigma: f64,
) -> [[f64; 2]; 4] {
    let mut lines = [(0.0f64, 0.0f64, 0.0f64); 4];
    let mut line_valid = [false; 4];

    for i in 0..4 {
        let next = (i + 1) % 4;
        let p1 = corners[i];
        let p2 = corners[next];

        if let Some((nx, ny, d)) = fit_edge_erf(img, p1, p2, sigma) {
            lines[i] = (nx, ny, d);
            line_valid[i] = true;
        }
    }

    if !line_valid.iter().all(|&v| v) {
        return *corners;
    }

    let mut refined = *corners;
    for i in 0..4 {
        let prev = (i + 3) % 4;
        let (a1, b1, c1) = lines[prev];
        let (a2, b2, c2) = lines[i];
        let det = a1 * b2 - a2 * b1;
        if det.abs() > 1e-6 {
            let x = (b1 * c2 - b2 * c1) / det;
            let y = (a2 * c1 - a1 * c2) / det;

            let dist_sq = (x - corners[i][0]).powi(2) + (y - corners[i][1]).powi(2);
            if dist_sq < 4.0 {
                refined[i] = [x, y];
            }
        }
    }
    refined
}

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

            let mut sum_g = 0.0;
            let mut count = 0;
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

    fn collect_samples(&self) -> Vec<(f64, f64, f64)> {
        let window = 2.5;
        let (x0, x1, y0, y1) = self.get_scan_bounds(window);
        let mut samples = Vec::new();

        for py in y0..=y1 {
            for px in x0..=x1 {
                let x = px as f64;
                let y = py as f64;
                let dist = self.nx * x + self.ny * y + self.d;
                if dist.abs() <= window {
                    let t = ((x - self.p1[0]) * self.dx + (y - self.p1[1]) * self.dy)
                        / (self.len * self.len);
                    if (-0.1..=1.1).contains(&t) {
                        let val = f64::from(self.img.get_pixel(px, py));
                        samples.push((x, y, val));
                    }
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
            let sqrt_pi = std::f64::consts::PI.sqrt();
            let k = (b - a) / (sqrt_pi * sigma);

            for &(x, y, img_val) in samples {
                let dist = self.nx * x + self.ny * y + self.d;
                let s = dist / sigma;
                if s.abs() <= 3.0 {
                    let model = (a + b) * 0.5 + (b - a) * 0.5 * crate::quad::erf_approx(s);
                    let residual = img_val - model;
                    let jac = k * (-s * s).exp();
                    sum_jtj += jac * jac;
                    sum_jt_res += jac * residual;
                }
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

    fn get_scan_bounds(&self, window: f64) -> (usize, usize, usize, usize) {
        let p2_0 = self.p1[0] + self.dx;
        let p2_1 = self.p1[1] + self.dy;
        let w_limit = (self.img.width - 2) as f64;
        let h_limit = (self.img.height - 2) as f64;

        let x0 = (self.p1[0].min(p2_0) - window - 0.5).clamp(1.0, w_limit) as usize;
        let x1 = (self.p1[0].max(p2_0) + window + 0.5).clamp(1.0, w_limit) as usize;
        let y0 = (self.p1[1].min(p2_1) - window - 0.5).clamp(1.0, h_limit) as usize;
        let y1 = (self.p1[1].max(p2_1) + window + 0.5).clamp(1.0, h_limit) as usize;
        (x0, x1, y0, y1)
    }
}

fn fit_edge_erf(
    img: &crate::image::ImageView,
    p1: [f64; 2],
    p2: [f64; 2],
    sigma: f64,
) -> Option<(f64, f64, f64)> {
    let mut fitter = EdgeFitter::new(img, p1, p2)?;
    fitter.scan_initial_d();
    let samples = fitter.collect_samples();
    if samples.len() < 10 {
        return None;
    }
    fitter.refine(&samples, sigma);
    Some((fitter.nx, fitter.ny, fitter.d))
}

pub(crate) fn compute_otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 128.0;
    }

    let n = values.len() as f64;
    let total_sum: f64 = values.iter().sum();
    let min_val = values.iter().copied().fold(f64::MAX, f64::min);
    let max_val = values.iter().copied().fold(f64::MIN, f64::max);

    if (max_val - min_val) < 1.0 {
        return (min_val + max_val) * 0.5;
    }

    let mut best_threshold = (min_val + max_val) * 0.5;
    let mut best_variance = 0.0;

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
        let variance = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);

        if variance > best_variance {
            best_variance = variance;
            best_threshold = t;
        }
    }

    best_threshold
}

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

        let wz = h20 * px + h21 * py + h22;
        let winv = rcp_nr(wz);

        let img_x = (h00 * px + h01 * py + h02) * winv;
        let img_y = (h10 * px + h11 * py + h12) * winv;

        if img_x < 0.0 || img_x >= w_limit || img_y < 0.0 || img_y >= h_limit {
            return false;
        }

        let ix = img_x.floor() as usize;
        let iy = img_y.floor() as usize;

        let v00 = roi.get(ix, iy);
        let v10 = roi.get(ix + 1, iy);
        let v01 = roi.get(ix, iy + 1);
        let v11 = roi.get(ix + 1, iy + 1);

        intensities[i] = f64::from(bilinear_interpolate_fixed(img_x, img_y, v00, v10, v01, v11));
    }
    true
}

/// Sample intensities for a quad from an image using a generic decoding strategy.
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

/// Sample intensities for a quad from an image using a pre-calculated homography.
pub fn sample_grid_soa_precomputed<S: crate::strategy::DecodingStrategy>(
    img: &crate::image::ImageView,
    roi: &RoiCache,
    homography: &Matrix3x3,
    decoder: &(impl TagDecoder + ?Sized),
) -> Option<S::Code> {
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

        thresholds[i] = 0.7 * global_threshold + 0.3 * quad_avg;
    }
    thresholds
}

/// Sample bit pattern for a quad from an image.
pub fn sample_grid(
    img: &crate::image::ImageView,
    arena: &Bump,
    detection: &crate::Detection,
    decoder: &(impl TagDecoder + ?Sized),
    _min_contrast: f64,
) -> Option<u64> {
    sample_grid_generic::<crate::strategy::HardStrategy>(img, arena, detection, decoder)
}

/// Rotate bit pattern 90 degrees.
#[allow(dead_code)]
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
#[tracing::instrument(skip_all, name = "pipeline::decode_pass")]
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

#[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
fn decode_batch_soa_generic<S: crate::strategy::DecodingStrategy>(
    batch: &mut crate::batch::DetectionBatch,
    n: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
) {
    use crate::batch::CandidateState;
    use rayon::prelude::*;

    let results: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            DECODE_ARENA.with_borrow_mut(|arena| {
                    arena.reset();

                    let corners = &batch.corners[i];
                    let homography = &batch.homographies[i];

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

                    let scales = [1.0, 0.9, 1.1];
                    let center = [
                        (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4.0,
                        (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4.0,
                    ];

                    for scale in scales {
                        let mut scaled_h_mat = Matrix3x3 {
                            data: [0.0; 9],
                            padding: [0.0; 7],
                        };

                        let current_homography: &Matrix3x3;

                        let mut best_h_in_scale = u32::MAX;
                        let mut best_match_in_scale: Option<(u32, u32, u8, S::Code, usize)> = None;

                        if (scale - 1.0f32).abs() > 1e-4 {
                            let mut scaled_corners = [Point2f::default(); 4];
                            for (j, item) in scaled_corners.iter_mut().enumerate().take(4) {
                                item.x = center[0] + (corners[j].x - center[0]) * scale;
                                item.y = center[1] + (corners[j].y - center[1]) * scale;
                            }

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
                                continue;
                            }
                        } else {
                            current_homography = homography;
                        }

                        for (decoder_idx, decoder) in decoders.iter().enumerate() {
                            if let Some(code) = sample_grid_soa_precomputed::<S>(
                                img,
                                &roi,
                                current_homography,
                                decoder.as_ref(),
                            ) && let Some((id, hamming, rot)) =
                                S::decode(&code, decoder.as_ref(), 255)
                            {
                                if hamming < best_h {
                                    best_h = hamming;
                                }

                                if hamming <= config.max_hamming_error
                                    && (best_match_in_scale.is_none() || hamming < best_h_in_scale)
                                {
                                    best_h_in_scale = hamming;
                                    best_match_in_scale =
                                        Some((id, hamming, rot, code, decoder_idx));
                                }
                            }
                        }
                        if let Some((id, hamming, rot, code, decoder_idx)) = best_match_in_scale {
                            best_id = id;
                            best_rot = rot;
                            best_code = Some(code);
                            let decoder = decoders[decoder_idx].as_ref();

                            if config.refinement_mode == crate::config::CornerRefinementMode::Erf {
                                let mut current_corners = [[0.0f64; 2]; 4];
                                for j in 0..4 {
                                    current_corners[j] =
                                        [f64::from(corners[j].x), f64::from(corners[j].y)];
                                }

                                let refined_corners = refine_corners_erf(
                                    img,
                                    &current_corners,
                                    config.subpixel_refinement_sigma,
                                );

                                let mut refined_corners_f32 = [Point2f::default(); 4];
                                for j in 0..4 {
                                    refined_corners_f32[j] = Point2f {
                                        x: refined_corners[j][0] as f32,
                                        y: refined_corners[j][1] as f32,
                                    };
                                }

                                let mut ref_h_mat = Matrix3x3 {
                                    data: [0.0; 9],
                                    padding: [0.0; 7],
                                };
                                if let Some(h_new) = Homography::square_to_quad(&refined_corners) {
                                    for (j, val) in h_new.h.iter().enumerate() {
                                        ref_h_mat.data[j] = *val as f32;
                                    }
                                } else {
                                    continue;
                                }

                                if let Some(code_ref) =
                                    sample_grid_soa_precomputed::<S>(img, &roi, &ref_h_mat, decoder)
                                    && let Some((id_ref, hamming_ref, _)) =
                                        S::decode(&code_ref, decoder, 255)
                                    && id_ref == id
                                    && hamming_ref <= hamming
                                {
                                    best_h = hamming_ref;
                                    best_code = Some(code_ref);
                                    return (
                                        CandidateState::Valid,
                                        best_id,
                                        best_rot,
                                        S::to_debug_bits(
                                            best_code.as_ref().expect("best_code exists"),
                                        ),
                                        best_h as f32,
                                        Some(refined_corners_f32),
                                    );
                                }
                            }

                            return (
                                CandidateState::Valid,
                                best_id,
                                best_rot,
                                S::to_debug_bits(best_code.as_ref().expect("best_code exists")),
                                hamming as f32,
                                None,
                            );
                        }

                        if best_h == 0 {
                            break;
                        }
                    }

                    let max_h_for_refine = if decoders.iter().any(|d| d.name() == "36h11") {
                        10
                    } else {
                        4
                    };

                    if best_h > config.max_hamming_error
                        && best_h <= max_h_for_refine
                        && best_code.is_some()
                    {
                        match config.refinement_mode {
                            crate::config::CornerRefinementMode::None
                            | crate::config::CornerRefinementMode::GridFit => {},
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
                                                        && let Some((id, hamming, rot)) =
                                                            S::decode(&code, decoder.as_ref(), 255)
                                                        && hamming < best_h
                                                    {
                                                        best_h = hamming;
                                                        current_corners = test_corners;
                                                        pass_improved = true;

                                                        if hamming <= config.max_hamming_error {
                                                            best_id = id;
                                                            best_rot = rot;
                                                            best_code = Some(code);

                                                            return (
                                                                CandidateState::Valid,
                                                                best_id,
                                                                best_rot,
                                                                S::to_debug_bits(
                                                                    best_code
                                                                        .as_ref()
                                                                        .expect("code exists"),
                                                                ),
                                                                best_h as f32,
                                                                Some(current_corners),
                                                            );
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
                            },
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
            batch.corners[i].copy_from_slice(&refined[..4]);
        }

        if state == CandidateState::Valid && rot > 0 {
            let mut temp_corners = [Point2f::default(); 4];
            for (j, item) in temp_corners.iter_mut().enumerate() {
                let src_idx = (j + usize::from(rot)) % 4;
                *item = batch.corners[i][src_idx];
            }
            batch.corners[i].copy_from_slice(&temp_corners[..4]);
        }
    }
}

/// A decoder for a specific tag family.
pub trait TagDecoder: Send + Sync {
    /// Returns the human-readable name of the family.
    fn name(&self) -> &str;
    /// Returns the dimension of the data grid (e.g., 6 for 36h11).
    fn dimension(&self) -> usize;
    /// Returns the total number of bits in the tag.
    fn bit_count(&self) -> usize;
    /// Returns the canonical sampling points in the [-1, 1] square.
    fn sample_points(&self) -> &[(f64, f64)];
    /// Decodes a bit pattern into (id, hamming, rotation) with a small default distance.
    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)>;
    /// Decodes a bit pattern into (id, hamming, rotation) with a specified max distance.
    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)>;
    /// Returns the canonical bit pattern for a given ID.
    fn get_code(&self, id: u16) -> Option<u64>;
    /// Returns the number of codes in the dictionary.
    fn num_codes(&self) -> usize;
    /// Returns an empty slice (legacy method for rotated codes).
    fn rotated_codes(&self) -> &[(u64, u16, u8)];
    /// Iterates over all candidate codes within a hamming distance.
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
    }
    fn bit_count(&self) -> usize {
        36
    }
    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_APRILTAG36H11
    }
    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h11)
            .decode(bits, 4)
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
        &[]
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

/// Decoder for the AprilTag 16h5 family.
pub struct AprilTag16h5;

impl TagDecoder for AprilTag16h5 {
    fn name(&self) -> &'static str {
        "16h5"
    }
    fn dimension(&self) -> usize {
        4
    }
    fn bit_count(&self) -> usize {
        16
    }
    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_APRILTAG16H5
    }
    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5)
            .decode(bits, 1)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }
    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }
    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5).get_code(id)
    }
    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5).len()
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
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag16h5)
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Decoder for the AprilTag 25h9 family.
pub struct AprilTag25h9;

impl TagDecoder for AprilTag25h9 {
    fn name(&self) -> &'static str {
        "25h9"
    }
    fn dimension(&self) -> usize {
        5
    }
    fn bit_count(&self) -> usize {
        25
    }
    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_APRILTAG25H9
    }
    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag25h9)
            .decode(bits, 3)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }
    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag25h9)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }
    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag25h9).get_code(id)
    }
    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag25h9).len()
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
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag25h9)
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Decoder for the AprilTag 36h10 family.
pub struct AprilTag36h10;

impl TagDecoder for AprilTag36h10 {
    fn name(&self) -> &'static str {
        "36h10"
    }
    fn dimension(&self) -> usize {
        6
    }
    fn bit_count(&self) -> usize {
        36
    }
    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_APRILTAG36H10
    }
    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h10)
            .decode(bits, 4)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }
    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h10)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }
    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h10).get_code(id)
    }
    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h10).len()
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
        crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h10)
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Decoder for the ArUco 4x4_50 dictionary.
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

/// Decoder for the ArUco 4x4_100 dictionary.
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

/// Convert a TagFamily enum to a boxed decoder instance.
#[must_use]
pub fn family_to_decoder(family: config::TagFamily) -> Box<dyn TagDecoder + Send + Sync> {
    match family {
        config::TagFamily::AprilTag16h5 => Box::new(AprilTag16h5),
        config::TagFamily::AprilTag25h9 => Box::new(AprilTag25h9),
        config::TagFamily::AprilTag36h10 => Box::new(AprilTag36h10),
        config::TagFamily::AprilTag36h11 => Box::new(AprilTag36h11),
        config::TagFamily::ArUco4x4_50 => Box::new(ArUco4x4_50),
        config::TagFamily::ArUco4x4_100 => Box::new(ArUco4x4_100),
    }
}

#[cfg(test)]
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
            let mut test_bits = dict.codes[(id_idx * 4) + rotation];
            test_bits ^= 1 << flip1;
            test_bits ^= 1 << flip2;
            let result = decoder.decode(test_bits);
            prop_assert!(result.is_some());
            let (decoded_id, _, _) = result.expect("decoded exists");
            prop_assert_eq!(decoded_id, u32::from(orig_id));
        }

        #[test]
        fn test_false_positive_resistance(bits in 0..u64::MAX) {
            let decoder = AprilTag36h11;
            if let Some((_id, hamming, _rot)) = decoder.decode(bits) {
                prop_assert!(hamming <= 4);
            }
        }

        #[test]
        fn prop_homography_projection(
            src in prop::collection::vec((-100.0..100.0, -100.0..100.0), 4),
            dst in prop::collection::vec((0.0..1000.0, 0.0..1000.0), 4)
        ) {
            let src_pts = [
                [src[0].0, src[0].1], [src[1].0, src[1].1],
                [src[2].0, src[2].1], [src[3].0, src[3].1],
            ];
            let dst_pts = [
                [dst[0].0, dst[0].1], [dst[1].0, dst[1].1],
                [dst[2].0, dst[2].1], [dst[3].0, dst[3].1],
            ];
            if let Some(h) = Homography::from_pairs(&src_pts, &dst_pts) {
                for i in 0..4 {
                    let p = h.project(src_pts[i]);
                    prop_assert!((p[0] - dst_pts[i][0]).abs() < 1e-3);
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
                .expect("Valid ID");
            let result = decoder.decode(code);
            assert!(result.is_some());
            let (id_out, _, _) = result.expect("decoded exists");
            assert_eq!(id_out, u32::from(id));
        }
    }

    #[test]
    fn test_all_codes_decode_36h10() {
        let decoder = AprilTag36h10;
        let dict = crate::dictionaries::get_dictionary(crate::config::TagFamily::AprilTag36h10);
        for id in 0..dict.len() as u16 {
            let code = dict.get_code(id).expect("Valid ID");
            let result = decoder.decode(code);
            assert!(result.is_some());
            let (id_out, _, _) = result.expect("decoded exists");
            assert_eq!(id_out, u32::from(id));
        }
    }

    #[test]
    fn test_grid_sampling() {
        let width = 64;
        let height = 64;
        let mut data = vec![0u8; width * height];
        // 8x8 grid, 36x36px tag centered at 32,32 => corners [14, 50]
        // TL=(14,14), TR=(50,14), BR=(50,50), BL=(14,50)

        // Border:
        for gy in 0..8 {
            for gx in 0..8 {
                if gx == 0 || gx == 7 || gy == 0 || gy == 7 {
                    for y in 0..4 {
                        for x in 0..4 {
                            let px = 14 + (f64::from(gx) * 4.5) as usize + x;
                            let py = 14 + (f64::from(gy) * 4.5) as usize + y;
                            if px < 64 && py < 64 {
                                data[py * width + px] = 0;
                            }
                        }
                    }
                }
            }
        }
        // Bit 0 (cell 1,1) -> White (canonical p = -0.625, -0.625)
        for y in 0..4 {
            for x in 0..4 {
                let px = 14 + (1.0 * 4.5) as usize + x;
                let py = 14 + (1.0 * 4.5) as usize + y;
                data[py * width + px] = 255;
            }
        }
        // Bit 35 (cell 6,6) -> Black (canonical p = 0.625, 0.625)
        for y in 0..4 {
            for x in 0..4 {
                let px = 14 + (6.0 * 4.5) as usize + x;
                let py = 14 + (6.0 * 4.5) as usize + y;
                data[py * width + px] = 0;
            }
        }
        let img =
            crate::image::ImageView::new(&data, width, height, width).expect("Image creation");
        let decoder = AprilTag36h11;
        let arena = Bump::new();
        let cand = crate::Detection {
            corners: [[14.0, 14.0], [50.0, 14.0], [50.0, 50.0], [14.0, 50.0]],
            ..Default::default()
        };
        let bits = sample_grid(&img, &arena, &cand, &decoder, 20.0).expect("sample fail");

        let h = Homography::square_to_quad(&cand.corners).expect("Homography creation");
        let pts = decoder.sample_points();
        for (i, p) in pts.iter().enumerate().take(1) {
            let img_p = h.project([p.0, p.1]);
            println!("Bit {i} canonical={p:?} -> image={img_p:?}");
            let val = img.sample_bilinear(img_p[0], img_p[1]);
            println!("Value at {img_p:?} = {val}");
        }

        assert_eq!(bits & 1, 1, "Bit 0 should be white(1)");
        assert_eq!((bits >> 35) & 1, 0, "Bit 35 should be black(0)");
    }

    #[test]
    fn test_homography_dlt() {
        let src = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let dst = [[10.0, 10.0], [20.0, 11.0], [19.0, 21.0], [9.0, 20.0]];
        let h = Homography::from_pairs(&src, &dst).expect("DLT fail");
        for i in 0..4 {
            let p = h.project(src[i]);
            assert!((p[0] - dst[i][0]).abs() < 1e-6);
            assert!((p[1] - dst[i][1]).abs() < 1e-6);
        }
    }

    use crate::config::TagFamily;
    use crate::image::ImageView;
    use crate::quad::extract_quads_fast;
    use crate::segmentation::label_components_with_stats;
    use crate::test_utils::{TestImageParams, generate_test_image_with_params};
    use crate::threshold::ThresholdEngine;
    use bumpalo::Bump;

    fn run_full_pipeline(tag_size: usize, canvas_size: usize, tag_id: u16) -> Vec<(u32, u32)> {
        let params = TestImageParams {
            family: TagFamily::AprilTag36h11,
            id: tag_id,
            tag_size,
            canvas_size,
            ..Default::default()
        };
        let (data, _) = generate_test_image_with_params(&params);
        let img =
            ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("Image creation");
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
                && let Some((id, hamming, _)) = decoder.decode(bits)
            {
                results.push((id, hamming));
            }
        }
        results
    }

    #[test]
    fn test_e2e_decoding_at_varying_sizes() {
        let canvas_size = 640;
        let test_id: u16 = 42;
        // Adjusted tag_sizes to ensure they are large enough for reliable detection in this simple synthetic test
        for tag_size in [100, 150, 200, 300] {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            assert!(
                decoded.iter().any(|(id, _)| *id == u32::from(test_id)),
                "size {tag_size} fail"
            );
        }
    }

    #[test]
    fn test_e2e_multiple_ids() {
        let canvas_size = 400;
        let tag_size = 150;
        for &test_id in &[0, 42, 100, 200, 500] {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            assert!(
                decoded.iter().any(|(id, _)| *id == u32::from(test_id)),
                "id {test_id} fail"
            );
        }
    }

    #[test]
    fn test_e2e_edge_ids() {
        let canvas_size = 400;
        let tag_size = 150;
        for &test_id in &[0, 586] {
            let decoded = run_full_pipeline(tag_size, canvas_size, test_id);
            assert!(
                decoded.iter().any(|(id, _)| *id == u32::from(test_id)),
                "edge id {test_id} fail"
            );
        }
    }
}
