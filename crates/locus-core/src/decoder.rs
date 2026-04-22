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
#[cfg(any(test, feature = "bench-internals"))]
use bumpalo::Bump;
use multiversion::multiversion;
use nalgebra::{SMatrix, SVector};

use crate::workspace::WORKSPACE_ARENA;

/// A 3x3 Homography matrix.
pub struct Homography {
    /// The 3x3 homography matrix.
    pub h: SMatrix<f64, 3, 3>,
}

/// A Digital Differential Analyzer (DDA) for incremental homography projection.
///
/// This avoids expensive matrix multiplications by using discrete partial derivatives
/// when stepping through a uniform grid in tag space.
// See `Homography::to_dda` for the dead-code rationale.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct HomographyDda {
    /// Current numerator for X coordinate.
    pub nx: f64,
    /// Current numerator for Y coordinate.
    pub ny: f64,
    /// Current denominator (perspective divide).
    pub d: f64,
    /// Partial derivative of nx with respect to u.
    pub dnx_du: f64,
    /// Partial derivative of ny with respect to u.
    pub dny_du: f64,
    /// Partial derivative of d with respect to u.
    pub dd_du: f64,
    /// Partial derivative of nx with respect to v.
    pub dnx_dv: f64,
    /// Partial derivative of ny with respect to v.
    pub dny_dv: f64,
    /// Partial derivative of d with respect to v.
    pub dd_dv: f64,
}

impl Homography {
    /// Convert the homography into a DDA state for a grid with step size (du, dv).
    /// Initial state is computed at (u0, v0) in canonical tag space.
    // Dead-code lint runs without target_feature gating, so the AVX2/NEON-only
    // consumer in `sample_grid_values_dda_simd` is invisible to it.
    #[allow(dead_code)]
    #[must_use]
    pub fn to_dda(&self, u0: f64, v0: f64, du: f64, dv: f64) -> HomographyDda {
        let h = self.h;
        let nx = h[(0, 0)] * u0 + h[(0, 1)] * v0 + h[(0, 2)];
        let ny = h[(1, 0)] * u0 + h[(1, 1)] * v0 + h[(1, 2)];
        let d = h[(2, 0)] * u0 + h[(2, 1)] * v0 + h[(2, 2)];

        HomographyDda {
            nx,
            ny,
            d,
            dnx_du: h[(0, 0)] * du,
            dny_du: h[(1, 0)] * du,
            dd_du: h[(2, 0)] * du,
            dnx_dv: h[(0, 1)] * dv,
            dny_dv: h[(1, 1)] * dv,
            dd_dv: h[(2, 1)] * dv,
        }
    }

    /// Compute homography from 4 source points to 4 destination points using DLT.
    /// Points are [x, y].
    #[cfg(any(test, feature = "bench-internals"))]
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

/// Compute homographies for all active quads in the batch using a pure-function SoA approach.
///
/// This uses `rayon` for data-parallel computation of the square-to-quad homographies.
/// Quads are defined by 4 corners in `corners` for each candidate index.
#[tracing::instrument(skip_all, name = "pipeline::homography_pass")]
pub fn compute_homographies_soa(
    corners: &[[Point2f; 4]],
    status_mask: &[crate::batch::CandidateState],
    homographies: &mut [Matrix3x3],
) {
    use crate::batch::CandidateState;
    use rayon::prelude::*;

    // Each homography maps from canonical square [(-1,-1), (1,-1), (1,1), (-1,1)] to image quads.
    homographies
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, h_out)| {
            if status_mask[i] != CandidateState::Active {
                h_out.data = [0.0; 9];
                h_out.padding = [0.0; 7];
                return;
            }

            let dst = [
                [f64::from(corners[i][0].x), f64::from(corners[i][0].y)],
                [f64::from(corners[i][1].x), f64::from(corners[i][1].y)],
                [f64::from(corners[i][2].x), f64::from(corners[i][2].y)],
                [f64::from(corners[i][3].x), f64::from(corners[i][3].y)],
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
    use crate::edge_refinement::{ErfEdgeFitter, RefineConfig, SampleConfig};

    let mut lines = [(0.0f64, 0.0f64, 0.0f64); 4];
    let mut line_valid = [false; 4];
    let sample_cfg = SampleConfig::for_decoder();
    let refine_cfg = RefineConfig::decoder_style(sigma);

    // Sub-pixel edge refinement for each of the 4 edges
    for i in 0..4 {
        let next = (i + 1) % 4;
        let p1 = corners[i];
        let p2 = corners[next];

        if let Some(mut fitter) = ErfEdgeFitter::new(img, p1, p2, false)
            && fitter.fit(arena, &sample_cfg, &refine_cfg)
        {
            lines[i] = fitter.line_params();
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

/// Maximum number of bits in a supported tag family payload.
const MAX_BIT_COUNT: usize = 64;

/// Sample values from the image using DDA-based coordinate generation and SIMD bilinear sampling.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn sample_grid_values_dda_simd(
    img: &crate::image::ImageView,
    roi: &RoiCache,
    h: &Homography,
    decoder: &(impl TagDecoder + ?Sized),
    intensities: &mut [f64],
) -> bool {
    let dim = decoder.dimension();
    let n = decoder.bit_count();
    let points = decoder.sample_points();
    if points.is_empty() {
        return false;
    }

    let _du = if dim > 1 {
        points[1].0 - points[0].0
    } else {
        0.0
    };
    let _dv = if dim > 1 {
        points[dim].1 - points[0].1
    } else {
        0.0
    };

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        use crate::simd::math::rcp_nr_v8;
        use crate::simd::sampler::sample_bilinear_v8;
        use std::arch::x86_64::*;

        let dda = h.to_dda(points[0].0, points[0].1, _du, _dv);

        let w_limit = _mm256_set1_ps(img.width as f32 - 1.0);
        let h_limit = _mm256_set1_ps(img.height as f32 - 1.0);

        let mut current_nx_row = dda.nx as f32;
        let mut current_ny_row = dda.ny as f32;
        let mut current_d_row = dda.d as f32;

        let dnx_du = dda.dnx_du as f32;
        let dny_du = dda.dny_du as f32;
        let dd_du = dda.dd_du as f32;

        let v_dnx_du = _mm256_set1_ps(dnx_du);
        let v_dny_du = _mm256_set1_ps(dny_du);
        let v_dd_du = _mm256_set1_ps(dd_du);
        let v_steps = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
        let v_half = _mm256_set1_ps(0.5);

        let mut idx = 0;
        for _y in 0..dim {
            let mut nx_start = current_nx_row;
            let mut ny_start = current_ny_row;
            let mut d_start = current_d_row;

            for _x in (0..dim).step_by(8) {
                let count = (dim - _x).min(8);

                // Vectorized coordinate generation (DDA)
                // x[i] = (nx + i*dnx_du)
                let v_nx_simd = _mm256_fmadd_ps(v_steps, v_dnx_du, _mm256_set1_ps(nx_start));
                let v_ny_simd = _mm256_fmadd_ps(v_steps, v_dny_du, _mm256_set1_ps(ny_start));
                let v_d_simd = _mm256_fmadd_ps(v_steps, v_dd_du, _mm256_set1_ps(d_start));

                // Perspective divide: (nx/d, ny/d)
                let v_winv = rcp_nr_v8(v_d_simd);
                let v_img_x_raw = _mm256_mul_ps(v_nx_simd, v_winv);
                let v_img_y_raw = _mm256_mul_ps(v_ny_simd, v_winv);

                // Offset by -0.5 to match bilinear logic center alignment
                let v_img_x = _mm256_sub_ps(v_img_x_raw, v_half);
                let v_img_y = _mm256_sub_ps(v_img_y_raw, v_half);

                // Bounds check: must be in [0, width - 1) for safe 2x2 bilinear fetch
                let v_zero = _mm256_setzero_ps();
                let mask_x = _mm256_and_ps(
                    _mm256_cmp_ps(v_img_x, v_zero, _CMP_GE_OQ),
                    _mm256_cmp_ps(v_img_x, w_limit, _CMP_LT_OQ),
                );
                let mask_y = _mm256_and_ps(
                    _mm256_cmp_ps(v_img_y, v_zero, _CMP_GE_OQ),
                    _mm256_cmp_ps(v_img_y, h_limit, _CMP_LT_OQ),
                );
                let mask = _mm256_movemask_ps(_mm256_and_ps(mask_x, mask_y));

                if (mask & ((1 << count) - 1)) != ((1 << count) - 1) {
                    return false;
                }

                // Restore original (non-subtracted) coords for sample_bilinear_v8 which handles the offset
                let mut v_img_x_arr = [0.0f32; 8];
                let mut v_img_y_arr = [0.0f32; 8];
                _mm256_storeu_ps(v_img_x_arr.as_mut_ptr(), v_img_x_raw);
                _mm256_storeu_ps(v_img_y_arr.as_mut_ptr(), v_img_y_raw);

                let mut sampled = [0.0f32; 8];
                sample_bilinear_v8(img, &v_img_x_arr, &v_img_y_arr, &mut sampled);

                for i in 0..count {
                    intensities[idx] = f64::from(sampled[i]);
                    idx += 1;
                }

                // Advance scalar row starts for next SIMD chunk
                nx_start += 8.0 * dnx_du;
                ny_start += 8.0 * dny_du;
                d_start += 8.0 * dd_du;
            }

            current_nx_row += dda.dnx_dv as f32;
            current_ny_row += dda.dny_dv as f32;
            current_d_row += dda.dd_dv as f32;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(unsafe_code)]
    // SAFETY: NEON intrinsics are safe on aarch64 with neon feature.
    unsafe {
        use crate::simd::sampler::sample_bilinear_v8;
        use std::arch::aarch64::*;

        let dda = h.to_dda(points[0].0, points[0].1, _du, _dv);

        let mut current_nx_row = dda.nx as f32;
        let mut current_ny_row = dda.ny as f32;
        let mut current_d_row = dda.d as f32;

        let dnx_du = dda.dnx_du as f32;
        let dny_du = dda.dny_du as f32;
        let dd_du = dda.dd_du as f32;

        let v_dnx_du = vdupq_n_f32(dnx_du);
        let v_dny_du = vdupq_n_f32(dny_du);
        let v_dd_du = vdupq_n_f32(dd_du);
        let v_steps_low = vld1q_f32([0.0, 1.0, 2.0, 3.0].as_ptr());
        let v_steps_high = vld1q_f32([4.0, 5.0, 6.0, 7.0].as_ptr());

        let mut idx = 0;
        for _y in 0..dim {
            let mut nx_start = current_nx_row;
            let mut ny_start = current_ny_row;
            let mut d_start = current_d_row;

            for _x in (0..dim).step_by(8) {
                let count = (dim - _x).min(8);

                // NEON perspective divide using vrecpeq_f32 + vrecpsq_f32
                let mut v_img_x = [0.0f32; 8];
                let mut v_img_y = [0.0f32; 8];

                for (chunk, v_steps) in [v_steps_low, v_steps_high].into_iter().enumerate() {
                    let v_nx_c = vfmaq_f32(vdupq_n_f32(nx_start), v_steps, v_dnx_du);
                    let v_ny_c = vfmaq_f32(vdupq_n_f32(ny_start), v_steps, v_dny_du);
                    let v_d_c = vfmaq_f32(vdupq_n_f32(d_start), v_steps, v_dd_du);

                    let v_winv = vrecpeq_f32(v_d_c);
                    let v_winv = vmulq_f32(v_winv, vrecpsq_f32(v_d_c, v_winv));

                    let img_x = vmulq_f32(v_nx_c, v_winv);
                    let img_y = vmulq_f32(v_ny_c, v_winv);

                    let offset = chunk * 4;
                    vst1q_f32(v_img_x.as_mut_ptr().add(offset), img_x);
                    vst1q_f32(v_img_y.as_mut_ptr().add(offset), img_y);
                }

                let mut sampled = [0.0f32; 8];
                sample_bilinear_v8(img, &v_img_x, &v_img_y, &mut sampled);

                for i in 0..count {
                    intensities[idx] = f64::from(sampled[i]);
                    idx += 1;
                }

                nx_start += 8.0 * dnx_du;
                ny_start += 8.0 * dny_du;
                d_start += 8.0 * dd_du;
            }

            current_nx_row += dda.dnx_dv as f32;
            current_ny_row += dda.dny_dv as f32;
            current_d_row += dda.dd_dv as f32;
        }
    }

    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    return sample_grid_values_optimized(img, h, roi, points, intensities, n);

    #[cfg(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    true
}

/// Sample values from the image using SIMD-optimized Fast-Math and ROI caching.
///
/// # Panics
/// Panics if the number of sample points exceeds `MAX_BIT_COUNT`.
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

        let img_x = (h00 * px + h01 * py + h02) * winv - 0.5;
        let img_y = (h10 * px + h11 * py + h12) * winv - 0.5;

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
///
/// # Panics
/// Panics if the number of sample points exceeds `MAX_BIT_COUNT`.
#[cfg(any(test, feature = "bench-internals"))]
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
    let mut intensities = [0.0f64; MAX_BIT_COUNT];
    let n = points.len().min(MAX_BIT_COUNT);
    assert!(
        points.len() <= MAX_BIT_COUNT,
        "Tag bit count ({}) exceeds static buffer size ({})",
        points.len(),
        MAX_BIT_COUNT
    );

    if !sample_grid_values_dda_simd(img, &roi, &homography, decoder, &mut intensities) {
        return None;
    }

    Some(S::from_intensities(
        &intensities[..n],
        &compute_adaptive_thresholds(&intensities[..n], points),
    ))
}

/// Sample the bit grid using Structure of Arrays (SoA) data and a precomputed ROI cache.
///
/// # Panics
/// Panics if the number of sample points exceeds `MAX_BIT_COUNT`.
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
    let mut intensities = [0.0f64; MAX_BIT_COUNT];
    let n = points.len().min(MAX_BIT_COUNT);
    assert!(
        points.len() <= MAX_BIT_COUNT,
        "Tag bit count ({}) exceeds static buffer size ({})",
        points.len(),
        MAX_BIT_COUNT
    );

    if !sample_grid_values_dda_simd(img, roi, &homography_obj, decoder, &mut intensities) {
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
#[cfg(any(test, feature = "bench-internals"))]
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

/// Rotate a square bit grid 90 degrees clockwise.
/// This is an O(1) bitwise operation but conceptually represents rotating the N x N pixel grid.
#[cfg(any(test, feature = "bench-internals"))]
#[must_use]
pub fn rotate90(bits: u64, dim: usize) -> u64 {
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

/// Sample the bit grid using scalar bilinear interpolation with distortion remapping.
///
/// Projects each canonical tag sample point through the ideal homography `h_ideal`
/// (computed from undistorted corners), then applies the camera distortion map to
/// convert the ideal pixel coordinate to the actual coordinate in the distorted image,
/// finally sampling the distorted image via bilinear interpolation.
///
/// This path is only called for non-rectified cameras (`!C::IS_RECTIFIED`). For rectified
/// cameras the faster SIMD path in [`sample_grid_soa_precomputed`] is used instead.
#[cfg(feature = "non_rectified")]
#[allow(clippy::similar_names)]
fn sample_grid_values_distorted<C: crate::camera::CameraModel>(
    img: &crate::image::ImageView,
    h_ideal: &Homography,
    decoder: &(impl TagDecoder + ?Sized),
    intrinsics: &crate::pose::CameraIntrinsics,
    model: &C,
    intensities: &mut [f64; MAX_BIT_COUNT],
) -> bool {
    let points = decoder.sample_points();
    if points.is_empty() {
        return false;
    }
    let hm = &h_ideal.h;
    let w_limit = (img.width as f64) - 1.0 - 1e-4;
    let h_limit = (img.height as f64) - 1.0 - 1e-4;

    for (i, (u, v)) in points.iter().enumerate() {
        let u = *u;
        let v = *v;

        let nx = hm[(0, 0)] * u + hm[(0, 1)] * v + hm[(0, 2)];
        let ny = hm[(1, 0)] * u + hm[(1, 1)] * v + hm[(1, 2)];
        let d = hm[(2, 0)] * u + hm[(2, 1)] * v + hm[(2, 2)];

        if d.abs() < 1e-8 {
            return false;
        }

        let px_ideal = nx / d;
        let py_ideal = ny / d;

        // Convert ideal pixel → normalized → apply distortion → distorted pixel.
        let xn = (px_ideal - intrinsics.cx) / intrinsics.fx;
        let yn = (py_ideal - intrinsics.cy) / intrinsics.fy;
        let [xd, yd] = model.distort(xn, yn);
        let px = xd * intrinsics.fx + intrinsics.cx;
        let py = yd * intrinsics.fy + intrinsics.cy;

        if px < 0.0 || px > w_limit || py < 0.0 || py > h_limit {
            return false;
        }

        let ix = px.floor() as usize;
        let iy = py.floor() as usize;
        let stride = img.stride;
        // SAFETY: bounds checked above; ix <= w_limit - 1 < width - 1, iy <= h_limit - 1 < height - 1.
        let v00 = unsafe { *img.data.get_unchecked(iy * stride + ix) };
        // SAFETY: bounds checked above; ix+1 <= w_limit < width, iy within height.
        let v10 = unsafe { *img.data.get_unchecked(iy * stride + ix + 1) };
        // SAFETY: bounds checked above; iy+1 <= h_limit < height, ix within width.
        let v01 = unsafe { *img.data.get_unchecked((iy + 1) * stride + ix) };
        // SAFETY: bounds checked above; ix+1 <= w_limit < width, iy+1 <= h_limit < height.
        let v11 = unsafe { *img.data.get_unchecked((iy + 1) * stride + ix + 1) };
        intensities[i] = f64::from(bilinear_interpolate_fixed(
            px as f32, py as f32, v00, v10, v01, v11,
        ));
    }
    true
}

/// Distortion-aware decode for a single candidate using scalar sampling.
///
/// Undistorts the detected corners to compute an ideal homography, then samples the
/// distorted image at the correctly distortion-mapped coordinates for each bit sample
/// point. Called by [`decode_batch_soa_with_camera`] for non-rectified cameras.
#[cfg(feature = "non_rectified")]
fn decode_candidate_distorted<
    S: crate::strategy::DecodingStrategy,
    C: crate::camera::CameraModel,
>(
    img: &crate::image::ImageView,
    corners: &[Point2f; 4],
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
    intrinsics: &crate::pose::CameraIntrinsics,
    model: &C,
) -> (crate::batch::CandidateState, u32, u8, u64, f32) {
    use crate::batch::CandidateState;

    let ideal: [[f64; 2]; 4] = core::array::from_fn(|j| {
        intrinsics.undistort_pixel(f64::from(corners[j].x), f64::from(corners[j].y))
    });

    let center = [
        (ideal[0][0] + ideal[1][0] + ideal[2][0] + ideal[3][0]) * 0.25,
        (ideal[0][1] + ideal[1][1] + ideal[2][1] + ideal[3][1]) * 0.25,
    ];

    let mut best_h = u32::MAX;
    let mut best_id = 0u32;
    let mut best_rot = 0u8;
    let mut best_bits = 0u64;

    for &scale in &[1.0f64, 0.9, 1.1] {
        let scaled: [[f64; 2]; 4] = core::array::from_fn(|j| {
            [
                center[0] + (ideal[j][0] - center[0]) * scale,
                center[1] + (ideal[j][1] - center[1]) * scale,
            ]
        });

        let Some(h_ideal) = Homography::square_to_quad(&scaled) else {
            continue;
        };

        let mut intensities = [0.0f64; MAX_BIT_COUNT];

        for decoder in decoders {
            let n = decoder.bit_count();
            if !sample_grid_values_distorted::<C>(
                img,
                &h_ideal,
                decoder.as_ref(),
                intrinsics,
                model,
                &mut intensities,
            ) {
                continue;
            }

            let pts = decoder.sample_points();
            let thresholds = compute_adaptive_thresholds(&intensities[..n], pts);
            let code = S::from_intensities(&intensities[..n], &thresholds);

            if let Some((id, hamming, rot)) = S::decode(&code, decoder.as_ref(), 255) {
                if hamming < best_h {
                    best_h = hamming;
                    best_bits = S::to_debug_bits(&code);
                    if hamming <= config.max_hamming_error {
                        best_id = id;
                        best_rot = rot;
                    }
                }
                if best_h == 0 {
                    break;
                }
            }
        }
        if best_h == 0 {
            break;
        }
    }

    if best_h <= config.max_hamming_error {
        (
            CandidateState::Valid,
            best_id,
            best_rot,
            best_bits,
            best_h as f32,
        )
    } else {
        (
            CandidateState::FailedDecode,
            0,
            0,
            if best_h == u32::MAX { 0 } else { best_bits },
            if best_h == u32::MAX {
                0.0
            } else {
                best_h as f32
            },
        )
    }
}

/// Distortion-aware batch decode for non-rectified cameras.
#[cfg(feature = "non_rectified")]
fn decode_batch_soa_with_camera_inner<
    S: crate::strategy::DecodingStrategy,
    C: crate::camera::CameraModel,
>(
    batch: &mut crate::batch::DetectionBatch,
    n: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
    intrinsics: &crate::pose::CameraIntrinsics,
    model: &C,
) {
    use crate::batch::CandidateState;
    use rayon::prelude::*;

    let results: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            if batch.status_mask[i] != CandidateState::Active {
                return (batch.status_mask[i], 0u32, 0u8, 0u64, batch.error_rates[i]);
            }
            let (state, id, rot, bits, err) = decode_candidate_distorted::<S, C>(
                img,
                &batch.corners[i],
                decoders,
                config,
                intrinsics,
                model,
            );
            (state, id, rot, bits, err)
        })
        .collect();

    for (i, (state, id, rot, payload, error_rate)) in results.into_iter().enumerate() {
        batch.status_mask[i] = state;
        batch.ids[i] = id;
        batch.payloads[i] = payload;
        batch.error_rates[i] = error_rate;

        // Reorder corners based on the decoded rotation (same convention as the SIMD path).
        if state == CandidateState::Valid && rot > 0 {
            let mut tmp = [Point2f::default(); 4];
            for (j, item) in tmp.iter_mut().enumerate() {
                let src = (j + usize::from(rot)) % 4;
                *item = batch.corners[i][src];
            }
            batch.corners[i] = tmp;
        }
    }
}

/// Decode all active candidates in the batch using the Structure of Arrays (SoA) layout.
///
/// This phase executes SIMD bilinear interpolation and Hamming error correction.
/// If a candidate fails decoding, its `status_mask` is flipped to `FailedDecode`.
#[tracing::instrument(skip_all, name = "pipeline::decoding_pass")]
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

/// Distortion-aware entry point for [`decode_batch_soa`].
///
/// When `C::IS_RECTIFIED = true` (i.e., [`PinholeModel`](crate::camera::PinholeModel)),
/// this delegates to the existing SIMD pipeline with zero overhead — the compiler
/// eliminates the distortion branch entirely via monomorphization.
///
/// When `C::IS_RECTIFIED = false`, each candidate's corners are undistorted to compute
/// an ideal homography. The bit grid is then sampled by projecting through the ideal
/// homography and applying the distortion map to get coordinates in the raw distorted
/// image, ensuring accurate bit sampling even under large fisheye distortion.
///
/// [`PinholeModel`]: crate::camera::PinholeModel
#[cfg(feature = "non_rectified")]
#[tracing::instrument(skip_all, name = "pipeline::decoding_pass_distortion")]
pub fn decode_batch_soa_with_camera<C: crate::camera::CameraModel>(
    batch: &mut crate::batch::DetectionBatch,
    n: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
    intrinsics: Option<&crate::pose::CameraIntrinsics>,
    model: &C,
) {
    if C::IS_RECTIFIED {
        // Zero-overhead path for rectified images: delegate to the existing SIMD pipeline.
        // The `if C::IS_RECTIFIED` is a compile-time constant; for PinholeModel the
        // compiler eliminates the else branch entirely via dead-code elimination.
        decode_batch_soa(batch, n, img, decoders, config);
    } else if let Some(intrinsics) = intrinsics {
        match config.decode_mode {
            crate::config::DecodeMode::Hard => {
                decode_batch_soa_with_camera_inner::<crate::strategy::HardStrategy, C>(
                    batch, n, img, decoders, config, intrinsics, model,
                );
            },
            crate::config::DecodeMode::Soft => {
                decode_batch_soa_with_camera_inner::<crate::strategy::SoftStrategy, C>(
                    batch, n, img, decoders, config, intrinsics, model,
                );
            },
        }
    } else {
        // No intrinsics provided — fall back to the standard path.
        decode_batch_soa(batch, n, img, decoders, config);
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
            if batch.status_mask[i] != CandidateState::Active {
                return (batch.status_mask[i], 0, 0, 0, batch.error_rates[i], None);
            }

            WORKSPACE_ARENA.with_borrow_mut(|arena| {
                    arena.reset();

                    let corners = &batch.corners[i];
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
                            | crate::config::CornerRefinementMode::Gwlf => {
                                // Gwlf not ported to SoA yet.
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
                        // Publish the first-pass best-seen bit pattern so the
                        // ROI rescue pass can check first-pass agreement even
                        // when the primary decode failed.
                        let fallback_payload =
                            best_overall_code.as_ref().map_or(0, S::to_debug_bits);
                        (
                            CandidateState::FailedDecode,
                            0,
                            0,
                            fallback_payload,
                            if best_h == u32::MAX { 0.0 } else { best_h as f32 },
                            None,
                        )
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
                batch.corners[i][j] = *corner;
            }
        }

        if state == CandidateState::Valid && rot > 0 {
            // Reorder corners based on rotation
            let mut temp_corners = [Point2f::default(); 4];
            for (j, item) in temp_corners.iter_mut().enumerate() {
                let src_idx = (j + usize::from(rot)) % 4;
                *item = batch.corners[i][src_idx];
            }
            for (j, item) in temp_corners.iter().enumerate() {
                batch.corners[i][j] = *item;
            }
        }
    }
}

/// Attempt a super-resolution rescue decode on a single candidate.
///
/// Runs on candidates that failed the primary decode pass (typically
/// `status_mask[idx] == FailedDecode`). Upscales the source-image ROI
/// around the candidate's quad, re-samples bit intensities against the
/// upscaled image, and decodes with a tighter `rescue_max_hamming` budget.
///
/// Returns `true` when the candidate was promoted to `Valid` — the batch
/// columns (`status_mask`, `ids`, `payloads`, `error_rates`, and rotated
/// `corners`) are written in place. Returns `false` when the rescue was
/// skipped (ROI too large, degenerate bbox, homography failure) or when
/// no decoder hit; the batch is left untouched.
///
/// `out`/`scratch` are caller-owned sub-slices of the frame's
/// `rescue_buf`. `out` needs `w*factor*h*factor + 3` bytes (the `+3` keeps
/// the AVX2 gather path active); `scratch` needs `w*factor*h` bytes for
/// `Lanczos3` and may be empty for `Bilinear`. Sizing is bounded by
/// `RoiRescuePolicy::max_roi_side_px` so callers can pre-allocate a
/// fixed-size budget at frame start.
#[tracing::instrument(skip_all, name = "pipeline::rescue_decode")]
#[allow(dead_code)] // consumed by B.6 rescue stage
pub fn decode_with_rescue(
    batch: &mut crate::batch::DetectionBatch,
    idx: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
    out: &mut [u8],
    scratch: &mut [u8],
) -> bool {
    match config.decode_mode {
        crate::config::DecodeMode::Hard => {
            decode_with_rescue_generic::<crate::strategy::HardStrategy>(
                batch, idx, img, decoders, config, out, scratch,
            )
        },
        crate::config::DecodeMode::Soft => {
            decode_with_rescue_generic::<crate::strategy::SoftStrategy>(
                batch, idx, img, decoders, config, out, scratch,
            )
        },
    }
}

#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
fn decode_with_rescue_generic<S: crate::strategy::DecodingStrategy>(
    batch: &mut crate::batch::DetectionBatch,
    idx: usize,
    img: &crate::image::ImageView,
    decoders: &[Box<dyn TagDecoder + Send + Sync>],
    config: &crate::config::DetectorConfig,
    out: &mut [u8],
    scratch: &mut [u8],
) -> bool {
    use crate::batch::CandidateState;
    use crate::image::upscale_roi_to_buf;

    let policy = &config.roi_rescue;
    let factor = policy.upscale_factor;
    let max_side = usize::from(policy.max_roi_side_px);

    let corners = batch.corners[idx];

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for p in &corners {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    if !min_x.is_finite() || !max_x.is_finite() {
        return false;
    }

    // 2-px Lanczos context padding, clamped to the source image.
    let pad = 2.0_f32;
    let col_lo = (min_x - pad).floor().max(0.0);
    let row_lo = (min_y - pad).floor().max(0.0);
    let last_col = (img.width as f32 - 1.0).max(0.0);
    let last_row = (img.height as f32 - 1.0).max(0.0);
    let col_hi = (max_x + pad).ceil().min(last_col);
    let row_hi = (max_y + pad).ceil().min(last_row);
    if col_hi <= col_lo || row_hi <= row_lo {
        return false;
    }

    let bx = col_lo as usize;
    let by = row_lo as usize;
    let bw = (col_hi as usize).saturating_sub(bx) + 1;
    let bh = (row_hi as usize).saturating_sub(by) + 1;

    // Skip super-resolution on large quads — the budget assumption behind
    // `max_rescues_per_frame * (side*factor)^2` breaks down, and a large
    // quad decoder failure is rarely an under-sampling problem.
    if bw > max_side || bh > max_side {
        return false;
    }

    let f = usize::from(factor);
    let ow = bw * f;
    let oh = bh * f;
    let out_need = ow * oh;
    let scratch_need = match policy.interpolation {
        crate::config::RescueInterpolation::Bilinear => 0,
        crate::config::RescueInterpolation::Lanczos3 => ow * bh,
    };
    if out.len() < out_need || scratch.len() < scratch_need {
        return false;
    }

    let Ok(upscaled) = upscale_roi_to_buf(
        img,
        (bx, by, bw, bh),
        out,
        factor,
        &mut scratch[..scratch_need],
        policy.interpolation,
    ) else {
        return false;
    };

    let roi_cache = RoiCache::from_upscaled(&upscaled.data[..out_need], ow, oh);

    // Homography in upscaled-local coordinates: translate corners by the
    // ROI origin and scale by the upscale factor.
    let f_f64 = f as f64;
    let origin = [bx as f64, by as f64];
    let dst = [
        [
            (f64::from(corners[0].x) - origin[0]) * f_f64,
            (f64::from(corners[0].y) - origin[1]) * f_f64,
        ],
        [
            (f64::from(corners[1].x) - origin[0]) * f_f64,
            (f64::from(corners[1].y) - origin[1]) * f_f64,
        ],
        [
            (f64::from(corners[2].x) - origin[0]) * f_f64,
            (f64::from(corners[2].y) - origin[1]) * f_f64,
        ],
        [
            (f64::from(corners[3].x) - origin[0]) * f_f64,
            (f64::from(corners[3].y) - origin[1]) * f_f64,
        ],
    ];
    let Some(h_new) = Homography::square_to_quad(&dst) else {
        return false;
    };
    let mut h_mat = Matrix3x3 {
        data: [0.0; 9],
        padding: [0.0; 7],
    };
    for (j, val) in h_new.h.iter().enumerate() {
        h_mat.data[j] = *val as f32;
    }

    let rescue_max = u32::from(policy.rescue_max_hamming);
    let first_pass_payload = batch.payloads[idx];

    for decoder in decoders {
        let Some(code) =
            sample_grid_soa_precomputed::<S>(&upscaled, &roi_cache, &h_mat, decoder.as_ref())
        else {
            continue;
        };
        let Some((id, hamming, rot)) = S::decode(&code, decoder.as_ref(), rescue_max) else {
            continue;
        };
        if hamming > rescue_max {
            continue;
        }

        if policy.require_first_pass_agreement {
            // Agreement = first-pass bit pattern decodes to the same ID
            // under the primary Hamming budget. `decode_full` with
            // `first_pass_payload == 0` on genuinely unseen candidates
            // will still reject (0 does not match a real tag except by
            // accident within `max_hamming_error`).
            let Some((fp_id, _, _)) =
                decoder.decode_full(first_pass_payload, config.max_hamming_error)
            else {
                continue;
            };
            if fp_id != id {
                continue;
            }
        }

        batch.status_mask[idx] = CandidateState::Valid;
        batch.ids[idx] = id;
        batch.payloads[idx] = S::to_debug_bits(&code);
        batch.error_rates[idx] = hamming as f32;

        if rot > 0 {
            let mut tmp = [Point2f::default(); 4];
            for (j, slot) in tmp.iter_mut().enumerate() {
                *slot = batch.corners[idx][(j + usize::from(rot)) % 4];
            }
            batch.corners[idx] = tmp;
        }

        return true;
    }

    false
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

/// Decoder for the ArUco 6x6_250 family.
pub struct ArUco6x6_250;

impl TagDecoder for ArUco6x6_250 {
    fn name(&self) -> &'static str {
        "6X6_250"
    }
    fn dimension(&self) -> usize {
        6
    }
    fn bit_count(&self) -> usize {
        36
    }

    fn sample_points(&self) -> &[(f64, f64)] {
        crate::dictionaries::POINTS_ARUCO6X6_250
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250)
            .decode(bits, 4)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn decode_full(&self, bits: u64, max_hamming: u32) -> Option<(u32, u32, u8)> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250)
            .decode(bits, max_hamming)
            .map(|(id, hamming, rot)| (u32::from(id), hamming, rot))
    }

    fn get_code(&self, id: u16) -> Option<u64> {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250).get_code(id)
    }

    fn num_codes(&self) -> usize {
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250).len()
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
        crate::dictionaries::get_dictionary(crate::config::TagFamily::ArUco6x6_250)
            .for_each_candidate_within_hamming(bits, max_hamming, callback);
    }
}

/// Convert a TagFamily enum to a boxed decoder instance.
#[must_use]
pub fn family_to_decoder(family: config::TagFamily) -> Box<dyn TagDecoder + Send + Sync> {
    match family {
        config::TagFamily::AprilTag16h5 => Box::new(AprilTag16h5),
        config::TagFamily::AprilTag36h11 => Box::new(AprilTag36h11),
        config::TagFamily::ArUco4x4_50 => Box::new(ArUco4x4_50),
        config::TagFamily::ArUco4x4_100 => Box::new(ArUco4x4_100),
        config::TagFamily::ArUco6x6_250 => Box::new(ArUco6x6_250),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
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
            let (id_out, _, _) = result.unwrap();
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

        let img = crate::image::ImageView::new(&data, width, height, width).unwrap();

        let decoder = AprilTag36h11;
        let arena = Bump::new();
        let cand = crate::Detection {
            corners: [[14.0, 14.0], [50.0, 14.0], [50.0, 50.0], [14.0, 50.0]],
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

    /// Test ArUco 6x6_250 dictionary integration.
    #[test]
    fn test_aruco_6x6_250_roundtrip() {
        let decoder = ArUco6x6_250;
        let test_ids = [0, 42, 100, 249];

        for &id in &test_ids {
            let code = decoder.get_code(id).expect("code should exist");
            let (decoded_id, hamming, rot) = decoder.decode(code).expect("should decode");
            assert_eq!(decoded_id, u32::from(id));
            assert_eq!(hamming, 0);
            assert_eq!(rot, 0);
        }
    }

    // ========================================================================
    // decode_with_rescue robustness
    // ========================================================================

    fn rescue_test_config() -> crate::config::DetectorConfig {
        let mut cfg = crate::config::DetectorConfig::default();
        cfg.roi_rescue.enabled = true;
        cfg.roi_rescue.upscale_factor = 2;
        cfg.roi_rescue.max_roi_side_px = 32;
        cfg.roi_rescue.rescue_max_hamming = 1;
        cfg.roi_rescue.require_first_pass_agreement = false;
        cfg.roi_rescue.interpolation = crate::config::RescueInterpolation::Bilinear;
        cfg
    }

    fn rescue_test_decoders() -> Vec<Box<dyn TagDecoder + Send + Sync>> {
        vec![Box::new(AprilTag36h11)]
    }

    #[test]
    fn test_decode_with_rescue_skips_oversized_bbox() {
        let width = 80;
        let height = 80;
        let data = vec![128u8; width * height];
        let img = ImageView::new(&data, width, height, width).expect("view");

        let mut batch = Box::new(crate::batch::DetectionBatch::new());
        // 64 px side > max_roi_side_px (32), must skip.
        batch.corners[0] = [
            Point2f { x: 8.0, y: 8.0 },
            Point2f { x: 72.0, y: 8.0 },
            Point2f { x: 72.0, y: 72.0 },
            Point2f { x: 8.0, y: 72.0 },
        ];
        batch.status_mask[0] = crate::batch::CandidateState::FailedDecode;

        let mut out = vec![0u8; 32 * 32 * 4];
        let mut scratch = vec![0u8; 32 * 32 * 2];

        let cfg = rescue_test_config();
        let decoders = rescue_test_decoders();
        let promoted =
            decode_with_rescue(&mut batch, 0, &img, &decoders, &cfg, &mut out, &mut scratch);
        assert!(!promoted, "oversized bbox must not rescue");
        assert_eq!(
            batch.status_mask[0],
            crate::batch::CandidateState::FailedDecode
        );
    }

    #[test]
    fn test_decode_with_rescue_rejects_tiny_buffer() {
        let width = 64;
        let height = 64;
        let data = vec![128u8; width * height];
        let img = ImageView::new(&data, width, height, width).expect("view");

        let mut batch = Box::new(crate::batch::DetectionBatch::new());
        batch.corners[0] = [
            Point2f { x: 10.0, y: 10.0 },
            Point2f { x: 30.0, y: 10.0 },
            Point2f { x: 30.0, y: 30.0 },
            Point2f { x: 10.0, y: 30.0 },
        ];
        batch.status_mask[0] = crate::batch::CandidateState::FailedDecode;

        // Undersized out buffer; must bail without touching the batch.
        let mut out = vec![0u8; 4];
        let mut scratch = vec![0u8; 4];

        let cfg = rescue_test_config();
        let decoders = rescue_test_decoders();
        let promoted =
            decode_with_rescue(&mut batch, 0, &img, &decoders, &cfg, &mut out, &mut scratch);
        assert!(!promoted);
        assert_eq!(
            batch.status_mask[0],
            crate::batch::CandidateState::FailedDecode
        );
    }

    #[test]
    fn test_decode_with_rescue_rejects_degenerate_bbox() {
        let width = 64;
        let height = 64;
        let data = vec![128u8; width * height];
        let img = ImageView::new(&data, width, height, width).expect("view");

        let mut batch = Box::new(crate::batch::DetectionBatch::new());
        // All four corners coincide: AABB collapses, must bail before upscale.
        batch.corners[0] = [
            Point2f { x: 20.0, y: 20.0 },
            Point2f { x: 20.0, y: 20.0 },
            Point2f { x: 20.0, y: 20.0 },
            Point2f { x: 20.0, y: 20.0 },
        ];
        batch.status_mask[0] = crate::batch::CandidateState::FailedDecode;

        let mut out = vec![0u8; 32 * 32 * 4];
        let mut scratch = vec![0u8; 32 * 32 * 2];

        let cfg = rescue_test_config();
        let decoders = rescue_test_decoders();
        let promoted =
            decode_with_rescue(&mut batch, 0, &img, &decoders, &cfg, &mut out, &mut scratch);
        // Degenerate AABB may produce a tiny upscaled ROI; the key invariant
        // is that nothing panics and the candidate is not promoted.
        assert!(!promoted);
        assert_eq!(
            batch.status_mask[0],
            crate::batch::CandidateState::FailedDecode
        );
    }
}
