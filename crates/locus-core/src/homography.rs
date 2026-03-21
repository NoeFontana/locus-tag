//! Projective geometry primitives for fiducial marker detection.
//!
//! This module provides the core homography abstraction used by both the decoder
//! (for bit sampling) and the pose estimator (for camera-space normalization).
//! It treats homography as a fundamental mathematical primitive, decoupled from
//! tag-family-specific logic.

use crate::batch::{CandidateState, Matrix3x3, Point2f};
use nalgebra::{SMatrix, SVector};

/// A 3x3 Homography matrix.
pub struct Homography {
    /// The 3x3 homography matrix.
    pub h: SMatrix<f64, 3, 3>,
}

/// A Digital Differential Analyzer (DDA) for incremental homography projection.
///
/// This avoids expensive matrix multiplications by using discrete partial derivatives
/// when stepping through a uniform grid in tag space.
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
    #[inline]
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
    #[inline]
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
    status_mask: &[CandidateState],
    homographies: &mut [Matrix3x3],
) {
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
