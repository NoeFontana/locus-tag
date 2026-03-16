//! Gradient-Weighted Line Fitting (GWLF) for sub-pixel corner refinement.

use crate::image::ImageView;
use nalgebra::{Matrix2, Matrix3, SMatrix, Vector2, Vector3};

/// Accumulator for gradient-weighted spatial moments of an edge.
#[derive(Clone, Copy, Debug, Default)]
pub struct MomentAccumulator {
    /// Sum of weights: sum(w_i)
    pub sum_w: f64,
    /// Sum of weighted x: sum(w_i * x_i)
    pub sum_wx: f64,
    /// Sum of weighted y: sum(w_i * y_i)
    pub sum_wy: f64,
    /// Sum of weighted x squared: sum(w_i * x_i^2)
    pub sum_wxx: f64,
    /// Sum of weighted y squared: sum(w_i * y_i^2)
    pub sum_wyy: f64,
    /// Sum of weighted x*y: sum(w_i * x_i * y_i)
    pub sum_wxy: f64,
}

impl MomentAccumulator {
    /// Create a new empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a weighted point to the accumulator.
    pub fn add(&mut self, x: f64, y: f64, w: f64) {
        self.sum_w += w;
        self.sum_wx += w * x;
        self.sum_wy += w * y;
        self.sum_wxx += w * x * x;
        self.sum_wyy += w * y * y;
        self.sum_wxy += w * x * y;
    }

    /// Compute the gradient-weighted centroid (cx, cy).
    #[must_use]
    pub fn centroid(&self) -> Option<Vector2<f64>> {
        if self.sum_w < 1e-9 {
            None
        } else {
            Some(Vector2::new(
                self.sum_wx / self.sum_w,
                self.sum_wy / self.sum_w,
            ))
        }
    }

    /// Compute the 2x2 gradient-weighted spatial covariance matrix.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn covariance(&self) -> Option<Matrix2<f64>> {
        let c = self.centroid()?;
        let s_w = self.sum_w;

        let s_xx = (self.sum_wxx / s_w) - (c.x * c.x);
        let s_yy = (self.sum_wyy / s_w) - (c.y * c.y);
        let s_xy = (self.sum_wxy / s_w) - (c.x * c.y);

        Some(Matrix2::new(s_xx, s_xy, s_xy, s_yy))
    }
}

/// A line in homogeneous coordinates l such that l^T * [x, y, 1]^T = 0.
#[derive(Clone, Copy, Debug)]
pub struct HomogeneousLine {
    /// Line parameters [nx, ny, d].
    pub l: Vector3<f64>,
    /// 3x3 covariance matrix of the line parameters.
    pub cov: Matrix3<f64>,
}

impl HomogeneousLine {
    /// Compute the intersection of two homogeneous lines.
    /// Returns (Cartesian corner, Cartesian 2x2 covariance).
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<(Vector2<f64>, Matrix2<f64>)> {
        // 1. Cross product intersection: c_hom = l1 x l2
        let c_hom = self.l.cross(&other.l);
        let w = c_hom.z;
        if w.abs() < 1e-9 {
            return None; // Parallel lines
        }

        // Phase B: Covariance of the Projective Intersection
        // Σ_ch = [l2]x * Σ_l1 * [l2]x^T + [l1]x * Σ_l2 * [l1]x^T
        let l1_x = self.l.cross_matrix();
        let l2_x = other.l.cross_matrix();
        let cov_ch = l2_x * self.cov * l2_x.transpose() + l1_x * other.cov * l1_x.transpose();

        // Phase C: Projection to the Affine Plane
        // c = [u/w, v/w]
        let x = c_hom.x / w;
        let y = c_hom.y / w;
        let corner = Vector2::new(x, y);

        // Jacobian of perspective division J_pi = (1/w) * [ 1 0 -u/w; 0 1 -v/w ]
        let mut j_pi = SMatrix::<f64, 2, 3>::zeros();
        let w_inv = 1.0 / w;
        j_pi[(0, 0)] = w_inv;
        j_pi[(0, 2)] = -x * w_inv;
        j_pi[(1, 1)] = w_inv;
        j_pi[(1, 2)] = -y * w_inv;

        let cov_c = j_pi * cov_ch * j_pi.transpose();

        Some((corner, cov_c))
    }
}

/// Result of analytic eigendecomposition of a 2x2 symmetric matrix.
pub struct EigenResult {
    /// Largest eigenvalue.
    pub l_max: f64,
    /// Smallest eigenvalue.
    pub l_min: f64,
    /// Eigenvector corresponding to the largest eigenvalue.
    pub v_max: Vector2<f64>,
    /// Eigenvector corresponding to the smallest eigenvalue.
    pub v_min: Vector2<f64>,
}

/// Solves the eigendecomposition of a 2x2 symmetric matrix [[a, b], [b, c]].
#[must_use]
#[allow(clippy::manual_midpoint)]
pub fn solve_2x2_symmetric(a: f64, b: f64, c: f64) -> EigenResult {
    let trace = a + c;
    let det = a * c - b * b;

    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let l_max = (trace + disc) / 2.0;
    let l_min = (trace - disc) / 2.0;

    // Smallest eigenvalue eigenvector (normal n)
    let v_min = if b.abs() > 1e-9 {
        let vx = b;
        let vy = l_min - a;
        Vector2::new(vx, vy).normalize()
    } else if a < c {
        Vector2::new(1.0, 0.0)
    } else {
        Vector2::new(0.0, 1.0)
    };

    // Largest eigenvalue eigenvector (tangent t)
    let v_max = Vector2::new(-v_min.y, v_min.x);

    EigenResult {
        l_max,
        l_min,
        v_max,
        v_min,
    }
}

/// Refine quad corners using Gradient-Weighted Line Fitting (GWLF).
///
/// Returns the refined corners [[x, y]; 4] and their 2x2 covariances [Matrix2; 4].
#[must_use]
#[allow(clippy::similar_names)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::type_complexity)]
pub fn refine_quad_gwlf_with_cov(
    img: &ImageView,
    coarse_corners: &[[f32; 2]; 4],
    alpha: f64,
) -> Option<([[f32; 2]; 4], [Matrix2<f64>; 4])> {
    let mut lines = [HomogeneousLine {
        l: Vector3::zeros(),
        cov: Matrix3::zeros(),
    }; 4];

    for i in 0..4 {
        let p0 = coarse_corners[i];
        let p1 = coarse_corners[(i + 1) % 4];

        let dx_edge = f64::from(p1[0] - p0[0]);
        let dy_edge = f64::from(p1[1] - p0[1]);
        let len = (dx_edge * dx_edge + dy_edge * dy_edge).sqrt();
        if len < 2.0 {
            return None;
        }

        let ux = dx_edge / len;
        let uy = dy_edge / len;
        let nx_coarse = -uy;
        let ny_coarse = ux;

        let mut acc = MomentAccumulator::new();
        let steps = (len * 2.0) as usize;

        // Adaptive Transversal Windowing: search band scales with edge length L.
        // Band = +/- max(2, alpha * L).
        let window_half_width = (alpha * len).max(2.0);
        let k_min = -window_half_width.round() as i32;
        let k_max = window_half_width.round() as i32;

        for s in 0..=steps {
            let t = (s as f64) / (steps as f64);
            let px = f64::from(p0[0]) + t * dx_edge;
            let py = f64::from(p0[1]) + t * dy_edge;

            for k in k_min..=k_max {
                let sx = px + f64::from(k) * nx_coarse;
                let sy = py + f64::from(k) * ny_coarse;

                if sx < 1.0
                    || sx >= (img.width - 2) as f64
                    || sy < 1.0
                    || sy >= (img.height - 2) as f64
                {
                    continue;
                }

                let g = img.sample_gradient_bilinear(sx, sy);
                let w = g[0] * g[0] + g[1] * g[1];

                // Weight noise floor
                if w > 100.0 {
                    acc.add(sx, sy, w);
                }
            }
        }

        let cov_spatial = acc.covariance()?;
        let res = solve_2x2_symmetric(
            cov_spatial[(0, 0)],
            cov_spatial[(0, 1)],
            cov_spatial[(1, 1)],
        );
        let n = res.v_min;
        let x_bar = acc.centroid()?;
        let w_total = acc.sum_w;

        // Phase A: Covariance of the Line Parameters
        // Σ_xbar = (λ_min / W) * I
        let sigma_xbar = Matrix2::identity().scale(res.l_min / w_total);
        // Σ_n = (λ_min / (W * (λ_max - λ_min))) * n_perp * n_perp^T
        let n_perp = res.v_max;
        let sigma_n = (n_perp * n_perp.transpose())
            .scale(res.l_min / (w_total * (res.l_max - res.l_min).max(1e-6)));

        // l = [n; -x_bar^T * n]
        // J_n = [I; -x_bar^T], J_xbar = [0; -n^T]
        let mut j_n = SMatrix::<f64, 3, 2>::zeros();
        j_n.fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&Matrix2::identity());
        j_n[(2, 0)] = -x_bar.x;
        j_n[(2, 1)] = -x_bar.y;

        let mut j_xbar = SMatrix::<f64, 3, 2>::zeros();
        j_xbar[(2, 0)] = -n.x;
        j_xbar[(2, 1)] = -n.y;

        let cov_l = j_n * sigma_n * j_n.transpose() + j_xbar * sigma_xbar * j_xbar.transpose();

        lines[i] = HomogeneousLine {
            l: Vector3::new(n.x, n.y, -x_bar.dot(&n)),
            cov: cov_l,
        };
    }

    let mut refined_corners = [[0.0f32; 2]; 4];
    let mut refined_covs = [Matrix2::zeros(); 4];

    for i in 0..4 {
        let l_prev = lines[(i + 3) % 4];
        let l_curr = lines[i];

        let (corner, cov) = l_prev.intersect(&l_curr)?;

        // Sanity check
        let dist_sq = (corner.x - f64::from(coarse_corners[i][0])).powi(2)
            + (corner.y - f64::from(coarse_corners[i][1])).powi(2);
        if dist_sq > 25.0 {
            // Relaxed sanity check for blur (5.0 px)
            return None;
        }

        refined_corners[i] = [corner.x as f32, corner.y as f32];
        refined_covs[i] = cov;
    }

    Some((refined_corners, refined_covs))
}

/// Compatibility wrapper for the existing API.
#[must_use]
pub fn refine_quad_gwlf(
    img: &ImageView,
    coarse_corners: &[[f32; 2]; 4],
    alpha: f64,
) -> Option<[[f32; 2]; 4]> {
    refine_quad_gwlf_with_cov(img, coarse_corners, alpha).map(|(c, _)| c)
}
