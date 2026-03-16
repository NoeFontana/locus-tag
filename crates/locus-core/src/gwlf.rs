//! Gradient-Weighted Line Fitting (GWLF) for sub-pixel corner refinement.

use crate::image::ImageView;

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
    pub fn centroid(&self) -> Option<(f64, f64)> {
        if self.sum_w < 1e-9 {
            None
        } else {
            Some((self.sum_wx / self.sum_w, self.sum_wy / self.sum_w))
        }
    }

    /// Compute the 2x2 gradient-weighted spatial covariance matrix.
    /// Returns [sigma_xx, sigma_xy, sigma_xy, sigma_yy]
    #[must_use]
    pub fn covariance(&self) -> Option<[f64; 4]> {
        let (cx, cy) = self.centroid()?;
        let s_w = self.sum_w;
        
        let s_xx = (self.sum_wxx / s_w) - (cx * cx);
        let s_yy = (self.sum_wyy / s_w) - (cy * cy);
        let s_xy = (self.sum_wxy / s_w) - (cx * cy);
        
        Some([s_xx, s_xy, s_xy, s_yy])
    }
}

/// A line in homogeneous coordinates [nx, ny, d] such that nx*x + ny*y + d = 0.
#[derive(Clone, Copy, Debug)]
pub struct HomogeneousLine {
    pub nx: f64,
    pub ny: f64,
    pub d: f64,
}

/// Result of analytic eigendecomposition of a 2x2 symmetric matrix.
pub struct EigenResult {
    pub min_eigenvalue: f64,
    pub min_eigenvector: (f64, f64),
}

/// Solves the eigendecomposition of a 2x2 symmetric matrix [[a, b], [b, c]]
/// and returns the result for the smallest eigenvalue.
#[must_use]
pub fn solve_2x2_symmetric_min_eigen(a: f64, b: f64, c: f64) -> EigenResult {
    // Characteristic equation: det(A - lambda*I) = 0
    // (a - lambda)(c - lambda) - b^2 = 0
    // lambda^2 - (a+c)lambda + (ac - b^2) = 0
    let trace = a + c;
    let det = a * c - b * b;
    
    // Quadratic formula: lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let l_min = (trace - disc) / 2.0;
    
    // Eigenvector (nx, ny) for l_min:
    // (a - l_min)nx + b*ny = 0
    // If b is not zero: ny = -(a - l_min)nx / b. Set nx = b, ny = -(a - l_min)
    // If b is zero: diagonal matrix.
    let (nx, ny) = if b.abs() > 1e-9 {
        let vx = b;
        let vy = l_min - a;
        let mag = (vx * vx + vy * vy).sqrt();
        (vx / mag, vy / mag)
    } else if a < c {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    
    EigenResult {
        min_eigenvalue: l_min,
        min_eigenvector: (nx, ny),
    }
}
