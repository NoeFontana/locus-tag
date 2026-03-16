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
    #[allow(clippy::similar_names)]
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
    /// X component of the unit normal vector.
    pub nx: f64,
    /// Y component of the unit normal vector.
    pub ny: f64,
    /// Scalar distance to the origin.
    pub d: f64,
}

impl HomogeneousLine {
    /// Compute the intersection of two homogeneous lines.
    /// Returns (x, y) in Cartesian coordinates.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<(f64, f64)> {
        // Cross product of l1 = [n1x, n1y, d1] and l2 = [n2x, n2y, d2]
        // c = l1 x l2 = [u, v, w]
        let u = self.ny * other.d - self.d * other.ny;
        let v = self.d * other.nx - self.nx * other.d;
        let w = self.nx * other.ny - self.ny * other.nx;

        if w.abs() < 1e-9 {
            None // Parallel lines
        } else {
            Some((u / w, v / w))
        }
    }
}

/// Result of analytic eigendecomposition of a 2x2 symmetric matrix.
pub struct EigenResult {
    /// Smallest eigenvalue.
    pub min_eigenvalue: f64,
    /// Eigenvector corresponding to the smallest eigenvalue.
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

/// Refine quad corners using Gradient-Weighted Line Fitting (GWLF).
///
/// Returns the refined corners [[x, y]; 4] or None if refinement fails.
#[must_use]
#[allow(clippy::similar_names)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_wrap)]
pub fn refine_quad_gwlf(img: &ImageView, coarse_corners: &[[f32; 2]; 4]) -> Option<[[f32; 2]; 4]> {
    let mut lines = [HomogeneousLine {
        nx: 0.0,
        ny: 0.0,
        d: 0.0,
    }; 4];

    for i in 0..4 {
        let p0 = coarse_corners[i];
        let p1 = coarse_corners[(i + 1) % 4];

        // Edge vector and length
        let dx = f64::from(p1[0] - p0[0]);
        let dy = f64::from(p1[1] - p0[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 2.0 {
            return None;
        }

        // Unit vector along the edge
        let ux = dx / len;
        let uy = dy / len;

        // Normal vector to the edge
        let nx_coarse = -uy;
        let ny_coarse = ux;

        let mut acc = MomentAccumulator::new();

        // Sample points along the edge
        let steps = (len as usize).max(2);
        for s in 0..=steps {
            let t = (s as f64) / (steps as f64);
            let px = f64::from(p0[0]) + t * dx;
            let py = f64::from(p0[1]) + t * dy;

            // Sample transversally
            for k in -2..=2 {
                let sx = px + f64::from(k) * nx_coarse;
                let sy = py + f64::from(k) * ny_coarse;

                let ix = sx.round() as i32;
                let iy = sy.round() as i32;

                let width_i32 = img.width as i32;
                let height_i32 = img.height as i32;

                if ix > 0 && ix < (width_i32 - 1) && iy > 0 && iy < (height_i32 - 1) {
                    let uix = ix as usize;
                    let uiy = iy as usize;
                    // Simple finite difference gradient
                    let g_x = f64::from(img.get_pixel(uix + 1, uiy))
                        - f64::from(img.get_pixel(uix - 1, uiy));
                    let g_y = f64::from(img.get_pixel(uix, uiy + 1))
                        - f64::from(img.get_pixel(uix, uiy - 1));

                    let w = g_x * g_x + g_y * g_y;
                    if w > 100.0 {
                        // Noise floor
                        acc.add(f64::from(ix), f64::from(iy), w);
                    }
                }
            }
        }

        let cov = acc.covariance()?;
        let res = solve_2x2_symmetric_min_eigen(cov[0], cov[1], cov[3]);
        let (nx, ny) = res.min_eigenvector;
        let (cx, cy) = acc.centroid()?;

        lines[i] = HomogeneousLine {
            nx,
            ny,
            d: -(nx * cx + ny * cy),
        };
    }

    // Intersect adjacent lines
    let mut refined = [[0.0f32; 2]; 4];
    for i in 0..4 {
        // Line i and line i-1 intersect at corner i
        let l_prev = lines[(i + 3) % 4];
        let l_curr = lines[i];

        let (ix, iy) = l_prev.intersect(&l_curr)?;

        // Sanity check: distance from coarse corner
        let dist_sq = (ix - f64::from(coarse_corners[i][0])).powi(2)
            + (iy - f64::from(coarse_corners[i][1])).powi(2);

        if dist_sq > 9.0 {
            // 3.0 pixel threshold
            return None;
        }

        refined[i] = [ix as f32, iy as f32];
    }

    Some(refined)
}
