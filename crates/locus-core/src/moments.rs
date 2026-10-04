//! Gradient-weighted spatial moments of edge pixels and the analytic 2×2 symmetric
//! eigenvector that turns them into an edge normal.
//!
//! Shared by the EdLines line fitter and the distortion-aware edge fit of quad extraction.

use nalgebra::{Matrix2, Vector2};

/// Accumulator for gradient-weighted spatial moments of an edge.
///
/// The shared `sum_` field prefix is intentional (each field is a moment sum).
// `allow`, not `expect`: `struct_field_names` fires only when this type is not
// re-exported (feature-dependent), so an expectation would be unfulfilled under
// `--all-features`.
#[allow(clippy::struct_field_names)]
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
    #[expect(
        clippy::similar_names,
        reason = "s_xx/s_yy/s_xy match the covariance-component math notation"
    )]
    pub fn covariance(&self) -> Option<Matrix2<f64>> {
        let c = self.centroid()?;
        let s_w = self.sum_w;

        let s_xx = (self.sum_wxx / s_w) - (c.x * c.x);
        let s_yy = (self.sum_wyy / s_w) - (c.y * c.y);
        let s_xy = (self.sum_wxy / s_w) - (c.x * c.y);

        Some(Matrix2::new(s_xx, s_xy, s_xy, s_yy))
    }
}

/// Unit eigenvector of the smallest eigenvalue of the 2×2 symmetric matrix `[[a, b], [b, c]]`.
///
/// Applied to the gradient-weighted spatial covariance of an edge, this is the edge normal
/// (the direction of least spread).
#[cfg(any(test, feature = "non_rectified"))]
#[must_use]
pub fn min_eigenvector_2x2_symmetric(a: f64, b: f64, c: f64) -> Vector2<f64> {
    let trace = a + c;
    let det = a * c - b * b;

    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let l_min = (trace - disc) / 2.0;

    if b.abs() > 1e-9 {
        Vector2::new(b, l_min - a).normalize()
    } else if a < c {
        Vector2::new(1.0, 0.0)
    } else {
        Vector2::new(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{MomentAccumulator, min_eigenvector_2x2_symmetric};

    #[test]
    fn moment_accumulation_horizontal_edge() {
        let mut acc = MomentAccumulator::new();

        // A horizontal edge at y = 10: (0, 10), (1, 10), (2, 10), all with weight 1.
        acc.add(0.0, 10.0, 1.0);
        acc.add(1.0, 10.0, 1.0);
        acc.add(2.0, 10.0, 1.0);

        let centroid = acc.centroid().unwrap_or_default();
        assert!((centroid.x - 1.0).abs() < 1e-9);
        assert!((centroid.y - 10.0).abs() < 1e-9);

        let cov = acc.covariance().unwrap_or_default();
        // sigma_xx = (0^2 + 1^2 + 2^2)/3 - 1^2 = 5/3 - 1 = 2/3
        assert!((cov[(0, 0)] - 0.666_666_666).abs() < 1e-6);
        assert!(cov[(0, 1)].abs() < 1e-9);
        assert!(cov[(1, 0)].abs() < 1e-9);
        assert!(cov[(1, 1)].abs() < 1e-9);
    }

    #[test]
    fn empty_accumulator_has_no_centroid() {
        let acc = MomentAccumulator::new();
        assert!(acc.centroid().is_none());
        assert!(acc.covariance().is_none());
    }

    #[test]
    fn min_eigenvector_of_symmetric_2x2() {
        // [[1, 0], [0, 2]]: the smallest eigenvalue 1 has eigenvector (1, 0).
        let v = min_eigenvector_2x2_symmetric(1.0, 0.0, 2.0);
        assert!((v.x - 1.0).abs() < 1e-9);
        assert!(v.y.abs() < 1e-9);

        // [[2, 0], [0, 1]]: the smallest eigenvalue 1 has eigenvector (0, 1).
        let v = min_eigenvector_2x2_symmetric(2.0, 0.0, 1.0);
        assert!(v.x.abs() < 1e-9);
        assert!((v.y - 1.0).abs() < 1e-9);

        // [[2, 1], [1, 2]]: lambda_min = 1 with eigenvector normalize(1, -1).
        let v = min_eigenvector_2x2_symmetric(2.0, 1.0, 2.0);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((v.x.abs() - inv_sqrt2).abs() < 1e-5);
        assert!((v.y.abs() - inv_sqrt2).abs() < 1e-5);
        assert!((v.x + v.y).abs() < 1e-5);
    }
}
