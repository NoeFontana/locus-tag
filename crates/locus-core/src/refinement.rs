//! Sub-pixel refinement using bivariate polynomial surface fitting.
//!
//! This module implements high-precision saddle point refinement for ChArUco
//! checkerboard corners. It uses a 2nd-order bivariate polynomial model:
//!
//! $$I(x,y) = a x^2 + b xy + c y^2 + d x + e y + f$$
//!
//! The refined saddle point is the location where the gradient is zero:
//!
//! $$\nabla I(x,y) = \begin{bmatrix} 2ax + by + d \\ bx + 2cy + e \end{bmatrix} = \mathbf{0}$$
//!
//! Solving for $(x,y)$ yields the sub-pixel refinement offset.

#![allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]

use crate::image::ImageView;
use nalgebra::{SMatrix, SVector};

/// A 2nd-order bivariate polynomial model for image intensity.
///
/// Represents $I(x,y) = a x^2 + b xy + c y^2 + d x + e y + f$.
#[derive(Debug, Clone, Copy, Default)]
pub struct BivariatePolynomial {
    /// Coefficient of $x^2$.
    pub a: f64,
    /// Coefficient of $xy$.
    pub b: f64,
    /// Coefficient of $y^2$.
    pub c: f64,
    /// Coefficient of $x$.
    pub d: f64,
    /// Coefficient of $y$.
    pub e: f64,
    /// Constant term $f$.
    pub f: f64,
}

impl BivariatePolynomial {
    /// Fits a bivariate polynomial to a local image patch using Least Squares.
    ///
    /// The input `window` should be a square patch centered around the coarse point.
    /// Returns the fitted coefficients.
    #[must_use]
    pub fn fit(view: &ImageView, cx: f64, cy: f64, radius: usize) -> Option<Self> {
        let mut ata = SMatrix::<f64, 6, 6>::zeros();
        let mut atb = SVector::<f64, 6>::zeros();

        let ix = cx.round() as isize;
        let iy = cy.round() as isize;
        let r = radius as isize;

        let mut count = 0;

        for dy in -r..=r {
            for dx in -r..=r {
                let px = ix + dx;
                let py = iy + dy;

                if px < 0 || py < 0 || px >= view.width as isize || py >= view.height as isize {
                    continue;
                }

                let intensity = f64::from(view.get_pixel(px as usize, py as usize));

                // Coordinates relative to the coarse center (cx, cy)
                let x = px as f64 + 0.5 - cx;
                let y = py as f64 + 0.5 - cy;

                let x2 = x * x;
                let y2 = y * y;
                let xy = x * y;

                let row = [x2, xy, y2, x, y, 1.0];

                for i in 0..6 {
                    for j in i..6 {
                        let val = row[i] * row[j];
                        ata[(i, j)] += val;
                        if i != j {
                            ata[(j, i)] += val;
                        }
                    }
                    atb[i] += row[i] * intensity;
                }
                count += 1;
            }
        }

        if count < 6 {
            return None;
        }

        ata.lu().solve(&atb).map(|p| Self {
            a: p[0],
            b: p[1],
            c: p[2],
            d: p[3],
            e: p[4],
            f: p[5],
        })
    }

    /// Solves for the saddle point where the gradient is zero.
    ///
    /// Returns the offset relative to the window center.
    #[must_use]
    pub fn solve_saddle_point(&self) -> Option<(f64, f64)> {
        // Gradient: [2ax + by + d, bx + 2cy + e] = [0, 0]
        // [2a b] [x] = [-d]
        // [b 2c] [y] = [-e]
        let det = 4.0 * self.a * self.c - self.b * self.b;
        if det.abs() < 1e-10 {
            return None;
        }

        let dx = (self.b * self.e - 2.0 * self.c * self.d) / det;
        let dy = (self.b * self.d - 2.0 * self.a * self.e) / det;

        Some((dx, dy))
    }
}

/// Refines a coarse corner estimate to sub-pixel precision.
#[must_use]
pub fn refine_saddle_point(view: &ImageView, x: f64, y: f64, radius: usize) -> Option<(f64, f64)> {
    let poly = BivariatePolynomial::fit(view, x, y, radius)?;
    let (dx, dy) = poly.solve_saddle_point()?;
    Some((x + dx, y + dy))
}

/// Trait for sub-pixel refinement strategies.
pub trait SubpixelRefiner {
    /// Refines a coarse estimate (x, y) in the given image view.
    /// Returns the refined (x, y) coordinates.
    #[must_use]
    fn refine(&self, view: &ImageView, x: f64, y: f64) -> Option<(f64, f64)>;
}

/// A sub-pixel refiner that uses bivariate polynomial surface fitting.
#[derive(Debug, Clone, Copy)]
pub struct PolynomialRefiner {
    /// Radius of the square search window.
    pub radius: usize,
}

impl PolynomialRefiner {
    /// Create a new polynomial refiner with the given radius.
    #[must_use]
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
}

impl SubpixelRefiner for PolynomialRefiner {
    fn refine(&self, view: &ImageView, x: f64, y: f64) -> Option<(f64, f64)> {
        refine_saddle_point(view, x, y, self.radius)
    }
}
