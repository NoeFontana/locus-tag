//! Zero-cost camera distortion model abstractions.
//!
//! This module defines the [`CameraModel`] trait and its three concrete implementations:
//! - [`PinholeModel`]: Ideal pinhole (no distortion). `IS_RECTIFIED = true` causes the
//!   compiler to eliminate all distortion branches in the hot path.
//! - [`BrownConradyModel`]: Standard polynomial radial + tangential distortion (OpenCV convention).
//! - [`KannalaBrandtModel`]: Equidistant fisheye projection model.
//!
//! All models operate on **normalized image coordinates** `(xn, yn)` — i.e., after
//! dividing pixel coordinates by the focal lengths and subtracting the principal point:
//! `xn = (px - cx) / fx`, `yn = (py - cy) / fy`.

/// A compile-time abstraction over camera distortion models.
///
/// Monomorphizing on this trait allows the compiler to completely eliminate all
/// distortion code when `IS_RECTIFIED = true` (e.g., for [`PinholeModel`]),
/// leaving zero overhead for the common rectified-image case.
pub trait CameraModel: Copy + Send + Sync + 'static {
    /// True iff the camera produces a rectified (undistorted) image.
    ///
    /// When `true`, the compiler can statically prove that `distort` and `undistort`
    /// are identity functions and will eliminate all branches guarded by `!C::IS_RECTIFIED`.
    const IS_RECTIFIED: bool;

    /// Map normalized ideal (undistorted) coordinates `(xn, yn)` to normalized
    /// distorted coordinates `(xd, yd)`.
    fn distort(&self, xn: f64, yn: f64) -> [f64; 2];

    /// Map normalized distorted (observed) coordinates `(xd, yd)` back to normalized
    /// ideal (undistorted) coordinates `(xu, yu)`.
    fn undistort(&self, xd: f64, yd: f64) -> [f64; 2];

    /// Compute the 2×2 Jacobian of the distortion map at `(xn, yn)`.
    ///
    /// Returns `[[∂xd/∂xn, ∂xd/∂yn], [∂yd/∂xn, ∂yd/∂yn]]`.
    ///
    /// Used in the LM Jacobian to correctly account for distortion derivatives.
    fn distort_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2];

    /// [`Self::undistort`], but `None` unless re-distorting lands back on the input within
    /// [`MAX_UNDISTORT_RESIDUAL`].
    ///
    /// This is the only safe way to unproject a point that will then be used as geometry.
    /// `undistort` cannot fail by signature, so outside the model's invertible domain it
    /// returns a point that is not a preimage of its argument; feeding that into a homography
    /// or a pose silently corrupts the result instead of dropping the measurement.
    #[inline]
    #[cfg(feature = "non_rectified")]
    fn undistort_checked(&self, xd: f64, yd: f64) -> Option<[f64; 2]> {
        let [xu, yu] = self.undistort(xd, yd);
        let [xc, yc] = self.distort(xu, yu);
        let dx = xc - xd;
        let dy = yc - yd;
        let residual_sq = dx * dx + dy * dy;
        // Finiteness is checked separately: a NaN fails every comparison, so the bound alone
        // would accept it.
        if residual_sq.is_finite() && residual_sq <= MAX_UNDISTORT_RESIDUAL * MAX_UNDISTORT_RESIDUAL
        {
            Some([xu, yu])
        } else {
            None
        }
    }
}

/// Ideal pinhole camera model (no distortion).
///
/// All methods are `#[inline]` no-ops. The compiler eliminates every
/// code path guarded by `!C::IS_RECTIFIED` at compile time, producing
/// zero-overhead monomorphized code for the standard rectified-image case.
#[derive(Clone, Copy, Debug, Default)]
pub struct PinholeModel;

impl CameraModel for PinholeModel {
    const IS_RECTIFIED: bool = true;

    #[inline]
    fn distort(&self, xn: f64, yn: f64) -> [f64; 2] {
        [xn, yn]
    }

    #[inline]
    fn undistort(&self, xd: f64, yd: f64) -> [f64; 2] {
        [xd, yd]
    }

    #[inline]
    fn distort_jacobian(&self, _xn: f64, _yn: f64) -> [[f64; 2]; 2] {
        [[1.0, 0.0], [0.0, 1.0]]
    }
}

#[cfg(feature = "non_rectified")]
/// Brown-Conrady (OpenCV) lens distortion model.
///
/// Distortion formula (operating on normalized coordinates):
/// ```text
/// r² = xn² + yn²
/// radial = 1 + k1·r² + k2·r⁴ + k3·r⁶
/// xd = xn·radial + 2·p1·xn·yn + p2·(r² + 2·xn²)
/// yd = yn·radial + p1·(r² + 2·yn²) + 2·p2·xn·yn
/// ```
///
/// Coefficient ordering matches OpenCV's `distCoeffs` convention: `[k1, k2, p1, p2, k3]`.
#[derive(Clone, Copy, Debug)]
pub struct BrownConradyModel {
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
    /// Radial distortion coefficient k3.
    pub k3: f64,
}

#[cfg(feature = "non_rectified")]
impl BrownConradyModel {
    /// Construct from a flat coefficient slice `[k1, k2, p1, p2, k3]`.
    ///
    /// # Errors
    /// Returns an error string if `coeffs.len() != 5`.
    pub fn from_coeffs(coeffs: &[f64]) -> Result<Self, &'static str> {
        if coeffs.len() != 5 {
            return Err("BrownConrady requires exactly 5 coefficients: [k1, k2, p1, p2, k3]");
        }
        Ok(Self {
            k1: coeffs[0],
            k2: coeffs[1],
            p1: coeffs[2],
            p2: coeffs[3],
            k3: coeffs[4],
        })
    }
}

/// How far `distort(undistort(u))` may land from `u`, in **normalized-plane** units, for the
/// inverse to be trusted.
///
/// Both models' [`CameraModel::undistort`] is infallible by signature: outside the model's
/// invertible domain it returns its best effort, which is not a preimage. Every consumer that
/// feeds an unprojected point into geometry must therefore re-distort and check against this
/// budget — and must reject non-finite residuals explicitly, since a comparison with NaN is
/// false and a bare `>` would *admit* them.
///
/// 2e-4 normalized is ~0.27 px at the shipped hub datasets' focal lengths, three orders above
/// a converged inverse (~1e-12 normalized) and three below the corner accuracy it protects. It
/// was originally chosen at 2x the `robustness/camera_geometry.rs` round-trip proptest envelope
/// (`< 1e-4`), so a diverging solve bails while well-conditioned points stay.
///
/// Public because it is observable contract, not a private tuning knob: it is the bound the
/// default [`CameraModel::undistort_checked`] applies, so anyone implementing the trait or
/// reading its `None` needs the number.
#[cfg(feature = "non_rectified")]
pub const MAX_UNDISTORT_RESIDUAL: f64 = 2e-4;

/// Radius below which every model in this module is the identity to within f64
/// precision, so the inversions can return their input. Shared by both models and
/// both directions, because it states one geometric fact: a point on the optical axis
/// has no radial direction to distort along.
#[cfg(feature = "non_rectified")]
const NEAR_AXIS_RADIUS: f64 = 1e-8;

/// Iteration cap of the Brown-Conrady radial Newton solve. Quadratic convergence from
/// `r_u = r_d` reaches f64 precision in 3-5 steps across the calibrations we ship.
#[cfg(feature = "non_rectified")]
const BC_RADIAL_MAX_ITERS: usize = 8;

/// **Relative** step tolerance of that solve. Relative, not absolute: at the wide-angle
/// radii a fisheye reaches (`r_u > 4.5`) an absolute 1e-15 sits below 1 ulp and can never
/// be met, so the loop would always burn its full iteration budget.
#[cfg(feature = "non_rectified")]
const BC_RADIAL_REL_TOL: f64 = 1e-14;

/// Iteration cap of the 2-D tangential polish (2-3 steps in practice).
#[cfg(feature = "non_rectified")]
const BC_POLISH_MAX_ITERS: usize = 4;

/// Squared-residual floor of that polish, in normalized units. 1e-24 is 1e-12 normalized,
/// i.e. ~1.4e-9 px at the shipped hub focal length — five orders below the
/// `quad::MAX_UNDISTORT_RESIDUAL` gate this feeds, and one polish step cheaper than
/// chasing the last few ulps.
#[cfg(feature = "non_rectified")]
const BC_POLISH_RESIDUAL_SQ: f64 = 1e-24;

/// Determinant floor below which the polish's 2x2 solve is treated as singular.
#[cfg(feature = "non_rectified")]
const BC_POLISH_MIN_DET: f64 = 1e-12;

#[cfg(feature = "non_rectified")]
impl BrownConradyModel {
    /// The radial map `g(r) = r·(1 + k1·r² + k2·r⁴ + k3·r⁶)` and its derivative.
    #[inline]
    fn radial_and_derivative(&self, r: f64) -> (f64, f64) {
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        (
            r * (1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6),
            1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r4 + 7.0 * self.k3 * r6,
        )
    }

    /// Forward map and its Jacobian in one pass, sharing the radial monomials.
    ///
    /// The Jacobian is **symmetric** — `∂xd/∂yn` and `∂yd/∂xn` are the same expression —
    /// so it is returned as `[a, b, d]` for `[[a, b], [b, d]]` and callers can use
    /// `det = a·d − b²`. Used by the polish, which would otherwise evaluate `r²/r⁴/r⁶`
    /// and `radial` twice per step.
    #[inline]
    fn distort_with_jacobian(&self, xn: f64, yn: f64) -> ([f64; 2], [f64; 3]) {
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        // ∂radial/∂r² = k1 + 2·k2·r² + 3·k3·r⁴
        let d_radial_dr2 = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;
        let two_xy_dr = 2.0 * xn * yn * d_radial_dr2;
        (
            [
                xn * radial + 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn),
                yn * radial + self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn,
            ],
            [
                radial + 2.0 * xn * xn * d_radial_dr2 + 2.0 * self.p1 * yn + 6.0 * self.p2 * xn,
                two_xy_dr + 2.0 * self.p1 * xn + 2.0 * self.p2 * yn,
                radial + 2.0 * yn * yn * d_radial_dr2 + 6.0 * self.p1 * yn + 2.0 * self.p2 * xn,
            ],
        )
    }
}

#[cfg(feature = "non_rectified")]
impl CameraModel for BrownConradyModel {
    const IS_RECTIFIED: bool = false;

    fn distort(&self, xn: f64, yn: f64) -> [f64; 2] {
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let xd = xn * radial + 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn);
        let yd = yn * radial + self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn;
        [xd, yd]
    }

    fn undistort(&self, xd: f64, yd: f64) -> [f64; 2] {
        // Two-stage inversion.
        //
        // Stage 1 solves the *radial* sub-problem, which is one-dimensional and strictly
        // monotone on `[0, r_turn)`, where `r_turn` is the first positive root of
        // `g'(r)`: find `r_u` with `g(r_u) = r_d`. Newton from `r_u = r_d` converges
        // quadratically, and is **exact** when `p1 = p2 = 0`, which is why stage 2 is
        // skipped entirely in that (common) case.
        //
        // Stage 2 is a 2-D Newton polish against the full forward map, absorbing the
        // tangential terms stage 1 ignores. It runs only if stage 1 converged.
        //
        // Both stages replace a 5-iteration *fixed point* (`xu ← (xd − dx(xu))/radial(xu)`,
        // only linearly convergent). At the coefficients of the shipped Brown-Conrady hub
        // dataset (`k1 = -0.28`, `k2 = 0.08`) that iteration had not converged after 5
        // steps and left **0.42 px** of round-trip error at the image corner — above
        // `quad.rs`'s `MAX_UNDISTORT_RESIDUAL`, so it silently discarded every candidate
        // touching ~1.8 % of the frame.
        //
        // **Out-of-domain inputs must not be polished.** Beyond `g(r_turn)` the radius
        // `r_d` has no preimage on the invertible branch, but the *full* map still has
        // one: the mirrored far branch, `g(-r) = -g(r)`. A polish started past the
        // turning point converges onto it and re-distorts to ~1e-14, so the caller's
        // residual gate would ACCEPT a point at several times the correct radius and the
        // wrong sign (`k1 = -0.1`, `r_d = 1.335` returns `r_u = -3.69`). Stage 1
        // therefore restores the last on-branch iterate and reports non-convergence,
        // which keeps a large residual and makes the gate reject — the behaviour the
        // superseded fixed point had by accident.
        let r_d = (xd * xd + yd * yd).sqrt();
        if r_d < NEAR_AXIS_RADIUS {
            return [xd, yd];
        }

        let mut r_u = r_d;
        let mut last_on_branch = r_d;
        let mut converged = false;
        for _ in 0..BC_RADIAL_MAX_ITERS {
            let (g, dg) = self.radial_and_derivative(r_u);
            if dg <= 0.0 {
                r_u = last_on_branch;
                break;
            }
            last_on_branch = r_u;
            let step = (g - r_d) / dg;
            r_u = (r_u - step).max(0.0);
            if step.abs() <= BC_RADIAL_REL_TOL * r_u.max(1.0) {
                converged = true;
                break;
            }
        }

        let scale = r_u / r_d;
        let mut xu = xd * scale;
        let mut yu = yd * scale;

        // Stage 1 is the exact inverse without tangential terms, and must not run when it
        // did not converge (see above).
        if !converged || (self.p1 == 0.0 && self.p2 == 0.0) {
            return [xu, yu];
        }

        for _ in 0..BC_POLISH_MAX_ITERS {
            let ([fx, fy], [a, b, d]) = self.distort_with_jacobian(xu, yu);
            let rx = fx - xd;
            let ry = fy - yd;
            if rx * rx + ry * ry < BC_POLISH_RESIDUAL_SQ {
                break;
            }
            // Symmetric Jacobian: det = a·d − b².
            let det = a * d - b * b;
            if det.abs() < BC_POLISH_MIN_DET {
                break;
            }
            xu -= (d * rx - b * ry) / det;
            yu -= (a * ry - b * rx) / det;
        }
        [xu, yu]
    }

    fn distort_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2] {
        // One expression, one place: `distort_with_jacobian` returns the symmetric
        // `[a, b, d]`; this expands it to the trait's dense 2x2.
        let (_, [a, b, d]) = self.distort_with_jacobian(xn, yn);
        [[a, b], [b, d]]
    }
}

#[cfg(feature = "non_rectified")]
/// Kannala-Brandt equidistant fisheye camera model.
///
/// Projection formula (operating on normalized coordinates):
/// ```text
/// r     = √(xn² + yn²)
/// θ     = atan(r)
/// θ_d   = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
/// xd    = (θ_d / r) · xn
/// yd    = (θ_d / r) · yn
/// ```
///
/// Coefficient ordering: `[k1, k2, k3, k4]`.
#[derive(Clone, Copy, Debug)]
pub struct KannalaBrandtModel {
    /// Fisheye distortion coefficient k1.
    pub k1: f64,
    /// Fisheye distortion coefficient k2.
    pub k2: f64,
    /// Fisheye distortion coefficient k3.
    pub k3: f64,
    /// Fisheye distortion coefficient k4.
    pub k4: f64,
}

/// Upper clamp on the recovered incidence angle θ before `tan(θ)` in KB `undistort`,
/// so a Newton iterate that diverges past the calibrated fisheye envelope produces a
/// large-but-finite radius (caught by the round-trip residual gate) rather than ±∞.
#[cfg(feature = "non_rectified")]
const MAX_KB_THETA: f64 = core::f64::consts::FRAC_PI_2 - 1e-3;

#[cfg(feature = "non_rectified")]
impl KannalaBrandtModel {
    /// Construct from a flat coefficient slice `[k1, k2, k3, k4]`.
    ///
    /// # Errors
    /// Returns an error string if `coeffs.len() != 4`.
    pub fn from_coeffs(coeffs: &[f64]) -> Result<Self, &'static str> {
        if coeffs.len() != 4 {
            return Err("KannalaBrandt requires exactly 4 coefficients: [k1, k2, k3, k4]");
        }
        Ok(Self {
            k1: coeffs[0],
            k2: coeffs[1],
            k3: coeffs[2],
            k4: coeffs[3],
        })
    }

    /// Evaluate the angle polynomial and its derivative at θ.
    /// Returns `(θ_d, dθ_d/dθ)`.
    #[inline]
    fn angle_poly(&self, theta: f64) -> (f64, f64) {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t2 * t4;
        let t8 = t4 * t4;
        let theta_d = theta * (1.0 + self.k1 * t2 + self.k2 * t4 + self.k3 * t6 + self.k4 * t8);
        let dtheta_d =
            1.0 + 3.0 * self.k1 * t2 + 5.0 * self.k2 * t4 + 7.0 * self.k3 * t6 + 9.0 * self.k4 * t8;
        (theta_d, dtheta_d)
    }
}

#[cfg(feature = "non_rectified")]
impl CameraModel for KannalaBrandtModel {
    const IS_RECTIFIED: bool = false;

    fn distort(&self, xn: f64, yn: f64) -> [f64; 2] {
        let r = (xn * xn + yn * yn).sqrt();
        if r < NEAR_AXIS_RADIUS {
            return [xn, yn];
        }
        // theta = atan(r) since the point is at depth z=1 in normalized coords
        let theta = r.atan();
        let (theta_d, _) = self.angle_poly(theta);
        let scale = theta_d / r;
        [xn * scale, yn * scale]
    }

    fn undistort(&self, xd: f64, yd: f64) -> [f64; 2] {
        let r_d = (xd * xd + yd * yd).sqrt();
        if r_d < NEAR_AXIS_RADIUS {
            return [xd, yd];
        }
        // Invert θ_d = poly(θ) via Newton's method to recover θ, then r = tan(θ).
        let mut theta = r_d; // initial guess: identity
        for _ in 0..10 {
            let (theta_d, d_theta_d) = self.angle_poly(theta);
            let f = theta_d - r_d;
            if f.abs() < 1e-12 {
                break;
            }
            theta -= f / d_theta_d.max(1e-8);
            theta = theta.max(0.0);
        }
        // r_undistorted = tan(θ). Clamp θ just below π/2 so a Newton iterate driven
        // into the divergent tail (extreme fisheye beyond the calibrated envelope)
        // cannot make `tan(θ)` blow up to ±∞; the resulting large-but-finite radius
        // is then caught by the undistort round-trip residual gate.
        let r_undist = theta.min(MAX_KB_THETA).tan();
        let scale = r_undist / r_d;
        [xd * scale, yd * scale]
    }

    #[expect(
        clippy::similar_names,
        reason = "partial-derivative vars (dxd_dxn/dxd_dyn/dyd_dxn/dyd_dyn) match the Jacobian math notation"
    )]
    fn distort_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2] {
        let r2 = xn * xn + yn * yn;
        let r = r2.sqrt();
        if r < NEAR_AXIS_RADIUS {
            // Near the optical axis, the equidistant model approaches identity.
            return [[1.0, 0.0], [0.0, 1.0]];
        }

        let theta = r.atan();
        let (theta_d, dtheta_d_dtheta) = self.angle_poly(theta);

        // scale = θ_d / r
        // ∂scale/∂xn = (∂θ_d/∂xn · r - θ_d · ∂r/∂xn) / r²
        //
        // ∂r/∂xn = xn / r
        // ∂θ/∂r  = 1 / (1 + r²)
        // ∂θ/∂xn = (xn / r) / (1 + r²)
        // ∂θ_d/∂xn = dθ_d_dθ · ∂θ/∂xn = dθ_d_dθ · xn / (r · (1 + r²))

        let one_plus_r2 = 1.0 + r2;
        let dthetad_dxn = dtheta_d_dtheta * xn / (r * one_plus_r2);
        let dthetad_dyn = dtheta_d_dtheta * yn / (r * one_plus_r2);

        let dscale_dxn = (dthetad_dxn * r - theta_d * (xn / r)) / r2;
        let dscale_dyn = (dthetad_dyn * r - theta_d * (yn / r)) / r2;

        // xd = scale · xn  →  ∂xd/∂xn = scale + xn · ∂scale/∂xn
        let dxd_dxn = theta_d / r + xn * dscale_dxn;
        let dxd_dyn = xn * dscale_dyn;
        let dyd_dxn = yn * dscale_dxn;
        let dyd_dyn = theta_d / r + yn * dscale_dyn;

        [[dxd_dxn, dxd_dyn], [dyd_dxn, dyd_dyn]]
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pinhole_is_identity() {
        let m = PinholeModel;
        let [xd, yd] = m.distort(0.3, -0.2);
        assert!((xd - 0.3).abs() < f64::EPSILON);
        assert!((yd - (-0.2)).abs() < f64::EPSILON);
        let [xu, yu] = m.undistort(0.3, -0.2);
        assert!((xu - 0.3).abs() < f64::EPSILON);
        assert!((yu - (-0.2)).abs() < f64::EPSILON);
    }

    #[cfg(feature = "non_rectified")]
    mod non_rectified {
        use super::*;

        /// Round-trip identity: distort then undistort should recover the original point.
        fn check_roundtrip<C: CameraModel>(model: &C, xn: f64, yn: f64, tol: f64) {
            let [xd, yd] = model.distort(xn, yn);
            let [xu, yu] = model.undistort(xd, yd);
            assert!((xu - xn).abs() < tol, "xn round-trip failed: {xu} vs {xn}");
            assert!((yu - yn).abs() < tol, "yn round-trip failed: {yu} vs {yn}");
        }

        /// Numerical Jacobian check via **central** differences, with the step
        /// scaled by the operating radius so the check stays meaningful at the
        /// fisheye periphery (where `r_u = tan(θ)` reaches ~11.7 on the hub
        /// dataset, far outside the `r ≤ 0.5` box the spot-checks below use).
        #[allow(clippy::similar_names)]
        fn check_jacobian<C: CameraModel>(model: &C, xn: f64, yn: f64) {
            let eps = 1e-6 * xn.hypot(yn).max(1.0);
            let jac = model.distort_jacobian(xn, yn);

            let [xp, yp_x] = model.distort(xn + eps, yn);
            let [xm, ym_x] = model.distort(xn - eps, yn);
            let [xp_y, yp] = model.distort(xn, yn + eps);
            let [xm_y, ym] = model.distort(xn, yn - eps);

            let num_dxd_dxn = (xp - xm) / (2.0 * eps);
            let num_dxd_dyn = (xp_y - xm_y) / (2.0 * eps);
            let num_dyd_dxn = (yp_x - ym_x) / (2.0 * eps);
            let num_dyd_dyn = (yp - ym) / (2.0 * eps);

            assert!(
                (jac[0][0] - num_dxd_dxn).abs() < 1e-5,
                "∂xd/∂xn: analytic={} numeric={num_dxd_dxn}",
                jac[0][0]
            );
            assert!(
                (jac[0][1] - num_dxd_dyn).abs() < 1e-5,
                "∂xd/∂yn: analytic={} numeric={num_dxd_dyn}",
                jac[0][1]
            );
            assert!(
                (jac[1][0] - num_dyd_dxn).abs() < 1e-5,
                "∂yd/∂xn: analytic={} numeric={num_dyd_dxn}",
                jac[1][0]
            );
            assert!(
                (jac[1][1] - num_dyd_dyn).abs() < 1e-5,
                "∂yd/∂yn: analytic={} numeric={num_dyd_dyn}",
                jac[1][1]
            );
        }

        #[test]
        fn brown_conrady_roundtrip() {
            let m = BrownConradyModel {
                k1: -0.3,
                k2: 0.1,
                p1: 0.001,
                p2: -0.002,
                k3: 0.0,
            };
            for &(xn, yn) in &[(0.1, 0.2), (-0.3, 0.15), (0.0, 0.4)] {
                check_roundtrip(&m, xn, yn, 1e-7);
            }
        }

        #[test]
        fn brown_conrady_jacobian() {
            let m = BrownConradyModel {
                k1: -0.3,
                k2: 0.1,
                p1: 0.001,
                p2: -0.002,
                k3: 0.0,
            };
            for &(xn, yn) in &[(0.1, 0.2), (-0.3, 0.15), (0.05, -0.05)] {
                check_jacobian(&m, xn, yn);
            }
        }

        #[test]
        fn kannala_brandt_roundtrip() {
            let m = KannalaBrandtModel {
                k1: 0.1,
                k2: -0.01,
                k3: 0.001,
                k4: 0.0,
            };
            for &(xn, yn) in &[(0.1, 0.2), (-0.3, 0.15), (0.5, 0.5)] {
                check_roundtrip(&m, xn, yn, 1e-7);
            }
        }

        #[test]
        fn kannala_brandt_jacobian() {
            let m = KannalaBrandtModel {
                k1: 0.1,
                k2: -0.01,
                k3: 0.001,
                k4: 0.0,
            };
            for &(xn, yn) in &[(0.1, 0.2), (-0.3, 0.15), (0.3, -0.3)] {
                check_jacobian(&m, xn, yn);
            }
        }

        #[test]
        fn kannala_brandt_undistort_stays_finite_at_extreme_radius() {
            // A large distorted radius drives the Newton iterate toward θ→π/2, where
            // an unclamped `tan(θ)` would blow up to ±∞. The MAX_KB_THETA clamp must
            // keep undistort finite (the round-trip residual gate then rejects it).
            let m = KannalaBrandtModel {
                k1: 0.1,
                k2: 0.01,
                k3: 0.0,
                k4: 0.0,
            };
            for &(xd, yd) in &[(5.0, 5.0), (50.0, 0.0), (0.0, 100.0)] {
                let out = m.undistort(xd, yd);
                assert!(
                    out[0].is_finite() && out[1].is_finite(),
                    "undistort must stay finite at extreme radius, got {out:?} for ({xd}, {yd})"
                );
            }
        }

        // ── The shipped hub datasets' own operating point ──────────────────
        //
        // `aprilgrid_distortion_{brown_conrady,kannala_brandt}_v1_1920x1080`.
        // The spot-checks above all sit inside `r ≤ 0.5`; the regression suite
        // runs a 1920×1080 frame whose corner reaches `r_d = 0.80` (Brown-Conrady)
        // and `θ = 85.1°` (Kannala-Brandt). Pinning the inverters *there* is what
        // the earlier benign points could not do.

        // Every literal below is read from the dataset, not chosen: `distortion_coeffs`,
        // `k_matrix` and `resolution` of the first TAG record of
        // `tests/data/hub_cache/aprilgrid_distortion_{brown_conrady,kannala_brandt}_v1_1920x1080/rich_truth.json`,
        // the subsets pinned by `xtask/datasets.toml` (`[hub]`, `revision`). Both K
        // matrices are `fx == fy` with the principal point at the exact frame centre, so
        // one `focal` is the camera, not an approximation. If a re-render or a pin bump
        // moves the operating point, update these together with the pin.
        const HUB_CX: f64 = 960.0;
        const HUB_CY: f64 = 540.0;
        const HUB_WIDTH: u32 = 1920;
        const HUB_HEIGHT: u32 = 1080;
        const HUB_BC_FOCAL: f64 = 1_371.022_086_472_430_1;
        const HUB_KB_FOCAL: f64 = 742.454_608_851_902_9;

        /// `distortion_coeffs` of the Brown-Conrady hub subset.
        fn hub_brown_conrady() -> BrownConradyModel {
            BrownConradyModel {
                k1: -0.28,
                k2: 0.08,
                p1: 0.0002,
                p2: 0.0001,
                k3: 0.0,
            }
        }

        /// `distortion_coeffs` of the Kannala-Brandt hub subset.
        fn hub_kannala_brandt() -> KannalaBrandtModel {
            KannalaBrandtModel {
                k1: -0.0035,
                k2: 0.0015,
                k3: -0.0003,
                k4: 0.0001,
            }
        }

        /// Worst `distort(undistort(u)) − u` over the frame, in **pixels**.
        fn max_round_trip_residual_px<C: CameraModel>(model: &C, focal: f64, stride: u32) -> f64 {
            let mut worst = 0.0_f64;
            let mut py = 0;
            while py <= HUB_HEIGHT {
                let mut px = 0;
                while px <= HUB_WIDTH {
                    let xd = (f64::from(px) - HUB_CX) / focal;
                    let yd = (f64::from(py) - HUB_CY) / focal;
                    if xd.hypot(yd) > 1e-9 {
                        let [xu, yu] = model.undistort(xd, yd);
                        let [xc, yc] = model.distort(xu, yu);
                        worst = worst.max((xc - xd).hypot(yc - yd) * focal);
                    }
                    px += stride;
                }
                py += stride;
            }
            worst
        }

        #[test]
        fn brown_conrady_round_trip_is_exact_over_the_hub_frame() {
            let worst = max_round_trip_residual_px(&hub_brown_conrady(), HUB_BC_FOCAL, 8);
            // Budget, not a round number: the polish stops at `BC_POLISH_RESIDUAL_SQ`
            // (1e-24 squared-normalized = 1e-12 normalized), which at this focal length is
            // 1.4e-9 px. The superseded fixed point scored 4.2e-1 px here, so this bound
            // is still eight orders of margin away from a convergence failure.
            assert!(
                worst < 1e-8,
                "Brown-Conrady round trip is {worst:.3e} px at worst over the hub frame; \
                 above 1e-8 px means the inverter stopped converging"
            );
        }

        #[test]
        fn kannala_brandt_round_trip_is_exact_over_the_hub_frame() {
            let worst = max_round_trip_residual_px(&hub_kannala_brandt(), HUB_KB_FOCAL, 8);
            assert!(
                worst < 1e-6,
                "Kannala-Brandt round trip is {worst:.3e} px at worst over the hub frame"
            );
        }

        #[test]
        fn jacobians_hold_at_the_hub_frame_periphery() {
            let bc = hub_brown_conrady();
            // r_u = 1.0055 at the image corner (field angle 45.2°).
            let r = 1.0055 / core::f64::consts::SQRT_2;
            check_jacobian(&bc, r, r);
            check_jacobian(&bc, 1.0055, 0.0);

            let kb = hub_kannala_brandt();
            // θ = 85.1° at the image corner, i.e. r_u = tan θ = 11.68.
            for theta_deg in [45.0_f64, 75.0, 85.1] {
                let r = theta_deg.to_radians().tan();
                check_jacobian(
                    &kb,
                    r / core::f64::consts::SQRT_2,
                    r / core::f64::consts::SQRT_2,
                );
            }
        }

        #[test]
        fn brown_conrady_distort_jacobian_is_symmetric() {
            // `distort_with_jacobian` returns `[a, b, d]` on the strength of this; the
            // polish then uses `det = a·d − b²`.
            let m = hub_brown_conrady();
            for &(xn, yn) in &[(0.31, -0.17), (0.9, 0.42), (-0.6, 0.6), (1.0055, 0.0)] {
                let j = m.distort_jacobian(xn, yn);
                assert!(
                    (j[0][1] - j[1][0]).abs() < f64::EPSILON,
                    "off-diagonals differ at ({xn}, {yn}): {} vs {}",
                    j[0][1],
                    j[1][0]
                );
            }
        }

        /// A radius past the model's radial turning point has no preimage on the
        /// invertible branch, but the full map still has the mirrored one
        /// (`g(-r) = -g(r)`), which re-distorts exactly. `undistort` must not return it:
        /// `quad.rs` only re-distorts and compares against `MAX_UNDISTORT_RESIDUAL`, so a
        /// far-branch point would be **accepted** at several times the correct radius and
        /// the wrong sign.
        #[test]
        fn brown_conrady_undistort_stays_on_the_invertible_branch() {
            const GATE: f64 = 2e-4; // quad::MAX_UNDISTORT_RESIDUAL

            // g(r) = r − 0.1·r³ turns at r = 1.826, where g = 1.217.
            let m = BrownConradyModel {
                k1: -0.1,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            };

            for &r_d in &[1.25_f64, 1.335, 1.5, 2.0, 5.0] {
                let [xu, yu] = m.undistort(r_d, 0.0);
                let [xc, yc] = m.distort(xu, yu);
                let residual = (xc - r_d).hypot(yc);
                assert!(
                    xu.is_finite() && yu.is_finite(),
                    "undistort must stay finite at r_d = {r_d}, got ({xu}, {yu})"
                );
                assert!(
                    xu >= 0.0,
                    "undistort returned the mirrored branch at r_d = {r_d}: xu = {xu}"
                );
                assert!(
                    residual > GATE,
                    "r_d = {r_d} is past the turning point, so the round-trip residual \
                     must exceed the caller's gate ({GATE:e}) and reject the point; \
                     got {residual:.3e} for r_u = {xu}"
                );
                assert!(
                    m.undistort_checked(r_d, 0.0).is_none(),
                    "undistort_checked must report failure past the turning point \
                     (r_d = {r_d}); every geometry consumer depends on it"
                );
            }

            // Just inside the branch it must still invert exactly.
            for &r_d in &[0.2_f64, 0.8, 1.2] {
                let [xu, yu] = m.undistort(r_d, 0.0);
                let [xc, yc] = m.distort(xu, yu);
                assert!(
                    (xc - r_d).hypot(yc) < 1e-12,
                    "r_d = {r_d} is inside the invertible branch and must round-trip"
                );
                assert!(
                    m.undistort_checked(r_d, 0.0).is_some(),
                    "undistort_checked must accept r_d = {r_d}, inside the branch"
                );
            }
        }

        #[test]
        fn brown_conrady_undistort_stays_finite_at_extreme_radius() {
            let m = hub_brown_conrady();
            for &(xd, yd) in &[(5.0, 5.0), (50.0, 0.0), (0.0, 100.0), (1e8, -1e8)] {
                let out = m.undistort(xd, yd);
                assert!(
                    out[0].is_finite() && out[1].is_finite(),
                    "undistort must stay finite at extreme radius, got {out:?} for ({xd}, {yd})"
                );
            }
        }

        #[test]
        fn brown_conrady_from_coeffs_validates_length() {
            assert!(BrownConradyModel::from_coeffs(&[0.0; 4]).is_err());
            assert!(BrownConradyModel::from_coeffs(&[0.0; 5]).is_ok());
            assert!(BrownConradyModel::from_coeffs(&[0.0; 6]).is_err());
        }

        #[test]
        fn kannala_brandt_from_coeffs_validates_length() {
            assert!(KannalaBrandtModel::from_coeffs(&[0.0; 3]).is_err());
            assert!(KannalaBrandtModel::from_coeffs(&[0.0; 4]).is_ok());
            assert!(KannalaBrandtModel::from_coeffs(&[0.0; 5]).is_err());
        }
    }
}
