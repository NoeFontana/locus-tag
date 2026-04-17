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
}

// ---------------------------------------------------------------------------
// PinholeModel — ideal rectified camera, zero runtime cost
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// BrownConradyModel — standard polynomial radial + tangential distortion
// ---------------------------------------------------------------------------

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
        // Iterative Newton refinement starting from the distorted point as initial guess.
        // 5 iterations is sufficient for sub-pixel accuracy under typical distortion magnitudes.
        let mut xu = xd;
        let mut yu = yd;
        for _ in 0..5 {
            let r2 = xu * xu + yu * yu;
            let r4 = r2 * r2;
            let r6 = r2 * r4;
            let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
            let dx = 2.0 * self.p1 * xu * yu + self.p2 * (r2 + 2.0 * xu * xu);
            let dy = self.p1 * (r2 + 2.0 * yu * yu) + 2.0 * self.p2 * xu * yu;
            xu = (xd - dx) / radial.max(1e-8);
            yu = (yd - dy) / radial.max(1e-8);
        }
        [xu, yu]
    }

    #[allow(clippy::similar_names)]
    fn distort_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2] {
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        // ∂radial/∂r² = k1 + 2·k2·r² + 3·k3·r⁴
        let d_radial_dr2 = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;

        // ∂xd/∂xn:
        //  xd = xn·radial + 2p1·xn·yn + p2·(r²+2xn²)
        //  ∂xd/∂xn = radial + xn·(∂radial/∂xn) + 2p1·yn + p2·(2xn + 4xn)
        //           = radial + 2xn²·d_radial_dr2 + 2p1·yn + 6p2·xn
        let dxd_dxn =
            radial + 2.0 * xn * xn * d_radial_dr2 + 2.0 * self.p1 * yn + 6.0 * self.p2 * xn;

        // ∂xd/∂yn:
        //  ∂xd/∂yn = xn·(∂radial/∂yn) + 2p1·xn + p2·2yn
        //           = 2xn·yn·d_radial_dr2 + 2p1·xn + 2p2·yn
        let dxd_dyn = 2.0 * xn * yn * d_radial_dr2 + 2.0 * self.p1 * xn + 2.0 * self.p2 * yn;

        // ∂yd/∂xn:
        //  yd = yn·radial + p1·(r²+2yn²) + 2p2·xn·yn
        //  ∂yd/∂xn = yn·(∂radial/∂xn) + p1·2xn + 2p2·yn
        //           = 2xn·yn·d_radial_dr2 + 2p1·xn + 2p2·yn
        let dyd_dxn = 2.0 * xn * yn * d_radial_dr2 + 2.0 * self.p1 * xn + 2.0 * self.p2 * yn;

        // ∂yd/∂yn:
        //  ∂yd/∂yn = radial + yn·(∂radial/∂yn) + p1·(2yn+4yn) + 2p2·xn
        //           = radial + 2yn²·d_radial_dr2 + 6p1·yn + 2p2·xn
        let dyd_dyn =
            radial + 2.0 * yn * yn * d_radial_dr2 + 6.0 * self.p1 * yn + 2.0 * self.p2 * xn;

        [[dxd_dxn, dxd_dyn], [dyd_dxn, dyd_dyn]]
    }
}

// ---------------------------------------------------------------------------
// KannalaBrandtModel — equidistant fisheye projection
// ---------------------------------------------------------------------------

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

impl CameraModel for KannalaBrandtModel {
    const IS_RECTIFIED: bool = false;

    fn distort(&self, xn: f64, yn: f64) -> [f64; 2] {
        let r = (xn * xn + yn * yn).sqrt();
        if r < 1e-8 {
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
        if r_d < 1e-8 {
            return [xd, yd];
        }
        // Invert θ_d = poly(θ) via Newton's method to recover θ, then r = tan(θ).
        let mut theta = r_d; // initial guess: identity
        for _ in 0..10 {
            let (theta_d, d_theta_d) = self.angle_poly(theta);
            let f = theta_d - r_d;
            let df = d_theta_d.max(1e-8);
            theta -= f / df;
            theta = theta.max(0.0);
        }
        // r_undistorted = tan(θ)
        let r_undist = theta.tan();
        let scale = r_undist / r_d;
        [xd * scale, yd * scale]
    }

    #[allow(clippy::similar_names)]
    fn distort_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2] {
        let r2 = xn * xn + yn * yn;
        let r = r2.sqrt();
        if r < 1e-8 {
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

    /// Round-trip identity: distort then undistort should recover the original point.
    fn check_roundtrip<C: CameraModel>(model: &C, xn: f64, yn: f64, tol: f64) {
        let [xd, yd] = model.distort(xn, yn);
        let [xu, yu] = model.undistort(xd, yd);
        assert!((xu - xn).abs() < tol, "xn round-trip failed: {xu} vs {xn}");
        assert!((yu - yn).abs() < tol, "yn round-trip failed: {yu} vs {yn}");
    }

    /// Numerical Jacobian check via finite differences.
    #[allow(clippy::similar_names)]
    fn check_jacobian<C: CameraModel>(model: &C, xn: f64, yn: f64) {
        let eps = 1e-6;
        let jac = model.distort_jacobian(xn, yn);

        let [x0, y0] = model.distort(xn, yn);
        let [x1, _] = model.distort(xn + eps, yn);
        let [_, y1] = model.distort(xn, yn + eps);
        let [x2, _] = model.distort(xn, yn + eps);
        let [_, y2] = model.distort(xn + eps, yn);

        let num_dxd_dxn = (x1 - x0) / eps;
        let num_dxd_dyn = (x2 - x0) / eps;
        let num_dyd_dxn = (y2 - y0) / eps;
        let num_dyd_dyn = (y1 - y0) / eps;

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
    fn pinhole_is_identity() {
        let m = PinholeModel;
        let [xd, yd] = m.distort(0.3, -0.2);
        assert!((xd - 0.3).abs() < f64::EPSILON);
        assert!((yd - (-0.2)).abs() < f64::EPSILON);
        let [xu, yu] = m.undistort(0.3, -0.2);
        assert!((xu - 0.3).abs() < f64::EPSILON);
        assert!((yu - (-0.2)).abs() < f64::EPSILON);
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
            // 5 Newton iterations; tolerance 1e-7 is well within sub-pixel accuracy.
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
