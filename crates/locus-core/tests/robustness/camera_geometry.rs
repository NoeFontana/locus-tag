#![cfg(feature = "non_rectified")]
#![allow(missing_docs)]
//! Tier A proptest invariants for camera distortion models.
//!
//! Covers:
//! - Round-trip identity: `distort(undistort(xd, yd)) ≈ (xd, yd)` within
//!   `1e-4` normalized units (≈ 0.08 px at fx=800 / ≈ 0.125 px at fx=500).
//!   This is the *downstream-impact* gate — residual round-trip error
//!   propagates directly into corner localization accuracy, so we pin it
//!   to sub-0.1-pixel at typical calibrations. The tested envelope is
//!   deliberately narrowed to the regime where 5-iter (BC) / 10-iter (KB)
//!   Newton converges to that level; the extractor's looser divergence
//!   guard (`MAX_RESIDUAL` in `quad.rs`) handles the tail.
//! - Collinearity preservation: a straight 3D line segment, projected through
//!   a distorted camera and then undistorted, recovers an approximately
//!   straight 2D arc (Pearson correlation ≥ `1 - 1e-6`).
//!
//! Both properties are load-bearing for the straight-space quad extraction
//! in `extract_quads_soa_with_camera`.

use locus_core::camera::{BrownConradyModel, CameraModel, KannalaBrandtModel};
use proptest::prelude::*;

/// Pearson correlation on two equal-length f64 slices. Used as a numerical
/// collinearity metric: a perfectly straight 2D point set in the undistorted
/// plane yields `|r| = 1.0`; departures < `1e-9` cover floating-point noise
/// plus the residual of the iterative undistort.
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mx;
        let dy = y - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom < 1e-18 {
        return 1.0;
    }
    sxy / denom
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "proptest-regressions/camera_geometry.txt",
            ),
        )),
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// Round-trip: the Newton undistort inverts distort to within 1e-4
    /// (≈ 0.08 px at fx=800) across a realistic Brown-Conrady envelope.
    /// Coefficient and coordinate bounds are narrowed to the regime where
    /// the 5-iter Newton solve converges to sub-0.1-px — beyond that
    /// window, the extractor's `MAX_RESIDUAL` guard handles bail-out.
    #[test]
    fn prop_brown_conrady_roundtrip(
        k1 in -0.25_f64..0.25,
        k2 in -0.08_f64..0.08,
        p1 in -0.003_f64..0.003,
        p2 in -0.003_f64..0.003,
        k3 in -0.03_f64..0.03,
        xn in -0.35_f64..0.35,
        yn in -0.35_f64..0.35,
    ) {
        let m = BrownConradyModel { k1, k2, p1, p2, k3 };
        let [xd, yd] = m.distort(xn, yn);
        let [xu, yu] = m.undistort(xd, yd);
        let dx = xu - xn;
        let dy = yu - yn;
        prop_assert!(
            (dx * dx + dy * dy).sqrt() < 1e-4,
            "BC round-trip failed at ({xn}, {yn}) with coeffs ({k1}, {k2}, {p1}, {p2}, {k3}): \
             recovered ({xu}, {yu})"
        );
    }

    /// Round-trip: same sub-0.1-px property for the Kannala-Brandt fisheye.
    /// Coordinate range extends further (KB's equidistant projection keeps
    /// Newton well-behaved at wider angles) but coefficients are tightened
    /// to keep the 10-iter solve inside the 1e-4 envelope.
    #[test]
    fn prop_kannala_brandt_roundtrip(
        k1 in -0.12_f64..0.12,
        k2 in -0.05_f64..0.05,
        k3 in -0.02_f64..0.02,
        k4 in -0.008_f64..0.008,
        xn in -0.7_f64..0.7,
        yn in -0.7_f64..0.7,
    ) {
        let m = KannalaBrandtModel { k1, k2, k3, k4 };
        let [xd, yd] = m.distort(xn, yn);
        let [xu, yu] = m.undistort(xd, yd);
        let dx = xu - xn;
        let dy = yu - yn;
        prop_assert!(
            (dx * dx + dy * dy).sqrt() < 1e-4,
            "KB round-trip failed at ({xn}, {yn}) with coeffs ({k1}, {k2}, {k3}, {k4}): \
             recovered ({xu}, {yu})"
        );
    }

    /// Collinearity preservation (Brown-Conrady): sample 100 points along a
    /// random 3D line segment, project through the distorted camera to get a
    /// curved 2D arc, undistort those arc points, and check that the result
    /// is numerically collinear in the undistorted plane.
    #[test]
    fn prop_brown_conrady_collinearity(
        k1 in -0.3_f64..0.3,
        k2 in -0.1_f64..0.1,
        p1 in -0.005_f64..0.005,
        p2 in -0.005_f64..0.005,
        k3 in -0.05_f64..0.05,
        a in -0.4_f64..0.4,
        b in -0.4_f64..0.4,
        c in -0.4_f64..0.4,
        d in -0.4_f64..0.4,
    ) {
        // Endpoints must be distinct.
        let dx_line = c - a;
        let dy_line = d - b;
        prop_assume!(dx_line * dx_line + dy_line * dy_line > 0.02);

        let m = BrownConradyModel { k1, k2, p1, p2, k3 };

        let n = 100;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            let xn = a + t * dx_line;
            let yn = b + t * dy_line;
            let [xd, yd] = m.distort(xn, yn);
            let [xu, yu] = m.undistort(xd, yd);
            xs.push(xu);
            ys.push(yu);
        }

        let r = pearson(&xs, &ys).abs();
        // 1-r should be dominated by floating-point + undistort residual noise.
        prop_assert!(
            r >= 1.0 - 1e-6 || (dx_line.abs() < 1e-6) || (dy_line.abs() < 1e-6),
            "BC collinearity violated: Pearson={r}"
        );
    }

    /// Collinearity preservation (Kannala-Brandt): same property under the
    /// equidistant fisheye projection. Notably lines not passing through the
    /// principal point bend visibly in pixel space but must straighten back
    /// out under `undistort`.
    #[test]
    fn prop_kannala_brandt_collinearity(
        k1 in -0.15_f64..0.15,
        k2 in -0.08_f64..0.08,
        k3 in -0.03_f64..0.03,
        k4 in -0.015_f64..0.015,
        a in -0.6_f64..0.6,
        b in -0.6_f64..0.6,
        c in -0.6_f64..0.6,
        d in -0.6_f64..0.6,
    ) {
        let dx_line = c - a;
        let dy_line = d - b;
        prop_assume!(dx_line * dx_line + dy_line * dy_line > 0.02);

        let m = KannalaBrandtModel { k1, k2, k3, k4 };

        let n = 100;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            let xn = a + t * dx_line;
            let yn = b + t * dy_line;
            let [xd, yd] = m.distort(xn, yn);
            let [xu, yu] = m.undistort(xd, yd);
            xs.push(xu);
            ys.push(yu);
        }

        let r = pearson(&xs, &ys).abs();
        prop_assert!(
            r >= 1.0 - 1e-6 || (dx_line.abs() < 1e-6) || (dy_line.abs() < 1e-6),
            "KB collinearity violated: Pearson={r}"
        );
    }
}
