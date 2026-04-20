#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::items_after_statements,
    missing_docs
)]
//! Property: the homography produced by `Homography::from_pairs` projects its
//! source points back to the destination points within DLT tolerance, and the
//! DDA ("Digital Differential Analyzer") reproduction of the grid agrees with
//! the direct matrix projection at every grid point within sub-pixel tolerance.
//!
//! The DDA is the SIMD-friendly perspective-divide helper used in the
//! bilinear-sampling kernel; its correctness is load-bearing for decoding.
//! Stepping through a grid by summing `(dnx_du, dny_du, dd_du)` and
//! `(dnx_dv, dny_dv, dd_dv)` must exactly mirror `H · [u, v, 1]` up to float
//! rounding.

use locus_core::bench_api::Homography;
use proptest::prelude::*;

/// Absolute sub-pixel tolerance for the DDA vs. direct projection comparison.
///
/// DDA accumulates rounding error; the ratio at each perspective divide is
/// stable, but summing partial derivatives is not. 1e-6 is tight enough to
/// catch a bogus derivative and loose enough for any reasonable FMA path.
const DDA_TOLERANCE_PX: f64 = 1e-6;

/// Reconstruction tolerance for `from_pairs` — must match the internal DLT
/// reconstruction tolerance in `Homography::from_pairs`.
const DLT_TOLERANCE_PX: f64 = 1e-2;

const GRID_STEPS: usize = 16;

/// A non-degenerate destination quad: start from a canonical rectangle and
/// perturb each corner by at most 10 pixels. Guarantees convexity by
/// construction for small perturbations.
fn dst_quad_strategy() -> impl Strategy<Value = [[f64; 2]; 4]> {
    let base = [
        [100.0, 100.0],
        [400.0, 100.0],
        [400.0, 400.0],
        [100.0, 400.0],
    ];
    proptest::array::uniform8(-10.0f64..10.0f64).prop_map(move |d| {
        [
            [base[0][0] + d[0], base[0][1] + d[1]],
            [base[1][0] + d[2], base[1][1] + d[3]],
            [base[2][0] + d[4], base[2][1] + d[5]],
            [base[3][0] + d[6], base[3][1] + d[7]],
        ]
    })
}

/// Project a canonical tag-space point through `h` using the DDA stepper.
///
/// `dda` is initialised at (u0, v0) and may be stepped by (i, j) units of
/// (du, dv). Summing `i * (dnx_du, dny_du, dd_du) + j * (dnx_dv, dny_dv, dd_dv)`
/// onto the base state and dividing by `d` reconstructs `H · [u0 + i*du, v0 + j*dv, 1]`.
fn dda_at_step(dda: &locus_core::bench_api::HomographyDda, i: usize, j: usize) -> [f64; 2] {
    let i = i as f64;
    let j = j as f64;
    let nx = dda.nx + i * dda.dnx_du + j * dda.dnx_dv;
    let ny = dda.ny + i * dda.dny_du + j * dda.dny_dv;
    let d = dda.d + i * dda.dd_du + j * dda.dd_dv;
    [nx / d, ny / d]
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "proptest-regressions/homography_roundtrip.txt",
            ),
        )),
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// The DLT must round-trip the four source→dst correspondences exactly.
    #[test]
    fn prop_from_pairs_roundtrip(dst in dst_quad_strategy()) {
        let src: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let Some(h) = Homography::from_pairs(&src, &dst) else {
            // Degenerate; not a property violation.
            return Ok(());
        };
        for (i, (s, d)) in src.iter().zip(dst.iter()).enumerate() {
            let p = h.project(*s);
            let dx = p[0] - d[0];
            let dy = p[1] - d[1];
            let err = (dx * dx + dy * dy).sqrt();
            prop_assert!(
                err <= DLT_TOLERANCE_PX,
                "corner {i}: from_pairs round-trip error {err} > {DLT_TOLERANCE_PX}",
            );
        }
    }

    /// The DDA stepper agrees with direct matrix projection at every point on
    /// a 16×16 grid.
    #[test]
    fn prop_dda_matches_direct_projection(dst in dst_quad_strategy()) {
        let src: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let Some(h) = Homography::from_pairs(&src, &dst) else {
            return Ok(());
        };

        let u0 = -1.0;
        let v0 = -1.0;
        let du = 2.0 / (GRID_STEPS - 1) as f64;
        let dv = 2.0 / (GRID_STEPS - 1) as f64;
        let dda = h.to_dda(u0, v0, du, dv);

        let mut worst_err = 0.0f64;
        for j in 0..GRID_STEPS {
            for i in 0..GRID_STEPS {
                let u = u0 + (i as f64) * du;
                let v = v0 + (j as f64) * dv;
                let direct = h.project([u, v]);
                let via_dda = dda_at_step(&dda, i, j);
                let dx = via_dda[0] - direct[0];
                let dy = via_dda[1] - direct[1];
                let err = (dx * dx + dy * dy).sqrt();
                worst_err = worst_err.max(err);
            }
        }
        prop_assert!(
            worst_err <= DDA_TOLERANCE_PX,
            "worst DDA vs direct error {worst_err} > {DDA_TOLERANCE_PX}",
        );
    }
}
