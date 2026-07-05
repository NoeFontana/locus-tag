#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Unit tests for Gradient-Weighted Line Fitting (GWLF) components.
use locus_core::ImageView;
use locus_core::bench_api::{
    HomogeneousLine, MomentAccumulator, refine_quad_gwlf, solve_2x2_symmetric,
};
use nalgebra::{Matrix3, Vector3};

/// Max Euclidean corner error (px) of a refined quad vs. its ground-truth corners.
fn max_corner_error(refined: &[[f32; 2]; 4], truth: &[[f64; 2]; 4]) -> f64 {
    (0..4)
        .map(|k| {
            (f64::from(refined[k][0]) - truth[k][0]).hypot(f64::from(refined[k][1]) - truth[k][1])
        })
        .fold(0.0f64, f64::max)
}

#[test]
fn test_gwlf_refinement_synthetic_square() {
    // Per-seed accuracy gate (see rationale below).
    const GATE_PX: f64 = 0.2;

    let width = 100;
    let height = 100;
    let mut data = vec![0u8; width * height];

    // 40x40 white square filling pixel columns/rows [20, 60) - its geometric
    // edges sit exactly at 20.0 and 60.0 in pixel-corner coordinates.
    for y in 20..60 {
        for x in 20..60 {
            data[y * width + x] = 255;
        }
    }
    let view = ImageView::new(&data, width, height, width).expect("valid image");
    let truth = [[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0]];

    // GWLF is sub-pixel accurate, but exhibits a small seed-position-dependent
    // quantization bias: the transversal intensity profile is sampled on an
    // integer grid offset from the seed. Measured worst case over symmetric
    // seeds {0.3, 0.5, 0.7, 1.0} px is ~0.16 px (usually < 0.05 px). We gate
    // every seed at 0.2 px - 5x tighter than the previous 1.0 px smoke
    // tolerance and enough to catch any real regression - and separately
    // require the refinement to *improve* on its seed. (Tighter, ~0.02 px
    // sub-pixel accuracy on anti-aliased ERF edges is covered by
    // `edge_refinement.rs`; a hard binary step carries no finer sub-pixel
    // information.)
    for j in [0.3f32, 0.5, 0.7, 1.0] {
        // Symmetric inward seed: each corner pushed toward the square center by j.
        let seed = [
            [20.0 + j, 20.0 + j],
            [60.0 - j, 20.0 + j],
            [60.0 - j, 60.0 - j],
            [20.0 + j, 60.0 - j],
        ];
        let refined = refine_quad_gwlf(&view, &seed, 0.01).expect("refinement should succeed");

        let refined_err = max_corner_error(&refined, &truth);
        let seed_err = max_corner_error(&seed, &truth);
        assert!(
            refined_err < GATE_PX,
            "seed +/-{j}px: refined corner error {refined_err:.4}px exceeds {GATE_PX}px gate \
             (refined = {refined:?})"
        );
        assert!(
            refined_err < seed_err,
            "seed +/-{j}px: refinement did not improve on seed \
             (refined {refined_err:.4}px vs seed {seed_err:.4}px)"
        );
    }
}

#[test]
fn test_gwlf_refinement_is_idempotent() {
    // Refining an already-refined quad must be (near) a fixed point: a second
    // GWLF pass should barely move the corners. A large second-pass drift would
    // signal an unstable line fit or a coordinate-convention mismatch between
    // the seed and the output.
    let width = 100;
    let height = 100;
    let mut data = vec![0u8; width * height];
    for y in 20..60 {
        for x in 20..60 {
            data[y * width + x] = 255;
        }
    }
    let view = ImageView::new(&data, width, height, width).expect("valid image");

    let seed = [[20.5, 20.5], [59.5, 20.5], [59.5, 59.5], [20.5, 59.5]];
    let pass1 = refine_quad_gwlf(&view, &seed, 0.01).expect("first pass");
    let pass2 = refine_quad_gwlf(&view, &pass1, 0.01).expect("second pass");

    // Second-pass drift vs the first refinement (both f32 corner arrays).
    let drift = (0..4)
        .map(|k| {
            (f64::from(pass2[k][0]) - f64::from(pass1[k][0]))
                .hypot(f64::from(pass2[k][1]) - f64::from(pass1[k][1]))
        })
        .fold(0.0f64, f64::max);

    // Measured ~0.017 px (residual seed-quantization bias, see
    // `test_gwlf_refinement_synthetic_square`); gate at 0.05 px.
    assert!(
        drift < 0.05,
        "second GWLF pass drifted {drift:.4}px from the first — refinement is not a fixed point"
    );
}

#[test]
fn test_gwlf_sanity_gate_fallback() {
    let width = 100;
    let height = 100;
    let data = vec![0u8; width * height];
    let view = ImageView::new(&data, width, height, width).expect("valid image");

    // Corners very far from any edges (all 0 image)
    let coarse = [[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0]];

    // Should return None because no gradients found or sanity gate triggered
    let refined = refine_quad_gwlf(&view, &coarse, 0.01);
    assert!(refined.is_none());
}

#[test]
fn test_line_intersection() {
    // x = 5 (Vertical line: 1*x + 0*y - 5 = 0)
    let l1 = HomogeneousLine {
        l: Vector3::new(1.0, 0.0, -5.0),
        cov: Matrix3::zeros(),
    };
    // y = 10 (Horizontal line: 0*x + 1*y - 10 = 0)
    let l2 = HomogeneousLine {
        l: Vector3::new(0.0, 1.0, -10.0),
        cov: Matrix3::zeros(),
    };

    let (corner, _cov) = l1.intersect(&l2).expect("should intersect");
    assert!((corner.x - 5.0).abs() < 1e-9);
    assert!((corner.y - 10.0).abs() < 1e-9);
}

#[test]
fn test_moment_accumulation_horizontal_edge() {
    let mut acc = MomentAccumulator::new();

    // Create a horizontal edge at y = 10.0
    // Points (0, 10), (1, 10), (2, 10) all with weight 1.0
    acc.add(0.0, 10.0, 1.0);
    acc.add(1.0, 10.0, 1.0);
    acc.add(2.0, 10.0, 1.0);

    let centroid = acc.centroid().expect("has points");
    assert!((centroid.x - 1.0).abs() < 1e-9);
    assert!((centroid.y - 10.0).abs() < 1e-9);

    let cov = acc.covariance().expect("has points");
    // sigma_xx = (0^2 + 1^2 + 2^2)/3 - 1^2 = 5/3 - 1 = 2/3
    assert!((cov[(0, 0)] - 0.666_666_666).abs() < 1e-6);
    assert!(cov[(0, 1)].abs() < 1e-9);
    assert!(cov[(1, 0)].abs() < 1e-9);
    assert!(cov[(1, 1)].abs() < 1e-9);
}

#[test]
fn test_analytic_eigendecomposition() {
    // Identity matrix: eigenvalues 1, 1. Smallest is 1.
    let res = solve_2x2_symmetric(1.0, 0.0, 1.0);
    assert!((res.l_min - 1.0).abs() < 1e-9);

    // Diagonal matrix [[1, 0], [0, 2]]: eigenvalues 1, 2. Smallest is 1, vector (1, 0) or (0, 1) depending on a < c
    // a=1, b=0, c=2. det=2, trace=3. disc=sqrt(9-8)=1. l_max=(3+1)/2=2, l_min=(3-1)/2=1.
    // v_min for a=1, c=2 is (1, 0)
    let res = solve_2x2_symmetric(1.0, 0.0, 2.0);
    assert!((res.l_min - 1.0).abs() < 1e-9);
    assert!((res.v_min.x - 1.0).abs() < 1e-9);
    assert!(res.v_min.y.abs() < 1e-9);

    // Diagonal matrix [[2, 0], [0, 1]]: eigenvalues 1, 2. Smallest is 1, vector (0, 1)
    let res = solve_2x2_symmetric(2.0, 0.0, 1.0);
    assert!((res.l_min - 1.0).abs() < 1e-9);
    assert!(res.v_min.x.abs() < 1e-9);
    assert!((res.v_min.y - 1.0).abs() < 1e-9);

    // Symmetric matrix [[2, 1], [1, 2]]:
    // lambda_min = 1.
    let res = solve_2x2_symmetric(2.0, 1.0, 2.0);
    assert!((res.l_min - 1.0).abs() < 1e-9);
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert!((res.v_min.x.abs() - inv_sqrt2).abs() < 1e-5);
    assert!((res.v_min.y.abs() - inv_sqrt2).abs() < 1e-5);
    // Since b=1 > 0, v_min = normalize(b, l_min - a) = normalize(1, 1 - 2) = normalize(1, -1)
    assert!((res.v_min.x + res.v_min.y).abs() < 1e-5);
}
