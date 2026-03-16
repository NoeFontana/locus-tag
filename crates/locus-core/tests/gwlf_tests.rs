//! Unit tests for Gradient-Weighted Line Fitting (GWLF) components.
use locus_core::ImageView;
use locus_core::gwlf::{
    HomogeneousLine, MomentAccumulator, refine_quad_gwlf, solve_2x2_symmetric_min_eigen,
};

#[test]
fn test_gwlf_refinement_synthetic_square() {
    let width = 100;
    let height = 100;
    let mut data = vec![0u8; width * height];

    // Create a 40x40 white square at (20, 20) to (60, 60)
    for y in 20..60 {
        for x in 20..60 {
            data[y * width + x] = 255;
        }
    }

    let view = ImageView::new(&data, width, height, width).expect("valid image");

    // Coarse corners slightly jittered from (20,20), (60,20), (60,60), (20,60)
    let coarse = [[21.5, 19.5], [59.0, 21.0], [60.5, 60.5], [19.0, 59.0]];

    let refined = refine_quad_gwlf(&view, &coarse).expect("refinement should succeed");

    // Should be close to the true edges
    assert!((refined[0][0] - 19.5).abs() < 1.0);
    assert!((refined[0][1] - 19.5).abs() < 1.0);
    assert!((refined[2][0] - 59.5).abs() < 1.0);
    assert!((refined[2][1] - 59.5).abs() < 1.0);
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
    let refined = refine_quad_gwlf(&view, &coarse);
    assert!(refined.is_none());
}

#[test]
fn test_line_intersection() {
    // x = 5 (Vertical line: 1*x + 0*y - 5 = 0)
    let l1 = HomogeneousLine {
        nx: 1.0,
        ny: 0.0,
        d: -5.0,
    };
    // y = 10 (Horizontal line: 0*x + 1*y - 10 = 0)
    let l2 = HomogeneousLine {
        nx: 0.0,
        ny: 1.0,
        d: -10.0,
    };

    let (ix, iy) = l1.intersect(&l2).expect("should intersect");
    assert!((ix - 5.0).abs() < 1e-9);
    assert!((iy - 10.0).abs() < 1e-9);
}

#[test]
fn test_moment_accumulation_horizontal_edge() {
    let mut acc = MomentAccumulator::new();

    // Create a horizontal edge at y = 10.0
    // Points (0, 10), (1, 10), (2, 10) all with weight 1.0
    acc.add(0.0, 10.0, 1.0);
    acc.add(1.0, 10.0, 1.0);
    acc.add(2.0, 10.0, 1.0);

    let (cx, cy) = acc.centroid().expect("has points");
    assert!((cx - 1.0).abs() < 1e-9);
    assert!((cy - 10.0).abs() < 1e-9);

    let cov = acc.covariance().expect("has points");
    // sigma_xx = (0^2 + 1^2 + 2^2)/3 - 1^2 = 5/3 - 1 = 2/3
    assert!((cov[0] - 0.666_666_666).abs() < 1e-6);
    assert!(cov[1].abs() < 1e-9);
    assert!(cov[2].abs() < 1e-9);
    assert!(cov[3].abs() < 1e-9);
}

#[test]
fn test_analytic_eigendecomposition() {
    // Identity matrix: eigenvalues 1, 1. Smallest is 1.
    let res = solve_2x2_symmetric_min_eigen(1.0, 0.0, 1.0);
    assert!((res.min_eigenvalue - 1.0).abs() < 1e-9);

    // Diagonal matrix [[1, 0], [0, 2]]: eigenvalues 1, 2. Smallest is 1, vector (1, 0)
    let res = solve_2x2_symmetric_min_eigen(1.0, 0.0, 2.0);
    assert!((res.min_eigenvalue - 1.0).abs() < 1e-9);
    assert!((res.min_eigenvector.0 - 1.0).abs() < 1e-9);
    assert!(res.min_eigenvector.1.abs() < 1e-9);

    // Diagonal matrix [[2, 0], [0, 1]]: eigenvalues 1, 2. Smallest is 1, vector (0, 1)
    let res = solve_2x2_symmetric_min_eigen(2.0, 0.0, 1.0);
    assert!((res.min_eigenvalue - 1.0).abs() < 1e-9);
    assert!(res.min_eigenvector.0.abs() < 1e-9);
    assert!((res.min_eigenvector.1 - 1.0).abs() < 1e-9);

    // Symmetric matrix [[2, 1], [1, 2]]:
    // lambda_min = 1.
    let res = solve_2x2_symmetric_min_eigen(2.0, 1.0, 2.0);
    assert!((res.min_eigenvalue - 1.0).abs() < 1e-9);
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert!((res.min_eigenvector.0.abs() - inv_sqrt2).abs() < 1e-5);
    assert!((res.min_eigenvector.1.abs() - inv_sqrt2).abs() < 1e-5);
    assert!((res.min_eigenvector.0 + res.min_eigenvector.1).abs() < 1e-5);
}
