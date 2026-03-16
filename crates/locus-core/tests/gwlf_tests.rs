use locus_core::gwlf::{MomentAccumulator, solve_2x2_symmetric_min_eigen, HomogeneousLine, refine_quad_gwlf};
use locus_core::ImageView;

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
    
    let view = ImageView::new(&data, width, height, width).unwrap();
    
    // Coarse corners slightly jittered from (20,20), (60,20), (60,60), (20,60)
    let coarse = [
        [21.5, 19.5],
        [59.0, 21.0],
        [60.5, 60.5],
        [19.0, 59.0],
    ];
    
    let refined = refine_quad_gwlf(&view, &coarse).unwrap();
    
    // Should be close to the true edges (19.5/20.5 depending on gradient implementation)
    // Finite difference on a sharp edge will put the max gradient between pixels.
    // For a transition from 0 (at 19) to 255 (at 20), (I(21)-I(19)) = 255.
    // Centroid will be around 20.0.
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
    let view = ImageView::new(&data, width, height, width).unwrap();
    
    // Corners very far from any edges (all 0 image)
    let coarse = [
        [20.0, 20.0],
        [60.0, 20.0],
        [60.0, 60.0],
        [20.0, 60.0],
    ];
    
    // Should return None because no gradients found or sanity gate triggered
    let refined = refine_quad_gwlf(&view, &coarse);
    assert!(refined.is_none());
}

#[test]
fn test_line_intersection() {
    // x = 5 (Vertical line: 1*x + 0*y - 5 = 0)
    let l1 = HomogeneousLine { nx: 1.0, ny: 0.0, d: -5.0 };
    // y = 10 (Horizontal line: 0*x + 1*y - 10 = 0)
    let l2 = HomogeneousLine { nx: 0.0, ny: 1.0, d: -10.0 };
    
    let (ix, iy) = l1.intersect(&l2).unwrap();
    assert_eq!(ix, 5.0);
    assert_eq!(iy, 10.0);
}

#[test]
fn test_moment_accumulation_horizontal_edge() {
    let mut acc = MomentAccumulator::new();
    
    // Create a horizontal edge at y = 10.0
    // Points (0, 10), (1, 10), (2, 10) all with weight 1.0
    acc.add(0.0, 10.0, 1.0);
    acc.add(1.0, 10.0, 1.0);
    acc.add(2.0, 10.0, 1.0);
    
    let (cx, cy) = acc.centroid().unwrap();
    assert_eq!(cx, 1.0);
    assert_eq!(cy, 10.0);
    
    let cov = acc.covariance().unwrap();
    // sigma_xx = (0^2 + 1^2 + 2^2)/3 - 1^2 = 5/3 - 1 = 2/3
    // sigma_yy = (10^2 + 10^2 + 10^2)/3 - 10^2 = 0
    // sigma_xy = (0*10 + 1*10 + 2*10)/3 - 1*10 = 30/3 - 10 = 0
    assert!((cov[0] - 0.666666666).abs() < 1e-6);
    assert_eq!(cov[1], 0.0);
    assert_eq!(cov[2], 0.0);
    assert_eq!(cov[3], 0.0);
}

#[test]
fn test_analytic_eigendecomposition() {
    // Identity matrix: eigenvalues 1, 1. Smallest is 1.
    let res = solve_2x2_symmetric_min_eigen(1.0, 0.0, 1.0);
    assert_eq!(res.min_eigenvalue, 1.0);
    
    // Diagonal matrix [[1, 0], [0, 2]]: eigenvalues 1, 2. Smallest is 1, vector (1, 0)
    let res = solve_2x2_symmetric_min_eigen(1.0, 0.0, 2.0);
    assert_eq!(res.min_eigenvalue, 1.0);
    assert_eq!(res.min_eigenvector, (1.0, 0.0));
    
    // Diagonal matrix [[2, 0], [0, 1]]: eigenvalues 1, 2. Smallest is 1, vector (0, 1)
    let res = solve_2x2_symmetric_min_eigen(2.0, 0.0, 1.0);
    assert_eq!(res.min_eigenvalue, 1.0);
    assert_eq!(res.min_eigenvector, (0.0, 1.0));
    
    // Symmetric matrix [[2, 1], [1, 2]]:
    // trace = 4, det = 3.
    // lambda^2 - 4*lambda + 3 = 0 -> (lambda-3)(lambda-1) = 0
    // lambda_min = 1.
    // (2 - 1)nx + 1*ny = 0 -> nx + ny = 0. e.g. (1/sqrt(2), -1/sqrt(2))
    let res = solve_2x2_symmetric_min_eigen(2.0, 1.0, 2.0);
    assert_eq!(res.min_eigenvalue, 1.0);
    assert!((res.min_eigenvector.0.abs() - 0.707106).abs() < 1e-5);
    assert!((res.min_eigenvector.1.abs() - 0.707106).abs() < 1e-5);
    assert!((res.min_eigenvector.0 + res.min_eigenvector.1).abs() < 1e-5);
}
