use locus_core::gwlf::{MomentAccumulator, solve_2x2_symmetric_min_eigen};

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
