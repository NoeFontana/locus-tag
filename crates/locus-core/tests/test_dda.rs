//! Tests for the Homography Digital Differential Analyzer (DDA).
use locus_core::bench_api::*;

#[test]
fn test_dda_step_calculation() {
    let dst = [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]];
    let h = Homography::square_to_quad(&dst).expect("valid homography");

    // Grid size for AprilTag 36h11 is 8x8
    let grid_size: i32 = 8;
    let delta = 2.0 / f64::from(grid_size - 1);

    let dda = h.to_dda(-1.0, -1.0, delta, delta);

    // Initial point (-1, -1)
    let mut nx = dda.nx;
    let mut ny = dda.ny;
    let mut d = dda.d;

    // Verify first point matches project(-1, -1)
    let p0 = h.project([-1.0, -1.0]);
    assert!((nx / d - p0[0]).abs() < 1e-6);
    assert!((ny / d - p0[1]).abs() < 1e-6);

    // Step u
    nx += dda.dnx_du;
    ny += dda.dny_du;
    d += dda.dd_du;

    // Verify second point matches project(-1 + delta, -1)
    let p1 = h.project([-1.0 + delta, -1.0]);
    assert!((nx / d - p1[0]).abs() < 1e-6);
    assert!((ny / d - p1[1]).abs() < 1e-6);

    // Step v (start from first point)
    let nx = dda.nx + dda.dnx_dv;
    let ny = dda.ny + dda.dny_dv;
    let d = dda.d + dda.dd_dv;

    let p_v = h.project([-1.0, -1.0 + delta]);
    assert!((nx / d - p_v[0]).abs() < 1e-6);
    assert!((ny / d - p_v[1]).abs() < 1e-6);
}
