use locus_core::decoder::Homography;
use proptest::prelude::*;

// ============================================================================
// Strategies
// ============================================================================

/// Strategy for a single 2D point [x, y] in a reasonable range.
fn point_strategy() -> impl Strategy<Value = [f64; 2]> {
    prop_oneof![
        // Standard coordinates (e.g. 640x480 or 4k)
        [(-2000.0..2000.0), (-2000.0..2000.0)],
        // Coordinates near 0
        [(-1.0..1.0), (-1.0..1.0)],
    ]
}

/// Strategy for invalid numerical values (NaN, Inf).
fn invalid_point_strategy() -> impl Strategy<Value = [[f64; 2]; 4]> {
    prop::collection::vec(
        prop_oneof![
            Just(f64::NAN),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
            (-1e10..1e10)
        ],
        8
    ).prop_map(|v| {
        [
            [v[0], v[1]],
            [v[2], v[3]],
            [v[4], v[5]],
            [v[6], v[7]],
        ]
    })
}

/// Helper to check if a quad is convex and CCW.
/// A quad (p0, p1, p2, p3) is convex if all internal angles are < 180 deg
/// and it's CCW if cross products of consecutive edges are positive.
fn is_convex_ccw(pts: &[[f64; 2]; 4]) -> bool {
    let mut cross_products = Vec::with_capacity(4);
    for i in 0..4 {
        let p0 = pts[i];
        let p1 = pts[(i + 1) % 4];
        let p2 = pts[(i + 2) % 4];

        let dx1 = p1[0] - p0[0];
        let dy1 = p1[1] - p0[1];
        let dx2 = p2[0] - p1[0];
        let dy2 = p2[1] - p1[1];

        let cp = dx1 * dy2 - dy1 * dx2;
        cross_products.push(cp);
    }

    // All cross products must have the same sign (convex)
    // and must be positive for CCW.
    cross_products.iter().all(|&cp| cp > 1e-9)
}

/// Strategy for a valid (convex, CCW) quadrilateral.
fn valid_quad_strategy() -> impl Strategy<Value = [[f64; 2]; 4]> {
    prop::collection::vec(point_strategy(), 4)
        .prop_map(|v| {
            let mut pts = [[0.0; 2]; 4];
            pts.copy_from_slice(&v);
            pts
        })
        .prop_filter("Must be convex and CCW", |pts| is_convex_ccw(pts))
}

/// Strategy for degenerate quads (strictly collinear or coincident).
fn collinear_quad_strategy() -> impl Strategy<Value = [[f64; 2]; 4]> {
    prop_oneof![
        // Coincident points: p0 == p1
        point_strategy().prop_map(|p| [p, p, [p[0] + 10.0, p[1]], [p[0], p[1] + 10.0]]),
        
        // Collinear points: p0, p1, p2 on the same line (y = p[1])
        point_strategy().prop_map(|p| {
            [p, [p[0] + 10.0, p[1]], [p[0] + 20.0, p[1]], [p[0] + 30.0, p[1]]]
        }),
        
        // All points same
        point_strategy().prop_map(|p| [p, p, p, p]),
    ]
}

// ============================================================================
// Properties
// ============================================================================

proptest! {
    /// Property: Homography correctly projects source points to destination points.
    #[test]
    fn test_homography_reprojection(
        src in valid_quad_strategy(),
        dst in valid_quad_strategy()
    ) {
        if let Some(h) = Homography::from_pairs(&src, &dst) {
            for i in 0..4 {
                let p_proj = h.project(src[i]);
                let err_x = (p_proj[0] - dst[i][0]).abs();
                let err_y = (p_proj[1] - dst[i][1]).abs();
                
                // Tolerance must scale with the magnitude of the coordinates.
                // For points ~2000, 1e-4 is reasonably tight for unnormalized DLT.
                let max_coord = dst[i][0].abs().max(dst[i][1].abs()).max(1.0);
                let tol = 1e-7 * max_coord;
                
                assert!(err_x < tol, "Reprojection error X too large: {} at point {} (tol: {})", err_x, i, tol);
                assert!(err_y < tol, "Reprojection error Y too large: {} at point {} (tol: {})", err_y, i, tol);
            }
        }
    }

    /// Property: square_to_quad is consistent with from_pairs using the unit square.
    #[test]
    fn test_square_to_quad_consistency(
        dst in valid_quad_strategy()
    ) {
        let src = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
        ];
        
        let h1 = Homography::from_pairs(&src, &dst);
        let h2 = Homography::square_to_quad(&dst);
        
        match (h1.as_ref(), h2.as_ref()) {
            (Some(h1_ref), Some(h2_ref)) => {
                // Test a sample point
                let p = [0.5, -0.2];
                let p1 = h1_ref.project(p);
                let p2 = h2_ref.project(p);
                
                assert!((p1[0] - p2[0]).abs() < 1e-9);
                assert!((p1[1] - p2[1]).abs() < 1e-9);
            },
            (None, None) => {},
            _ => panic!("Consistency mismatch between from_pairs and square_to_quad"),
        }
    }

    /// Property: Collinear points (Phase 2: Line Test) MUST return None.
    #[test]
    fn test_homography_line_test_singularity(
        src in collinear_quad_strategy(),
        dst in valid_quad_strategy()
    ) {
        let h = Homography::from_pairs(&src, &dst);
        
        // Rigorous check: Collinear source points should NOT result in a valid homography.
        // If it returns Some, it's a "ghost" solution due to numerical noise in LU.
        if let Some(_h_val) = h {
            // Check reprojection error - it should be high or it's a miracle
            // Actually, we expect it to be None.
            panic!("DLT returned a solution for collinear points: {:?}", src);
        }
    }

    /// Property: Invalid numbers (NaN, Inf) never panic (Phase 2: NaN Check).
    #[test]
    fn test_homography_nan_safety(
        src in invalid_point_strategy(),
        dst in invalid_point_strategy()
    ) {
        // This should never panic
        let _ = Homography::from_pairs(&src, &dst);
        let _ = Homography::square_to_quad(&src);
    }

    /// Property: Extreme slant (Phase 3: Horizon Effect).
    /// Simulates a 1x1m tag at 10m distance, rotated by 80-85 degrees.
    #[test]
    fn test_homography_extreme_slant_precision(
        slant_deg in 80.0..85.0f64,
        z_dist in 5.0..15.0f64
    ) {
        // Intrinsics: 640x480 camera
        let fx = 600.0;
        let fy = 600.0;
        let cx = 320.0;
        let cy = 240.0;

        // Tag corners in 3D (1x1m centered at origin)
        let world_corners = [
            [-0.5, -0.5, 0.0],
            [ 0.5, -0.5, 0.0],
            [ 0.5,  0.5, 0.0],
            [-0.5,  0.5, 0.0],
        ];

        // Rotation: slant around Y axis
        let theta = slant_deg.to_radians();
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Project corners to image
        let mut img_corners = [[0.0; 2]; 4];
        for i in 0..4 {
            let wx = world_corners[i][0];
            let wy = world_corners[i][1];
            
            // Rotate around Y: x' = x*cos + z*sin, z' = -x*sin + z*cos
            // Translation: z' += z_dist
            let rx = wx * cos_t;
            let ry = wy;
            let rz = -wx * sin_t + z_dist;
            
            img_corners[i][0] = fx * (rx / rz) + cx;
            img_corners[i][1] = fy * (ry / rz) + cy;
        }

        // Target: canonical square
        let src_sq = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
        ];

        if let Some(h) = Homography::from_pairs(&src_sq, &img_corners) {
            // Project center (0, 0)
            let p_center = h.project([0.0, 0.0]);
            
            // Ground truth center projection
            let g_cx = fx * (0.0 / z_dist) + cx;
            let g_cy = fy * (0.0 / z_dist) + cy;
            
            let err_x = (p_center[0] - g_cx).abs();
            let err_y = (p_center[1] - g_cy).abs();
            let err_total = (err_x * err_x + err_y * err_y).sqrt();
            
            // Phase 3: Precision Gate - Error must be < 0.5 pixels
            assert!(err_total < 0.5, "Extreme slant ({:.1} deg) center error too large: {:.4}px", slant_deg, err_total);
        }
    }

    /// Property: Optimizer Cross-Check (Phase 3).
    /// square_to_quad must be identical to from_pairs(unit_square, dst).
    #[test]
    fn test_homography_optimizer_cross_check(
        dst in valid_quad_strategy()
    ) {
        let src_sq = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]
        ];

        let h_gen = Homography::from_pairs(&src_sq, &dst);
        let h_opt = Homography::square_to_quad(&dst);

        match (h_gen, h_opt) {
            (Some(g), Some(o)) => {
                for i in 0..3 {
                    for j in 0..3 {
                        let diff = (g.h[(i, j)] - o.h[(i, j)]).abs();
                        assert!(diff < 1e-6, "Optimizer divergence at ({}, {}): diff={:.2e}", i, j, diff);
                    }
                }
            },
            (None, None) => {},
            _ => panic!("Solver consensus failed between DLT and optimized solver"),
        }
    }
}
