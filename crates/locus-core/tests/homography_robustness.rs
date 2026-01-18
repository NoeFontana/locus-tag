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

/// Strategy for degenerate quads (collinear or coincident).
fn degenerate_quad_strategy() -> impl Strategy<Value = [[f64; 2]; 4]> {
    prop_oneof![
        // Coincident points: p0 == p1
        point_strategy().prop_map(|p| [p, p, [p[0] + 10.0, p[1]], [p[0], p[1] + 10.0]]),
        
        // Collinear points: p0, p1, p2 on the same line
        point_strategy().prop_map(|p| {
            [p, [p[0] + 10.0, p[1]], [p[0] + 20.0, p[1]], [p[0], p[1] + 10.0]]
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

    /// Property: Degenerate quads never panic and return None or handle gracefully (no NaN).
    #[test]
    fn test_homography_degenerate_safety(
        src in degenerate_quad_strategy(),
        dst in valid_quad_strategy()
    ) {
        let h = Homography::from_pairs(&src, &dst);
        if let Some(h_val) = h {
            let p = h_val.project([0.0, 0.0]);
            assert!(!p[0].is_nan());
            assert!(!p[1].is_nan());
            assert!(!p[0].is_infinite());
            assert!(!p[1].is_infinite());
        }
    }
}
