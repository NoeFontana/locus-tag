#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Robustness tests for geometry solvers and pose estimation.
use locus_core::bench_api::Homography;
use locus_core::pose::estimate_tag_pose;
use locus_core::{CameraIntrinsics, PoseEstimationMode};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("proptest-regressions/geometry.txt"))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_homography_survives_degenerate_quads(
        pts in proptest::array::uniform4(
            (-10000.0_f64..10000.0_f64, -10000.0_f64..10000.0_f64)
        )
    ) {
        let corners = [
            [pts[0].0, pts[0].1],
            [pts[1].0, pts[1].1],
            [pts[2].0, pts[2].1],
            [pts[3].0, pts[3].1],
        ];

        // This might return None for degenerate quads, but it shouldn't panic.
        let _h = Homography::square_to_quad(&corners);
    }

    #[test]
    fn prop_pose_estimation_survives_degenerate_quads(
        pts in proptest::array::uniform4(
            (-10000.0_f64..10000.0_f64, -10000.0_f64..10000.0_f64)
        ),
        tag_size in 0.0_f64..10.0_f64,
        mode in prop_oneof![Just(PoseEstimationMode::Fast), Just(PoseEstimationMode::Accurate)]
    ) {
        let corners = [
            [pts[0].0, pts[0].1],
            [pts[1].0, pts[1].1],
            [pts[2].0, pts[2].1],
            [pts[3].0, pts[3].1],
        ];

        let intrinsics = CameraIntrinsics {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        };

        // The estimation can return None, but it shouldn't panic on collinear/degenerate points.
        let _pose = estimate_tag_pose(&intrinsics, &corners, tag_size, None, mode);

        // Assert survival
        prop_assert!(true);
    }
}
