#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Robustness tests for geometry solvers and pose estimation.
use locus_core::CameraIntrinsics;
use locus_core::bench_api::Homography;
use locus_core::pose::estimate_tag_pose;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("proptest-regressions/geometry.txt"))),
        ..ProptestConfig::default()
    })]

    /// Degenerate quads must yield either `None` or a *fully finite* homography
    /// — never a NaN/Inf-laden matrix. `square_to_quad` guards this by rejecting
    /// solutions whose reprojection error is non-finite (this is also what
    /// defends the unguarded perspective-divide in `Homography::project` when a
    /// point maps near the line at infinity, w ~ 0). This property is the
    /// numerical-robustness contract behind the previous no-op smoke test.
    #[test]
    fn prop_homography_degenerate_quads_never_nan(
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

        // Must not panic, and if a homography is returned it must be finite.
        if let Some(h) = Homography::square_to_quad(&corners) {
            for (i, v) in h.h.iter().enumerate() {
                prop_assert!(
                    v.is_finite(),
                    "square_to_quad returned a non-finite entry h[{i}] = {v} for corners {corners:?}",
                );
            }
        }
    }

    /// Degenerate/collinear corners must yield either `None` or a pose whose
    /// rotation and translation are entirely finite — never NaN/Inf. This
    /// exercises the IPPE analytic SVD path (including its frontal-view
    /// Gram-Schmidt branch) on pathological inputs.
    #[test]
    fn prop_pose_estimation_degenerate_quads_never_nan(
        pts in proptest::array::uniform4(
            (-10000.0_f64..10000.0_f64, -10000.0_f64..10000.0_f64)
        ),
        tag_size in 0.0_f64..10.0_f64,
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
            distortion: locus_core::pose::DistortionCoeffs::None,
        };

        if let (Some(pose), _) = estimate_tag_pose(&intrinsics, &corners, tag_size, None) {
            for v in pose.translation.iter() {
                prop_assert!(
                    v.is_finite(),
                    "estimate_tag_pose returned non-finite translation {:?} for corners {corners:?}",
                    pose.translation,
                );
            }
            for v in pose.rotation.iter() {
                prop_assert!(
                    v.is_finite(),
                    "estimate_tag_pose returned a non-finite rotation entry {v} for corners {corners:?}",
                );
            }
        }
    }
}
