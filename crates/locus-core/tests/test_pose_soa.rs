#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
use locus_core::bench_api::*;
use locus_core::config::PoseEstimationMode;

#[test]
fn test_pose_refinement_soa_empty() {
    let mut batch = DetectionBatch::new();
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);

    refine_poses_soa(
        &mut batch,
        0,
        &intrinsics,
        0.1,
        None,
        PoseEstimationMode::Fast,
    );
}
