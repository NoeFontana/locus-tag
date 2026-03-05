use locus_core::config::PoseEstimationMode;
use locus_core::bench_api::*;
use locus_core::ImageView;

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
        PoseEstimationMode::Fast
    );
}
