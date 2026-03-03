//! Tests for the SoA Pose Refinement.

use locus_core::config::PoseEstimationMode;
use locus_core::pose::{CameraIntrinsics, refine_poses_soa};
use locus_core::{CandidateState, DetectionBatch};

#[test]
fn test_refine_poses_soa_interface() {
    let mut batch = DetectionBatch::new();

    // Setup a valid candidate at index 0 (unit square)
    batch.status_mask[0] = CandidateState::Valid;
    batch.corners[0].x = -10.0;
    batch.corners[0].y = -10.0;
    batch.corners[1].x = 10.0;
    batch.corners[1].y = -10.0;
    batch.corners[2].x = 10.0;
    batch.corners[2].y = 10.0;
    batch.corners[3].x = -10.0;
    batch.corners[3].y = 10.0;

    let v = 1;
    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
    let tag_size = 0.16;

    refine_poses_soa(
        &mut batch,
        v,
        &intrinsics,
        tag_size,
        None,
        PoseEstimationMode::Fast,
    );

    // Check if pose was populated (translation should be non-zero if not looking at origin)
    // Actually for a centered tag, translation Z will be non-zero.
    assert!(batch.poses[0].data[2] > 0.0);
}
