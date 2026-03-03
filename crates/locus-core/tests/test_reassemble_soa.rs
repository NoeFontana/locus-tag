//! Tests for the SoA Reassembly.

use locus_core::{CandidateState, DetectionBatch, Point2f, Pose6D};

#[test]
fn test_reassemble_soa() {
    let mut batch = DetectionBatch::new();

    // Setup a valid candidate
    batch.status_mask[0] = CandidateState::Valid;
    batch.ids[0] = 42;
    batch.corners[0] = Point2f { x: 1.0, y: 2.0 };
    batch.corners[1] = Point2f { x: 3.0, y: 4.0 };
    batch.corners[2] = Point2f { x: 5.0, y: 6.0 };
    batch.corners[3] = Point2f { x: 7.0, y: 8.0 };
    batch.payloads[0] = 0x12345678;
    batch.error_rates[0] = 1.0;
    // Mock pose: tx=1, ty=2, tz=3, qx=0, qy=0, qz=0, qw=1 (identity rotation)
    batch.poses[0] = Pose6D {
        data: [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
    };

    let detections = batch.reassemble(1);

    assert_eq!(detections.len(), 1);
    let d = &detections[0];
    assert_eq!(d.id, 42);
    assert_eq!(d.corners[0], [1.0, 2.0]);
    assert_eq!(d.bits, 0x12345678);
    assert_eq!(d.hamming, 1);

    let pose = d.pose.as_ref().unwrap();
    assert_eq!(pose.translation.x, 1.0);
    assert_eq!(pose.translation.y, 2.0);
    assert_eq!(pose.translation.z, 3.0);
}
