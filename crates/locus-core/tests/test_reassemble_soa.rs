#![allow(
    missing_docs,
    dead_code,
    clippy::unwrap_used,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::float_cmp
)]
use locus_core::bench_api::*;

#[test]
fn test_reassemble_single() {
    let mut batch = DetectionBatch::new();
    batch.status_mask[0] = CandidateState::Valid;
    batch.ids[0] = 42;
    batch.corners[0] = Point2f { x: 1.0, y: 2.0 };
    batch.corners[1] = Point2f { x: 3.0, y: 4.0 };
    batch.corners[2] = Point2f { x: 5.0, y: 6.0 };
    batch.corners[3] = Point2f { x: 7.0, y: 8.0 };
    batch.payloads[0] = 0x1234_5678;
    batch.error_rates[0] = 1.0;

    batch.poses[0] = Pose6D {
        data: [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        padding: 0.0,
    };

    let detections = batch.reassemble(1);
    assert_eq!(detections.len(), 1);
    let d = &detections[0];
    assert_eq!(d.id, 42);
    assert_eq!(d.corners[0], [1.0, 2.0]);
    assert!(d.pose.is_some());
}
