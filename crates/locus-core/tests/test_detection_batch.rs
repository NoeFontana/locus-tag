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
    clippy::return_self_not_must_use
)]
use locus_core::bench_api::DetectionBatch;

#[test]
fn test_batch_allocation() {
    let batch = DetectionBatch::new();
    assert_eq!(batch.capacity(), 1024);
}

#[test]
fn test_batch_alignment() {
    let batch = DetectionBatch::new();
    let corners_ptr = batch.corners.as_ptr() as usize;
    assert_eq!(corners_ptr % 32, 0);

    let homographies_ptr = batch.homographies.as_ptr() as usize;
    assert_eq!(homographies_ptr % 32, 0);
}
