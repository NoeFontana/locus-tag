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
use locus_core::bench_api::{DetectionBatch, FunnelStatus};

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

#[test]
fn test_funnel_status_initialization() {
    let batch = DetectionBatch::new();
    assert_eq!(batch.funnel_status[0], FunnelStatus::None);
}
