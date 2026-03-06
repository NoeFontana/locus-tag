//! Smoke tests for ICRA dataset loading and detection.
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
use locus_core::{Detector, ImageView, PoseEstimationMode};

#[cfg(feature = "bench-internals")]
mod common;

#[test]
fn test_smoke_icra_dataset() {
    let mut detector = Detector::new();
    let img_data = vec![0u8; 640 * 480];
    let input_view = ImageView::new(&img_data, 640, 480, 640).expect("valid image");

    let _res = detector.detect(&input_view, None, None, PoseEstimationMode::Fast);

    // Smoke test: should not crash
}
