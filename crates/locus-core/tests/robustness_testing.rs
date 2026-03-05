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

#[test]
fn test_robustness_noise() {
    let mut detector = Detector::new();
    let data = vec![128u8; 100 * 100];
    let img = ImageView::new(&data, 100, 100, 100).unwrap();

    let detections = detector.detect(&img, None, None, PoseEstimationMode::Fast);
    assert!(detections.is_empty());
}
