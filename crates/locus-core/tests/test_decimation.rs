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
use locus_core::bench_api::*;
use locus_core::{DetectorBuilder, ImageView, PoseEstimationMode, TagFamily};

#[cfg(feature = "bench-internals")]
#[test]
fn test_decimation_accuracy() {
    let canvas_size = 640;
    let tag_id = 0;
    let tag_size_px = 100;
    let family = TagFamily::AprilTag36h11;

    #[cfg(feature = "bench-internals")]
    {
        let (data, gt_corners) =
            generate_synthetic_test_image(family, tag_id, tag_size_px, canvas_size, 0.0);
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

        // 1. Detect with decimation 1
        let mut detector = DetectorBuilder::new()
            .with_family(family)
            .with_decimation(1)
            .build();

        let detections = detector.detect(&img, None, None, PoseEstimationMode::Fast);
        assert!(!detections.is_empty());
        let corners1 = [
            [f64::from(detections.corners[0][0].x), f64::from(detections.corners[0][0].y)],
            [f64::from(detections.corners[0][1].x), f64::from(detections.corners[0][1].y)],
            [f64::from(detections.corners[0][2].x), f64::from(detections.corners[0][2].y)],
            [f64::from(detections.corners[0][3].x), f64::from(detections.corners[0][3].y)],
        ];
        let err1 = compute_corner_error(&corners1, &gt_corners);

        // 2. Detect with decimation 2
        let mut detector2 = DetectorBuilder::new()
            .with_family(family)
            .with_decimation(2)
            .build();

        let detections2 = detector2.detect(&img, None, None, PoseEstimationMode::Fast);
        assert!(!detections2.is_empty());
        let corners2 = [
            [f64::from(detections2.corners[0][0].x), f64::from(detections2.corners[0][0].y)],
            [f64::from(detections2.corners[0][1].x), f64::from(detections2.corners[0][1].y)],
            [f64::from(detections2.corners[0][2].x), f64::from(detections2.corners[0][2].y)],
            [f64::from(detections2.corners[0][3].x), f64::from(detections2.corners[0][3].y)],
        ];
        let err2 = compute_corner_error(&corners2, &gt_corners);

        println!("Error D1: {err1}, Error D2: {err2}");
        // Decimation should maintain reasonable sub-pixel accuracy
        assert!(err2 < 1.5);
    }
}
