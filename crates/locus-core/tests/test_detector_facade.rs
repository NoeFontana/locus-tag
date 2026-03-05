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
use locus_core::{Detector, DetectorBuilder, ImageView, PoseEstimationMode, TagFamily};

#[cfg(feature = "bench-internals")]
use locus_core::bench_api::*;

#[test]
fn test_detector_builder_basic() {
    let detector = DetectorBuilder::new()
        .with_decimation(2)
        .with_family(TagFamily::AprilTag36h11)
        .with_threads(4)
        .build();

    assert_eq!(detector.config().decimation, 2);
    assert_eq!(detector.config().nthreads, 4);
}

#[test]
fn test_detector_new_default() {
    let mut detector = Detector::new();
    assert_eq!(detector.config().decimation, 1);

    // Test detection on empty image
    let data = vec![0u8; 100 * 100];
    let img = ImageView::new(&data, 100, 100, 100).unwrap();
    let detections = detector.detect(&img, None, None, PoseEstimationMode::Fast);
    assert!(detections.is_empty());
}

#[test]
fn test_detector_multiple_families() {
    let mut detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .with_family(TagFamily::ArUco4x4_50)
        .build();

    let canvas_size = 200;
    #[cfg(feature = "bench-internals")]
    {
        // Generate AprilTag
        let (data, _) =
            generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, 50, canvas_size, 0.0);
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();
        let detections = detector.detect(&img, None, None, PoseEstimationMode::Fast);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections.ids[0], 0);

        // Generate ArUco
        let (data2, _) =
            generate_synthetic_test_image(TagFamily::ArUco4x4_50, 5, 50, canvas_size, 0.0);
        let img2 = ImageView::new(&data2, canvas_size, canvas_size, canvas_size).unwrap();
        let detections2 = detector.detect(&img2, None, None, PoseEstimationMode::Fast);
        assert_eq!(detections2.len(), 1);
        assert_eq!(detections2.ids[0], 5);
    }
}

#[test]
fn test_detector_decimation() {
    let mut detector = DetectorBuilder::new().with_decimation(2).build();

    let canvas_size = 200;
    #[cfg(feature = "bench-internals")]
    {
        let (data, _) = generate_synthetic_test_image(
            TagFamily::AprilTag36h11,
            0,
            80, // Large enough to survive decimation
            canvas_size,
            0.0,
        );
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();
        let detections = detector.detect(&img, None, None, PoseEstimationMode::Fast);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections.ids[0], 0);
    }
}
