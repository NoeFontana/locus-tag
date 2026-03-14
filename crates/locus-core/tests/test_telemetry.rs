#![allow(missing_docs)]
use locus_core::test_utils::generate_synthetic_test_image;
use locus_core::{Detector, ImageView, TagFamily};

#[test]
fn test_capture_invalid_quads() {
    let canvas_size = 200;
    let (mut data, _gt_corners) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 42, 100, canvas_size, 0.0);

    // Corrupt bits to fail decode but stay a valid quad
    for y in 80..120 {
        for x in 80..120 {
            data[y * canvas_size + x] = 255;
        }
    }

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("valid image");
    let mut detector = Detector::builder()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let batch = detector.detect(
        &img,
        None,
        None,
        locus_core::config::PoseEstimationMode::Fast,
        true,
    ).unwrap();

    assert_eq!(batch.len(), 0);
    assert!(!batch.rejected_corners.is_empty());
}

#[test]
fn test_subpixel_jitter_telemetry() {
    let canvas_size = 200;
    let (data, _gt_corners) = generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        42,
        100,
        canvas_size,
        5.0, // Noisy
    );

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("valid image");
    let mut detector = Detector::builder()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let batch = detector.detect(
        &img,
        None,
        None,
        locus_core::config::PoseEstimationMode::Fast,
        true,
    ).unwrap();

    assert!(!batch.is_empty());
    let telemetry = batch.telemetry.expect("telemetry should be present");
    assert!(!telemetry.subpixel_jitter_ptr.is_null());
}

#[test]
fn test_failed_decode_telemetry() {
    let canvas_size = 200;
    let (mut data, _gt_corners) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 42, 100, canvas_size, 0.0);

    for y in 80..120 {
        for x in 80..120 {
            data[y * canvas_size + x] = 255;
        }
    }

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("valid image");
    let mut detector = Detector::builder()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let batch = detector.detect(
        &img,
        None,
        None,
        locus_core::config::PoseEstimationMode::Fast,
        true,
    ).unwrap();

    assert_eq!(batch.len(), 0);
    assert_eq!(batch.rejected_corners.len(), 1);
    assert!(batch.rejected_error_rates[0] > 0.0);
}

#[test]
fn test_reprojection_error_telemetry() {
    let canvas_size = 200;
    let (data, _gt_corners) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 42, 100, canvas_size, 0.0);

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("valid image");
    let mut detector = Detector::builder()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let intrinsics = locus_core::CameraIntrinsics::new(100.0, 100.0, 100.0, 100.0);
    let tag_size = 0.16;

    let batch = detector.detect(
        &img,
        Some(&intrinsics),
        Some(tag_size),
        locus_core::config::PoseEstimationMode::Fast,
        true,
    ).unwrap();

    assert!(!batch.is_empty());
    let telemetry = batch.telemetry.expect("telemetry should be present");
    assert!(!telemetry.reprojection_errors_ptr.is_null());
}
