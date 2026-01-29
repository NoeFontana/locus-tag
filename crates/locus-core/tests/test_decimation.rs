#![allow(missing_docs, clippy::unwrap_used)]
use locus_core::config::TagFamily;
use locus_core::test_utils::{
    TestImageParams, compute_corner_error, generate_test_image_with_params,
};
use locus_core::{DetectOptions, Detector, ImageView};

#[test]
fn test_decimation_basic() {
    let canvas_size = 640;
    let tag_size = 120;
    let params = TestImageParams {
        family: TagFamily::AprilTag36h11,
        id: 0,
        tag_size,
        canvas_size,
        ..Default::default()
    };

    let (data, gt_corners) = generate_test_image_with_params(&params);
    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

    let config = locus_core::config::DetectorConfig::builder()
        .enable_bilateral(false)
        .enable_sharpening(false)
        .quad_min_fill_ratio(0.1)
        .build();
    let mut detector = Detector::with_config(config);

    // Test with decimation = 2
    let options = DetectOptions::builder().decimation(2).build();

    let (detections, stats) = detector.detect_with_stats_and_options(&img, &options);

    assert!(
        !detections.is_empty(),
        "Tag should be detected with decimation=2"
    );

    let det_corners = detections[0].corners;
    let error = compute_corner_error(&det_corners, &gt_corners);

    println!("Decimation 2 error: {error:.4}px");
    println!("Stats: {stats:?}");

    // We expect high accuracy even with decimation because of high-res refinement
    assert!(error < 1.0, "Error {error:.4}px too high for decimation=2");
}

#[test]
fn test_decimation_vs_baseline() {
    let canvas_size = 640;
    let tag_size = 100;
    let params = TestImageParams {
        family: TagFamily::AprilTag36h11,
        id: 0,
        tag_size,
        canvas_size,
        ..Default::default()
    };

    let (data, gt_corners) = generate_test_image_with_params(&params);
    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

    let config = locus_core::config::DetectorConfig::builder()
        .enable_bilateral(false)
        .enable_sharpening(false)
        .quad_min_fill_ratio(0.1)
        .build();
    let mut detector = Detector::with_config(config);

    // Baseline (decimation = 1)
    let (detections1, stats1) = detector.detect_with_stats(&img);
    assert!(!detections1.is_empty());
    let error1 = compute_corner_error(&detections1[0].corners, &gt_corners);

    // Decimation = 2
    let options2 = DetectOptions::builder().decimation(2).build();
    let (detections2, stats2) = detector.detect_with_stats_and_options(&img, &options2);
    assert!(!detections2.is_empty());
    let error2 = compute_corner_error(&detections2[0].corners, &gt_corners);

    println!(
        "Baseline (D=1) - Error: {:.4}px, Time: {:.2}ms",
        error1, stats1.total_ms
    );
    println!(
        "Decimation (D=2) - Error: {:.4}px, Time: {:.2}ms",
        error2, stats2.total_ms
    );

    // Verify that D=2 is significantly faster for thresholding
    assert!(
        stats2.threshold_ms < stats1.threshold_ms,
        "Thresholding should be faster with decimation"
    );
}
