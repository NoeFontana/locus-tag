#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::uninlined_format_args)]

use locus_core::Detector;
use locus_core::test_utils::{compute_corner_error, generate_test_image};

#[test]
fn test_accuracy_synthetic() {
    let canvas_size = 640;
    // Test with multiple sizes
    let test_cases = [
        (locus_core::config::TagFamily::AprilTag36h11, 0, 150),
        (locus_core::config::TagFamily::AprilTag36h11, 1, 193), // Use 193 to avoid exact tile alignment
        (locus_core::config::TagFamily::ArUco4x4_50, 5, 121),
    ];

    for (family, tag_id, size) in test_cases {
        let (data, gt_corners) = locus_core::test_utils::generate_test_image(
            family,
            tag_id as u16,
            size,
            canvas_size,
            0.0,
        );
        let img = locus_core::image::ImageView::new(&data, canvas_size, canvas_size, canvas_size)
            .unwrap();

        let mut detector = Detector::new();
        detector.set_families(&[family]);
        let detections = detector.detect(&img);

        assert!(
            !detections.is_empty(),
            "Failed to detect {:?} ID {} at size {}",
            family,
            tag_id,
            size
        );
        assert_eq!(detections.len(), 1);
        let det = &detections[0];
        assert_eq!(det.id, tag_id as u32);

        let err = compute_corner_error(&det.corners, &gt_corners);
        println!(
            "Family {:?} ID {}: Corner Error = {} px",
            family, tag_id, err
        );

        // Target sub-pixel accuracy: < 1.0px with noiseless synthetic tags
        // Note: Sharp synthetic edges often introduce a systematic ~0.8px bias
        // due to sub-pixel refinement assumptions.
        assert!(err < 1.0, "Corner error too high: {} px", err);
    }
}

#[test]
fn test_pose_accuracy() {
    let canvas_size = 640;
    let tag_id = 0;
    let tag_size_px = 150;
    let tag_size_m = 0.16;
    let family = locus_core::config::TagFamily::AprilTag36h11;

    let fx = 800.0;
    let fy = 800.0;
    let cx = 320.0;
    let cy = 240.0;

    // Generate synthetic tag with valid bit pattern
    let (data, _gt_corners) =
        locus_core::test_utils::generate_test_image(family, tag_id, tag_size_px, canvas_size, 0.0);
    let img =
        locus_core::image::ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

    let mut detector = Detector::new();
    detector.set_families(&[family]);
    let detections = detector.detect(&img);

    assert!(!detections.is_empty(), "Tag should be found and decoded");
    let det = &detections[0];

    // Manually call pose estimation (or it could be done via detect_with_options)
    let intrinsics = locus_core::pose::CameraIntrinsics::new(fx, fy, cx, cy);
    let pose = locus_core::pose::estimate_tag_pose(&intrinsics, &det.corners, tag_size_m);

    assert!(pose.is_some(), "Pose should be estimated from corners");
    let pose = pose.unwrap();

    // Check if translation is reasonable
    assert!(pose.translation.z > 0.0, "Z translation should be positive");
    assert!(
        pose.translation.z < 10.0,
        "Z translation should be reasonable"
    );

    // Check that rotation is a valid SO(3) matrix
    let det_r = pose.rotation.determinant();
    assert!((det_r - 1.0).abs() < 1e-6);
}
