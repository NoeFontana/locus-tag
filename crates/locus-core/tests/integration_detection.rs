use locus_core::bench_api::*;
use locus_core::{PoseEstimationMode, Detector, DetectorBuilder, TagFamily, ImageView};

#[cfg(feature = "bench-internals")]

#[test]
fn test_accuracy_synthetic() {
    let canvas_size = 640;
    let test_cases = [
        (TagFamily::AprilTag36h11, 0, 150),
        (TagFamily::AprilTag36h11, 1, 193),
        (TagFamily::AprilTag41h12, 10, 150),
        (TagFamily::ArUco4x4_50, 5, 121),
    ];

    for (family, tag_id, size) in test_cases {
        #[cfg(feature = "bench-internals")]
        {
            let (data, gt_corners) = generate_synthetic_test_image(
                family,
                tag_id as u16,
                size,
                canvas_size,
                0.0,
            );
            let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

            let mut detector = DetectorBuilder::new()
                .with_family(family)
                .build();
            let detections = detector.detect(&img, None, None, PoseEstimationMode::Fast);

            assert!(!detections.is_empty());
            let det = &detections[0];
            assert_eq!(det.id, tag_id as u32);

            let err = compute_corner_error(&det.corners, &gt_corners);
            assert!(err < 1.0);
        }
    }
}

#[test]
fn test_pose_accuracy() {
    let canvas_size = 640;
    let tag_id = 0;
    let tag_size_px = 150;
    let tag_size_m = 0.16;
    let family = TagFamily::AprilTag36h11;

    #[cfg(feature = "bench-internals")]
    {
        let (data, _) = generate_synthetic_test_image(
            family,
            tag_id,
            tag_size_px,
            canvas_size,
            0.0,
        );
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

        let mut detector = DetectorBuilder::new()
            .with_family(family)
            .build();
        let detections = detector.detect(&img, None, None, PoseEstimationMode::Fast);

        assert!(!detections.is_empty());
        let det = &detections[0];

        let intrinsics = locus_core::CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);
        let (pose, _) = estimate_tag_pose(
            &intrinsics,
            &det.corners,
            tag_size_m,
            Some(&img),
            locus_core::config::PoseEstimationMode::Fast,
        );

        assert!(pose.is_some());
        assert!(pose.unwrap().translation.z > 0.0);
    }
}
