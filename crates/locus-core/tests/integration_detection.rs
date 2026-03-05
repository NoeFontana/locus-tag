#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::panic)]

use locus_core::{Detector, DetectorBuilder, TagFamily, ImageView};

#[cfg(feature = "bench-internals")]
use locus_core::bench_api as internals;

#[test]
fn test_accuracy_synthetic() {
    let canvas_size = 640;
    // Test with multiple sizes
    let test_cases = [
        (TagFamily::AprilTag36h11, 0, 150),
        (TagFamily::AprilTag36h11, 1, 193),
        (TagFamily::AprilTag41h12, 10, 150),
        (TagFamily::ArUco4x4_50, 5, 121),
    ];

    for (family, tag_id, size) in test_cases {
        #[cfg(feature = "bench-internals")]
        {
            let (data, gt_corners) = internals::generate_synthetic_test_image(
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

            let err = internals::compute_corner_error(&det.corners, &gt_corners);
            println!(
                "Family {:?} ID {}: Corner Error = {} px",
                family, tag_id, err
            );
            assert!(err < 1.0, "Corner error too high: {} px", err);
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

    let fx = 800.0;
    let fy = 800.0;
    let cx = 320.0;
    let cy = 240.0;

    #[cfg(feature = "bench-internals")]
    {
        let (data, _) = internals::generate_synthetic_test_image(
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
        let detections = detector.detect(&img);

        assert!(!detections.is_empty());
        let det = &detections[0];

        let intrinsics = locus_core::CameraIntrinsics::new(fx, fy, cx, cy);
        let (pose, _) = internals::estimate_tag_pose(
            &intrinsics,
            &det.corners,
            tag_size_m,
            Some(&img),
            locus_core::config::PoseEstimationMode::Fast,
        );

        assert!(pose.is_some());
        let pose = pose.unwrap();
        assert!(pose.translation.z > 0.0);
    }
}
