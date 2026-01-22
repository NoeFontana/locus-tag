#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::uninlined_format_args)]

use locus_core::Detector;
use locus_core::test_utils::compute_corner_error;

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
        let (data, gt_corners) = locus_core::test_utils::generate_synthetic_test_image(
            family,
            tag_id as u16,
            size,
            canvas_size,
            0.0,
        );
        let img = locus_core::image::ImageView::new(&data, canvas_size, canvas_size, canvas_size)
            .unwrap();

        let config = locus_core::config::DetectorConfig::builder()
            .threshold_min_range(0) // Synthetic images have 0 noise
            .quad_min_fill_ratio(0.1) // Synthetic tags might yield thin borders -> low fill
            .build();
        let mut detector = Detector::with_config(config);
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
    let (data, _gt_corners) = locus_core::test_utils::generate_synthetic_test_image(
        family,
        tag_id,
        tag_size_px,
        canvas_size,
        0.0,
    );
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

#[cfg(feature = "extended-tests")]
mod extended {
    use super::*;
    use locus_core::config::{DetectOptions, TagFamily};
    use locus_core::image::ImageView;
    use locus_core::test_utils::SceneBuilder;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_complex_scene_accuracy_ideal() {
        let width = 1280;
        let height = 720;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for i in 0..10 {
            let mut builder = SceneBuilder::new(width, height)
                .with_noise(0.0)
                .with_blur(0.0);

            let families = [
                TagFamily::AprilTag36h11,
                TagFamily::ArUco4x4_50,
                TagFamily::ArUco4x4_100,
                TagFamily::AprilTag16h5,
            ];

            for &family in &families {
                builder.add_random_tag(&mut rng, family, (100.0, 150.0));
            }
            builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (80.0, 120.0));

            let (data, placements) = builder.build();
            let img = ImageView::new(&data, width, height, width).unwrap();

            let mut detector = Detector::new();
            let options = DetectOptions {
                families: families.to_vec(),
                ..Default::default()
            };

            let detections = detector.detect_with_options(&img, &options);

            assert_eq!(
                detections.len(),
                placements.len(),
                "Failed to detect all tags in scene {}",
                i
            );

            for placement in &placements {
                let _det = detections
                    .iter()
                    .find(|d| d.id == placement.id)
                    .expect(&format!(
                        "Tag ID {} not found in detections in scene {}",
                        placement.id, i
                    ));
            }
        }
    }

    #[test]
    fn test_rotation_invariance_verification() {
        let width = 640;
        let height = 480;
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let mut builder = SceneBuilder::new(width, height);
        builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (200.0, 250.0));

        let (data, placements) = builder.build();
        let img = ImageView::new(&data, width, height, width).unwrap();

        let mut detector = Detector::new();
        let detections = detector.detect(&img);

        assert!(!detections.is_empty(), "Failed to detect rotated tag");
        assert_eq!(detections[0].id, placements[0].id);
    }
}
