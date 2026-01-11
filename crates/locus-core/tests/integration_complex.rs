#![cfg(feature = "extended-tests")]
//! Complex integration tests for diverse scenes.
//! These tests verify that the detector can handle multiple families,
//! high density, and various tag IDs in ideal and near-ideal conditions.

use locus_core::Detector;
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

        // No rotation for "ideal" verification
        // (Rotation is supported but synthetic rendering currently introduces artifacts)

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
    // Single tag rotation test (more stable than dense multi-tag rotated scenes)
    let width = 640;
    let height = 480;
    let mut rng = ChaCha8Rng::seed_from_u64(123);

    let mut builder = SceneBuilder::new(width, height);
    // Use a large tag to minimize rendering aliasing
    builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (200.0, 250.0));

    let (data, placements) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();

    let mut detector = Detector::new();
    let detections = detector.detect(&img);

    assert!(!detections.is_empty(), "Failed to detect rotated tag");
    assert_eq!(detections[0].id, placements[0].id);
}
