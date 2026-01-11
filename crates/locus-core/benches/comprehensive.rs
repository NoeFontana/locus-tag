//! Comprehensive benchmarking suite for complex scenes.

use divan::bench;
use locus_core::Detector;
use locus_core::config::{DetectOptions, TagFamily};
use locus_core::image::ImageView;
use locus_core::test_utils::SceneBuilder;

fn main() {
    divan::main();
}

/// Benchmark detection in a complex scene with multiple families and tags.
#[bench]
fn bench_mixed_scene_multiple_tags(bencher: divan::Bencher) {
    let width = 1280;
    let height = 720;

    // Generate a scene once
    let mut builder = SceneBuilder::new(width, height)
        .with_noise(2.0)
        .with_blur(0.5);

    let mut rng = rand::thread_rng();

    // Add various tags
    builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (50.0, 100.0));
    builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (80.0, 120.0));
    builder.add_random_tag(&mut rng, TagFamily::ArUco4x4_50, (60.0, 90.0));
    builder.add_random_tag(&mut rng, TagFamily::ArUco4x4_100, (70.0, 110.0));
    builder.add_random_tag(&mut rng, TagFamily::AprilTag16h5, (40.0, 70.0));

    let (data, _placements) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();

    let mut detector = Detector::new();
    let options = DetectOptions {
        families: vec![
            TagFamily::AprilTag36h11,
            TagFamily::ArUco4x4_50,
            TagFamily::ArUco4x4_100,
            TagFamily::AprilTag16h5,
        ],
        ..Default::default()
    };

    bencher.bench_local(move || detector.detect_with_options(&img, &options));
}

/// Benchmark detection with high tag density (stress test quad extraction).
#[bench]
fn bench_dense_scene_20_tags(bencher: divan::Bencher) {
    let width = 1280;
    let height = 720;

    let mut builder = SceneBuilder::new(width, height)
        .with_background(100)
        .with_noise(1.0);

    let mut rng = rand::thread_rng();
    for _ in 0..20 {
        builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (40.0, 60.0));
    }

    let (data, _placements) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();

    let mut detector = Detector::new();
    let options = DetectOptions {
        families: vec![TagFamily::AprilTag36h11],
        ..Default::default()
    };

    bencher.bench_local(move || detector.detect_with_options(&img, &options));
}

/// Benchmark detection robustness under high noise.
#[bench]
fn bench_noisy_scene(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;

    let mut builder = SceneBuilder::new(width, height).with_noise(10.0); // Significant noise

    let mut rng = rand::thread_rng();
    builder.add_random_tag(&mut rng, TagFamily::ArUco4x4_50, (100.0, 150.0));

    let (data, _placements) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();

    let mut detector = Detector::new();
    let options = DetectOptions {
        families: vec![TagFamily::ArUco4x4_50],
        ..Default::default()
    };

    bencher.bench_local(move || detector.detect_with_options(&img, &options));
}
