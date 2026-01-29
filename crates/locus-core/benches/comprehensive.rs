#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::ptr_arg)]

//! Comprehensive benchmarking suite for complex scenes and pipeline stages.

use bumpalo::Bump;
use divan::bench;
use locus_core::Detector;
use locus_core::config::{DetectOptions, TagFamily};
use locus_core::image::ImageView;
use locus_core::test_utils::{SceneBuilder, TagPlacement};

fn main() {
    divan::main();
}

// =============================================================================
// PIPELINE STAGES (Granular Benchmarks)
// =============================================================================

#[bench]
fn bench_thresholding_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let mut builder = SceneBuilder::new(width, height);
    builder.add_tag(TagPlacement {
        family: TagFamily::AprilTag36h11,
        id: 0,
        center_x: width as f64 / 2.0,
        center_y: height as f64 / 2.0,
        size: 100.0,
        rotation_rad: 0.0,
    });
    let (data, _) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let engine = locus_core::threshold::ThresholdEngine::new();
    let mut output = vec![0u8; width * height];

    let arena = Bump::new();
    bencher.bench_local(move || {
        let stats = engine.compute_tile_stats(&arena, &img);
        engine.apply_threshold(&arena, &img, &stats, &mut output);
    });
}

#[bench]
fn bench_segmentation_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let mut builder = SceneBuilder::new(width, height);
    builder.add_tag(TagPlacement {
        family: TagFamily::AprilTag36h11,
        id: 0,
        center_x: width as f64 / 2.0,
        center_y: height as f64 / 2.0,
        size: 100.0,
        rotation_rad: 0.0,
    });
    let (data, _) = builder.build();
    let mut binarized = vec![0u8; width * height];
    let engine = locus_core::threshold::ThresholdEngine::new();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let arena = Bump::new();
    let stats = engine.compute_tile_stats(&arena, &img);
    engine.apply_threshold(&arena, &img, &stats, &mut binarized);

    bencher.bench_local(move || {
        let local_arena = Bump::new();
        locus_core::segmentation::label_components(&local_arena, &binarized, width, height, true);
    });
}

#[bench]
fn bench_quad_extraction_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let mut builder = SceneBuilder::new(width, height);
    builder.add_tag(TagPlacement {
        family: TagFamily::AprilTag36h11,
        id: 0,
        center_x: width as f64 / 2.0,
        center_y: height as f64 / 2.0,
        size: 100.0,
        rotation_rad: 0.0,
    });
    let (data, _) = builder.build();
    let arena = Bump::new();
    let mut binarized = vec![0u8; width * height];
    let engine = locus_core::threshold::ThresholdEngine::new();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let stats = engine.compute_tile_stats(&arena, &img);
    engine.apply_threshold(&arena, &img, &stats, &mut binarized);

    let labels = locus_core::segmentation::label_components_with_stats(
        &arena, &binarized, width, height, true,
    );

    bencher.bench_local(move || {
        let local_arena = Bump::new();
        locus_core::quad::extract_quads_with_config(
            &local_arena,
            &img,
            &labels,
            &locus_core::config::DetectorConfig::default(),
            1,
            &img,
        );
    });
}

// =============================================================================
// FULL PIPELINE (E2E Benchmarks)
// =============================================================================

#[bench]
fn bench_full_detect_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let mut builder = SceneBuilder::new(width, height);
    builder.add_tag(TagPlacement {
        family: TagFamily::AprilTag36h11,
        id: 0,
        center_x: width as f64 / 2.0,
        center_y: height as f64 / 2.0,
        size: 100.0,
        rotation_rad: 0.0,
    });
    let (data, _) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || detector.detect(&img));
}

/// Benchmark detection in a complex scene with multiple families and tags.
#[bench]
fn bench_mixed_scene_multiple_tags(bencher: divan::Bencher) {
    let width = 1280;
    let height = 720;

    let mut builder = SceneBuilder::new(width, height)
        .with_noise(2.0)
        .with_blur(0.5);

    let mut rng = rand::thread_rng();

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

    let mut builder = SceneBuilder::new(width, height).with_noise(10.0);

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
