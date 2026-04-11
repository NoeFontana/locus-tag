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
    clippy::return_self_not_must_use,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::ptr_arg
)]

//! Comprehensive benchmarking suite for complex scenes and pipeline stages.

use bumpalo::Bump;
use divan::bench;
use locus_core::Detector;
use locus_core::ImageView;
use locus_core::PoseEstimationMode;
use locus_core::TagFamily;
use locus_core::bench_api::ThresholdEngine;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

// =============================================================================
// REAL WORLD DATASET (ICRA 2020) - Multi-Resolution
// =============================================================================

mod utils;
use utils::BenchDataset;

const RESOLUTIONS: &[(usize, usize)] = &[
    (640, 480),   // VGA
    (1280, 720),  // 720p
    (1920, 1080), // 1080p
    (3840, 2160), // 4K
];

#[bench(args = RESOLUTIONS)]
fn bench_thresholding(bencher: divan::Bencher, &(width, height): &(usize, usize)) {
    // SETUP PHASE (Not timed)
    let dataset = BenchDataset::load_and_resize_icra_frame("forward", 0, width, height);
    let img = ImageView::new(&dataset.raw_data, width, height, width).unwrap();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);
    let mut output = vec![0u8; width * height];
    let mut arena = Bump::new();

    // MEASUREMENT PHASE (Timed)
    bencher.bench_local(move || {
        arena.reset();
        let stats = engine.compute_tile_stats(&arena, &img);
        engine.apply_threshold(&arena, &img, &stats, &mut output);
    });
}

#[bench(args = RESOLUTIONS)]
fn bench_segmentation(bencher: divan::Bencher, &(width, height): &(usize, usize)) {
    // SETUP PHASE (Not timed)
    let dataset = BenchDataset::load_and_resize_icra_frame("forward", 0, width, height);
    let img = ImageView::new(&dataset.raw_data, width, height, width).unwrap();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);
    let mut binarized = vec![0u8; width * height];
    let mut threshold_map = vec![0u8; width * height];
    let setup_arena = Bump::new();

    let stats = engine.compute_tile_stats(&setup_arena, &img);
    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &stats,
        &mut binarized,
        &mut threshold_map,
    );

    // MEASUREMENT PHASE (Timed)
    bencher.bench_local(move || {
        let arena = Bump::new();
        let _label_result =
            locus_core::bench_api::label_components_lsl(&arena, &img, &threshold_map, true, 16);
    });
}

#[bench(args = RESOLUTIONS)]
fn bench_quad_extraction(bencher: divan::Bencher, &(width, height): &(usize, usize)) {
    // SETUP PHASE (Not timed)
    let dataset = BenchDataset::load_and_resize_icra_frame("forward", 0, width, height);
    let img = ImageView::new(&dataset.raw_data, width, height, width).unwrap();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);
    let mut binarized = vec![0u8; width * height];
    let mut threshold_map = vec![0u8; width * height];
    let setup_arena = Bump::new();

    let stats = engine.compute_tile_stats(&setup_arena, &img);
    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &stats,
        &mut binarized,
        &mut threshold_map,
    );

    let labels = locus_core::bench_api::label_components_with_stats(
        &setup_arena,
        &binarized,
        width,
        height,
        true,
    );

    // MEASUREMENT PHASE (Timed)
    bencher.bench_local(move || {
        let local_arena = Bump::new();
        locus_core::bench_api::extract_quads_with_config(
            &local_arena,
            &img,
            &labels,
            &locus_core::DetectorConfig::default(),
            1,
            &img,
        );
    });
}

#[bench]
fn bench_icra_full_pipeline(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let config = locus_core::DetectorConfig::production_default();
    let mut detector = Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);

    bencher.bench_local(move || {
        let _ = detector
            .detect(&img, None, None, PoseEstimationMode::Fast, false)
            .unwrap();
    });
}

#[bench]
fn bench_icra_decoding_soa(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let mut batch = BenchDataset::generate_bench_batch(50, 200);
    let n = 250;
    let config = locus_core::DetectorConfig::production_default();
    let decoders = vec![locus_core::bench_api::family_to_decoder(
        TagFamily::AprilTag36h11,
    )];

    bencher.bench_local(move || {
        locus_core::bench_api::decode_batch_soa(&mut batch, n, &img, &decoders, &config);
    });
}

// =============================================================================
// SYNTHETIC SCENES
// =============================================================================

/// Benchmark detection in a complex scene with multiple families and tags.
#[bench]
fn bench_mixed_scene_multiple_tags(bencher: divan::Bencher) {
    use locus_core::bench_api::SceneBuilder;
    let width = 1280;
    let height = 720;

    let mut builder = SceneBuilder::new(width, height)
        .with_noise(2.0)
        .with_blur(0.5);

    let mut rng = rand::rng();

    builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (50.0, 100.0));
    builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (80.0, 120.0));
    builder.add_random_tag(&mut rng, TagFamily::ArUco4x4_50, (60.0, 90.0));
    builder.add_random_tag(&mut rng, TagFamily::ArUco4x4_100, (70.0, 110.0));

    let (data, _placements) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();

    let mut detector = Detector::new();
    detector.set_families(&[
        TagFamily::AprilTag36h11,
        TagFamily::ArUco4x4_50,
        TagFamily::ArUco4x4_100,
    ]);

    bencher.bench_local(move || {
        let _ = detector
            .detect(&img, None, None, PoseEstimationMode::Fast, false)
            .unwrap();
    });
}

/// Benchmark detection with high tag density (stress test quad extraction).
#[bench]
fn bench_dense_scene_20_tags(bencher: divan::Bencher) {
    use locus_core::bench_api::SceneBuilder;
    let width = 1280;
    let height = 720;

    let mut builder = SceneBuilder::new(width, height)
        .with_background(100)
        .with_noise(1.0);

    let mut rng = rand::rng();
    for _ in 0..20 {
        builder.add_random_tag(&mut rng, TagFamily::AprilTag36h11, (40.0, 60.0));
    }

    let (data, _placements) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();

    let mut detector = Detector::new();
    detector.set_families(&[TagFamily::AprilTag36h11]);

    bencher.bench_local(move || {
        let _ = detector
            .detect(&img, None, None, PoseEstimationMode::Fast, false)
            .unwrap();
    });
}

/// Benchmark detection robustness under high noise.
#[bench]
fn bench_noisy_scene(bencher: divan::Bencher) {
    use locus_core::bench_api::SceneBuilder;
    let width = 640;
    let height = 480;

    let mut builder = SceneBuilder::new(width, height).with_noise(10.0);

    let mut rng = rand::rng();
    builder.add_random_tag(&mut rng, TagFamily::ArUco4x4_50, (100.0, 150.0));

    let (data, _placements) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();

    let mut detector = Detector::new();
    detector.set_families(&[TagFamily::ArUco4x4_50]);

    bencher.bench_local(move || {
        let _ = detector
            .detect(&img, None, None, PoseEstimationMode::Fast, false)
            .unwrap();
    });
}
