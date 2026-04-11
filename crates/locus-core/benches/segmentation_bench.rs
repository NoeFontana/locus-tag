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
    clippy::return_self_not_must_use
)]

mod utils;

use bumpalo::Bump;
use divan::bench;
use locus_core::ImageView;
use locus_core::bench_api::ThresholdEngine;
use utils::BenchDataset;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench]
fn bench_segmentation_real_icra_threshold_model(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let setup_arena = Bump::new();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);

    let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
    let mut threshold_map = vec![0u8; dataset.width * dataset.height];
    let mut binarized = vec![0u8; dataset.width * dataset.height];

    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &tile_stats,
        &mut binarized,
        &mut threshold_map,
    );

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _label_result =
            locus_core::bench_api::label_components_lsl(&arena, &img, &threshold_map, true, 16);
    });
}

#[bench]
fn bench_segmentation_real_icra_threshold_model_1080p(bencher: divan::Bencher) {
    let dataset = BenchDataset::load_and_resize_icra_frame("forward", 0, 1920, 1080);
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let setup_arena = Bump::new();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);

    let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
    let mut threshold_map = vec![0u8; dataset.width * dataset.height];
    let mut binarized = vec![0u8; dataset.width * dataset.height];

    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &tile_stats,
        &mut binarized,
        &mut threshold_map,
    );

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _label_result =
            locus_core::bench_api::label_components_lsl(&arena, &img, &threshold_map, true, 16);
    });
}

#[bench]
fn bench_segmentation_real_icra_threshold_model_4k(bencher: divan::Bencher) {
    let dataset = BenchDataset::load_and_resize_icra_frame("forward", 0, 3840, 2160);
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let setup_arena = Bump::new();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);

    let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
    let mut threshold_map = vec![0u8; dataset.width * dataset.height];
    let mut binarized = vec![0u8; dataset.width * dataset.height];

    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &tile_stats,
        &mut binarized,
        &mut threshold_map,
    );

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _label_result =
            locus_core::bench_api::label_components_lsl(&arena, &img, &threshold_map, true, 16);
    });
}
