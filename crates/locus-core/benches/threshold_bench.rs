#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
mod utils;

use divan::bench;
use locus_core::bench_api::ThresholdEngine;
use locus_core::{DetectorConfig, ImageView};
use utils::BenchDataset;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench]
fn bench_threshold_real_icra_stats(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let config = DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);
    let arena = bumpalo::Bump::new();

    bencher.bench_local(move || {
        let _ = engine.compute_tile_stats(&arena, &img);
    });
}

#[bench]
fn bench_threshold_real_icra_apply(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let config = DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);
    let arena_init = bumpalo::Bump::new();
    let stats = engine.compute_tile_stats(&arena_init, &img).to_vec();
    let mut output = vec![0u8; dataset.width * dataset.height];
    let mut arena = bumpalo::Bump::new();

    bencher.bench_local(move || {
        arena.reset();
        engine.apply_threshold(&arena, &img, &stats, &mut output);
    });
}

/// `LocalMean` threshold map (radius 7, the Liu4K/EuRoC candidate) on the 1080p ICRA frame,
/// telemetry off (empty binary output) as in production.
#[bench]
fn bench_threshold_real_icra_local_mean(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let mut config = DetectorConfig::default();
    config.threshold_mode = locus_core::config::ThresholdMode::LocalMean;
    config.threshold_local_mean_radius = 7;
    let engine = ThresholdEngine::from_config(&config);
    let arena_init = bumpalo::Bump::new();
    let stats = engine.compute_tile_stats(&arena_init, &img).to_vec();
    let mut threshold_map = vec![0u8; dataset.width * dataset.height];
    let mut arena = bumpalo::Bump::new();

    bencher.bench_local(move || {
        arena.reset();
        engine.apply_threshold_with_map(&arena, &img, &stats, &mut [], &mut threshold_map);
    });
}

/// The tile threshold map as production builds it: telemetry off, so `binary_output` is empty.
/// This is the shape `detect()` calls (`detector.rs` sizes the binarized buffer to zero unless
/// debug telemetry is on), and the one the pipeline's latency actually depends on —
/// `bench_threshold_real_icra_apply` above measures the telemetry shape instead, which also
/// binarizes the whole frame and expands a per-tile validity mask nothing else reads.
#[bench]
fn bench_threshold_real_icra_apply_map(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let config = DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);
    let arena_init = bumpalo::Bump::new();
    let stats = engine.compute_tile_stats(&arena_init, &img).to_vec();
    let mut threshold_map = vec![0u8; dataset.width * dataset.height];
    let mut arena = bumpalo::Bump::new();

    bencher.bench_local(move || {
        arena.reset();
        engine.apply_threshold_with_map(&arena, &img, &stats, &mut [], &mut threshold_map);
    });
}
