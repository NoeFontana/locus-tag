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

use divan::bench;
use locus_core::threshold::ThresholdEngine;
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
