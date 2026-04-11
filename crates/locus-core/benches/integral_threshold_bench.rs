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
use bumpalo::Bump;
use divan::bench;
use locus_core::ImageView;
use locus_core::bench_api::compute_gradient_map;
use locus_core::bench_api::generate_checkered;
use locus_core::bench_api::{
    adaptive_threshold_gradient_window, adaptive_threshold_integral, compute_integral_image,
};

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench]
fn bench_integral_image_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut integral = vec![0u64; (width + 1) * (height + 1)];

    bencher.bench_local(move || {
        compute_integral_image(&img, &mut integral);
    });
}

#[bench]
fn bench_adaptive_integral_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut integral = vec![0u64; (width + 1) * (height + 1)];
    compute_integral_image(&img, &mut integral);
    let mut output = vec![0u8; width * height];

    bencher.bench_local(move || {
        adaptive_threshold_integral(&img, &integral, &mut output, 6, 3);
    });
}

#[bench]
fn bench_adaptive_gradient_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut integral = vec![0u64; (width + 1) * (height + 1)];
    compute_integral_image(&img, &mut integral);
    let mut gradient = vec![0u8; width * height];
    compute_gradient_map(&img, &mut gradient);
    let mut output = vec![0u8; width * height];

    bencher.bench_local(move || {
        adaptive_threshold_gradient_window(&img, &gradient, &integral, &mut output, 2, 7, 40, 3);
    });
}

#[bench]
fn bench_tile_threshold_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let config = locus_core::DetectorConfig::builder()
        .refinement_mode(locus_core::CornerRefinementMode::Erf)
        .build();
    let engine = locus_core::bench_api::ThresholdEngine::from_config(&config);
    let mut arena = Bump::new();

    bencher.bench_local(move || {
        arena.reset();
        let _stats = engine.compute_tile_stats(&arena, &img);
    });
}
