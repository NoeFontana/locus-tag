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
use bumpalo::Bump;
use divan::bench;
use locus_core::ImageView;
use locus_core::PoseEstimationMode;
use locus_core::config::CornerRefinementMode;
use std::path::Path;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

fn load_icra_image() -> (Vec<u8>, usize, usize) {
    let path = Path::new("../../tests/data/icra2020/forward/pure_tags_images/0000.png");
    let img = image::open(path)
        .expect("Failed to open ICRA image")
        .to_luma8();
    let (width, height) = img.dimensions();
    (img.into_raw(), width as usize, height as usize)
}

#[bench]
fn bench_preprocessing_real(bencher: divan::Bencher) {
    let (data, width, height) = load_icra_image();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let config = locus_core::DetectorConfig::builder()
        .refinement_mode(CornerRefinementMode::Erf)
        .build();
    let engine = locus_core::threshold::ThresholdEngine::from_config(&config);

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _stats = engine.compute_tile_stats(&arena, &img);
    });
}

#[bench]
fn bench_segmentation_real(bencher: divan::Bencher) {
    let (data, width, height) = load_icra_image();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let setup_arena = Bump::new();
    let config = locus_core::DetectorConfig::builder()
        .refinement_mode(CornerRefinementMode::Erf)
        .build();
    let engine = locus_core::threshold::ThresholdEngine::from_config(&config);

    let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
    let mut binarized = vec![0u8; width * height];
    let mut threshold_map = vec![0u8; width * height];
    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &tile_stats,
        &mut binarized,
        &mut threshold_map,
    );

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _label_result = locus_core::bench_api::label_components_threshold_model(
            &arena,
            &data,
            width,
            &threshold_map,
            width,
            height,
            true,
            16,
            1,
        );
    });
}

#[bench]
fn bench_quad_extraction_real(bencher: divan::Bencher) {
    let (data, width, height) = load_icra_image();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let setup_arena = Bump::new();
    let config = locus_core::DetectorConfig::builder()
        .refinement_mode(CornerRefinementMode::Erf)
        .build();
    let engine = locus_core::threshold::ThresholdEngine::from_config(&config);

    let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
    let mut binarized = vec![0u8; width * height];
    let mut threshold_map = vec![0u8; width * height];
    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &tile_stats,
        &mut binarized,
        &mut threshold_map,
    );
    let label_result = locus_core::bench_api::label_components_threshold_model(
        &setup_arena,
        &data,
        width,
        &threshold_map,
        width,
        height,
        true,
        16,
        1,
    );

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _quads = locus_core::bench_api::extract_quads_with_config(
            &arena,
            &img,
            &label_result,
            &config,
            1,
            &img,
        );
    });
}

#[bench]
fn bench_full_pipeline_real(bencher: divan::Bencher) {
    let (data, width, height) = load_icra_image();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let config = locus_core::DetectorConfig::builder()
        .refinement_mode(CornerRefinementMode::Erf)
        .build();
    let mut detector = locus_core::Detector::with_config(config);

    bencher.bench_local(move || {
        let _detections = detector.detect(&img, None, None, PoseEstimationMode::Fast);
    });
}
