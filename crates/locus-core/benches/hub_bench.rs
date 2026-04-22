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
//! Benchmarking over Hub datasets with per-stage breakdown.
//! Uses the bench-internals feature to access private pipeline stages.

use bumpalo::Bump;
use divan::Bencher;
use locus_core::bench_api::{
    DetectionBatch, ThresholdEngine, extract_quads_soa, label_components_lsl,
};
use locus_core::{DetectorConfig, ImageView};
use std::path::PathBuf;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::main();
}

fn get_hub_dir() -> PathBuf {
    std::env::var("LOCUS_HUB_DATASET_DIR")
        .map_or_else(|_| PathBuf::from("tests/data/hub_cache"), PathBuf::from)
}

fn load_image(dataset: &str, filename: &str) -> (Vec<u8>, usize, usize) {
    let path = get_hub_dir().join(dataset).join("images").join(filename);
    let img = image::open(path)
        .expect("Failed to open hub image")
        .to_luma8();
    let (width, height) = img.dimensions();
    (img.into_raw(), width as usize, height as usize)
}

macro_rules! hub_benchmarks {
    ($mod_name:ident, $dataset:expr) => {
        mod $mod_name {
            use super::*;

            #[divan::bench]
            fn thresholding(bencher: Bencher) {
                let (data, width, height) = load_image($dataset, "scene_0000_cam_0000.png");
                let img = ImageView::new(&data, width, height, width).unwrap();
                let config = DetectorConfig::default();
                let engine = ThresholdEngine::from_config(&config);

                bencher.bench_local(move || {
                    let arena = Bump::new();
                    let mut binarized = vec![0u8; width * height];
                    let mut threshold_map = vec![0u8; width * height];
                    let tile_stats = engine.compute_tile_stats(&arena, &img);
                    engine.apply_threshold_with_map(
                        &arena,
                        &img,
                        &tile_stats,
                        &mut binarized,
                        &mut threshold_map,
                    );
                });
            }

            #[divan::bench]
            fn segmentation(bencher: Bencher) {
                let (data, width, height) = load_image($dataset, "scene_0000_cam_0000.png");
                let img = ImageView::new(&data, width, height, width).unwrap();
                let config = DetectorConfig::default();
                let engine = ThresholdEngine::from_config(&config);
                let setup_arena = Bump::new();
                let mut binarized = vec![0u8; width * height];
                let mut threshold_map = vec![0u8; width * height];
                let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
                engine.apply_threshold_with_map(
                    &setup_arena,
                    &img,
                    &tile_stats,
                    &mut binarized,
                    &mut threshold_map,
                );

                bencher.bench_local(move || {
                    let arena = Bump::new();
                    let _label_result = label_components_lsl(
                        &arena,
                        &img,
                        &threshold_map,
                        true,
                        config.quad_min_area,
                    );
                });
            }

            #[divan::bench]
            fn quad_extraction(bencher: Bencher) {
                let (data, width, height) = load_image($dataset, "scene_0000_cam_0000.png");
                let img = ImageView::new(&data, width, height, width).unwrap();
                let config = DetectorConfig::default();
                let engine = ThresholdEngine::from_config(&config);
                let setup_arena = Bump::new();
                let mut binarized = vec![0u8; width * height];
                let mut threshold_map = vec![0u8; width * height];
                let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
                engine.apply_threshold_with_map(
                    &setup_arena,
                    &img,
                    &tile_stats,
                    &mut binarized,
                    &mut threshold_map,
                );
                let label_result = label_components_lsl(
                    &setup_arena,
                    &img,
                    &threshold_map,
                    true,
                    config.quad_min_area,
                );

                bencher.bench_local(move || {
                    let _arena = Bump::new();
                    let mut batch = DetectionBatch::new();
                    let _n = extract_quads_soa(
                        &mut batch,
                        &img,
                        &label_result,
                        &config,
                        1,
                        &img,
                        &img,
                        6,
                        false,
                    );
                });
            }

            #[divan::bench]
            fn full_pipeline(bencher: Bencher) {
                let (data, width, height) = load_image($dataset, "scene_0000_cam_0000.png");
                let img = ImageView::new(&data, width, height, width).unwrap();
                let mut detector = locus_core::Detector::new();
                detector.set_families(&[locus_core::TagFamily::AprilTag36h11]);

                bencher.bench_local(move || {
                    let _detections = detector
                        .detect(
                            &img,
                            None,
                            None,
                            locus_core::PoseEstimationMode::Fast,
                            false,
                        )
                        .unwrap();
                });
            }
        }
    };
}

hub_benchmarks!(res_480p, "locus_v1_tag36h11_640x480");
hub_benchmarks!(res_720p, "locus_v1_tag36h11_1280x720");
hub_benchmarks!(res_1080p, "locus_v1_tag36h11_1920x1080");
