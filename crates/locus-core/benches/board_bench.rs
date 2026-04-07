//! Micro-benchmark for `BoardEstimator::estimate()`.
//!
//! Isolates the LO-RANSAC + AW-LM board pose estimation step from the
//! upstream detection pipeline.  A realistic `DetectionBatch` is captured
//! once from the AprilGrid Golden v1 hub dataset during setup, then the
//! benchmark loop drives only `estimate()` — giving pristine IPC / L1-cache
//! numbers for the solver itself.
//!
//! Run (single-threaded as mandated by the micro-benchmarking guide):
//! ```bash
//! LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
//!   cargo bench --bench board_bench --features bench-internals -- --threads 1
//! ```
#![allow(
    missing_docs,
    dead_code,
    clippy::unwrap_used,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

use divan::Bencher;
use locus_core::{
    AprilGridTopology, CameraIntrinsics, Detector, DetectorConfig, ImageView, PoseEstimationMode,
    TagFamily,
    board::{BoardEstimator, LoRansacConfig},
};
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    // Single-threaded mandate: prevents Rayon / OS scheduler from thrashing L1.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

// ── Dataset helpers ──────────────────────────────────────────────────────────

fn hub_dir() -> PathBuf {
    std::env::var("LOCUS_HUB_DATASET_DIR").map_or_else(
        |_| {
            // CWD for bench binaries is the package root (crates/locus-core/),
            // so relative paths must be anchored via CARGO_MANIFEST_DIR.
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/hub_cache")
        },
        PathBuf::from,
    )
}

/// Parses the board config and camera intrinsics from `rich_truth.json`.
fn load_board_meta(dataset: &str) -> (Arc<AprilGridTopology>, CameraIntrinsics) {
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct Entry {
        record_type: String,
        #[serde(default)]
        k_matrix: Vec<Vec<f64>>,
        #[serde(default)]
        board_definition: Option<BoardDef>,
    }
    #[derive(Deserialize)]
    struct BoardDef {
        #[serde(rename = "type")]
        board_type: String,
        rows: usize,
        cols: usize,
        square_size_mm: f64,
        marker_size_mm: f64,
    }

    let path = hub_dir().join(dataset).join("rich_truth.json");
    let raw: Vec<Entry> = serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();

    let mut board_cfg: Option<Arc<AprilGridTopology>> = None;
    let mut intrinsics = None;

    for e in &raw {
        if e.record_type == "BOARD" {
            if board_cfg.is_none()
                && let Some(ref d) = e.board_definition
            {
                let sq_m = d.square_size_mm / 1000.0;
                let mk_m = d.marker_size_mm / 1000.0;
                board_cfg = Some(Arc::new(if d.board_type.contains("charuco") {
                    // Adapt ChAruco marker table for use with BoardEstimator.
                    let topo = locus_core::board::CharucoTopology::new(
                        d.rows,
                        d.cols,
                        sq_m,
                        mk_m,
                        usize::MAX,
                    )
                    .unwrap();
                    AprilGridTopology::from_obj_points(
                        topo.rows,
                        topo.cols,
                        topo.marker_length,
                        topo.obj_points,
                    )
                } else {
                    AprilGridTopology::new(d.rows, d.cols, sq_m - mk_m, mk_m, usize::MAX).unwrap()
                }));
            }
            if intrinsics.is_none() && e.k_matrix.len() >= 2 {
                intrinsics = Some(CameraIntrinsics::new(
                    e.k_matrix[0][0],
                    e.k_matrix[1][1],
                    e.k_matrix[0][2],
                    e.k_matrix[1][2],
                ));
            }
        }
        if board_cfg.is_some() && intrinsics.is_some() {
            break;
        }
    }

    (board_cfg.unwrap(), intrinsics.unwrap())
}

// ── Benchmark: isolated estimate() ──────────────────────────────────────────

/// Benchmarks `BoardEstimator::estimate()` in isolation using a real detection
/// batch captured from a representative AprilGrid Golden v1 frame.
///
/// The detection pipeline runs once in setup (not timed); only the board pose
/// estimation step is measured.
#[divan::bench]
fn bench_board_estimate_aprilgrid(bencher: Bencher) {
    const DATASET: &str = "aprilgrid_golden_v1";
    const IMAGE: &str = "scene_0010_cam_0000.png";

    let (board_config, intrinsics) = load_board_meta(DATASET);
    let mut estimator = BoardEstimator::new(Arc::clone(&board_config));

    // Run detector once to get a realistic batch — NOT timed.
    let img_path = hub_dir().join(DATASET).join("images").join(IMAGE);
    let luma = image::open(img_path).unwrap().to_luma8();
    let (w, h) = luma.dimensions();
    let raw = luma.into_raw();
    let img_view = ImageView::new(&raw, w as usize, h as usize, w as usize).unwrap();

    let mut detector = Detector::with_config(DetectorConfig::production_default());
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let _ = detector
        .detect(
            &img_view,
            Some(&intrinsics),
            Some(board_config.marker_length),
            PoseEstimationMode::Accurate,
            false,
        )
        .unwrap();

    let mut batch = detector.bench_api_get_batch_cloned();
    let v = batch.partition(batch.capacity());

    println!("\n  [setup] valid tags in batch: {v}");

    // MEASUREMENT: only estimate() is timed.
    bencher.bench_local(|| {
        let _ = estimator.estimate(&batch.view(v), &intrinsics);
    });
}

/// Same benchmark with a tighter LO-RANSAC config (fewer iterations) to show
/// the cost/accuracy trade-off surface.
#[divan::bench]
fn bench_board_estimate_aprilgrid_fast(bencher: Bencher) {
    const DATASET: &str = "aprilgrid_golden_v1";
    const IMAGE: &str = "scene_0010_cam_0000.png";

    let (board_config, intrinsics) = load_board_meta(DATASET);

    let fast_ransac = LoRansacConfig {
        k_max: 20,
        ..LoRansacConfig::default()
    };
    let mut estimator =
        BoardEstimator::new(Arc::clone(&board_config)).with_lo_ransac_config(fast_ransac);

    let img_path = hub_dir().join(DATASET).join("images").join(IMAGE);
    let luma = image::open(img_path).unwrap().to_luma8();
    let (w, h) = luma.dimensions();
    let raw = luma.into_raw();
    let img_view = ImageView::new(&raw, w as usize, h as usize, w as usize).unwrap();

    let mut detector = Detector::with_config(DetectorConfig::production_default());
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let _ = detector
        .detect(
            &img_view,
            Some(&intrinsics),
            Some(board_config.marker_length),
            PoseEstimationMode::Accurate,
            false,
        )
        .unwrap();

    let mut batch = detector.bench_api_get_batch_cloned();
    let v = batch.partition(batch.capacity());

    bencher.bench_local(|| {
        let _ = estimator.estimate(&batch.view(v), &intrinsics);
    });
}

/// Benchmarks the full detect + estimate pipeline end-to-end on a hub frame.
///
/// This is the Tier-1-equivalent latency number for board pose estimation.
/// Compare with `bench_board_estimate_aprilgrid` to understand the solver's
/// share of total frame latency.
#[divan::bench]
fn bench_board_full_pipeline_aprilgrid(bencher: Bencher) {
    const DATASET: &str = "aprilgrid_golden_v1";
    const IMAGE: &str = "scene_0010_cam_0000.png";

    let (board_config, intrinsics) = load_board_meta(DATASET);
    let mut estimator = BoardEstimator::new(Arc::clone(&board_config));

    let img_path = hub_dir().join(DATASET).join("images").join(IMAGE);
    let luma = image::open(img_path).unwrap().to_luma8();
    let (w, h) = luma.dimensions();
    let raw = luma.into_raw();
    let img_view = ImageView::new(&raw, w as usize, h as usize, w as usize).unwrap();

    let mut detector = Detector::with_config(DetectorConfig::production_default());
    detector.set_families(&[TagFamily::AprilTag36h11]);

    bencher.bench_local(|| {
        let _ = detector
            .detect(
                &img_view,
                Some(&intrinsics),
                Some(board_config.marker_length),
                PoseEstimationMode::Accurate,
                false,
            )
            .unwrap();
        let mut batch = detector.bench_api_get_batch_cloned();
        let v = batch.partition(batch.capacity());
        let _ = estimator.estimate(&batch.view(v), &intrinsics);
    });
}
