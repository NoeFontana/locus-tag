#![cfg(feature = "dhat-heap")]
#![allow(clippy::expect_used, clippy::unwrap_used, dead_code, missing_docs)]
//! DHAT allocation-growth gate for the opt-in `AdaptivePpb` router + ROI
//! rescue path.
//!
//! `AdaptivePpb` + rescue legitimately grow the arena / `rescue_buf` for the
//! first handful of frames while the buffers settle at their steady-state
//! capacity. Once stable the per-frame allocation rate must not increase.
//! Mirrors the drift-rate check in `allocation_zero_alloc_static.rs`, with a
//! longer warmup to let the rescue-buffer resize settle.
//!
//! Runs only with `--features dhat-heap`.

use locus_core::config::{AdaptivePpbConfig, QuadExtractionPolicy, RoiRescuePolicy};
use locus_core::{DetectOptions, Detector, DetectorConfig, ImageView, TagFamily};
use std::path::PathBuf;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const WARMUP_FRAMES: u64 = 20;
const WINDOW_FRAMES: u64 = 40;
const DRIFT_TOLERANCE: f64 = 1.05;

fn load_fixture() -> (Vec<u8>, usize, usize) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/icra2020/0037.png");
    let img = image::open(&path).expect("fixture png").into_luma8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w as usize, h as usize)
}

fn run_frames(detector: &mut Detector, img: &ImageView, options: &DetectOptions, count: u64) {
    for _ in 0..count {
        detector
            .detect(img, None, None, options.pose_estimation_mode, false)
            .unwrap();
    }
}

#[test]
fn adaptive_rescue_bounded_growth() {
    let _profiler = dhat::Profiler::builder().testing().build();

    let (data, w, h) = load_fixture();
    let img = ImageView::new(&data, w, h, w).unwrap();
    let config = DetectorConfig {
        quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig::default()),
        roi_rescue: RoiRescuePolicy {
            enabled: true,
            rescue_max_hamming: 1,
            ..RoiRescuePolicy::default()
        },
        ..DetectorConfig::default()
    };
    let mut detector = Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let options = DetectOptions::default();

    run_frames(&mut detector, &img, &options, WARMUP_FRAMES);
    let s0 = dhat::HeapStats::get();

    run_frames(&mut detector, &img, &options, WINDOW_FRAMES);
    let s1 = dhat::HeapStats::get();

    run_frames(&mut detector, &img, &options, WINDOW_FRAMES);
    let s2 = dhat::HeapStats::get();

    let early_blocks = s1.total_blocks - s0.total_blocks;
    let late_blocks = s2.total_blocks - s1.total_blocks;
    let early_rate = early_blocks as f64 / WINDOW_FRAMES as f64;
    let late_rate = late_blocks as f64 / WINDOW_FRAMES as f64;
    let limit = early_rate * DRIFT_TOLERANCE;

    eprintln!(
        "adaptive+rescue: early={early_blocks} blocks over {WINDOW_FRAMES} frames ({early_rate:.1}/frame), \
         late={late_blocks} blocks ({late_rate:.1}/frame), limit={limit:.1}/frame"
    );

    assert!(
        late_rate <= limit,
        "allocation rate drifted: early={early_rate:.1}/frame, late={late_rate:.1}/frame \
         (limit={limit:.1}/frame, tolerance={DRIFT_TOLERANCE}×)"
    );
}
