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
//! End-to-end benchmarks for the ROI rescue pipeline.
//!
//! The rescue stage is disabled by default and must add zero measurable
//! latency when off; `bench_detection_without_rescue` establishes the
//! baseline and `bench_detection_with_rescue` measures the worst-case
//! on-frame overhead with rescue enabled on a real ICRA frame.
mod utils;

use divan::bench;
use locus_core::config::RoiRescuePolicy;
use locus_core::{DetectOptions, Detector, DetectorConfig, ImageView, TagFamily};
use utils::BenchDataset;

fn main() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
    divan::Divan::from_args().threads([1]).run_benches();
}

fn run(detector: &mut Detector, img: &ImageView<'_>, options: &DetectOptions) {
    let _ = detector
        .detect(
            img,
            options.intrinsics.as_ref(),
            options.tag_size,
            options.pose_estimation_mode,
            false,
        )
        .expect("detection");
}

#[bench]
fn bench_detection_without_rescue(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let mut detector = Detector::with_config(DetectorConfig::default());
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let options = DetectOptions::default();
    bencher.bench_local(move || run(&mut detector, &img, &options));
}

#[bench]
fn bench_detection_with_rescue(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    // Rescue budget must be strictly smaller than the first-pass budget.
    // Default max_hamming_error = 2 → rescue budget = 1.
    let config = DetectorConfig {
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
    bencher.bench_local(move || run(&mut detector, &img, &options));
}

/// Upper bound for rescue pressure: `max_rescues_per_frame = 16` on the
/// ICRA forward frame. Swap for a hand-crafted mosaic once the
/// `locus_ppb_sweep_v1` dataset lands.
#[bench]
fn bench_rescue_16_candidates(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let config = DetectorConfig {
        roi_rescue: RoiRescuePolicy {
            enabled: true,
            rescue_max_hamming: 1,
            max_rescues_per_frame: 16,
            ..RoiRescuePolicy::default()
        },
        ..DetectorConfig::default()
    };
    let mut detector = Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let options = DetectOptions::default();
    bencher.bench_local(move || run(&mut detector, &img, &options));
}
