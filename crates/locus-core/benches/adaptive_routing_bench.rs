#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
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
//! Benchmarks for the AdaptivePpb router.
//!
//! * `bench_ppb_estimate` — tight scalar loop isolating the per-candidate
//!   `min(bbox_w, bbox_h) as f32 / outer_dim as f32` cost. The plan's
//!   non-negotiable is that under `QuadExtractionPolicy::Static` there's
//!   zero measurable latency delta vs. today; this bench upper-bounds
//!   the scalar overhead path takes when routing IS enabled.
//! * `bench_detection_static` / `bench_detection_adaptive` — end-to-end
//!   detector runs on an ICRA frame, comparing the two policies. Gates
//!   the ±5% regression budget on `bench_detection_static` in CI.
mod utils;

use divan::bench;
use locus_core::bench_api::ComponentStats;
use locus_core::config::{AdaptivePpbConfig, QuadExtractionPolicy};
use locus_core::{DetectOptions, Detector, DetectorConfig, ImageView, TagFamily};
use utils::BenchDataset;

fn main() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
    divan::Divan::from_args().threads([1]).run_benches();
}

fn synthesize_stats(n: usize) -> Vec<ComponentStats> {
    (0..n)
        .map(|i| {
            let side = 4 + (i as u16 % 60);
            ComponentStats {
                min_x: 0,
                max_x: side,
                min_y: 0,
                max_y: side,
                pixel_count: u32::from(side) * u32::from(side),
                first_pixel_x: 0,
                first_pixel_y: 0,
                m10: 0,
                m01: 0,
                m20: 0,
                m02: 0,
                m11: 0,
            }
        })
        .collect()
}

#[bench]
fn bench_ppb_estimate(bencher: divan::Bencher) {
    let stats = synthesize_stats(1024);
    // AprilTag 36h11 outer dim = 8 bits (inner grid 6 + 2 border cells).
    let outer_dim = 8.0_f32;

    bencher.bench_local(|| {
        let mut acc = 0.0_f32;
        for s in &stats {
            let w = u32::from(s.max_x - s.min_x) + 1;
            let h = u32::from(s.max_y - s.min_y) + 1;
            let ppb = w.min(h) as f32 / outer_dim;
            acc += divan::black_box(ppb);
        }
        divan::black_box(acc);
    });
}

fn run_detection(detector: &mut Detector, img: &ImageView<'_>, options: &DetectOptions) {
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
fn bench_detection_static(bencher: divan::Bencher) {
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

    bencher.bench_local(move || {
        run_detection(&mut detector, &img, &options);
    });
}

#[bench]
fn bench_detection_adaptive(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let config = DetectorConfig {
        quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig::default()),
        ..DetectorConfig::default()
    };
    let mut detector = Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let options = DetectOptions::default();

    bencher.bench_local(move || {
        run_detection(&mut detector, &img, &options);
    });
}
