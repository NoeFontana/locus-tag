#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::print_stdout,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! ICRA forward frame-0..5 rejection-stage diagnostic.
//!
//! Frames 0000.png–0005.png on the ICRA forward dataset detect ~half their
//! tags under `standard` and *zero* under `high_accuracy`. This binary
//! attributes the rejection: did the funnel cull pre-decode, or did decode
//! fail post-funnel? Run on demand:
//!
//! ```text
//! LOCUS_ICRA_DATASET_DIR=tests/data/icra2020 \
//!   cargo test --release --features bench-internals \
//!   --test icra_forward_diagnostic -- --ignored --nocapture
//! ```
//!
//! Output is printed to stdout; nothing is written to disk.

mod common;

use locus_core::bench_api::FunnelStatus;
use locus_core::{Detector, DetectorConfig, ImageView, PoseEstimationMode};

const FRAMES: &[&str] = &[
    "0000.png", "0001.png", "0002.png", "0003.png", "0004.png", "0005.png",
];
const PROFILES: &[&str] = &["standard", "high_accuracy"];

#[test]
#[ignore = "diagnostic — run manually"]
fn diagnose_icra_forward_frames_0_5() {
    let Some(root) = common::resolve_dataset_root() else {
        println!("LOCUS_ICRA_DATASET_DIR unset and no fallback found — skipping.");
        return;
    };
    let img_dir = root.join("forward").join("pure_tags_images");
    if !img_dir.is_dir() {
        println!("ICRA forward image directory not found: {img_dir:?}");
        return;
    }

    println!("=== ICRA forward diagnostic — frames 0..5 ===");
    for profile in PROFILES {
        let cfg = DetectorConfig::from_profile(profile);
        let mut detector = Detector::with_config(cfg);
        println!("\n--- profile: {profile} ---");
        for frame in FRAMES {
            let path = img_dir.join(frame);
            if !path.is_file() {
                println!("  {frame}: missing");
                continue;
            }
            let raw = image::open(&path).expect("decode png").into_luma8();
            let (w, h) = raw.dimensions();
            let bytes = raw.into_raw();
            let view = ImageView::new(&bytes, w as usize, h as usize, w as usize)
                .expect("valid image view");
            let det = detector
                .detect(&view, None, None, PoseEstimationMode::Fast, true)
                .expect("detect ok");
            let n_valid = det.len();
            let n_rejected = det.rejected_corners.len();
            let funnel_hist = histogram(det.rejected_funnel_status);
            let size_hist = corner_size_histogram(det.rejected_corners);
            let hamming_hist =
                decode_hamming_histogram(det.rejected_funnel_status, det.rejected_error_rates);
            println!(
                "  {frame}: valid={n_valid} rejected={n_rejected}  funnel={funnel_hist}  rejected_size_hist={size_hist}  decode_hamming={hamming_hist}"
            );
        }
    }
}

fn histogram(states: &[FunnelStatus]) -> String {
    let mut none = 0usize;
    let mut pass_contrast = 0usize;
    let mut rej_contrast = 0usize;
    let mut rej_sampling = 0usize;
    for s in states {
        match s {
            FunnelStatus::None => none += 1,
            FunnelStatus::PassedContrast => pass_contrast += 1,
            FunnelStatus::RejectedContrast => rej_contrast += 1,
            FunnelStatus::RejectedSampling => rej_sampling += 1,
        }
    }
    format!(
        "None={none} PassContrast={pass_contrast} RejContrast={rej_contrast} RejSampling={rej_sampling}"
    )
}

/// Histograms the best Hamming distance recorded for each rejected candidate
/// that *passed* the funnel and reached the decode stage.
///
/// Per `decoder.rs:1539`, `error_rate == 0.0` on a `FailedDecode` candidate is a
/// sentinel meaning `best_h == u32::MAX` — i.e., bit sampling never produced a
/// usable code (e.g., homography projected out of bounds). Anything else is the
/// minimum Hamming distance to any dictionary entry across all rotations.
fn decode_hamming_histogram(funnel: &[FunnelStatus], error_rates: &[f32]) -> String {
    debug_assert_eq!(funnel.len(), error_rates.len());
    let mut no_sample = 0usize;
    let mut bins = [0usize; 5];
    let edges = [3.0_f32, 6.0, 11.0, 21.0];
    for (s, &e) in funnel.iter().zip(error_rates) {
        if !matches!(s, FunnelStatus::PassedContrast) {
            continue;
        }
        if e <= 0.0 {
            no_sample += 1;
            continue;
        }
        let mut placed = false;
        for (i, &edge) in edges.iter().enumerate() {
            if e < edge {
                bins[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            bins[4] += 1;
        }
    }
    format!(
        "no_sample={no_sample} h<3={} 3-5={} 6-10={} 11-20={} >=21={}",
        bins[0], bins[1], bins[2], bins[3], bins[4],
    )
}

fn corner_size_histogram(corners: &[[locus_core::bench_api::Point2f; 4]]) -> String {
    let mut bins = [0usize; 6];
    let edges = [8.0_f32, 12.0, 18.0, 28.0, 48.0];
    for c in corners {
        let xs = [c[0].x, c[1].x, c[2].x, c[3].x];
        let ys = [c[0].y, c[1].y, c[2].y, c[3].y];
        let dx = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - xs.iter().copied().fold(f32::INFINITY, f32::min);
        let dy = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - ys.iter().copied().fold(f32::INFINITY, f32::min);
        let side = dx.max(dy);
        let mut placed = false;
        for (i, &e) in edges.iter().enumerate() {
            if side < e {
                bins[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            bins[5] += 1;
        }
    }
    format!(
        "<8={} 8-12={} 12-18={} 18-28={} 28-48={} >=48={}",
        bins[0], bins[1], bins[2], bins[3], bins[4], bins[5]
    )
}
