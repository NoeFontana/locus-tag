#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! Negative-dataset regression: `locus_negative_v1` must produce **zero**
//! detections across every shipped profile × policy combination.
//!
//! The dataset is 500 tag-free scenes with rich textured backgrounds
//! (clutter, brick, foliage). Any non-zero detection count is a false
//! positive and blocks merge — particularly important for the opt-in
//! `AdaptivePpb` router and the ROI-rescue path, which expand the set of
//! candidates that reach the decoder.
//!
//! The dataset is synced via `tools/cli.py bench prepare`. This test
//! auto-skips when `LOCUS_HUB_DATASET_DIR/locus_negative_v1` is absent.

use locus_core::config::{AdaptivePpbConfig, QuadExtractionPolicy, RoiRescuePolicy};
use locus_core::{DetectOptions, Detector, DetectorConfig, ImageView, TagFamily};

mod common;

use common::hub::{DatasetProvider, HubProvider};
use common::resolve_hub_root;

const HUB_CONFIG: &str = "locus_negative_v1";

struct Scenario {
    label: &'static str,
    config: DetectorConfig,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            label: "standard",
            config: DetectorConfig::from_profile("standard"),
        },
        Scenario {
            label: "grid",
            config: DetectorConfig::from_profile("grid"),
        },
        Scenario {
            label: "high_accuracy",
            config: DetectorConfig::from_profile("high_accuracy"),
        },
        Scenario {
            label: "max_recall_adaptive",
            config: DetectorConfig::from_profile("max_recall_adaptive"),
        },
        Scenario {
            label: "standard+rescue",
            config: DetectorConfig {
                roi_rescue: RoiRescuePolicy {
                    enabled: true,
                    rescue_max_hamming: 1,
                    ..RoiRescuePolicy::default()
                },
                ..DetectorConfig::from_profile("standard")
            },
        },
        Scenario {
            label: "adaptive+rescue",
            config: DetectorConfig {
                quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(
                    AdaptivePpbConfig::default(),
                ),
                roi_rescue: RoiRescuePolicy {
                    enabled: true,
                    rescue_max_hamming: 1,
                    ..RoiRescuePolicy::default()
                },
                ..DetectorConfig::default()
            },
        },
    ]
}

#[test]
fn negative_dataset_zero_false_positives() {
    let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") else {
        println!("Skipping regression_negative_dataset: LOCUS_HUB_DATASET_DIR unset.");
        return;
    };
    let dataset_path = resolve_hub_root(&hub_dir).join(HUB_CONFIG);
    if !dataset_path.exists() {
        println!(
            "Skipping regression_negative_dataset: {HUB_CONFIG} not in cache. \
             Run `tools/cli.py bench prepare`."
        );
        return;
    }
    let Some(provider) = HubProvider::new(&dataset_path) else {
        panic!("{HUB_CONFIG} exists but has no images/rich_truth.json");
    };

    let options = DetectOptions::default();
    let mut failures: Vec<String> = Vec::new();

    for scenario in scenarios() {
        let mut detector = Detector::with_config(scenario.config);
        detector.set_families(&[TagFamily::AprilTag36h11]);

        let mut total_fp = 0usize;
        let mut worst_cases: Vec<(String, usize)> = Vec::new();

        for (fname, data, w, h, _gt) in provider.iter() {
            let img = ImageView::new(&data, w, h, w).expect("valid image");
            let detections = detector
                .detect(
                    &img,
                    options.intrinsics.as_ref(),
                    options.tag_size,
                    options.pose_estimation_mode,
                    false,
                )
                .expect("detection");
            let n = detections.len();
            if n > 0 {
                total_fp += n;
                worst_cases.push((fname, n));
            }
        }

        if total_fp > 0 {
            worst_cases.sort_by(|a, b| b.1.cmp(&a.1));
            let worst: Vec<String> = worst_cases
                .iter()
                .take(5)
                .map(|(f, n)| format!("    {f}: {n} FPs"))
                .collect();
            failures.push(format!(
                "{}: {total_fp} false positives across {} image(s)\n{}",
                scenario.label,
                worst_cases.len(),
                worst.join("\n")
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "regression_negative_dataset detected false positives:\n{}",
        failures.join("\n")
    );
}
