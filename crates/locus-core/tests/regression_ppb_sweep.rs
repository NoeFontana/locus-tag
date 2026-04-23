#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! PPB (pixels-per-bit) sweep regression.
//!
//! Loops over each PPB bin under `locus_ppb_sweep_v1` and runs four
//! configurations against each bin:
//!   1. shipped `standard` profile
//!   2. shipped `high_accuracy` profile
//!   3. hand-crafted Static + ContourRdp + Erf
//!   4. hand-crafted Static + EdLines + None
//!
//! Emits one insta snapshot per scenario. The aggregated curves drive the
//! empirical threshold selection for `max_recall_adaptive`: smallest T such
//! that for all bins >= T, `recall_edlines >= recall_contourrdp - 0.01` AND
//! `rmse_edlines <= rmse_contourrdp`.
//!
//! Bins are auto-discovered as any immediate subdirectory of the sweep root
//! that contains both `images/` and `rich_truth.json`. Sorted alphabetically
//! for snapshot determinism.
//!
//! The dataset is synced via `tools/cli.py bench prepare`. This test
//! auto-skips when `LOCUS_HUB_DATASET_DIR/locus_ppb_sweep_v1` is absent.

use locus_core::config::{
    CornerRefinementMode, DecodeMode, QuadExtractionMode, QuadExtractionPolicy,
};
use locus_core::{DetectOptions, Detector, DetectorConfig, ImageView, Pose, TagFamily};
use nalgebra::{UnitQuaternion, Vector3};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

mod common;

use common::hub::{DatasetProvider, HubProvider, calculate_percentiles, serialize_rmse};
use common::resolve_hub_root;

const HUB_CONFIG: &str = "locus_ppb_sweep_v1";

#[derive(Serialize)]
struct BinSummary {
    bin: String,
    images: usize,
    mean_recall: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_reprojection_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p50_translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p90_translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p50_rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p90_rotation_error: f64,
}

#[derive(Serialize)]
struct ScenarioReport {
    scenario: String,
    bins: Vec<BinSummary>,
}

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
            label: "high_accuracy",
            config: DetectorConfig::from_profile("high_accuracy"),
        },
        Scenario {
            label: "static_contour_erf",
            config: DetectorConfig {
                quad_extraction_policy: QuadExtractionPolicy::Static,
                quad_extraction_mode: QuadExtractionMode::ContourRdp,
                refinement_mode: CornerRefinementMode::Erf,
                decode_mode: DecodeMode::Hard,
                ..DetectorConfig::default()
            },
        },
        Scenario {
            label: "static_edlines_none",
            config: DetectorConfig {
                quad_extraction_policy: QuadExtractionPolicy::Static,
                quad_extraction_mode: QuadExtractionMode::EdLines,
                refinement_mode: CornerRefinementMode::None,
                decode_mode: DecodeMode::Hard,
                ..DetectorConfig::default()
            },
        },
    ]
}

fn discover_bins(sweep_root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(sweep_root) else {
        return Vec::new();
    };
    let mut bins: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("images").is_dir() && p.join("rich_truth.json").is_file())
        .collect();
    bins.sort();
    bins
}

fn evaluate_bin(bin_name: &str, provider: &HubProvider, config: DetectorConfig) -> BinSummary {
    let mut detector = Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let options = DetectOptions::default();

    let mut total_recall = 0.0;
    let mut total_rmse = 0.0;
    let mut total_repro_rmse = 0.0;
    let mut count = 0usize;
    let mut translation_errors: Vec<f64> = Vec::new();
    let mut rotation_errors: Vec<f64> = Vec::new();

    for (_fname, data, w, h, gt) in provider.iter() {
        let img = ImageView::new(&data, w, h, w).expect("valid image");
        let intrinsics = gt.intrinsics.or(options.intrinsics);
        let tag_size = gt.tag_size.or(options.tag_size);

        let detections = detector
            .detect(
                &img,
                intrinsics.as_ref(),
                tag_size,
                options.pose_estimation_mode,
                false,
            )
            .expect("detection");

        let mut image_rmse_sum = 0.0;
        let mut image_repro_rmse_sum = 0.0;
        let mut match_count = 0u32;
        let mut pose_match_count = 0u32;
        let mut found_ids = BTreeSet::new();

        for i in 0..detections.len() {
            let det_id = detections.ids[i];
            let det_corners = detections.corners[i];
            let det_corners_f64: [[f64; 2]; 4] = [
                [f64::from(det_corners[0].x), f64::from(det_corners[0].y)],
                [f64::from(det_corners[1].x), f64::from(det_corners[1].y)],
                [f64::from(det_corners[2].x), f64::from(det_corners[2].y)],
                [f64::from(det_corners[3].x), f64::from(det_corners[3].y)],
            ];
            let cx = det_corners_f64.iter().map(|p| p[0]).sum::<f64>() / 4.0;
            let cy = det_corners_f64.iter().map(|p| p[1]).sum::<f64>() / 4.0;

            let Some(gt_corners) = gt.tags.get(&det_id) else {
                continue;
            };
            let g_cx: f64 = gt_corners.iter().map(|p| p[0]).sum::<f64>() / 4.0;
            let g_cy: f64 = gt_corners.iter().map(|p| p[1]).sum::<f64>() / 4.0;
            if (cx - g_cx).powi(2) + (cy - g_cy).powi(2) >= 100.0 * 100.0 {
                continue;
            }

            let mut rmse_sq = 0.0;
            for k in 0..4 {
                rmse_sq += (det_corners_f64[k][0] - gt_corners[k][0]).powi(2)
                    + (det_corners_f64[k][1] - gt_corners[k][1]).powi(2);
            }
            image_rmse_sum += (rmse_sq / 4.0).sqrt();
            match_count += 1;
            found_ids.insert(det_id);

            let p = detections.poses[i].data;
            if p[2] <= 0.0 {
                continue;
            }
            let det_t = Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
            let det_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                f64::from(p[6]),
                f64::from(p[3]),
                f64::from(p[4]),
                f64::from(p[5]),
            ));

            if let (Some(gt_pose), Some(intr), Some(size)) =
                (gt.poses.get(&det_id), intrinsics, tag_size)
            {
                let q_gt = UnitQuaternion::from_matrix(&gt_pose.rotation);
                translation_errors.push((det_t - gt_pose.translation).norm());
                rotation_errors.push(det_q.angle_to(&q_gt).to_degrees());
                pose_match_count += 1;

                let h = size * 0.5;
                let model_corners = [
                    Vector3::new(-h, -h, 0.0),
                    Vector3::new(h, -h, 0.0),
                    Vector3::new(h, h, 0.0),
                    Vector3::new(-h, h, 0.0),
                ];
                let est_pose = Pose::new(det_q.to_rotation_matrix().into_inner(), det_t);
                let mut repro_sq = 0.0;
                for k in 0..4 {
                    let proj = est_pose.project(&model_corners[k], &intr);
                    repro_sq +=
                        (proj[0] - gt_corners[k][0]).powi(2) + (proj[1] - gt_corners[k][1]).powi(2);
                }
                image_repro_rmse_sum += (repro_sq / 4.0).sqrt();
            }
        }

        let recall = if gt.tags.is_empty() {
            1.0
        } else {
            found_ids.len() as f64 / gt.tags.len() as f64
        };
        let avg_rmse = if match_count > 0 {
            image_rmse_sum / f64::from(match_count)
        } else {
            0.0
        };
        let avg_repro = if pose_match_count > 0 {
            image_repro_rmse_sum / f64::from(pose_match_count)
        } else {
            0.0
        };

        total_recall += recall;
        total_rmse += avg_rmse;
        total_repro_rmse += avg_repro;
        count += 1;
    }

    let (p50_t, p90_t, _) = calculate_percentiles(&mut translation_errors);
    let (p50_r, p90_r, _) = calculate_percentiles(&mut rotation_errors);
    let n = count.max(1) as f64;

    BinSummary {
        bin: bin_name.to_string(),
        images: count,
        mean_recall: total_recall / n,
        mean_rmse: total_rmse / n,
        mean_reprojection_rmse: total_repro_rmse / n,
        p50_translation_error: p50_t,
        p90_translation_error: p90_t,
        p50_rotation_error: p50_r,
        p90_rotation_error: p90_r,
    }
}

#[test]
fn ppb_sweep_per_scenario_curves() {
    let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") else {
        println!("Skipping regression_ppb_sweep: LOCUS_HUB_DATASET_DIR unset.");
        return;
    };
    let sweep_root = resolve_hub_root(&hub_dir).join(HUB_CONFIG);
    if !sweep_root.exists() {
        println!(
            "Skipping regression_ppb_sweep: {HUB_CONFIG} not in cache. \
             Run `tools/cli.py bench prepare`."
        );
        return;
    }
    let bins = discover_bins(&sweep_root);
    assert!(
        !bins.is_empty(),
        "{HUB_CONFIG} exists at {} but no PPB bins were found (expected \
         subdirectories with images/ and rich_truth.json)",
        sweep_root.display()
    );

    let providers: Vec<(String, HubProvider)> = bins
        .iter()
        .map(|p| {
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let provider = HubProvider::new(p).expect("bin must have images/rich_truth.json");
            (name, provider)
        })
        .collect();

    let mut per_scenario: BTreeMap<&'static str, ScenarioReport> = BTreeMap::new();
    for scenario in scenarios() {
        let mut report = ScenarioReport {
            scenario: scenario.label.to_string(),
            bins: Vec::with_capacity(providers.len()),
        };
        for (name, provider) in &providers {
            report
                .bins
                .push(evaluate_bin(name, provider, scenario.config));
        }
        per_scenario.insert(scenario.label, report);
    }

    for (label, report) in &per_scenario {
        insta::assert_yaml_snapshot!(format!("ppb_sweep__{label}"), report);
    }
}
