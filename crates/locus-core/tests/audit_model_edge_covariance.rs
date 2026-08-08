//! Phase-A calibration audit for the model-edge pose covariance (honest SoA
//! covariance emission — the ship gate).
//!
//! For every matched tag on the hub render-tag corpus it runs the production
//! edge-refined detect() path, computes the combined edge+corner body-frame
//! covariance ([`bench_model_edge_covariance`]), forms the body-frame SE(3)
//! residual against ground truth `δ = [R_gtᵀ(t_det−t_gt); log(R_gtᵀR_det)]`, and
//! dumps `(δ, Σ, d²)` per scene to a JSON the Python stats step
//! (`tools/bench/model_edge_cov_stats.py`) aggregates into mean d² / KL(‖χ²(6)) /
//! per-axis ratios. A well-calibrated Σ gives `d² ~ χ²(6)` (mean 6). Baseline (the
//! 4-corner weighted LM) is mean d²=714.7, KL=13.93.
//!
//! Requires `LOCUS_HUB_DATASET_DIR`; skips gracefully otherwise. Bench-internals.
//! Run: `LOCUS_HUB_DATASET_DIR=tests/data/hub_cache cargo test --release \
//!   -p locus-core --features bench-internals --test audit_model_edge_covariance \
//!   -- --nocapture`

#![cfg(feature = "bench-internals")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_lines)]

mod common;

use std::path::PathBuf;

use locus_core::bench_api::bench_model_edge_covariance;
use locus_core::{DetectorConfig, ImageView, Pose, TagFamily};
use nalgebra::{Matrix6, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector6};

use common::hub::{DatasetProvider, HubProvider};

#[derive(serde::Serialize)]
struct Sample {
    image: String,
    tag_id: u32,
    delta: [f64; 6],
    cov: Vec<f64>,
    d2: f64,
    d2_per_axis: [f64; 6],
}

#[test]
fn audit_model_edge_covariance() {
    let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") else {
        eprintln!("skip: set LOCUS_HUB_DATASET_DIR to run the covariance audit");
        return;
    };
    let config_name = std::env::var("MODEL_EDGE_COV_CONFIG")
        .unwrap_or_else(|_| "locus_v1_tag36h11_1920x1080".into());
    let dataset_path = common::resolve_hub_root(&hub_dir).join(&config_name);
    let Some(provider) = HubProvider::new(&dataset_path) else {
        eprintln!("skip: dataset {config_name} not found under {hub_dir}");
        return;
    };

    let mut config = DetectorConfig::from_profile("high_accuracy");
    config.pose_edge_refinement_enabled = true;
    let sigma_n_sq = config.sigma_n_sq;
    let sigma_edge_scale: f64 = std::env::var("MODEL_EDGE_SIGMA_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let sigma_corner_scale: f64 = std::env::var("MODEL_EDGE_CORNER_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let block_diagonal = std::env::var("MODEL_EDGE_BLOCK").is_ok();
    let mut detector = locus_core::Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);

    let mut samples: Vec<Sample> = Vec::new();
    for (filename, data, width, height, gt) in provider.iter() {
        let img = ImageView::new(&data, width, height, width).expect("valid image");
        let Some(intr) = gt.intrinsics else { continue };
        let Some(tag_size) = gt.tag_size else {
            continue;
        };
        let detections = detector
            .detect(&img, Some(&intr), Some(tag_size), false)
            .expect("detection failed");

        for i in 0..detections.len() {
            let det_id = detections.ids[i];
            let Some(gt_pose) = gt.poses.get(&det_id) else {
                continue;
            };
            let p = detections.poses[i].data; // [tx,ty,tz, qx,qy,qz,qw]
            if p.iter().all(|&v| v == 0.0) {
                continue; // no pose emitted
            }
            let t_det = Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
            let q_det = UnitQuaternion::new_normalize(Quaternion::new(
                f64::from(p[6]),
                f64::from(p[3]),
                f64::from(p[4]),
                f64::from(p[5]),
            ));
            let r_det = q_det.to_rotation_matrix().into_inner();
            let corners_f = detections.corners[i];
            let corners = [
                [f64::from(corners_f[0].x), f64::from(corners_f[0].y)],
                [f64::from(corners_f[1].x), f64::from(corners_f[1].y)],
                [f64::from(corners_f[2].x), f64::from(corners_f[2].y)],
                [f64::from(corners_f[3].x), f64::from(corners_f[3].y)],
            ];

            let refined = Pose::new(r_det, t_det);
            let Some(cov) = bench_model_edge_covariance(
                &intr,
                &img,
                6,
                tag_size,
                &refined,
                &corners,
                sigma_n_sq,
                sigma_edge_scale,
                sigma_corner_scale,
                block_diagonal,
            ) else {
                continue; // edge refinement / covariance unavailable for this tag
            };
            let sigma = Matrix6::from_fn(|r, c| cov[r][c]);
            let Some(sigma_inv) = sigma.try_inverse() else {
                continue;
            };

            // Body-frame SE(3) residual vs GT: δ = [R_gtᵀ(t_det−t_gt); log(R_gtᵀR_det)].
            let rgt_t = gt_pose.rotation.transpose();
            let d_t = rgt_t * (t_det - gt_pose.translation);
            let d_r = Rotation3::from_matrix_unchecked(rgt_t * r_det).scaled_axis();
            let delta = Vector6::new(d_t.x, d_t.y, d_t.z, d_r.x, d_r.y, d_r.z);

            let d2 = (delta.transpose() * sigma_inv * delta)[(0, 0)];
            let d2_per_axis = core::array::from_fn(|k| {
                if cov[k][k] > 0.0 {
                    delta[k] * delta[k] / cov[k][k]
                } else {
                    0.0
                }
            });

            samples.push(Sample {
                image: filename.clone(),
                tag_id: det_id,
                delta: core::array::from_fn(|k| delta[k]),
                cov: (0..36).map(|k| cov[k / 6][k % 6]).collect(),
                d2,
                d2_per_axis,
            });
        }
    }

    assert!(
        !samples.is_empty(),
        "no covariance samples collected — check dataset + edge refinement"
    );

    // Quick in-test summary (Python does the full KL/per-axis aggregation).
    let mut d2s: Vec<f64> = samples.iter().map(|s| s.d2).collect();
    d2s.sort_by(f64::total_cmp);
    let mean = d2s.iter().sum::<f64>() / d2s.len() as f64;
    let median = d2s[d2s.len() / 2];
    eprintln!(
        "model-edge covariance audit: n={}  mean d²={:.2}  median d²={:.2}  (χ²(6) ideal: mean 6)",
        d2s.len(),
        mean,
        median
    );

    let out_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../diagnostics/model_edge_cov_audit");
    std::fs::create_dir_all(&out_dir).expect("create diagnostics dir");
    let out = out_dir.join(format!("samples_{config_name}.json"));
    std::fs::write(&out, serde_json::to_string(&samples).expect("serialize"))
        .expect("write samples");
    eprintln!("wrote {} samples to {}", samples.len(), out.display());
}
