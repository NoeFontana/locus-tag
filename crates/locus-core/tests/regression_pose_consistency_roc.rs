#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::unwrap_used,
    missing_docs
)]
//! Pose-consistency gate ROC sweep.
//!
//! Drives the χ² gate directly with controlled isotropic noise so we can
//! verify that the *realized* false-positive rate matches the *modeled*
//! FPR derived from the χ²(2) and χ²(1) critical values.
//!
//! The acceptance assertion is the hard gate: at the chosen production
//! `pose_consistency_fpr = 1e-3`, the realized FPR on this synthetic
//! dataset must lie in `[1e-4, 1e-2]` (within one decade of modeled). If
//! the assertion fires, the χ² assumption is breaking on this codebase
//! and the snapshot must be reviewed before merge.
//!
//! The test deliberately *avoids* loading ICRA / Hub fixtures: the recall
//! and precision deltas at the production threshold are already covered
//! by the existing `regression_icra2020`, `regression_render_tag`, and
//! `regression_distortion_hub` snapshots; adding them here would just
//! duplicate that work without exercising the gate's noise model.

use locus_core::PoseEstimationMode;
use locus_core::bench_api::{CameraIntrinsics, Pose, bench_pose_consistency_d2, estimate_tag_pose};
use nalgebra::{Matrix2, Matrix3, Vector3};

fn centered_tag_corners(tag_size: f64) -> [Vector3<f64>; 4] {
    let h = tag_size * 0.5;
    [
        Vector3::new(-h, -h, 0.0),
        Vector3::new(h, -h, 0.0),
        Vector3::new(h, h, 0.0),
        Vector3::new(-h, h, 0.0),
    ]
}
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Serialize;

const SIGMA_PX: f64 = 1.0;
const SIGMA_SQ: f64 = SIGMA_PX * SIGMA_PX;
const N_TRIALS: usize = 20_000;
const FPR_SWEEP: &[f64] = &[1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5];
const ACCEPTANCE_FPR: f64 = 1.0e-3;
/// One decade either side of the modeled rate (matches the plan's
/// `[1e-4, 1e-2]` window for `acceptance_fpr = 1e-3`).
const ACCEPTANCE_LO: f64 = 1.0e-4;
const ACCEPTANCE_HI: f64 = 1.0e-2;

#[derive(Serialize)]
struct SweepRow {
    fpr: f64,
    n_samples: usize,
    realized_fpr: f64,
}

#[derive(Serialize)]
struct SweepReport {
    sigma_px: f64,
    n_trials: usize,
    rows: Vec<SweepRow>,
}

fn project(intrinsics: &CameraIntrinsics, pose: &Pose, p_world: &Vector3<f64>) -> [f64; 2] {
    let p_cam = pose.rotation * p_world + pose.translation;
    [
        intrinsics.fx * (p_cam.x / p_cam.z) + intrinsics.cx,
        intrinsics.fy * (p_cam.y / p_cam.z) + intrinsics.cy,
    ]
}

/// One synthetic trial: project a known pose, perturb each corner by
/// independent Gaussian(0, σ²) noise, **fit** a pose by LM (so residuals
/// genuinely span 8 obs − 6 fitted = 2 DOF, matching the χ²(2) model
/// production assumes), then run the gate at every sweep FPR.
///
/// Returns `Some(per-fpr Vec<bool>)` when the LM fit succeeds (gate ran),
/// or `None` when it fails (trial is silently dropped from the denominator).
fn one_trial(
    intrinsics: &CameraIntrinsics,
    tag_size: f64,
    pose: &Pose,
    rng: &mut ChaCha20Rng,
) -> Option<Vec<bool>> {
    let obj_pts = centered_tag_corners(tag_size);
    let mut observed = [[0.0_f64; 2]; 4];
    for i in 0..4 {
        let ideal = project(intrinsics, pose, &obj_pts[i]);
        // Box-Muller from two uniforms (cheap; we don't need rand_distr here).
        let (nx, ny) = sample_gaussian_pair(rng, SIGMA_PX);
        observed[i] = [ideal[0] + nx, ideal[1] + ny];
    }

    // LM-refit before measuring d² so the χ²(2) DOF assumption holds.
    let (fitted, _) = estimate_tag_pose(
        intrinsics,
        &observed,
        tag_size,
        None,
        PoseEstimationMode::Fast,
    );
    let fitted = fitted?;

    // Isotropic information matrix matching the noise model.
    let inv_sigma2 = 1.0 / SIGMA_SQ;
    let info = Matrix2::new(inv_sigma2, 0.0, 0.0, inv_sigma2);
    let info_matrices: [Matrix2<f64>; 4] = [info; 4];

    Some(
        FPR_SWEEP
            .iter()
            .map(|&fpr| {
                let (accepted, _agg, _max) = bench_pose_consistency_d2(
                    intrinsics,
                    &observed,
                    &info_matrices,
                    tag_size,
                    &fitted,
                    fpr,
                );
                !accepted
            })
            .collect(),
    )
}

fn sample_gaussian_pair(rng: &mut ChaCha20Rng, sigma: f64) -> (f64, f64) {
    // Box-Muller. `RngExt::random` on f64 returns a uniform in [0, 1).
    let mut u1: f64 = rng.random();
    if u1 < 1e-300 {
        u1 = 1e-300;
    }
    let u2: f64 = rng.random();
    let mag = sigma * (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (mag * theta.cos(), mag * theta.sin())
}

#[test]
fn pose_consistency_roc_sweep() {
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;

    // Fixed nominal pose: 0.5 m in front of the camera, slight rotation.
    let rotation = Matrix3::identity();
    let translation = Vector3::new(0.05, -0.03, 0.5);
    let pose = Pose::new(rotation, translation);

    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_FFEE_5EED_BABE);
    let mut rejections = vec![0_usize; FPR_SWEEP.len()];
    let mut n_used = 0_usize;

    for _ in 0..N_TRIALS {
        let Some(trial) = one_trial(&intrinsics, tag_size, &pose, &mut rng) else {
            continue;
        };
        n_used += 1;
        for (acc, rejected) in rejections.iter_mut().zip(trial.iter()) {
            if *rejected {
                *acc += 1;
            }
        }
    }
    assert!(
        n_used as f64 / N_TRIALS as f64 > 0.9,
        "LM converged on only {n_used}/{N_TRIALS} trials — sweep is degenerate",
    );

    let rows: Vec<SweepRow> = FPR_SWEEP
        .iter()
        .zip(rejections.iter())
        .map(|(&fpr, &rejected)| SweepRow {
            fpr,
            n_samples: n_used,
            realized_fpr: rejected as f64 / n_used as f64,
        })
        .collect();

    // Acceptance gate: realized FPR at the production threshold must lie
    // within one decade of modeled.
    let row = rows
        .iter()
        .find(|r| (r.fpr - ACCEPTANCE_FPR).abs() < f64::EPSILON)
        .expect("sweep must include the production fpr");
    assert!(
        row.realized_fpr >= ACCEPTANCE_LO && row.realized_fpr <= ACCEPTANCE_HI,
        "Realized FPR {:.2e} at modeled fpr={:.0e} is outside the [{:.0e}, {:.0e}] \
         acceptance window. The χ²(2) assumption is breaking — review the snapshot \
         and consider an empirical_d2_threshold fallback (see \
         docs/engineering/track2_precision_threshold.md).",
        row.realized_fpr,
        ACCEPTANCE_FPR,
        ACCEPTANCE_LO,
        ACCEPTANCE_HI,
    );

    let report = SweepReport {
        sigma_px: SIGMA_PX,
        n_trials: N_TRIALS,
        rows,
    };
    insta::assert_yaml_snapshot!("pose_consistency_roc__synthetic_isotropic", report);
}
