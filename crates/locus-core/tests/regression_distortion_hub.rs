#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! Regression tests for distortion-aware detection on hub datasets.
//!
//! Covers Brown-Conrady and Kannala-Brandt distortion models.
//! Distortion coefficients and intrinsics are read from `rich_truth.json`
//! (`distortion_model` + `dist_coeffs` fields) stored alongside the dataset.

use locus_core::{DetectOptions, PoseEstimationMode, TagFamily};

mod common;

use common::hub::{
    DatasetProvider, HubProvider, RegressionHarness, build_intrinsics, load_rich_truth_entries,
};

// ============================================================================
// Runner
// ============================================================================

/// Runs a hub distortion dataset test.
///
/// Intrinsics (including distortion coefficients) are read exclusively from
/// `rich_truth.json` — these datasets do not use `provenance.json`. Pose-mode
/// coverage lives in `regression_render_tag.rs::pose_mode_variants`; this
/// suite exercises the undistortion path only, so it runs Accurate mode.
fn run_distortion_hub_test(config_name: &str, family: TagFamily) {
    let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") else {
        println!("Skipping distortion hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
        return;
    };

    let root = common::resolve_hub_root(&hub_dir);
    let dataset_path = root.join(config_name);

    if !dataset_path.exists() {
        println!("Dataset not found in cache: {config_name}. Skipping.");
        return;
    }

    let Some(provider) = HubProvider::new(&dataset_path) else {
        println!("Failed to load dataset: {config_name}. Skipping.");
        return;
    };

    // Distortion datasets embed intrinsics + distortion coefficients in
    // rich_truth.json. Build a fallback for the options in case any image
    // lacks per-entry intrinsics (HubProvider already sets gt.intrinsics).
    let mut options = DetectOptions {
        pose_estimation_mode: PoseEstimationMode::Accurate,
        ..Default::default()
    };
    if let Some(entries) = load_rich_truth_entries(&dataset_path.join("rich_truth.json")) {
        if let Some(first) = entries.first()
            && let Some(k) = first.k_matrix
        {
            options.intrinsics = Some(build_intrinsics(
                k,
                first.distortion_model.as_deref(),
                first.dist_coeffs.as_deref(),
            ));
        }
        if let Some(first) = entries.first()
            && let Some(size_mm) = first.tag_size_mm
        {
            options.tag_size = Some(size_mm / 1000.0);
        }
    }

    let snapshot = format!("hub_{}", provider.name());

    RegressionHarness::new(snapshot)
        .with_profile("standard")
        .with_families(vec![family])
        .with_options(options)
        .run(provider);
}

// ============================================================================
// Brown-Conrady distortion
// ============================================================================

#[test]
fn regression_hub_distortion_brown_conrady() {
    let _guard = common::telemetry::init("regression_hub_distortion_brown_conrady");
    run_distortion_hub_test(
        "aprilgrid_distortion_brown_conrady_v1_1920x1080",
        TagFamily::AprilTag36h11,
    );
}

// ============================================================================
// Kannala-Brandt fisheye distortion
// ============================================================================

#[test]
fn regression_hub_distortion_kannala_brandt() {
    let _guard = common::telemetry::init("regression_hub_distortion_kannala_brandt");
    run_distortion_hub_test(
        "aprilgrid_distortion_kannala_brandt_v1_1920x1080",
        TagFamily::AprilTag36h11,
    );
}
