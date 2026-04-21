#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::type_complexity,
    clippy::unnecessary_debug_formatting,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! Regression tests for rendered tags from the hub.

use locus_core::{CameraIntrinsics, DetectOptions, PoseEstimationMode, TagFamily};

mod common;

use common::hub::{DatasetProvider, HubEntry, HubProvider, RegressionHarness, build_intrinsics};

// ============================================================================
// Test Runners
// ============================================================================

fn run_hub_test(
    config_name: &str,
    family: TagFamily,
    mode: PoseEstimationMode,
    refinement: Option<locus_core::config::CornerRefinementMode>,
) {
    if let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") {
        let root = common::resolve_hub_root(&hub_dir);
        let dataset_path = root.join(config_name);

        if !dataset_path.exists() {
            println!("Dataset not found in cache: {config_name}. Skipping.");
            return;
        }

        if let Some(provider) = HubProvider::new(&dataset_path) {
            let mut options = DetectOptions::default();
            let metadata_path = dataset_path.join("provenance.json");
            let rich_path = dataset_path.join("rich_truth.json");

            if metadata_path.exists() {
                let metadata_str = std::fs::read_to_string(metadata_path).unwrap();
                let meta: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();

                if let Some(intrinsics) = meta.get("camera_intrinsics") {
                    let fx = intrinsics["fx"].as_f64().unwrap();
                    let fy = intrinsics["fy"].as_f64().unwrap();
                    let cx = intrinsics["cx"].as_f64().unwrap();
                    let cy = intrinsics["cy"].as_f64().unwrap();
                    options.intrinsics = Some(CameraIntrinsics::new(fx, fy, cx, cy));
                }

                if let Some(tag_size_mm) = meta.get("tag_size_mm") {
                    options.tag_size = Some(tag_size_mm.as_f64().unwrap() / 1000.0);
                }
            }

            if (options.intrinsics.is_none() || options.tag_size.is_none()) && rich_path.exists() {
                let file = std::fs::File::open(&rich_path).unwrap();
                let entries: Vec<HubEntry> = serde_json::from_reader(file).unwrap();
                if let Some(first) = entries.first() {
                    if options.intrinsics.is_none()
                        && let Some(k) = first.k_matrix
                    {
                        options.intrinsics = Some(build_intrinsics(
                            k,
                            first.distortion_model.as_deref(),
                            first.dist_coeffs.as_deref(),
                        ));
                    }
                    if options.tag_size.is_none()
                        && let Some(size_mm) = first.tag_size_mm
                    {
                        options.tag_size = Some(size_mm / 1000.0);
                    }
                }
            }

            options.pose_estimation_mode = mode;

            let mode_suffix = match mode {
                PoseEstimationMode::Fast => "_fast",
                PoseEstimationMode::Accurate => "",
            };

            let refinement_suffix =
                if let Some(locus_core::config::CornerRefinementMode::Gwlf) = refinement {
                    "_gwlf"
                } else {
                    ""
                };

            let snapshot = format!(
                "hub_{}{}{}",
                provider.name(),
                mode_suffix,
                refinement_suffix
            );
            let mut harness = RegressionHarness::new(snapshot)
                .with_profile("standard")
                .with_families(vec![family])
                .with_options(options);

            if let Some(r) = refinement {
                harness = harness.with_refinement_mode(r);
            }

            harness.run(provider);
        }
    } else {
        println!("Skipping hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
    }
}

fn run_hub_test_tuned(
    config_name: &str,
    family: TagFamily,
    mode: PoseEstimationMode,
    snapshot_suffix: &str,
    max_elongation: f64,
    min_density: f64,
    quad_mode: locus_core::config::QuadExtractionMode,
) {
    run_hub_test_tuned_r(
        config_name,
        family,
        mode,
        snapshot_suffix,
        max_elongation,
        min_density,
        quad_mode,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_hub_test_tuned_r(
    config_name: &str,
    family: TagFamily,
    mode: PoseEstimationMode,
    snapshot_suffix: &str,
    max_elongation: f64,
    min_density: f64,
    quad_mode: locus_core::config::QuadExtractionMode,
    refinement: Option<locus_core::config::CornerRefinementMode>,
) {
    if let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") {
        let root = common::resolve_hub_root(&hub_dir);
        let dataset_path = root.join(config_name);

        if !dataset_path.exists() {
            println!("Dataset not found in cache: {config_name}. Skipping.");
            return;
        }

        if let Some(provider) = HubProvider::new(&dataset_path) {
            let mut options = DetectOptions::default();
            let metadata_path = dataset_path.join("provenance.json");
            let rich_path = dataset_path.join("rich_truth.json");

            if metadata_path.exists() {
                let metadata_str = std::fs::read_to_string(metadata_path).unwrap();
                let meta: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();
                if let Some(intrinsics) = meta.get("camera_intrinsics") {
                    let fx = intrinsics["fx"].as_f64().unwrap();
                    let fy = intrinsics["fy"].as_f64().unwrap();
                    let cx = intrinsics["cx"].as_f64().unwrap();
                    let cy = intrinsics["cy"].as_f64().unwrap();
                    options.intrinsics = Some(CameraIntrinsics::new(fx, fy, cx, cy));
                }
                if let Some(tag_size_mm) = meta.get("tag_size_mm") {
                    options.tag_size = Some(tag_size_mm.as_f64().unwrap() / 1000.0);
                }
            }
            if (options.intrinsics.is_none() || options.tag_size.is_none()) && rich_path.exists() {
                let file = std::fs::File::open(&rich_path).unwrap();
                let entries: Vec<HubEntry> = serde_json::from_reader(file).unwrap();
                if let Some(first) = entries.first() {
                    if options.intrinsics.is_none()
                        && let Some(k) = first.k_matrix
                    {
                        options.intrinsics = Some(build_intrinsics(
                            k,
                            first.distortion_model.as_deref(),
                            first.dist_coeffs.as_deref(),
                        ));
                    }
                    if options.tag_size.is_none()
                        && let Some(size_mm) = first.tag_size_mm
                    {
                        options.tag_size = Some(size_mm / 1000.0);
                    }
                }
            }

            options.pose_estimation_mode = mode;
            let mode_suffix = match mode {
                PoseEstimationMode::Fast => "_fast",
                PoseEstimationMode::Accurate => "",
            };
            let snapshot = format!("hub_{}{}{}", provider.name(), mode_suffix, snapshot_suffix);
            let mut harness = RegressionHarness::new(snapshot)
                .with_profile("standard")
                .with_families(vec![family])
                .with_options(options)
                .with_moments_culling(max_elongation, min_density)
                .with_quad_extraction_mode(quad_mode);
            if let Some(r) = refinement {
                harness = harness.with_refinement_mode(r);
            }
            harness.run(provider);
        }
    } else {
        println!("Skipping hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
    }
}

fn run_hub_test_highaccuracy(config_name: &str, family: TagFamily) {
    if let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") {
        let root = common::resolve_hub_root(&hub_dir);
        let dataset_path = root.join(config_name);

        if !dataset_path.exists() {
            println!("Dataset not found in cache: {config_name}. Skipping.");
            return;
        }

        if let Some(provider) = HubProvider::new(&dataset_path) {
            let mut options = DetectOptions::default();
            let metadata_path = dataset_path.join("provenance.json");
            let rich_path = dataset_path.join("rich_truth.json");

            if metadata_path.exists() {
                let metadata_str = std::fs::read_to_string(metadata_path).unwrap();
                let meta: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();
                if let Some(intrinsics) = meta.get("camera_intrinsics") {
                    let fx = intrinsics["fx"].as_f64().unwrap();
                    let fy = intrinsics["fy"].as_f64().unwrap();
                    let cx = intrinsics["cx"].as_f64().unwrap();
                    let cy = intrinsics["cy"].as_f64().unwrap();
                    options.intrinsics = Some(CameraIntrinsics::new(fx, fy, cx, cy));
                }
                if let Some(tag_size_mm) = meta.get("tag_size_mm") {
                    options.tag_size = Some(tag_size_mm.as_f64().unwrap() / 1000.0);
                }
            }
            if (options.intrinsics.is_none() || options.tag_size.is_none()) && rich_path.exists() {
                let file = std::fs::File::open(&rich_path).unwrap();
                let entries: Vec<HubEntry> = serde_json::from_reader(file).unwrap();
                if let Some(first) = entries.first() {
                    if options.intrinsics.is_none()
                        && let Some(k) = first.k_matrix
                    {
                        options.intrinsics = Some(build_intrinsics(
                            k,
                            first.distortion_model.as_deref(),
                            first.dist_coeffs.as_deref(),
                        ));
                    }
                    if options.tag_size.is_none()
                        && let Some(size_mm) = first.tag_size_mm
                    {
                        options.tag_size = Some(size_mm / 1000.0);
                    }
                }
            }
            options.pose_estimation_mode = PoseEstimationMode::Accurate;

            let snapshot = format!("hub_{}_highaccuracy", provider.name());
            RegressionHarness::new(snapshot)
                .with_profile("high_accuracy")
                .with_families(vec![family])
                .with_options(options)
                .run(provider);
        }
    } else {
        println!("Skipping hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
    }
}

// ── Accurate mode (Structure Tensor + Weighted LM) ───────────────────────────

#[test]
fn regression_hub_tag36h11_640x480() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_640x480");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_640x480",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        None,
    );
}

#[test]
fn regression_hub_tag36h11_720p() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        None,
    );
}

#[test]
fn regression_hub_tag36h11_1080p() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        None,
    );
}

#[test]
fn regression_hub_tag36h11_1080p_gwlf() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_gwlf");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        Some(locus_core::config::CornerRefinementMode::Gwlf),
    );
}

#[test]
fn regression_hub_tag36h11_2160p() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        None,
    );
}

// ── Fast mode (Trust-Region LM + Huber M-Estimator) ──────────────────────────

#[test]
fn regression_hub_fast_tag36h11_640x480() {
    let _guard = common::telemetry::init("regression_hub_fast_tag36h11_640x480");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_640x480",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Fast,
        None,
    );
}

#[test]
fn regression_hub_fast_tag36h11_720p() {
    let _guard = common::telemetry::init("regression_hub_fast_tag36h11_720p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Fast,
        None,
    );
}

#[test]
fn regression_hub_fast_tag36h11_1080p() {
    let _guard = common::telemetry::init("regression_hub_fast_tag36h11_1080p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Fast,
        None,
    );
}

#[test]
fn regression_hub_fast_tag36h11_2160p() {
    let _guard = common::telemetry::init("regression_hub_fast_tag36h11_2160p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Fast,
        None,
    );
}

// ── Algorithm tuning variants (moments culling + EDLines) ────────────────────

#[test]
fn regression_hub_tag36h11_720p_moments_culling() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_moments_culling");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_moments_culling",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::ContourRdp,
    );
}

#[test]
fn regression_hub_tag36h11_720p_edlines() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_edlines");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_720p_edlines_moments() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_edlines_moments");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_moments",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_720p_edlines_none() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_edlines_none");
    run_hub_test_tuned_r(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_none",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
        Some(locus_core::config::CornerRefinementMode::None),
    );
}

#[test]
fn regression_hub_tag36h11_720p_edlines_gwlf() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_edlines_gwlf");
    run_hub_test_tuned_r(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_gwlf",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
        Some(locus_core::config::CornerRefinementMode::Gwlf),
    );
}

#[test]
fn regression_hub_tag36h11_1080p_moments_culling() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_moments_culling");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_moments_culling",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::ContourRdp,
    );
}

#[test]
fn regression_hub_tag36h11_1080p_edlines() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_edlines");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_1080p_edlines_moments() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_edlines_moments");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_moments",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_2160p_moments_culling() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p_moments_culling");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_moments_culling",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::ContourRdp,
    );
}

#[test]
fn regression_hub_tag36h11_2160p_edlines() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p_edlines");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_2160p_edlines_moments() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p_edlines_moments");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_moments",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

// ── HighAccuracy (EdLines GN + covariance propagation + Weighted LM) ───────

#[test]
fn regression_hub_tag36h11_640x480_highaccuracy() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_640x480_highaccuracy");
    run_hub_test_highaccuracy(
        "single_tag_locus_v1_tag36h11_640x480",
        TagFamily::AprilTag36h11,
    );
}

#[test]
fn regression_hub_tag36h11_720p_highaccuracy() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_highaccuracy");
    run_hub_test_highaccuracy(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
    );
}

#[test]
fn regression_hub_tag36h11_1080p_highaccuracy() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_highaccuracy");
    run_hub_test_highaccuracy(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
    );
}

#[test]
fn regression_hub_tag36h11_2160p_highaccuracy() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p_highaccuracy");
    run_hub_test_highaccuracy(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
    );
}
