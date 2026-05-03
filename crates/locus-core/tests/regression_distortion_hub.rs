#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::naive_bytecount,
    clippy::panic,
    clippy::single_match_else,
    clippy::unwrap_used,
    dead_code,
    missing_docs,
    unsafe_code
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

#[cfg(feature = "non_rectified")]
use locus_core::{
    ConfigError, Detector, DetectorConfig, DetectorError, ImageView,
    config::{CornerRefinementMode, QuadExtractionMode},
    pose::CameraIntrinsics,
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

// ============================================================================
// Routing-dispatch coverage
// ============================================================================
//
// The two tests below pin the distortion-routing dispatch in
// `crates/locus-core/src/detector.rs::run_detection_pipeline`:
//
//   1. `max_recall_adaptive` (AdaptivePpb policy, high-PPB route = EdLines):
//      under distorted intrinsics the high-PPB branch is silently degraded to
//      ContourRdp + Erf via `force_low_route=true` in `resolve_route`. This
//      test asserts the silent-fallback semantics so a regression that lets
//      EdLines reach the distortion path (or that strips Erf refinement on
//      the high branch) trips the suite.
//
//   2. Static `EdLines` profile + distorted intrinsics: the dispatch errors
//      eagerly with `ConfigError::EdLinesUnsupportedWithDistortion`. This
//      test asserts the variant pattern (not a display string) so the gate
//      can be renamed without churn.

/// `max_recall_adaptive` profile + distorted intrinsics: every active
/// candidate must be routed through the low (ContourRdp) branch even when
/// PPB would otherwise select the high (EdLines) branch. Telemetry hook:
/// `DetectionBatch.routed_to[..N] == 0` (all low route) when
/// `debug_telemetry=true`.
#[cfg(feature = "non_rectified")]
#[test]
fn test_max_recall_adaptive_falls_back_under_distortion() {
    let _guard = common::telemetry::init("test_max_recall_adaptive_falls_back_under_distortion");

    let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") else {
        println!("Skipping: set LOCUS_HUB_DATASET_DIR to run.");
        return;
    };

    let dataset_path =
        common::resolve_hub_root(&hub_dir).join("aprilgrid_distortion_brown_conrady_v1_1920x1080");
    if !dataset_path.exists() {
        println!("Dataset not found in cache. Skipping.");
        return;
    }

    let Some(provider) = HubProvider::new(&dataset_path) else {
        println!("Failed to load dataset. Skipping.");
        return;
    };

    let entries = match load_rich_truth_entries(&dataset_path.join("rich_truth.json")) {
        Some(e) => e,
        None => {
            println!("Failed to load rich_truth.json. Skipping.");
            return;
        },
    };
    let Some(first_entry) = entries.first() else {
        println!("rich_truth.json is empty. Skipping.");
        return;
    };
    let Some(k) = first_entry.k_matrix else {
        println!("rich_truth.json missing k_matrix. Skipping.");
        return;
    };
    let intrinsics = build_intrinsics(
        k,
        first_entry.distortion_model.as_deref(),
        first_entry.dist_coeffs.as_deref(),
    );
    // The fixture is curated to be Brown-Conrady distorted; if that ever
    // breaks (e.g. dataset re-rendered as pinhole), abort instead of silently
    // degrading the test into a no-op covering the rectified path.
    assert!(
        intrinsics.distortion.is_distorted(),
        "fixture intrinsics must be distorted to exercise the fallback path"
    );
    let tag_size = first_entry.tag_size_mm.map(|mm| mm / 1000.0);

    let config = DetectorConfig::from_profile("max_recall_adaptive");
    let mut detector = Detector::builder()
        .with_config(config)
        .with_family(TagFamily::AprilTag36h11)
        .build();

    // Run on the first scene only — we're asserting routing semantics, not
    // accuracy across the dataset (that's covered by the snapshot tests above).
    let Some((fname, data, width, height, gt)) = provider.iter().next() else {
        println!("Provider yielded no images. Skipping.");
        return;
    };

    let img = match ImageView::new(&data, width, height, width) {
        Ok(v) => v,
        Err(e) => panic!("invalid image view for {fname}: {e:?}"),
    };

    let view = detector
        .detect(
            &img,
            Some(&intrinsics),
            tag_size,
            PoseEstimationMode::Accurate,
            true, // debug_telemetry: required to populate routed_to telemetry
        )
        .unwrap_or_else(|e| panic!("detection failed under distortion: {e:?}"));

    // Decode-count floor: for a representative aprilgrid frame the fallback
    // must still recover a meaningful fraction of the ground-truth tags. We
    // keep the floor at one quarter to leave headroom for fixture refreshes
    // while still failing loud if the high-PPB branch decays to "no decodes".
    let decoded = view.len();
    let gt_count = gt.tags.len();
    let floor = (gt_count / 4).max(1);
    assert!(
        decoded >= floor,
        "distorted-fallback decode count too low: got {decoded} for {fname} \
         (gt={gt_count}, floor={floor})"
    );

    // Telemetry hook: `routed_to[i] == 0` is the low (ContourRdp + Erf) route.
    // Under AdaptivePpb on a distorted camera, `extract_single_quad_with_camera`
    // forces `force_low_route=true`, so every active candidate must be 0. A
    // value of 1 would indicate the high (EdLines) branch leaked through;
    // ROUTED_TO_STATIC (u8::MAX) would indicate the policy was misread as
    // Static.
    let telemetry = view
        .telemetry
        .expect("debug_telemetry=true must populate the telemetry payload");
    let n = telemetry.num_routed;
    assert!(
        n > 0,
        "expected at least one routed candidate, got num_routed=0"
    );
    assert!(
        !telemetry.routed_to_ptr.is_null(),
        "routed_to_ptr must be non-null when debug_telemetry is set"
    );
    // SAFETY: `routed_to_ptr` points into `DetectionBatch.routed_to` (length
    // MAX_CANDIDATES); `num_routed` is the N from Phase A and is bounded by
    // MAX_CANDIDATES; the payload's lifetime is tied to `view`, which still
    // borrows the detector's frame context here.
    let routed = unsafe { std::slice::from_raw_parts(telemetry.routed_to_ptr, n) };
    let high_route_count = routed.iter().filter(|&&r| r == 1).count();
    let low_route_count_dbg = routed.iter().filter(|&&r| r == 0).count();
    println!(
        "test_max_recall_adaptive_falls_back_under_distortion: scene={fname} \
         decoded={decoded} gt={gt_count} num_routed={n} \
         low_route={low_route_count_dbg} high_route={high_route_count}"
    );
    assert_eq!(
        high_route_count, 0,
        "AdaptivePpb on distorted intrinsics must route 0 candidates to the high (EdLines) branch, got {high_route_count}/{n}"
    );
    // Sanity: at least some candidates should carry the low-route label
    // (0); ROUTED_TO_STATIC (u8::MAX) here would mean the AdaptivePpb policy
    // wasn't recognised by the route resolver.
    let low_route_count = routed.iter().filter(|&&r| r == 0).count();
    assert!(
        low_route_count > 0,
        "expected at least one low-route candidate under AdaptivePpb, got 0/{n} (sentinels={}); routing telemetry not populated correctly",
        routed.iter().filter(|&&r| r == u8::MAX).count()
    );
}

/// Static `EdLines` extraction + distorted intrinsics must error eagerly with
/// `ConfigError::EdLinesUnsupportedWithDistortion`. This is coverage for the
/// gate at the top of `run_detection_pipeline`. We don't load a hub dataset:
/// the gate fires before any pixel work, so a synthetic blank image is fine.
#[cfg(feature = "non_rectified")]
#[test]
fn test_static_edlines_errors_under_distortion() {
    let _guard = common::telemetry::init("test_static_edlines_errors_under_distortion");

    // Static EdLines: matches `DetectorConfig::static_uses_edlines() == true`.
    // Pair with `Refinement::None` to satisfy the EdLines+Erf compatibility
    // check at config validation time (otherwise the builder would panic
    // before we reach the distortion gate we're trying to cover).
    let config = DetectorConfig::builder()
        .quad_extraction_mode(QuadExtractionMode::EdLines)
        .refinement_mode(CornerRefinementMode::None)
        .build();
    assert!(
        config.static_uses_edlines(),
        "test precondition: config must trip the static-EdLines gate"
    );

    let mut detector = Detector::builder()
        .with_config(config)
        .with_family(TagFamily::AprilTag36h11)
        .build();

    // Tiny synthetic frame — the gate triggers before thresholding, so the
    // contents don't matter. Smaller than the threshold tile to keep the
    // failure clearly attributable to the gate rather than to a downstream
    // pipeline assertion that could mask the test.
    let pixels = vec![0u8; 64 * 64];
    let img = ImageView::new(&pixels, 64, 64, 64).expect("valid image view");

    let intrinsics = CameraIntrinsics::with_brown_conrady(
        1000.0, 1000.0, 32.0, 32.0, -0.3, 0.1, 0.001, -0.002, 0.0,
    );
    assert!(
        intrinsics.distortion.is_distorted(),
        "test precondition: intrinsics must report distortion"
    );

    let result = detector.detect(
        &img,
        Some(&intrinsics),
        None,
        PoseEstimationMode::Fast,
        false,
    );

    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("Static EdLines on distorted intrinsics must error eagerly"),
    };
    assert!(
        matches!(
            err,
            DetectorError::Config(ConfigError::EdLinesUnsupportedWithDistortion)
        ),
        "expected DetectorError::Config(EdLinesUnsupportedWithDistortion), got {err:?}"
    );
}
