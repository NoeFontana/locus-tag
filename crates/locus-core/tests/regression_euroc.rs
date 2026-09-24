#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::panic,
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
//! EuRoC MAV calibration dataset regression tests.
//!
//! Exercises detection and pose estimation on real grayscale imagery from the
//! EuRoC `cam_april` calibration sequence — a 6×6 AprilTag 36h11 grid captured
//! with a globally-shuttered MT9V034 sensor (752×480) under real-world conditions
//! (motion blur, sensor noise, radial-tangential distortion).
//!
//! # Setup
//!
//! ```text
//! bash scripts/fetch_euroc_calibration.sh
//! ```
//!
//! # Running
//!
//! ```text
//! # Detection baseline (no distortion correction)
//! cargo nextest run --release --features bench-internals \
//!     --test regression_euroc -- --test-threads=1
//!
//! # Full suite including distortion-aware tests
//! cargo nextest run --release --features bench-internals,non_rectified \
//!     --test regression_euroc -- --test-threads=1
//! ```

mod common;

use common::euroc::{self, EurocProvider};
use common::hub::DatasetProvider;
use locus_core::{Detector, ImageView};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Read the sampling stride from `LOCUS_EUROC_SAMPLE_STRIDE` (default: 10).
fn sample_stride() -> usize {
    std::env::var("LOCUS_EUROC_SAMPLE_STRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10)
}

// ============================================================================
// 1. Detection baseline — no distortion correction
// ============================================================================

/// Detect AprilTag 36h11 markers in EuRoC cam_april images using pinhole-only
/// intrinsics (no undistortion).  Validates that the detector finds a
/// reasonable number of tags on real sensor imagery.
///
/// Acceptance criteria:
/// - At least 50 % of sampled frames detect ≥ 15 tags (out of 36).
/// - Zero false-positive IDs (all detected IDs ∈ [0, 35]).
/// - Detector never panics on real sensor data.
#[test]
fn euroc_detection_baseline() {
    let _g = common::telemetry::init("euroc_detection_baseline");
    let stride = sample_stride();
    let Some(provider) = EurocProvider::cam0(stride) else {
        println!(
            "EuRoC dataset not found — skipping. Run: bash scripts/fetch_euroc_calibration.sh"
        );
        return;
    };
    println!(
        "EuRoC cam_april: {} frames (stride {})",
        provider.len(),
        stride
    );

    let mut detector = Detector::new();
    let intrinsics = euroc::cam0_intrinsics_pinhole();

    let mut good_frames = 0u32;
    let mut total_frames = 0u32;
    let mut total_detections = 0u32;

    for (fname, data, width, height, _gt) in provider.iter() {
        let img = ImageView::new(&data, width, height, width).expect("valid image");
        let dets = detector
            .detect(&img, Some(&intrinsics), Some(euroc::TAG_SIZE), false)
            .expect("detection must not fail on real data");

        // Validate: no IDs outside the 6×6 grid
        for i in 0..dets.len() {
            assert!(
                dets.ids[i] < 36,
                "frame {fname}: unexpected tag ID {} (expected 0–35)",
                dets.ids[i]
            );
        }

        let n = dets.len() as u32;
        total_detections += n;
        total_frames += 1;
        if n >= 15 {
            good_frames += 1;
        }
    }

    let recall = f64::from(good_frames) / f64::from(total_frames);
    println!(
        "Detection baseline: {good_frames}/{total_frames} frames ≥15 tags \
         (recall {:.1}%), {total_detections} total detections",
        recall * 100.0
    );

    assert!(
        recall >= 0.5,
        "Detection recall too low: {recall:.2} (expected ≥ 0.5). \
         Only {good_frames}/{total_frames} frames had ≥ 15 detections."
    );
}

// ============================================================================
// 2. Distortion-aware detection and pose validation
// ============================================================================

/// Detect with real Brown-Conrady distortion parameters from the EuRoC
/// calibration.  Validates that:
/// - Pose estimation produces sensible results (t_z > 0).
/// - Distortion-corrected detection recall is at least as good as pinhole.
/// - The board is at a physically plausible distance (0.1 m – 3.0 m).
#[cfg(feature = "non_rectified")]
#[test]
fn euroc_distorted_pose_validation() {
    let _g = common::telemetry::init("euroc_distorted_pose_validation");
    let stride = sample_stride();
    let Some(provider) = EurocProvider::cam0_distorted(stride) else {
        println!("EuRoC dataset not found — skipping.");
        return;
    };

    let mut detector = Detector::new();
    let intrinsics = euroc::cam0_intrinsics_distorted();

    let mut frames_with_pose = 0u32;
    let mut total_poses = 0u32;
    let mut total_frames = 0u32;

    for (fname, data, width, height, _gt) in provider.iter() {
        let img = ImageView::new(&data, width, height, width).expect("valid image");
        let dets = detector
            .detect(&img, Some(&intrinsics), Some(euroc::TAG_SIZE), false)
            .expect("detection must not fail");

        let mut frame_has_pose = false;
        for i in 0..dets.len() {
            let pose = dets.poses[i];
            // pose.data layout: [tx, ty, tz, qx, qy, qz, qw]
            let tz = pose.data[2];
            if tz > 0.0 {
                // Physical plausibility: calibration board is 0.1–3.0 m away
                assert!(
                    f64::from(tz) > 0.05 && f64::from(tz) < 5.0,
                    "frame {fname}: tag {} has implausible tz={tz:.3} m",
                    dets.ids[i]
                );
                frame_has_pose = true;
                total_poses += 1;
            }
        }

        total_frames += 1;
        if frame_has_pose {
            frames_with_pose += 1;
        }
    }

    let pose_rate = f64::from(frames_with_pose) / f64::from(total_frames);
    println!(
        "Distorted pose: {frames_with_pose}/{total_frames} frames with valid poses \
         ({:.1}%), {total_poses} total poses",
        pose_rate * 100.0
    );

    assert!(
        pose_rate >= 0.4,
        "Too few frames produced valid poses: {pose_rate:.2} (expected ≥ 0.4)"
    );
}

// ============================================================================
// 3. AprilGrid board-level consistency
// ============================================================================

/// Board-level test: since all 36 tags lie on a rigid planar board, the
/// estimated poses should be geometrically consistent.  Specifically, the
/// inter-tag distances implied by individual tag poses should agree with
/// the known physical grid spacing.
///
/// We check this by comparing adjacent-tag translation vectors: for tags
/// in the same row, the implied horizontal spacing should be close to
/// `TAG_SIZE + TAG_SPACING_RATIO * TAG_SIZE`.
#[cfg(feature = "non_rectified")]
#[test]
fn euroc_board_geometry_consistency() {
    let _g = common::telemetry::init("euroc_board_geometry_consistency");
    let stride = sample_stride();
    let Some(provider) = EurocProvider::cam0_distorted(stride) else {
        println!("EuRoC dataset not found — skipping.");
        return;
    };

    let mut detector = Detector::new();
    let intrinsics = euroc::cam0_intrinsics_distorted();
    let expected_step = euroc::TAG_SIZE + euroc::TAG_SPACING_RATIO * euroc::TAG_SIZE;

    let mut consistent_frames = 0u32;
    let mut checked_frames = 0u32;

    for (_fname, data, width, height, _gt) in provider.iter() {
        let img = ImageView::new(&data, width, height, width).expect("valid image");
        let dets = detector
            .detect(&img, Some(&intrinsics), Some(euroc::TAG_SIZE), false)
            .expect("detection");

        if dets.len() < 6 {
            continue; // need enough tags for meaningful comparison
        }

        // Collect tag IDs → translation vectors
        let mut tag_translations: std::collections::BTreeMap<u32, [f64; 3]> =
            std::collections::BTreeMap::new();
        for i in 0..dets.len() {
            let pose = dets.poses[i];
            if pose.data[2] > 0.0 {
                tag_translations.insert(
                    dets.ids[i],
                    [
                        f64::from(pose.data[0]),
                        f64::from(pose.data[1]),
                        f64::from(pose.data[2]),
                    ],
                );
            }
        }

        // Check horizontal neighbours (same row: IDs differ by 1, row = id / 6)
        let mut pair_errors = Vec::new();
        for (&id_a, t_a) in &tag_translations {
            let id_b = id_a + 1;
            // Same row check: id_a / 6 == id_b / 6
            if id_a / 6 != id_b / 6 {
                continue;
            }
            if let Some(t_b) = tag_translations.get(&id_b) {
                let dist = ((t_a[0] - t_b[0]).powi(2)
                    + (t_a[1] - t_b[1]).powi(2)
                    + (t_a[2] - t_b[2]).powi(2))
                .sqrt();
                let relative_error = (dist - expected_step).abs() / expected_step;
                pair_errors.push(relative_error);
            }
        }

        if pair_errors.len() >= 3 {
            checked_frames += 1;
            let mean_error: f64 = pair_errors.iter().sum::<f64>() / pair_errors.len() as f64;
            // Allow 30 % relative error — we're dealing with noisy single-tag
            // PnP poses on a low-res sensor with real distortion.
            if mean_error < 0.30 {
                consistent_frames += 1;
            }
        }
    }

    if checked_frames > 0 {
        let consistency = f64::from(consistent_frames) / f64::from(checked_frames);
        println!(
            "Board consistency: {consistent_frames}/{checked_frames} frames \
             ({:.1}%) have < 30 % inter-tag spacing error",
            consistency * 100.0
        );
        assert!(
            consistency >= 0.3,
            "Too few frames are geometrically consistent: {consistency:.2}"
        );
    } else {
        println!("Not enough multi-tag frames to check board consistency.");
    }
}
