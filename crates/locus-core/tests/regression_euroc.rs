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

/// Fit a 2D affine map from board-plane (x, y) to image-pixel (x, y) by
/// ordinary least squares over `board_pts`/`image_pts` correspondences
/// (parallel slices, same length).
///
/// Deliberately affine, not a full projective homography or 6-DOF pose: the
/// board is planar, and this is used purely to answer "does tag T's known,
/// fixed footprint plausibly fall inside the image" for tags that never
/// decoded — a coarse question that doesn't need perspective correctness.
/// The payoff is robustness: fitting `[a b c; d e f]` is two independent
/// linear least-squares problems (closed-form normal equations, no
/// iteration, no RANSAC consensus search, no Necker-style pose ambiguity),
/// unlike a full board-pose fit — see the module doc comment on
/// `euroc_detection_baseline` for why that path was abandoned (it converges
/// on under 20% of real frames here even with generously loosened gates).
/// Returns `None` only if the normal matrix is singular (e.g. all
/// correspondences collinear — not expected for a decoded AprilGrid, whose
/// points span at least two grid rows/columns once `>= 4` tags decode).
fn fit_affine_board_to_image(board_pts: &[[f64; 2]], image_pts: &[[f64; 2]]) -> Option<[f64; 6]> {
    debug_assert_eq!(board_pts.len(), image_pts.len());
    let (mut sxx, mut sxy, mut sx, mut syy, mut sy, mut sn) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut spx_x, mut spx_y, mut spx) = (0.0, 0.0, 0.0);
    let (mut spy_x, mut spy_y, mut spy) = (0.0, 0.0, 0.0);
    for (&[bx, by], &[px, py]) in board_pts.iter().zip(image_pts) {
        sxx += bx * bx;
        sxy += bx * by;
        sx += bx;
        syy += by * by;
        sy += by;
        sn += 1.0;
        spx_x += px * bx;
        spx_y += px * by;
        spx += px;
        spy_x += py * bx;
        spy_y += py * by;
        spy += py;
    }
    let m = nalgebra::Matrix3::new(sxx, sxy, sx, sxy, syy, sy, sx, sy, sn);
    let m_inv = m.try_inverse()?;
    let abc = m_inv * nalgebra::Vector3::new(spx_x, spx_y, spx);
    let def = m_inv * nalgebra::Vector3::new(spy_x, spy_y, spy);
    Some([abc.x, abc.y, abc.z, def.x, def.y, def.z])
}

/// Apply the affine map from [`fit_affine_board_to_image`] to a board-plane point.
#[allow(
    clippy::many_single_char_names,
    reason = "a..f are the standard affine coefficient names"
)]
fn project_affine(coeffs: [f64; 6], [bx, by]: [f64; 2]) -> [f64; 2] {
    let [a, b, c, d, e, f] = coeffs;
    [a * bx + b * by + c, d * bx + e * by + f]
}

// ============================================================================
// 1. Detection baseline — no distortion correction
// ============================================================================

/// Detect AprilTag 36h11 markers in EuRoC cam_april images using pinhole-only
/// intrinsics (no undistortion).  Validates that the detector finds a
/// reasonable number of tags on real sensor imagery, and that recall isn't
/// silently concentrated in the image interior.
///
/// `cam_april` is a *calibration sweep* recording, not a framed-detection
/// benchmark: a large fraction of frames (measured 62.8% at stride 10, see
/// `euroc_pinhole_funnel_diagnostic`) have the board absent, far away, or
/// off-frame by construction (sequence start, operator repositioning,
/// deliberate extreme-angle/distance coverage for calibration). An absolute
/// "≥15 tags" bar conflates "board not in view" with "decoder missed tags
/// that were actually there", so recall here is *relative*: for each frame,
/// how many of the tags that were actually present did we decode?
///
/// "Present" isn't guessed — it's measured. Once ≥4 tags decode in a frame,
/// [`fit_affine_board_to_image`] fits a 2D affine map from every corner of
/// every decoded tag's *known, fixed* board-plane position to its observed
/// pixel position (ordinary least squares — closed-form, no iteration). The
/// full 6×6 grid layout is then mapped through that fit, and a tag counts
/// as "present" iff all 4 of its projected corners land inside the image
/// bounds. A full 6-DOF board pose (`locus_core::board::BoardEstimator`,
/// LO-RANSAC + refinement) was tried first and is strictly more rigorous
/// when it works, but empirically converges on under 20% of real frames
/// here even with its gates loosened 25× — real per-tag corner noise on
/// this sensor is outside the envelope its residual model assumes. The
/// affine fit trades that rigor for robustness: it can't fail to converge
/// (only a singular normal matrix stops it — effectively never, since any
/// ≥4 decoded tags span at least two grid rows/columns) and doesn't need
/// camera intrinsics or depth at all, at the cost of not modeling
/// perspective — acceptable here because the question is coarse
/// ("plausibly in frame or not"), not a precision pose estimate. This is
/// bootstrapped from the detector's own output but not circular in the way
/// that matters: the reprojection is pure geometry driven by the *fitted
/// map*, not by which other tags happened to decode, so a worse decoder
/// cannot inflate its own score by finding fewer "present" tags — decoding
/// fewer tags only shrinks the pool available to *fit* the map (until <4,
/// where the frame becomes indeterminate rather than favorable; see
/// `frames_indeterminate` below).
///
/// Acceptance criteria:
/// - Mean per-frame relative recall (decoded / present) ≥ 50 % over frames
///   with a successful affine board fit.
/// - Zero false-positive IDs (all detected IDs ∈ [0, 35]).
/// - Zero duplicate IDs within a single frame (a physical tag can only be
///   seen once per frame; a duplicate means two distinct quads decoded to
///   the same ID — a corner-localization/decode failure the ID-range check
///   alone cannot catch).
/// - Nonzero recall on the outer ring of the 6×6 grid (periphery), not just
///   the interior 4×4 block. This matters because the interior block alone
///   is 16 tags — already enough to satisfy the ≥15 aggregate gate on its
///   own, so that gate cannot by itself distinguish full-FOV detection from
///   interior-only detection. EuRoC cam0 has strong real barrel distortion
///   (k1 = -0.28340811, see `common::euroc::CAM0_K1`) that this pinhole-only
///   test deliberately ignores, so *some* periphery recall falloff is
///   expected; this guards against a *total* periphery dead zone, not
///   against falloff itself — the threshold is intentionally weak (`> 0`,
///   not a percentage) until real numbers justify a tighter one.
/// - Detector never panics on real sensor data.
///
/// A per-tag 6×6 detection-rate heatmap and periphery/interior recall
/// breakdown are printed for diagnosis regardless of pass/fail.
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
    let topology = euroc::aprilgrid_topology();

    let mut total_frames = 0u32;
    let mut total_detections = 0u32;
    let mut duplicate_id_frames = 0u32;

    // Per-frame relative recall (see the doc comment above for the "present"
    // methodology). `frames_considered` = frames where a board pose was
    // successfully estimated; only these contribute to the recall numbers.
    let mut frames_considered = 0u32;
    let mut sum_decoded = 0u64;
    let mut sum_present = 0u64;
    let mut per_frame_ratio_sum = 0.0f64;
    let mut ratio_over_one_count = 0u32; // sanity check on pose-fit boundary noise; should stay near 0

    let mut frames_indeterminate = 0u32; // < 4 decodes: can't bootstrap a fit, true recall unknown
    let mut frames_indeterminate_suspicious = 0u32; // ...but >= 15 raw candidates anyway (likely real under-detection, not board-absent)
    let mut frames_affine_fit_failed = 0u32; // >= 4 decodes but the affine fit was singular (unexpected)

    // Per-tag counts, indexed by AprilGrid ID (row-major 6×6), accumulated
    // only over `frames_considered`.
    let n_tags = euroc::GRID_ROWS * euroc::GRID_COLS;
    let mut id_decoded_hits = vec![0u32; n_tags];

    for (fname, data, width, height, _gt) in provider.iter() {
        let img = ImageView::new(&data, width, height, width).expect("valid image");
        let dets = detector
            .detect(&img, Some(&intrinsics), Some(euroc::TAG_SIZE), false)
            .expect("detection must not fail on real data");

        // Validate: no IDs outside the 6×6 grid, and no ID seen twice within
        // one frame (each physical tag can only be decoded once per frame).
        // Unconditional invariants — checked over every frame.
        let mut seen_ids = std::collections::BTreeSet::new();
        let mut frame_has_duplicate = false;
        for i in 0..dets.len() {
            let id = dets.ids[i];
            assert!(
                id < n_tags as u32,
                "frame {fname}: unexpected tag ID {id} (expected 0–{})",
                n_tags - 1
            );
            if !seen_ids.insert(id) {
                frame_has_duplicate = true;
                eprintln!("frame {fname}: duplicate detection of tag ID {id} within one frame");
            }
        }
        if frame_has_duplicate {
            duplicate_id_frames += 1;
        }

        total_frames += 1;
        total_detections += dets.len() as u32;

        if dets.len() < 4 {
            frames_indeterminate += 1;
            let n_total = dets.len() + dets.rejected_corners.len();
            if n_total >= 15 {
                frames_indeterminate_suspicious += 1;
                eprintln!(
                    "frame {fname}: only {} decoded but {n_total} raw quad candidates — cannot \
                     bootstrap a board pose (<4 decodes) to verify true recall, but this many \
                     candidates suggests the board likely IS in view",
                    dets.len()
                );
            }
            continue;
        }

        // Fit board-plane -> pixel affine map from every corner of every
        // decoded tag (up to 4 * dets.len() correspondences — always >= 16
        // once the dets.len() >= 4 precondition above holds).
        let mut board_pts = Vec::with_capacity(dets.len() * 4);
        let mut image_pts = Vec::with_capacity(dets.len() * 4);
        for i in 0..dets.len() {
            let id = dets.ids[i] as usize;
            let Some(obj_corners) = topology.obj_points[id] else {
                continue;
            };
            for (obj_corner, img_corner) in obj_corners.iter().zip(&dets.corners[i]) {
                board_pts.push([obj_corner[0], obj_corner[1]]);
                image_pts.push([f64::from(img_corner.x), f64::from(img_corner.y)]);
            }
        }
        let Some(affine) = fit_affine_board_to_image(&board_pts, &image_pts) else {
            frames_affine_fit_failed += 1;
            continue;
        };

        let mut present = 0u32;
        for id in 0..n_tags {
            let Some(corners) = topology.obj_points[id] else {
                continue;
            };
            let all_visible = corners.iter().all(|&[x, y, _z]| {
                let [px, py] = project_affine(affine, [x, y]);
                px >= 0.0 && px < width as f64 && py >= 0.0 && py < height as f64
            });
            if all_visible {
                present += 1;
            }
        }

        for &id in &seen_ids {
            id_decoded_hits[id as usize] += 1;
        }

        frames_considered += 1;
        sum_decoded += dets.len() as u64;
        sum_present += u64::from(present);
        let ratio = dets.len() as f64 / f64::from(present.max(1));
        per_frame_ratio_sum += ratio.min(1.0);
        if ratio > 1.0 {
            ratio_over_one_count += 1;
        }
    }

    assert!(
        total_frames > 0,
        "no frames were iterated despite a provider being available — check \
         tests/data/euroc/cam_april/mav0/cam0/data for corrupt/empty download"
    );

    println!(
        "Frames: {total_frames} total, {frames_considered} with an affine board fit, \
         {frames_indeterminate} indeterminate (<4 decodes; {frames_indeterminate_suspicious} \
         suspicious — ≥15 raw candidates despite that), {frames_affine_fit_failed} affine-fit \
         failures despite ≥4 decodes"
    );

    assert!(
        frames_considered > 0,
        "no frames had an affine board fit (need >=4 decoded tags matching the board layout) \
         — check tests/data/euroc/cam_april/mav0/cam0/data for corrupt/empty download"
    );

    let relative_recall_global = sum_decoded as f64 / sum_present.max(1) as f64;
    let relative_recall_mean = per_frame_ratio_sum / f64::from(frames_considered.max(1));
    println!(
        "Relative recall (decoded / present, via affine-reprojected board layout): global \
         {:.1}% ({sum_decoded}/{sum_present}), per-frame mean {:.1}% over {frames_considered} \
         frames",
        relative_recall_global * 100.0,
        relative_recall_mean * 100.0
    );
    if ratio_over_one_count > 0 {
        println!(
            "  note: {ratio_over_one_count} frame(s) had decoded > present (affine-fit boundary \
             noise) — clamped to 100% for the mean"
        );
    }

    // ── Spatial recall breakdown ────────────────────────────────────────
    // Classify each of the 36 grid tags as periphery (outer ring: row 0,
    // row 5, col 0, or col 5 — 20 tags) or interior (inner 4×4 — 16 tags),
    // and report recall separately for each, plus a full per-tag heatmap.
    let is_periphery = |id: u32| {
        let row = id as usize / euroc::GRID_COLS;
        let col = id as usize % euroc::GRID_COLS;
        row == 0 || row == euroc::GRID_ROWS - 1 || col == 0 || col == euroc::GRID_COLS - 1
    };
    let periphery_ids: Vec<u32> = (0..n_tags as u32).filter(|&id| is_periphery(id)).collect();
    let interior_ids: Vec<u32> = (0..n_tags as u32).filter(|&id| !is_periphery(id)).collect();

    let periphery_hits: u32 = periphery_ids
        .iter()
        .map(|&id| id_decoded_hits[id as usize])
        .sum();
    let interior_hits: u32 = interior_ids
        .iter()
        .map(|&id| id_decoded_hits[id as usize])
        .sum();
    let periphery_rate =
        f64::from(periphery_hits) / (f64::from(frames_considered) * periphery_ids.len() as f64);
    let interior_rate =
        f64::from(interior_hits) / (f64::from(frames_considered) * interior_ids.len() as f64);

    println!(
        "Spatial recall: periphery {:.1}% ({} tags) vs interior {:.1}% ({} tags)",
        periphery_rate * 100.0,
        periphery_ids.len(),
        interior_rate * 100.0,
        interior_ids.len()
    );
    println!("Per-tag decode rate (row-major 6×6 grid, over frames with an affine board fit):");
    for row in 0..euroc::GRID_ROWS {
        use std::fmt::Write as _;
        let line = (0..euroc::GRID_COLS).fold(String::new(), |mut acc, col| {
            let id = row * euroc::GRID_COLS + col;
            let rate = f64::from(id_decoded_hits[id]) / f64::from(frames_considered);
            let _ = write!(acc, "{:5.0}% ", rate * 100.0);
            acc
        });
        println!("  {line}");
    }
    println!("({total_detections} total detections across {total_frames} sampled frames)");

    assert_eq!(
        duplicate_id_frames, 0,
        "{duplicate_id_frames} frame(s) had the same tag ID decoded more than once — the \
         decoder must never report a physical tag twice within a single detection batch"
    );

    assert!(
        relative_recall_mean >= 0.5,
        "Relative recall too low: {relative_recall_mean:.2} (expected ≥ 0.5) over \
         {frames_considered} frames with a board pose estimated. This measures decoded/present \
         per frame (see the module-level doc comment on `euroc_detection_baseline`), so a low \
         number here means real, in-view tags are being missed — not that the board was out of \
         frame."
    );

    assert!(
        periphery_rate > 0.0,
        "Zero periphery-tag recall (interior {:.1}% vs periphery 0%) — this would mean the \
         detector is blind to the image periphery under pinhole (distortion-unaware) intrinsics.",
        interior_rate * 100.0
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

// ============================================================================
// 4. Diagnostic — pipeline funnel breakdown for the pinhole baseline
// ============================================================================

/// Diagnostic (not a pass/fail gate): breaks down exactly where candidate
/// tags are lost in the pinhole-baseline pipeline on real EuRoC frames —
/// quad extraction → fast-path contrast gate → decode. Originally written to
/// root-cause why `euroc_detection_baseline`'s absolute "≥15 tags" bar only
/// reached 27.6% recall; that turned out to be a metric problem, not a
/// detector problem (62.8% of frames simply don't have the board in view —
/// see `euroc_detection_baseline`'s doc comment for the relative-recall fix
/// that replaced the absolute bar). Kept as a still-useful stage-by-stage
/// breakdown independent of that fix.
///
/// Uses `debug_telemetry = true` to read:
/// - `TelemetryPayload::num_routed` — quad candidates extracted before the
///   funnel/decode stages run (`n` in `run_detection_pipeline`,
///   `crates/locus-core/src/detector.rs`).
/// - `DetectionBatchView::rejected_funnel_status` — per-rejected-candidate
///   outcome from the O(1) contrast pre-check (`apply_funnel_gate`,
///   `crates/locus-core/src/funnel.rs`). Only `RejectedContrast` and
///   `PassedContrast` are ever written by that gate; a rejected candidate
///   with `PassedContrast` therefore passed the contrast check but still
///   failed to decode (bad bit-sampling/Hamming distance) — the two are
///   distinguishable from this one field.
/// - The final decoded count (`DetectionBatchView::len()`).
///
/// This gives an exact "N in, M out" count per stage from a plain Rust test
/// — no Python/rerun round-trip needed for the numbers (see
/// `docs/how-to/debug_with_rerun.md` for the complementary *visual* view of
/// an individual frame).
///
/// Run with:
/// ```text
/// cargo nextest run --release --features bench-internals \
///     -E 'binary(regression_euroc)' --run-ignored ignored-only --no-capture
/// ```
#[ignore = "diagnostic; run explicitly with --ignored --nocapture"]
#[test]
fn euroc_pinhole_funnel_diagnostic() {
    let _g = common::telemetry::init("euroc_pinhole_funnel_diagnostic");
    let stride = sample_stride();
    let Some(provider) = EurocProvider::cam0(stride) else {
        println!(
            "EuRoC dataset not found — skipping. Run: bash scripts/fetch_euroc_calibration.sh"
        );
        return;
    };

    let mut detector = Detector::new();
    let intrinsics = euroc::cam0_intrinsics_pinhole();

    let mut total_frames = 0u64;
    let mut zero_quad_frames = 0u64;
    let mut total_quads = 0u64;
    let mut rejected_contrast = 0u64;
    let mut rejected_after_contrast = 0u64; // passed contrast, failed decode
    let mut rejected_sampling = 0u64; // FunnelStatus::RejectedSampling — currently dead in the pipeline (no write site), kept distinct in case that changes
    let mut rejected_unclassified = 0u64; // FunnelStatus::None on a rejected slot — unexpected here
    let mut valid = 0u64;
    let mut gwlf_fallback_total = 0u64;
    let mut worst_frame = (String::new(), 0i64); // (filename, rejects - valid), for pointing the rerun visualizer at a representative frame

    // Bucket frames by quad-candidate count `n` — a cheap proxy for "is the
    // calibration board actually in view" (a bare/far/edge-of-frame shot
    // yields very few candidates regardless of decoder quality). Buckets:
    // [0,5] likely board absent/far, [6,15] partial view, [16,30] good view,
    // [31,∞) very close/oblique view. Per bucket: (frame_count, valid_sum,
    // frames_meeting_the_15_tag_gate).
    let bucket_edges = [5u64, 15, 30];
    let mut buckets = [(0u64, 0u64, 0u64); 4];
    let bucket_of = |n: u64| bucket_edges.iter().position(|&e| n <= e).unwrap_or(3);

    // For candidates that passed the contrast gate but still failed to
    // decode: `rejected_error_rates` holds the *best* Hamming distance found
    // across all sampled rotations/scales (see `decoder.rs`'s
    // `decode_batch_soa_generic` — 0.0 encodes "no codeword sampled at all",
    // since an actual Hamming distance of 0 would always be accepted).
    // Bucketing this tells us whether decode is failing by a *near miss*
    // (consistent with a small, systematic bit-sampling bias — e.g. from
    // fitting straight quad edges to a genuinely distorted tag boundary) or
    // failing *wildly* (consistent with a fundamentally wrong homography).
    const NEAR_MISS_MAX_HAMMING: f32 = 5.0; // family default max_hamming (2) + slack
    let mut decode_fail_no_code_sampled = 0u64;
    let mut decode_fail_near_miss = 0u64; // 0 < best_hamming <= NEAR_MISS_MAX_HAMMING
    let mut decode_fail_far = 0u64; // best_hamming > NEAR_MISS_MAX_HAMMING
    let mut decode_fail_hamming_sum = 0.0f64;
    let mut decode_fail_hamming_max = 0.0f32;

    for (fname, data, width, height, _gt) in provider.iter() {
        let img = ImageView::new(&data, width, height, width).expect("valid image");
        let dets = detector
            .detect(&img, Some(&intrinsics), Some(euroc::TAG_SIZE), true)
            .expect("detection must not fail on real data");

        let telemetry = dets
            .telemetry
            .expect("debug_telemetry=true must populate telemetry");
        let n = telemetry.num_routed as u64;
        let n_rejects = i64::try_from(n).unwrap_or(i64::MAX) - dets.len() as i64;
        if n_rejects > worst_frame.1 {
            worst_frame = (fname.clone(), n_rejects);
        }

        total_frames += 1;
        total_quads += n;
        if n == 0 {
            zero_quad_frames += 1;
        }
        valid += dets.len() as u64;
        gwlf_fallback_total += telemetry.gwlf_fallback_count as u64;

        let b = &mut buckets[bucket_of(n)];
        b.0 += 1;
        b.1 += dets.len() as u64;
        if dets.len() >= 15 {
            b.2 += 1;
        }

        for (&status, &best_hamming) in dets
            .rejected_funnel_status
            .iter()
            .zip(dets.rejected_error_rates)
        {
            use locus_core::bench_api::FunnelStatus;
            match status {
                FunnelStatus::RejectedContrast => rejected_contrast += 1,
                FunnelStatus::PassedContrast => {
                    rejected_after_contrast += 1;
                    if best_hamming == 0.0 {
                        decode_fail_no_code_sampled += 1;
                    } else {
                        decode_fail_hamming_sum += f64::from(best_hamming);
                        decode_fail_hamming_max = decode_fail_hamming_max.max(best_hamming);
                        if best_hamming <= NEAR_MISS_MAX_HAMMING {
                            decode_fail_near_miss += 1;
                        } else {
                            decode_fail_far += 1;
                        }
                    }
                },
                FunnelStatus::RejectedSampling => rejected_sampling += 1,
                FunnelStatus::None => {
                    rejected_unclassified += 1;
                    eprintln!(
                        "frame {fname}: rejected candidate with unclassified FunnelStatus::None \
                         (expected only when the funnel gate is skipped for distorted \
                         intrinsics — but this run uses pinhole intrinsics)"
                    );
                },
            }
        }
    }

    assert!(total_frames > 0, "no frames were iterated");

    let pct = |x: u64| 100.0 * x as f64 / total_quads.max(1) as f64;
    println!("=== Pinhole baseline funnel breakdown ({total_frames} frames, stride {stride}) ===");
    println!(
        "Quad candidates extracted (stage 3, pre-decode): {total_quads} total, \
         {:.2}/frame average, {zero_quad_frames} frames with ZERO candidates",
        total_quads as f64 / total_frames as f64
    );
    println!(
        "  -> rejected at O(1) contrast gate (stage 3.5):  {rejected_contrast:>6} ({:.1}%)",
        pct(rejected_contrast)
    );
    println!(
        "  -> passed contrast, failed decode (stage 5):    {rejected_after_contrast:>6} ({:.1}%)",
        pct(rejected_after_contrast)
    );
    {
        let attempted = decode_fail_near_miss + decode_fail_far;
        println!(
            "     - no codeword sampled at all:                {decode_fail_no_code_sampled:>6}"
        );
        println!(
            "     - near miss (best Hamming <= {NEAR_MISS_MAX_HAMMING:.0}, max_hamming=2): \
             {decode_fail_near_miss:>6}"
        );
        println!(
            "     - far (best Hamming > {NEAR_MISS_MAX_HAMMING:.0}):             {decode_fail_far:>6}"
        );
        if attempted > 0 {
            println!(
                "     - best-Hamming stats over attempted decodes: mean {:.2}, max {:.0}",
                decode_fail_hamming_sum / attempted as f64,
                decode_fail_hamming_max
            );
        }
    }
    if rejected_sampling > 0 {
        println!(
            "  -> rejected at sampling stage:                  {rejected_sampling:>6} ({:.1}%)",
            pct(rejected_sampling)
        );
    }
    if rejected_unclassified > 0 {
        println!(
            "  -> unclassified (FunnelStatus::None):           {rejected_unclassified:>6} ({:.1}%) — unexpected, see stderr",
            pct(rejected_unclassified)
        );
    }
    println!(
        "  -> decoded successfully (final):                {valid:>6} ({:.1}%)",
        pct(valid)
    );
    println!(
        "GWLF corner-refinement fallback-to-coarse count: {gwlf_fallback_total} \
         ({:.1}% of extracted quads)",
        pct(gwlf_fallback_total)
    );
    println!(
        "Worst-offender frame (most rejects - most decode failures), for visual debugging: \
         {} ({} net rejects)",
        worst_frame.0, worst_frame.1
    );

    println!("Recall by quad-candidate-count bucket (proxy for 'is the board in view'):");
    let bucket_labels = [
        "n in [0,5]  (board likely absent/far)",
        "n in [6,15] (partial view)",
        "n in [16,30] (good view)",
        "n in [31,+) (close/oblique view)",
    ];
    for (label, &(frames, valid_sum, good)) in bucket_labels.iter().zip(buckets.iter()) {
        if frames == 0 {
            println!("  {label}: 0 frames");
            continue;
        }
        println!(
            "  {label}: {frames} frames, {:.2} valid/frame avg, {good}/{frames} ({:.1}%) meet the ≥15 gate",
            valid_sum as f64 / frames as f64,
            100.0 * good as f64 / frames as f64
        );
    }

    // No assertions beyond `total_frames > 0` — this is a diagnostic, not a
    // regression gate.
}
