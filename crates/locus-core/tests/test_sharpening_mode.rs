#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]
//! Detector-level guard for `threshold.sharpening_mode`.
//!
//! The filter unit tests in `filter.rs` prove the two kernels differ. They do
//! not prove that `DetectorConfig::sharpening_mode` still reaches the filter:
//! hard-coding `SharpeningMode::Standard` at the `laplacian_sharpen` call site
//! in `detector.rs` leaves every other Rust and Python test green, so the
//! opt-in could silently become a no-op in a later pipeline refactor.
//!
//! These tests close that gap on the pipeline the mode actually affects, and
//! assert the *behaviour* the mode exists for rather than just the plumbing:
//! on a low-key scene, stock sharpening loses the tag and `ShootLimited`
//! recovers it.

use locus_core::bench_api::generate_synthetic_test_image;
use locus_core::config::SharpeningMode;
use locus_core::{DetectorBuilder, ImageView, TagFamily};

const CANVAS: usize = 640;
const TAG_SIZE: usize = 160;
const TAG_ID: u16 = 0;

/// Build a deterministic "low-key" scene: a dark tag on a mid-grey, textured
/// background that is only moderately brighter than the tag's black border.
///
/// This is the regime the sharpening overshoot breaks. The tile thresholder
/// binarises against `(min + max) / 2` over a 3x3 tile neighbourhood, so stock
/// sharpening's overshoot at the tag border inflates the local maximum, lifts
/// that midpoint above the background level, and the background floods into
/// the border ring as one connected component.
///
/// Deterministic on purpose — no `rand`, and `noise_sigma` is left at 0
/// because `generate_synthetic_test_image` draws its noise from a thread RNG.
fn low_key_scene(bg: u8, dark: u8, texture: u8, period: usize) -> Vec<u8> {
    let (base, _) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, TAG_ID, TAG_SIZE, CANVAS, 0.0);

    let mut out = vec![0u8; CANVAS * CANVAS];
    for y in 0..CANVAS {
        for x in 0..CANVAS {
            // Checkerboard texture so the background is not flat; a flat
            // background gives the tile thresholder an unrealistically clean
            // min/max and the failure mode does not reproduce.
            let bright = if (x / period + y / period).is_multiple_of(2) {
                bg.saturating_add(texture)
            } else {
                bg.saturating_sub(texture)
            };
            out[y * CANVAS + x] = if base[y * CANVAS + x] > 127 {
                bright
            } else {
                dark
            };
        }
    }
    out
}

/// Detected tag ids for one sharpening configuration.
fn detect_ids(data: &[u8], sharpening: bool, mode: SharpeningMode) -> Vec<u32> {
    let img = ImageView::new(data, CANVAS, CANVAS, CANVAS).unwrap();
    let mut detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .with_sharpening(sharpening)
        .with_sharpening_mode(mode)
        .build();
    detector
        .detect(&img, None, None, false)
        .map(|batch| batch.ids.to_vec())
        .unwrap_or_default()
}

/// The mode must change what the detector produces — and change it in the
/// direction the PR claims: stock sharpening loses the low-key tag, the
/// shoot-limited filter recovers it, matching "sharpening off" as the
/// theoretical ceiling.
#[test]
fn sharpening_mode_changes_detector_output_on_low_key_scene() {
    let data = low_key_scene(100, 30, 16, 7);

    let stock = detect_ids(&data, true, SharpeningMode::Standard);
    let limited = detect_ids(&data, true, SharpeningMode::ShootLimited);
    let off = detect_ids(&data, false, SharpeningMode::Standard);

    assert!(
        stock.is_empty(),
        "fixture no longer exercises the overshoot failure: stock sharpening found {stock:?}"
    );
    assert_eq!(
        limited,
        vec![u32::from(TAG_ID)],
        "ShootLimited did not recover the low-key tag — is config.sharpening_mode still \
         reaching filter::laplacian_sharpen?"
    );
    assert_eq!(
        off,
        vec![u32::from(TAG_ID)],
        "sharpening-off is the ceiling ShootLimited is supposed to approach"
    );
}

/// A second, independent operating point, so the guard does not rest on one
/// hand-tuned fixture.
#[test]
fn sharpening_mode_changes_detector_output_second_operating_point() {
    let data = low_key_scene(80, 30, 16, 3);

    assert!(
        detect_ids(&data, true, SharpeningMode::Standard).is_empty(),
        "stock sharpening unexpectedly recovered the tag"
    );
    assert_eq!(
        detect_ids(&data, true, SharpeningMode::ShootLimited),
        vec![u32::from(TAG_ID)]
    );
}

/// `Standard` is the default, so an untouched builder must behave exactly like
/// an explicit `Standard` — this is the byte-identity claim at detector level.
#[test]
fn default_sharpening_mode_matches_explicit_standard() {
    let data = low_key_scene(100, 30, 16, 7);
    let img = ImageView::new(&data, CANVAS, CANVAS, CANVAS).unwrap();

    let mut default_detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .with_sharpening(true)
        .build();
    let default_ids: Vec<u32> = default_detector
        .detect(&img, None, None, false)
        .map(|b| b.ids.to_vec())
        .unwrap_or_default();

    assert_eq!(
        default_ids,
        detect_ids(&data, true, SharpeningMode::Standard),
        "the default sharpening mode drifted away from Standard"
    );
}
