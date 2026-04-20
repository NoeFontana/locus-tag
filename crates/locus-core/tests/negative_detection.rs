#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    missing_docs
)]
//! Negative-detection fixtures: the detector must return zero detections —
//! and not panic — on curated inputs that contain no fiducial markers.
//!
//! QR codes and text documents are intentionally excluded: adding a QR or
//! text-rendering dev-dep for a single fixture class would violate the
//! dependency hygiene rule in `.agent/rules/constraints.md §5`.
//!
//! The stability-loop test is the structural proxy for "no heap allocations
//! outside the arena": a `#[global_allocator]` counting allocator is out of
//! scope because the bumpalo contract is already enforced structurally.

use locus_core::{DetectorBuilder, ImageView, PoseEstimationMode};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
/// SIMD gather kernels may perform 32-bit loads on 8-bit data — require 3
/// bytes of end-padding (see `src/image.rs` has_simd_padding).
const SIMD_PADDING: usize = 3;

fn padded(mut data: Vec<u8>) -> Vec<u8> {
    data.resize(WIDTH * HEIGHT + SIMD_PADDING, 0);
    data
}

fn noise_frame(seed: u64) -> Vec<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut data = vec![0u8; WIDTH * HEIGHT];
    for px in &mut data {
        *px = rng.random();
    }
    data
}

fn gradient_frame<F: Fn(usize, usize) -> u8>(f: F) -> Vec<u8> {
    let mut data = vec![0u8; WIDTH * HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            data[y * WIDTH + x] = f(x, y);
        }
    }
    data
}

fn run_and_assert_empty(data: &[u8], label: &str) {
    let mut detector = DetectorBuilder::new().build();
    let image =
        ImageView::new(data, WIDTH, HEIGHT, WIDTH).expect("ImageView construction must succeed");
    let detections = detector
        .detect(&image, None, None, PoseEstimationMode::Fast, false)
        .unwrap_or_else(|e| panic!("detect() failed on {label}: {e:?}"));
    assert!(
        detections.is_empty(),
        "Expected zero detections on {label}, got {}",
        detections.len()
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[test]
fn negative_uniform_gray() {
    run_and_assert_empty(&padded(vec![0x80; WIDTH * HEIGHT]), "uniform gray 0x80");
}

#[test]
fn negative_uniform_black() {
    run_and_assert_empty(&padded(vec![0x00; WIDTH * HEIGHT]), "uniform black");
}

#[test]
fn negative_uniform_white() {
    run_and_assert_empty(&padded(vec![0xFF; WIDTH * HEIGHT]), "uniform white");
}

#[test]
fn negative_pure_noise_seeded() {
    run_and_assert_empty(
        &padded(noise_frame(0xDEAD_BEEF_CAFE_BABE)),
        "seeded uniform random noise",
    );
}

#[test]
fn negative_gradient_horizontal() {
    run_and_assert_empty(
        &padded(gradient_frame(|x, _| (x & 0xFF) as u8)),
        "horizontal gradient",
    );
}

#[test]
fn negative_gradient_vertical() {
    run_and_assert_empty(
        &padded(gradient_frame(|_, y| (y & 0xFF) as u8)),
        "vertical gradient",
    );
}

#[test]
fn negative_gradient_diagonal() {
    run_and_assert_empty(
        &padded(gradient_frame(|x, y| ((x + y) & 0xFF) as u8)),
        "diagonal gradient",
    );
}

#[test]
fn negative_checkerboard_non_charuco() {
    run_and_assert_empty(
        &padded(locus_core::bench_api::generate_checkered(WIDTH, HEIGHT)),
        "plain checkerboard (no ChArUco markers)",
    );
}

#[test]
fn negative_salt_and_pepper() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5A17_DEAD_C0DE_BEEF);
    let mut data = vec![0x80u8; WIDTH * HEIGHT];
    for px in &mut data {
        let r: f32 = rng.random();
        if r < 0.025 {
            *px = 0;
        } else if r < 0.05 {
            *px = 255;
        }
    }
    run_and_assert_empty(&padded(data), "5% salt-and-pepper noise");
}

// ---------------------------------------------------------------------------
// Zero-allocation stability loop
// ---------------------------------------------------------------------------

#[test]
fn negative_repeated_detection_no_buffer_growth() {
    let buf = padded(noise_frame(0xFEED_FACE_C0DE_CAFE));
    let image =
        ImageView::new(&buf, WIDTH, HEIGHT, WIDTH).expect("ImageView construction must succeed");

    let mut detector = DetectorBuilder::new().build();

    // Warm-up: upscale_buf is sized on the first call; we then assert that
    // subsequent calls do not grow it. Proxy for zero arena-external allocs.
    let first = detector
        .detect(&image, None, None, PoseEstimationMode::Fast, false)
        .expect("first detection must succeed");
    assert!(first.is_empty());
    let baseline_cap = detector.state().upscale_buf.capacity();

    for iter in 1..100 {
        let detections = detector
            .detect(&image, None, None, PoseEstimationMode::Fast, false)
            .expect("repeated detection must succeed");
        assert!(detections.is_empty(), "iter {iter}: unexpected detections");
        let cap = detector.state().upscale_buf.capacity();
        assert_eq!(
            cap, baseline_cap,
            "iter {iter}: upscale buffer grew {baseline_cap} → {cap}",
        );
    }
}
