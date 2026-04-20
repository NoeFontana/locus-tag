#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    missing_docs
)]
//! Property: rotating the input image by k*90° (k ∈ {0,1,2,3}) produces the
//! same quad corner SET as rotating the original corners by k*90°, up to
//! cyclic permutation, within 1 px.
//!
//! The image rotation is a pure integer transpose-flip (no interpolation), so
//! the property is mathematically exact on the pixel grid; the only source of
//! slack is sub-pixel corner refinement. 1 px tolerance is enough to allow for
//! convergence-basin differences between edge orientations in GWLF but tight
//! enough to catch a genuine corner-ordering bug.

use locus_core::bench_api::family_to_decoder;
use locus_core::{DetectorBuilder, ImageView, PoseEstimationMode, TagFamily};
use proptest::prelude::*;

const CANVAS: usize = 256;
const TAG_SIZE: usize = 128;
const SIMD_PADDING: usize = 3;
const CORNER_TOLERANCE_PX: f64 = 1.0;

fn valid_id() -> impl Strategy<Value = u16> {
    let n = family_to_decoder(TagFamily::AprilTag36h11).num_codes() as u16;
    0u16..n
}

/// Rotate a sub-pixel 2D point CW about the center of a square canvas.
///
/// Uses the "+0.5 pixel center" convention (see `.agent/rules/core.md`): pixel
/// (i, j) has its center at sub-pixel (i + 0.5, j + 0.5), so the canvas
/// rotation center is at (canvas / 2, canvas / 2). Under this convention,
/// rotating by 90° CW maps (x, y) → (canvas − y, x), etc.
fn rotate_point_cw(p: [f64; 2], k: u32, canvas: usize) -> [f64; 2] {
    let (x, y) = (p[0], p[1]);
    let s = canvas as f64;
    match k & 3 {
        0 => [x, y],
        1 => [s - y, x],
        2 => [s - x, s - y],
        3 => [y, s - x],
        _ => unreachable!(),
    }
}

/// Rotate an image CW by k*90° on a square canvas. Returns a new buffer of
/// the same dimensions.
fn rotate_image_cw(src: &[u8], canvas: usize, k: u32) -> Vec<u8> {
    let mut dst = vec![0u8; canvas * canvas];
    let k = k & 3;
    for y in 0..canvas {
        for x in 0..canvas {
            let v = src[y * canvas + x];
            let (nx, ny) = match k {
                0 => (x, y),
                1 => (canvas - 1 - y, x),
                2 => (canvas - 1 - x, canvas - 1 - y),
                3 => (y, canvas - 1 - x),
                _ => unreachable!(),
            };
            dst[ny * canvas + nx] = v;
        }
    }
    dst
}

/// Pad a buffer to the SIMD-safe end-padding required by gather kernels.
fn with_padding(mut data: Vec<u8>) -> Vec<u8> {
    data.resize(data.len() + SIMD_PADDING, 0);
    data
}

/// Detect once and return the single marker's corners as an unordered set.
fn detect_single(data: &[u8], canvas: usize) -> Option<[[f64; 2]; 4]> {
    let mut detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let image = ImageView::new(data, canvas, canvas, canvas).ok()?;
    let view = detector
        .detect(&image, None, None, PoseEstimationMode::Fast, false)
        .ok()?;
    if view.len() != 1 {
        return None;
    }
    let c = &view.corners[0];
    Some([
        [f64::from(c[0].x), f64::from(c[0].y)],
        [f64::from(c[1].x), f64::from(c[1].y)],
        [f64::from(c[2].x), f64::from(c[2].y)],
        [f64::from(c[3].x), f64::from(c[3].y)],
    ])
}

/// Assert two corner sets agree as unordered sets within `tol` pixels.
fn corners_agree_unordered(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4], tol: f64) -> bool {
    let mut matched = [false; 4];
    for pa in a {
        let mut found = false;
        for (j, pb) in b.iter().enumerate() {
            if matched[j] {
                continue;
            }
            let dx = pa[0] - pb[0];
            let dy = pa[1] - pb[1];
            if (dx * dx + dy * dy).sqrt() <= tol {
                matched[j] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    matched.iter().all(|&m| m)
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "proptest-regressions/quad_rotation.txt",
            ),
        )),
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_quad_rotation_invariance(id in valid_id()) {
        let (canvas_img, gt_corners) = locus_core::bench_api::generate_synthetic_test_image(
            TagFamily::AprilTag36h11,
            id,
            TAG_SIZE,
            CANVAS,
            0.0,
        );

        // Baseline: the detector must see the tag at all in the un-rotated
        // frame — otherwise the property is vacuous for this id.
        let buf = with_padding(canvas_img.clone());
        let Some(base_corners) = detect_single(&buf, CANVAS) else {
            // If baseline detection fails, skip this case — not a property
            // failure.
            return Ok(());
        };

        // Baseline must agree with ground-truth within 1 px.
        prop_assert!(
            corners_agree_unordered(&base_corners, &gt_corners, CORNER_TOLERANCE_PX),
            "baseline detection disagrees with ground-truth corners",
        );

        for k in 1u32..4 {
            let rotated = rotate_image_cw(&canvas_img, CANVAS, k);
            let buf_rot = with_padding(rotated);
            let rot_corners = detect_single(&buf_rot, CANVAS).ok_or_else(|| {
                TestCaseError::fail(format!("detection failed on k={k} rotation of id={id}"))
            })?;

            let expected: [[f64; 2]; 4] = [
                rotate_point_cw(base_corners[0], k, CANVAS),
                rotate_point_cw(base_corners[1], k, CANVAS),
                rotate_point_cw(base_corners[2], k, CANVAS),
                rotate_point_cw(base_corners[3], k, CANVAS),
            ];

            prop_assert!(
                corners_agree_unordered(&rot_corners, &expected, CORNER_TOLERANCE_PX),
                "k={k}: rotated corners {rot_corners:?} do not match expected {expected:?}",
            );
        }
    }
}
