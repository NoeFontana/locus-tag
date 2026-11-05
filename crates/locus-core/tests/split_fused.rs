//! `quad.split_fused`: two markers whose corners are joined by an ink neck form one component;
//! the split recovers both.
#![cfg(feature = "bench-internals")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_sign_loss,
    missing_docs
)]

use locus_core::bench_api::generate_synthetic_test_image;
use locus_core::{Detector, DetectorConfig, ImageView, TagFamily};

const CANVAS: usize = 240;
const TAG: usize = 80;
const IDS: [u16; 2] = [3, 11];

/// Two tag36h11 markers meeting corner to corner at (120, 120), joined by the two pixels that
/// turn their diagonal contact into a 4-connected neck, as ink spread does.
fn fused_pair() -> Vec<u8> {
    let mut canvas = vec![255u8; CANVAS * CANVAS];
    for (k, &id) in IDS.iter().enumerate() {
        let (tag, corners) =
            generate_synthetic_test_image(TagFamily::AprilTag36h11, id, TAG, 2 * TAG, 0.0);
        let (x0, y0) = (corners[0][0] as usize, corners[0][1] as usize);
        let at = 120 - TAG + k * TAG;
        for y in 0..TAG {
            for x in 0..TAG {
                canvas[(at + y) * CANVAS + at + x] = tag[(y0 + y) * 2 * TAG + x0 + x];
            }
        }
    }
    canvas[119 * CANVAS + 120] = 0;
    canvas[120 * CANVAS + 119] = 0;
    canvas
}

fn detected_ids(config: DetectorConfig, data: &[u8]) -> Vec<u32> {
    let mut detector = Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let img = ImageView::new(data, CANVAS, CANVAS, CANVAS).unwrap();
    let batch = detector.detect(&img, None, None, false).unwrap();
    let mut ids: Vec<u32> = batch.ids[..batch.len()].to_vec();
    ids.sort_unstable();
    ids
}

#[test]
fn grid_splits_two_markers_fused_at_a_corner() {
    let data = fused_pair();
    let grid = DetectorConfig::from_profile("grid");
    assert!(grid.quad_split_fused);
    assert_eq!(detected_ids(grid, &data), vec![3, 11]);
}

#[test]
fn without_the_split_the_fused_pair_is_not_fully_recovered() {
    let data = fused_pair();
    let config = DetectorConfig {
        quad_split_fused: false,
        ..DetectorConfig::from_profile("grid")
    };
    assert!(detected_ids(config, &data).len() < 2);
}
