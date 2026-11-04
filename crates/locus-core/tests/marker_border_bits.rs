//! `decoder.marker_border_bits`: a marker printed with a wider border than its family's
//! one-bit layout (Kalibr's AprilGrid prints tag36h11 with two) decodes on its true lattice.
#![cfg(feature = "bench-internals")]
#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use locus_core::bench_api::generate_synthetic_test_image;
use locus_core::{Detector, DetectorConfig, ImageView, TagFamily};

const CANVAS: usize = 240;
const ID: u32 = 7;

/// A tag36h11 marker with a two-cell border: the one-bit synthetic tag (8 cells of 10 px) with
/// one more black cell painted around it, so its outline spans 10 cells.
fn two_bit_tag() -> (Vec<u8>, [[f64; 2]; 4]) {
    let (mut data, inner) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, ID as u16, 80, CANVAS, 0.0);
    let cell = 10.0;
    let (x0, y0) = (inner[0][0] - cell, inner[0][1] - cell);
    let (x1, y1) = (inner[2][0] + cell, inner[2][1] + cell);
    for y in y0 as usize..y1 as usize {
        for x in x0 as usize..x1 as usize {
            let inside = (x as f64) >= inner[0][0]
                && (x as f64) < inner[2][0]
                && (y as f64) >= inner[0][1]
                && (y as f64) < inner[2][1];
            if !inside {
                data[y * CANVAS + x] = 0;
            }
        }
    }
    (data, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
}

fn detect(data: &[u8], border_bits: u8) -> Vec<(u32, [[f64; 2]; 4], f32)> {
    let config = DetectorConfig {
        decoder_marker_border_bits: border_bits,
        ..DetectorConfig::default()
    };
    let mut detector = Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let img = ImageView::new(data, CANVAS, CANVAS, CANVAS).unwrap();
    let batch = detector.detect(&img, None, None, false).unwrap();
    (0..batch.len())
        .map(|i| {
            let c = batch.corners[i].map(|p| [f64::from(p.x), f64::from(p.y)]);
            (batch.ids[i], c, batch.error_rates[i])
        })
        .collect()
}

#[test]
fn declared_border_decodes_the_true_lattice_without_errors() {
    let (data, outline) = two_bit_tag();
    let found = detect(&data, 2);
    assert_eq!(found.len(), 1, "{found:?}");
    let (id, corners, hamming) = found[0];
    assert_eq!(id, ID);
    assert_eq!(hamming, 0.0, "every bit read at its cell centre");
    // The reported quad is the printed outline, whatever the border.
    let extent = |c: &[[f64; 2]; 4], k: usize| {
        let v: Vec<f64> = c.iter().map(|p| p[k]).collect();
        v.iter().copied().fold(f64::MIN, f64::max) - v.iter().copied().fold(f64::MAX, f64::min)
    };
    for k in 0..2 {
        assert!(
            (extent(&corners, k) - extent(&outline, k)).abs() < 1.0,
            "{corners:?} vs {outline:?}"
        );
    }
}

#[test]
fn default_border_reads_a_one_bit_marker_unchanged() {
    let (data, _) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, ID as u16, 80, CANVAS, 0.0);
    let found = detect(&data, 1);
    assert_eq!(found.len(), 1);
    assert_eq!((found[0].0, found[0].2), (ID, 0.0));
}

#[test]
fn border_bits_outside_the_supported_range_are_rejected() {
    for bits in [0, locus_core::config::MAX_MARKER_BORDER_BITS + 1] {
        let config = DetectorConfig {
            decoder_marker_border_bits: bits,
            ..DetectorConfig::default()
        };
        assert!(config.validate().is_err(), "{bits}");
    }
}
