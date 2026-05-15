//! The detector-level GWLF pass is route-aware: it fires on every
//! candidate whose route-resolved refinement is `Gwlf`, independently
//! of the static `config.refinement_mode`. These tests pin that
//! invariant — they used to fail because the pass was gated on the
//! static field alone, which silently dropped `AdaptivePpb` routes.

#![allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::unwrap_used,
    missing_docs
)]

use locus_core::bench_api::generate_synthetic_test_image;
use locus_core::config::{
    AdaptivePpbConfig, CornerRefinementMode, DetectorConfig, QuadExtractionMode,
    QuadExtractionPolicy,
};
use locus_core::{Detector, ImageView, TagFamily};

fn detect_and_read_gwlf_telemetry(config: DetectorConfig) -> (usize, usize, f32) {
    let canvas = 400;
    let (data, _gt) = generate_synthetic_test_image(TagFamily::AprilTag36h11, 42, 200, canvas, 0.0);
    let img = ImageView::new(&data, canvas, canvas, canvas).expect("valid image");
    let mut detector = Detector::with_config(config);
    let batch = detector
        .detect(&img, None, None, true)
        .expect("detection should not error");
    let telem = batch.telemetry.expect("debug_telemetry=true requested");
    (batch.len(), telem.gwlf_fallback_count, telem.gwlf_avg_delta)
}

#[test]
fn adaptiveppb_with_gwlf_routes_fires_detector_gwlf() {
    let config = DetectorConfig::builder()
        .refinement_mode(CornerRefinementMode::Erf)
        .quad_extraction_policy(QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
            threshold: 2.5,
            low_extraction: QuadExtractionMode::ContourRdp,
            high_extraction: QuadExtractionMode::ContourRdp,
            low_refinement: CornerRefinementMode::Gwlf,
            high_refinement: CornerRefinementMode::Gwlf,
        }))
        .build();

    let (n, fallback, avg_delta) = detect_and_read_gwlf_telemetry(config);
    assert!(n > 0, "synthetic tag should be detected");
    assert!(
        avg_delta > 0.0 || fallback > 0,
        "AdaptivePpb+Gwlf route should fire the detector-level GWLF pass; \
         got gwlf_avg_delta={avg_delta:.6} fallback={fallback}"
    );
}

#[test]
fn static_non_gwlf_does_not_fire_detector_gwlf() {
    let config = DetectorConfig::builder()
        .refinement_mode(CornerRefinementMode::Erf)
        .build();

    let (n, fallback, avg_delta) = detect_and_read_gwlf_telemetry(config);
    assert!(n > 0, "synthetic tag should be detected");
    assert_eq!(fallback, 0, "Erf path should not fire the GWLF pass");
    assert_eq!(avg_delta, 0.0, "Erf path should not fire the GWLF pass");
}

#[test]
fn adaptiveppb_without_gwlf_does_not_fire_detector_gwlf() {
    let config = DetectorConfig::builder()
        .quad_extraction_policy(QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
            threshold: 2.5,
            low_extraction: QuadExtractionMode::ContourRdp,
            high_extraction: QuadExtractionMode::ContourRdp,
            low_refinement: CornerRefinementMode::Erf,
            high_refinement: CornerRefinementMode::None,
        }))
        .build();

    let (n, fallback, avg_delta) = detect_and_read_gwlf_telemetry(config);
    assert!(n > 0, "synthetic tag should be detected");
    assert_eq!(fallback, 0, "no Gwlf-route candidate, no GWLF pass");
    assert_eq!(avg_delta, 0.0, "no Gwlf-route candidate, no GWLF pass");
}
