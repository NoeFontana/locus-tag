//! End-to-end verification that the shipped JSON profiles deserialize
//! via the serde shim into `DetectorConfig` values matching hand-transcribed
//! expectations.
//!
//! This is the tripwire against silent profile drift — a reviewer editing a
//! profile JSON sees a loud failure here if the edit diverges from the
//! documented semantics.

#![cfg(feature = "profiles")]
#![allow(
    clippy::bool_assert_comparison,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

use locus_core::config::{
    AdaptivePpbConfig, CornerRefinementMode, DecodeMode, DetectorConfig, QuadExtractionMode,
    QuadExtractionPolicy, SegmentationConnectivity,
};

/// Fields that *all three* shipped profiles carry at the current repo defaults.
fn assert_shared_defaults(cfg: &DetectorConfig) {
    let d = DetectorConfig::default();
    assert_eq!(cfg.threshold_min_range, d.threshold_min_range);
    assert_eq!(cfg.enable_adaptive_window, d.enable_adaptive_window);
    assert_eq!(cfg.threshold_min_radius, d.threshold_min_radius);
    assert_eq!(cfg.threshold_max_radius, d.threshold_max_radius);
    assert_eq!(
        cfg.adaptive_threshold_constant,
        d.adaptive_threshold_constant
    );
    assert_eq!(
        cfg.adaptive_threshold_gradient_threshold,
        d.adaptive_threshold_gradient_threshold
    );
    assert_eq!(cfg.quad_min_area, d.quad_min_area);
    assert_eq!(cfg.quad_max_aspect_ratio, d.quad_max_aspect_ratio);
    assert_eq!(cfg.quad_min_fill_ratio, d.quad_min_fill_ratio);
    assert_eq!(cfg.quad_max_fill_ratio, d.quad_max_fill_ratio);
    assert_eq!(cfg.quad_min_edge_length, d.quad_min_edge_length);
    assert_eq!(cfg.subpixel_refinement_sigma, d.subpixel_refinement_sigma);
    assert_eq!(cfg.max_hamming_error, d.max_hamming_error);
    assert_eq!(cfg.gwlf_transversal_alpha, d.gwlf_transversal_alpha);
    assert_eq!(cfg.huber_delta_px, d.huber_delta_px);
    assert_eq!(cfg.tikhonov_alpha_max, d.tikhonov_alpha_max);
    assert_eq!(cfg.sigma_n_sq, d.sigma_n_sq);
    assert_eq!(cfg.structure_tensor_radius, d.structure_tensor_radius);
    assert_eq!(cfg.segmentation_margin, d.segmentation_margin);
    assert_eq!(cfg.upscale_factor, d.upscale_factor);
    // Per-call orchestration fields are never set from a profile.
    assert_eq!(cfg.decimation, d.decimation);
    assert_eq!(cfg.nthreads, d.nthreads);
}

#[test]
fn standard_profile_matches_former_builder() {
    let cfg = DetectorConfig::from_profile("standard");

    // Standard-specific overrides.
    assert_eq!(cfg.threshold_tile_size, 8);
    assert_eq!(cfg.enable_sharpening, true);
    assert_eq!(cfg.quad_max_elongation, 20.0);
    assert_eq!(cfg.quad_min_density, 0.15);
    assert_eq!(cfg.quad_min_edge_score, 4.0);
    assert_eq!(cfg.refinement_mode, CornerRefinementMode::Erf);
    assert_eq!(cfg.decoder_min_contrast, 20.0);
    assert_eq!(cfg.decode_mode, DecodeMode::Hard);
    assert_eq!(cfg.quad_extraction_mode, QuadExtractionMode::ContourRdp);
    assert_eq!(
        cfg.segmentation_connectivity,
        SegmentationConnectivity::Eight
    );

    assert_shared_defaults(&cfg);
    cfg.validate().expect("standard profile must validate");
}

#[test]
fn grid_profile_matches_former_builder() {
    let cfg = DetectorConfig::from_profile("grid");

    // Grid-specific overrides.
    assert_eq!(cfg.threshold_tile_size, 8);
    assert_eq!(cfg.enable_sharpening, false);
    assert_eq!(cfg.quad_max_elongation, 20.0);
    assert_eq!(cfg.quad_min_density, 0.15);
    assert_eq!(cfg.quad_min_edge_score, 2.0);
    assert_eq!(cfg.refinement_mode, CornerRefinementMode::Erf);
    assert_eq!(cfg.decoder_min_contrast, 10.0);
    assert_eq!(cfg.decode_mode, DecodeMode::Hard);
    assert_eq!(cfg.quad_extraction_mode, QuadExtractionMode::ContourRdp);
    assert_eq!(
        cfg.segmentation_connectivity,
        SegmentationConnectivity::Four
    );

    assert_shared_defaults(&cfg);
    cfg.validate().expect("grid profile must validate");
}

#[test]
fn high_accuracy_profile_routes_low_ppb_to_contour_rdp() {
    let cfg = DetectorConfig::from_profile("high_accuracy");

    // High-accuracy overrides.
    assert_eq!(cfg.threshold_tile_size, 8);
    assert_eq!(cfg.enable_sharpening, false);
    assert_eq!(cfg.quad_max_elongation, 20.0);
    assert_eq!(cfg.quad_min_density, 0.15);
    assert_eq!(cfg.quad_min_edge_score, 4.0);
    assert_eq!(cfg.refinement_mode, CornerRefinementMode::None);
    assert_eq!(cfg.decoder_min_contrast, 20.0);
    assert_eq!(cfg.decode_mode, DecodeMode::Hard);
    assert_eq!(cfg.quad_extraction_mode, QuadExtractionMode::EdLines);
    assert_eq!(
        cfg.segmentation_connectivity,
        SegmentationConnectivity::Eight
    );

    match cfg.quad_extraction_policy {
        QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
            threshold,
            low_extraction,
            high_extraction,
            low_refinement,
            high_refinement,
        }) => {
            assert!(
                threshold > 1.0 && threshold < 5.0,
                "threshold must be in validator bounds (1,5), got {threshold}"
            );
            assert_eq!(low_extraction, QuadExtractionMode::ContourRdp);
            assert_eq!(high_extraction, QuadExtractionMode::EdLines);
            assert_eq!(low_refinement, CornerRefinementMode::Erf);
            assert_eq!(high_refinement, CornerRefinementMode::None);
        },
        QuadExtractionPolicy::Static => {
            panic!("high_accuracy profile must carry AdaptivePpb policy, got Static");
        },
    }

    assert_shared_defaults(&cfg);
    cfg.validate().expect("high_accuracy profile must validate");
}

#[test]
fn max_recall_adaptive_profile_enables_adaptive_router() {
    let cfg = DetectorConfig::from_profile("max_recall_adaptive");

    match cfg.quad_extraction_policy {
        QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
            threshold,
            low_extraction,
            high_extraction,
            low_refinement,
            high_refinement,
        }) => {
            assert!(
                threshold > 1.0 && threshold < 5.0,
                "threshold must be in validator bounds (1,5), got {threshold}"
            );
            assert_eq!(low_extraction, QuadExtractionMode::ContourRdp);
            assert_eq!(high_extraction, QuadExtractionMode::EdLines);
            assert_eq!(low_refinement, CornerRefinementMode::Erf);
            assert_eq!(high_refinement, CornerRefinementMode::None);
        },
        QuadExtractionPolicy::Static => {
            panic!("max_recall_adaptive profile must carry AdaptivePpb policy, got Static");
        },
    }

    // Runtime-overridden fields still round-trip (readable via debug tools).
    assert_eq!(cfg.quad_extraction_mode, QuadExtractionMode::ContourRdp);
    assert_eq!(cfg.refinement_mode, CornerRefinementMode::Erf);
    assert_eq!(cfg.decode_mode, DecodeMode::Hard);
    assert_eq!(
        cfg.segmentation_connectivity,
        SegmentationConnectivity::Eight
    );

    cfg.validate()
        .expect("max_recall_adaptive profile must validate");
}

#[test]
fn from_profile_json_accepts_minimal_document() {
    // Loader must tolerate groups being absent and fall back to
    // `DetectorConfig::default()` per-field.
    let cfg = DetectorConfig::from_profile_json(r#"{"name": "minimal"}"#)
        .expect("minimal JSON must deserialize");
    assert_eq!(cfg, DetectorConfig::default());
}

#[test]
#[should_panic(expected = "Unknown shipped profile")]
fn from_profile_panics_on_unknown_name() {
    let _ = DetectorConfig::from_profile("does_not_exist");
}
