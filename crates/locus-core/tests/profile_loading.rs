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
    AdaptivePpbConfig, CornerRefinementMode, DetectorConfig, EdLinesImbalanceGatePolicy,
    QuadExtractionMode, QuadExtractionPolicy, SegmentationConnectivity,
};

/// Fields that *all three* shipped profiles carry at the current repo defaults.
fn assert_shared_defaults(cfg: &DetectorConfig) {
    let d = DetectorConfig::default();
    assert_eq!(cfg.threshold_min_range, d.threshold_min_range);
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
    // `quad_min_area` is profile-specific (clean-render profiles raise it to
    // suppress small textured-quad false positives); asserted per-profile.
    // `max_hamming_error` is profile-specific (high_accuracy tightens to 1 to
    // close the synthetic-corpus FP at hamming=2); asserted per-profile.
    assert_eq!(cfg.quad_max_aspect_ratio, d.quad_max_aspect_ratio);
    assert_eq!(cfg.quad_min_fill_ratio, d.quad_min_fill_ratio);
    assert_eq!(cfg.quad_max_fill_ratio, d.quad_max_fill_ratio);
    assert_eq!(cfg.quad_min_edge_length, d.quad_min_edge_length);
    assert_eq!(cfg.subpixel_refinement_sigma, d.subpixel_refinement_sigma);
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
    assert_eq!(cfg.quad_min_area, 36);
    assert_eq!(cfg.quad_max_elongation, 20.0);
    assert_eq!(cfg.quad_min_density, 0.15);
    assert_eq!(cfg.quad_min_edge_score, 4.0);
    assert_eq!(cfg.refinement_mode, CornerRefinementMode::Erf);
    assert_eq!(cfg.decoder_min_contrast, 20.0);
    assert_eq!(cfg.quad_extraction_mode, QuadExtractionMode::ContourRdp);
    assert_eq!(cfg.max_hamming_error, None);
    assert_eq!(
        cfg.segmentation_connectivity,
        SegmentationConnectivity::Eight
    );
    // Edge refinement is a high_accuracy-only opt-in; standard stays off.
    assert_eq!(cfg.pose_edge_refinement_enabled, false);

    assert_shared_defaults(&cfg);
    cfg.validate().expect("standard profile must validate");
}

#[test]
fn grid_profile_matches_former_builder() {
    let cfg = DetectorConfig::from_profile("grid");

    // Grid-specific overrides.
    assert_eq!(cfg.threshold_tile_size, 8);
    assert_eq!(cfg.enable_sharpening, false);
    assert_eq!(cfg.quad_min_area, 36);
    assert_eq!(cfg.quad_max_elongation, 20.0);
    assert_eq!(cfg.quad_min_density, 0.15);
    assert_eq!(cfg.quad_min_edge_score, 2.0);
    assert_eq!(cfg.refinement_mode, CornerRefinementMode::Erf);
    assert_eq!(cfg.decoder_min_contrast, 10.0);
    assert_eq!(cfg.quad_extraction_mode, QuadExtractionMode::ContourRdp);
    assert_eq!(cfg.max_hamming_error, None);
    assert_eq!(
        cfg.segmentation_connectivity,
        SegmentationConnectivity::Four
    );
    // Edge refinement is a high_accuracy-only opt-in; grid stays off.
    assert_eq!(cfg.pose_edge_refinement_enabled, false);

    assert_shared_defaults(&cfg);
    cfg.validate().expect("grid profile must validate");
}

#[test]
fn high_accuracy_profile_routes_low_ppb_to_contour_rdp() {
    let cfg = DetectorConfig::from_profile("high_accuracy");

    // High-accuracy overrides.
    assert_eq!(cfg.threshold_tile_size, 8);
    assert_eq!(cfg.enable_sharpening, false);
    assert_eq!(cfg.quad_min_area, 400);
    assert_eq!(cfg.quad_max_elongation, 20.0);
    assert_eq!(cfg.quad_min_density, 0.15);
    assert_eq!(cfg.quad_min_edge_score, 4.0);
    assert_eq!(cfg.refinement_mode, CornerRefinementMode::None);
    assert_eq!(cfg.decoder_min_contrast, 20.0);
    assert_eq!(cfg.max_hamming_error, Some(1));
    assert_eq!(cfg.quad_extraction_mode, QuadExtractionMode::EdLines);
    assert_eq!(
        cfg.segmentation_connectivity,
        SegmentationConnectivity::Eight
    );
    // Model-edge pose refinement is shipped on for high_accuracy (v0.7.0); lock
    // it so an accidental JSON revert is caught loudly. standard/grid stay off.
    assert_eq!(cfg.pose_edge_refinement_enabled, true);

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

#[test]
fn to_profile_json_round_trips_every_field() {
    // Serialize → deserialize must be the identity over every profile field.
    // Every field is set to a *distinct, non-default, valid* value so that a
    // mis-wired flat↔nested mapping (e.g. writing `quad_min_area` into the
    // `max_area` slot, or swapping two same-typed fields) produces a mismatch —
    // the schema-parity test only guards key sets, not value wiring, so this is
    // the value-level guard.
    let cfg = DetectorConfig {
        // Threshold
        threshold_tile_size: 12,
        threshold_min_range: 5,
        enable_sharpening: false,
        threshold_min_radius: 3,
        threshold_max_radius: 9,
        adaptive_threshold_constant: 4,
        adaptive_threshold_gradient_threshold: 42,
        // Quad
        quad_min_area: 25,
        quad_max_aspect_ratio: 4.5,
        quad_min_fill_ratio: 0.2,
        quad_max_fill_ratio: 0.9,
        quad_min_edge_length: 6.0,
        quad_min_edge_score: 1.5,
        subpixel_refinement_sigma: 0.7,
        upscale_factor: 2,
        quad_max_elongation: 18.0,
        quad_min_density: 0.25,
        quad_extraction_mode: QuadExtractionMode::ContourRdp,
        edlines_imbalance_gate: EdLinesImbalanceGatePolicy::Enabled,
        // Decoder
        decoder_min_contrast: 12.5,
        refinement_mode: CornerRefinementMode::Erf,
        max_hamming_error: Some(2),
        gwlf_transversal_alpha: 0.03,
        // Pose
        huber_delta_px: 1.75,
        tikhonov_alpha_max: 0.3,
        sigma_n_sq: 3.5,
        structure_tensor_radius: 4,
        pose_consistency_fpr: 1e-3,
        pose_consistency_gate_sigma_px: 0.75,
        pose_consistency_min_decisive_ratio: 7.5,
        outlier_drop_d2_threshold: 25.0,
        // Segmentation
        segmentation_connectivity: SegmentationConnectivity::Four,
        segmentation_margin: 2,
        // Policy (exercises the AdaptivePpb externally-tagged enum subtree)
        quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
            threshold: 3.25,
            low_extraction: QuadExtractionMode::ContourRdp,
            high_extraction: QuadExtractionMode::EdLines,
            low_refinement: CornerRefinementMode::Erf,
            high_refinement: CornerRefinementMode::None,
        }),
        // `decimation` / `nthreads` are per-call orchestration, not profile
        // fields — left at default (they are intentionally not round-tripped).
        ..DetectorConfig::default()
    };

    let json = cfg.to_profile_json().expect("serialize config");
    let round_tripped = DetectorConfig::from_profile_json(&json).expect("re-parse config");
    assert_eq!(round_tripped, cfg);
}

#[test]
fn to_profile_json_round_trips_infinite_decisive_ratio() {
    // `f64::INFINITY` is the documented value that disables the χ² escape
    // clause. JSON has no number form for it, so it must round-trip as `null`
    // (matching Pydantic's inf↔null handling) rather than breaking the boundary.
    let cfg = DetectorConfig {
        pose_consistency_min_decisive_ratio: f64::INFINITY,
        ..DetectorConfig::default()
    };

    let json = cfg.to_profile_json().expect("serialize config");
    assert!(
        json.contains("\"pose_consistency_min_decisive_ratio\":null"),
        "infinite ratio must serialize as JSON null; got: {json}"
    );

    let round_tripped = DetectorConfig::from_profile_json(&json).expect("re-parse config");
    assert_eq!(
        round_tripped.pose_consistency_min_decisive_ratio,
        f64::INFINITY
    );
    assert_eq!(round_tripped, cfg);
}
