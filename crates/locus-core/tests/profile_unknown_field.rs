//! Tripwire for the `#[serde(deny_unknown_fields)]` attribute on the shim
//! structs: a misspelled or unrecognised JSON field must be rejected, not
//! silently dropped. This is the Rust-side guard against config drift when
//! a user copies a profile and adds a typo.

#![cfg(feature = "profiles")]
#![allow(clippy::panic, clippy::panic_in_result_fn)]

use locus_core::config::DetectorConfig;
use locus_core::error::ConfigError;

#[test]
fn unknown_root_field_rejected() {
    let json = r#"{ "name": "x", "bogus": 1 }"#;
    match DetectorConfig::from_profile_json(json) {
        Err(ConfigError::ProfileParse(msg)) => assert!(
            msg.contains("bogus"),
            "error message must mention offending field, got: {msg}"
        ),
        other => panic!("expected ProfileParse error, got: {other:?}"),
    }
}

#[test]
fn unknown_nested_field_rejected() {
    let json = r#"{ "threshold": { "tile_size": 8, "does_not_exist": true } }"#;
    match DetectorConfig::from_profile_json(json) {
        Err(ConfigError::ProfileParse(msg)) => assert!(
            msg.contains("does_not_exist"),
            "error message must mention offending nested field, got: {msg}"
        ),
        other => panic!("expected ProfileParse error, got: {other:?}"),
    }
}

#[test]
fn malformed_json_rejected() {
    let json = r#"{ "name": "x", "threshold": {"#;
    assert!(matches!(
        DetectorConfig::from_profile_json(json),
        Err(ConfigError::ProfileParse(_))
    ));
}

#[test]
fn removed_v0_9_keys_rejected() {
    // Keys removed in v0.9 are unknown fields now: a stale profile fails loudly
    // instead of silently losing its setting.
    for json in [
        r#"{ "extends": null }"#,
        r#"{ "threshold": { "min_range": 10 } }"#,
        r#"{ "threshold": { "constant": 15 } }"#,
        r#"{ "quad": { "refine_before_decode": false } }"#,
        r#"{ "decoder": { "gwlf_transversal_alpha": 0.01 } }"#,
        r#"{ "pose": { "pose_consistency_min_decisive_ratio": 5.0 } }"#,
    ] {
        assert!(
            matches!(
                DetectorConfig::from_profile_json(json),
                Err(ConfigError::ProfileParse(_))
            ),
            "{json} must be rejected"
        );
    }
}

#[test]
fn removed_enum_forms_rejected() {
    // The `Gwlf` refinement mode and the boolean imbalance-gate form are gone.
    for json in [
        r#"{ "decoder": { "refinement_mode": "Gwlf" } }"#,
        r#"{ "quad": { "edlines_imbalance_gate": true } }"#,
    ] {
        assert!(
            matches!(
                DetectorConfig::from_profile_json(json),
                Err(ConfigError::ProfileParse(_))
            ),
            "{json} must be rejected"
        );
    }
}

#[test]
fn edlines_with_erf_rejected_by_validate() {
    // Cross-group compat is enforced post-parse via `DetectorConfig::validate`.
    let json = r#"{
        "name": "x",
        "quad": {
            "min_area": 16, "max_aspect_ratio": 10.0,
            "min_fill_ratio": 0.10, "max_fill_ratio": 0.98,
            "min_edge_length": 4.0, "min_edge_score": 4.0,
            "subpixel_refinement_sigma": 0.6, "upscale_factor": 1,
            "max_elongation": 0.0, "min_density": 0.0,
            "extraction_mode": "EdLines"
        },
        "decoder": {
            "min_contrast": 20.0, "refinement_mode": "Erf",
            "max_hamming_error": 2
        }
    }"#;
    assert!(matches!(
        DetectorConfig::from_profile_json(json),
        Err(ConfigError::EdLinesIncompatibleWithErf)
    ));
}
