#![allow(missing_docs)]
use locus_core::config::{CornerRefinementMode, DetectorConfig};

#[test]
fn test_gwlf_config_option() {
    let config = DetectorConfig::builder()
        .refinement_mode(CornerRefinementMode::Gwlf)
        .build();

    assert_eq!(config.refinement_mode, CornerRefinementMode::Gwlf);
}
