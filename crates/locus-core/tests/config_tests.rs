use locus_core::config::{CornerRefinementMode, DetectorConfig};

#[test]
fn test_gwlf_config_option() {
    // This test will fail to compile initially because Gwlf is not in the enum
    let config = DetectorConfig::builder()
        .refinement_mode(CornerRefinementMode::Gwlf)
        .build();
    
    assert_eq!(config.refinement_mode, CornerRefinementMode::Gwlf);
}
