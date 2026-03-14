//! Robustness tests for the public API boundaries.
use locus_core::{DetectorBuilder, DetectorConfig, ImageView};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("proptest-regressions/api.txt"))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_api_survives_invalid_buffer_strides(
        (width, height, stride, data_len) in (
            16..640usize,
            16..480usize,
            0..1000usize,
            0..1_000_000_usize
        ).prop_flat_map(|(w, h, s, dl)| {
            (Just(w), Just(h), Just(s), Just(dl))
        })
    ) {
        // Red phase verified, intentional panic removed.

        let data = vec![0u8; data_len];

        // This simulates a user passing an invalid buffer/stride combination from Python/C
        // It MUST return an Err and MUST NOT panic.
        let result = ImageView::new(&data, width, height, stride);

        if let Ok(image) = result {
            // If by some chance the random combination IS valid, processing it shouldn't panic
            let config = DetectorConfig::default();
            let mut detector = DetectorBuilder::new().with_config(config).build();
            let _ = detector.detect(&image, None, None, locus_core::PoseEstimationMode::Fast, false).expect("detection failed");
        }

        prop_assert!(true);
    }
}
