//! Robustness tests for memory bounds and allocation limits.
use locus_core::{DetectorBuilder, DetectorConfig, ImageView};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("proptest-regressions/memory.txt"))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_detector_survives_extreme_dimensions(
        width in 1..100_000usize,
        height in 1..100_000usize
    ) {
        // Red phase verified, intentional panic removed.

        // We don't allocate a full image of this size (it would OOM the test runner).
        // We just ensure the ImageView constructor or detector initialization doesn't panic
        // inappropriately. We give it a dummy slice.
        let data = vec![0u8; 16];

        // If ImageView creation fails due to bounds/size mismatch, that's fine (it returns Result).
        // The point is it shouldn't panic.
        if let Ok(image) = ImageView::new(&data, width, height, width) {
            let config = DetectorConfig {
                threshold_tile_size: 5,
                ..DetectorConfig::default()
            };

            let mut detector = DetectorBuilder::new().with_config(config).build();            // It might fail gracefully or return empty, but shouldn't panic.
            let _ = detector.detect(&image, None, None, locus_core::PoseEstimationMode::Fast, false);
        }

        prop_assert!(true);
    }
}
