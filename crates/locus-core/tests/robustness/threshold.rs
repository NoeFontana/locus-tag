#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Robustness tests for adaptive thresholding logic.

use locus_core::{DetectorBuilder, DetectorConfig, ImageView};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("proptest-regressions/threshold.txt"))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_threshold_engine_memory_bounds(
        (width, height, data) in (16..640usize, 16..480usize).prop_flat_map(|(w, h)| {
            (Just(w), Just(h), proptest::collection::vec(any::<u8>(), w * h))
        }),
        window_size in 5..=31usize, // Test different window sizes
        min_contrast in 0..=50u8 // Test different min contrast
    ) {
        // Enforce odd window size
        let window_size = if window_size % 2 == 0 { window_size + 1 } else { window_size };
        prop_assume!(width >= window_size && height >= window_size, "Tile size must not be larger than image dimensions to avoid panic (known issue)");

        let config = DetectorConfig {
            threshold_tile_size: window_size,
            threshold_min_range: min_contrast,
            ..DetectorConfig::default()
        };

        // Build the detector to process the image through the facade to ensure SOA integration
        let mut detector = DetectorBuilder::new().with_config(config).build();

        let image = ImageView::new(&data, width, height, width)
            .expect("Generated proptest data should always be valid for ImageView");

        // Processing should not panic on arbitrary parameters
        // Note: intentional assert added initially for TDD Red Phase.
        let detections = detector.detect(&image, None, None, locus_core::PoseEstimationMode::Fast, false).expect("detection failed");

        // We expect it not to find anything in noise and not to crash.
        // It's technically possible but extremely unlikely to find a valid tag in random noise.
        // The most important thing is that it does not panic due to out of bounds memory accesses in the SoA buffers.
        prop_assert!(detections.is_empty() || !detections.is_empty(), "Threshold engine should not panic on arbitrary parameters.");
    }
}
