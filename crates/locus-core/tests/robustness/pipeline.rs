//! Robustness tests for the end-to-end perception pipeline.
use locus_core::{DetectorBuilder, ImageView, PoseEstimationMode};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("proptest-regressions/pipeline.txt"))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_pipeline_handles_pure_noise(
        (width, height, data) in (16..640usize, 16..480usize).prop_flat_map(|(w, h)| {
            (Just(w), Just(h), proptest::collection::vec(any::<u8>(), w * h))
        })
    ) {
        let mut detector = DetectorBuilder::new().build();

        // Feed pure noise into the detector.
        // It shouldn't panic, but should return 0 detections or gracefully handle it.
        // To make the test fail initially, we can add an intentional assert that will fail on some noise.
        let image = ImageView::new(&data, width, height, width).expect("ImageView creation should succeed with generated data");
        let detections = detector.detect(&image, None, None, PoseEstimationMode::Fast, false).expect("detection failed");

        // The pipeline should survive and return zero valid detections from pure noise.
        prop_assert!(detections.is_empty(), "Expected no valid tags in pure noise.");
    }
}
