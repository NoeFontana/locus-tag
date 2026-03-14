//! Robustness tests for concurrency and thread safety.

use locus_core::{DetectorBuilder, ImageView};
use proptest::prelude::*;
use std::sync::Arc;
use std::thread;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("proptest-regressions/concurrency.txt"))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_detector_pipeline_concurrent_reads(
        (width, height, data) in (16..640usize, 16..480usize).prop_flat_map(|(w, h)| {
            (Just(w), Just(h), proptest::collection::vec(any::<u8>(), w * h))
        }),
        num_threads in 2..=8usize
    ) {
        // Red phase verified, removing intentional panic.

        let image_data = Arc::new(data);
        let mut handles = vec![];

        for _ in 0..num_threads {
            let data_clone = Arc::clone(&image_data);

            handles.push(thread::spawn(move || {
                let mut detector = DetectorBuilder::new().build();
                let image = ImageView::new(&data_clone, width, height, width)
                    .expect("Generated proptest data should always be valid for ImageView");
                // We don't care about the result, just that it doesn't panic or deadlock
                // when multiple threads process the exact same underlying byte slice simultaneously.
                detector.detect(&image, None, None, locus_core::PoseEstimationMode::Fast, false).unwrap();
            }));
        }

        for handle in handles {
            handle.join().expect("Threads should exit cleanly");
        }

        // Assert survival
        prop_assert!(true);
    }
}
