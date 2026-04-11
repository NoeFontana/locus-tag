#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Robustness tests for tag decoding logic.

use locus_core::TagFamily;
use locus_core::bench_api::family_to_decoder;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Direct("proptest-regressions/decoder.txt"))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_decoder_survives_random_payloads(
        payload in any::<u64>(),
        family in prop_oneof![
            Just(TagFamily::AprilTag16h5),
            Just(TagFamily::AprilTag36h11),
            Just(TagFamily::ArUco4x4_50),
            Just(TagFamily::ArUco4x4_100),
        ],
        max_hamming in 0..=3u32
    ) {
        let decoder = family_to_decoder(family);

        // decoder should survive and not panic on any payload
        let _ = decoder.decode_full(payload, max_hamming);

        // Assert survival
        prop_assert!(true);
    }
}
