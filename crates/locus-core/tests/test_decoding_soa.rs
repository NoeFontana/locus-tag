//! Tests for the SoA Decoding Pass.

use locus_core::decoder::decode_batch_soa;
use locus_core::{
    CandidateState, DetectionBatch, DetectorConfig, ImageView, TagFamily, family_to_decoder,
};

#[test]
fn test_decode_batch_soa_interface() {
    let mut batch = DetectionBatch::new();
    let pixels = vec![0u8; 100 * 100];
    let img = ImageView::new(&pixels, 100, 100, 100).expect("Failed to create ImageView");
    let config = DetectorConfig::default();

    // Setup one active candidate
    batch.status_mask[0] = CandidateState::Active;
    // Degenerate homography will fail decoding
    batch.homographies[0].data = [0.0; 9];

    let n = 1;

    let decoder = family_to_decoder(TagFamily::AprilTag36h11);
    let decoders = vec![decoder];

    // This should fail to compile because decode_batch_soa is not defined yet.
    decode_batch_soa(&mut batch, n, &img, &decoders, &config);

    // Should be FailedDecode if it failed to decode
    assert_eq!(batch.status_mask[0], CandidateState::FailedDecode);
}
