#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
use locus_core::bench_api::*;
use locus_core::{DetectorConfig, ImageView, TagFamily};

#[test]
fn test_decoding_soa_empty() {
    let mut batch = DetectionBatch::new();
    let data = vec![0u8; 100 * 100];
    let img = ImageView::new(&data, 100, 100, 100).unwrap();
    let config = DetectorConfig::default();
    let decoders = vec![family_to_decoder(TagFamily::AprilTag36h11)];

    decode_batch_soa(&mut batch, 0, &img, &decoders, &config);
}
