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
use bumpalo::Bump;
use locus_core::bench_api::DetectionBatch;
use locus_core::bench_api::LabelResult;
use locus_core::{DetectorConfig, ImageView};

#[test]
fn test_quad_extraction_soa_empty() {
    let mut batch = DetectionBatch::new();
    let data = vec![0u8; 100 * 100];
    let img = ImageView::new(&data, 100, 100, 100).unwrap();
    let config = DetectorConfig::default();

    let labels = vec![0u32; 100 * 100];
    let label_result = LabelResult {
        labels: &labels,
        component_stats: Vec::new(),
    };

    let arena = Bump::new();
    let (n, _) = locus_core::bench_api::extract_quads_soa(
        &arena,
        &mut batch,
        &img,
        &label_result,
        &config,
        1,
        &img,
        false,
    );

    assert_eq!(n, 0);
}
