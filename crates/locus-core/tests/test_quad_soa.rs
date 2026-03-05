use locus_core::bench_api::*;
use locus_core::bench_api::DetectionBatch;
use locus_core::segmentation::LabelResult;
use locus_core::{PoseEstimationMode, DetectorConfig, ImageView};

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

    let n = locus_core::bench_api::extract_quads_soa(
        &mut batch,
        &img,
        &label_result,
        &config,
        1,
        &img
    );

    assert_eq!(n, 0);
}
