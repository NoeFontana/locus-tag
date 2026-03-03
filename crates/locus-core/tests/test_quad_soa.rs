//! Tests for the SoA Quad Extraction.

use locus_core::{DetectionBatch, ImageView, DetectorConfig};
use locus_core::quad::extract_quads_soa;
use locus_core::segmentation::LabelResult;

#[test]
fn test_extract_quads_soa_interface() {
    let pixels = vec![0u8; 100 * 100];
    let img = ImageView::new(&pixels, 100, 100, 100).unwrap();
    let labels = vec![0u32; 100 * 100];
    let label_result = LabelResult {
        labels: &labels,
        component_stats: Vec::new(),
    };
    let config = DetectorConfig::default();
    let mut batch = DetectionBatch::new();

    // This should fail to compile because extract_quads_soa is not defined yet.
    let n = extract_quads_soa(&mut batch, &img, &label_result, &config, 1, &img);
    
    assert_eq!(n, 0);
}
