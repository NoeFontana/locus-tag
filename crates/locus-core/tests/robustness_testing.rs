use locus_core::{Detector, DetectorBuilder, TagFamily, ImageView};
use locus_core::bench_api::*;

#[test]
fn test_robustness_noise() {
    let mut detector = Detector::new();
    let data = vec![128u8; 100 * 100];
    let img = ImageView::new(&data, 100, 100, 100).unwrap();
    
    let detections = detector.detect(&img);
    assert!(detections.is_empty());
}
