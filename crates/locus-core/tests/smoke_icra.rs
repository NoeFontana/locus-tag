use locus_core::bench_api::*;
use locus_core::{Detector, DetectOptions, TagFamily, ImageView};

#[cfg(feature = "bench-internals")]

mod common;

#[test]
fn test_smoke_icra_dataset() {
    let mut detector = Detector::new();
    let img_data = vec![0u8; 640 * 480];
    let input_view = ImageView::new(&img_data, 640, 480, 640).unwrap();
    
    let options = DetectOptions::all_families();
    let res = detector.detect_with_stats_and_options(&input_view, &options);
    
    // Smoke test: should not crash
    assert!(res.stats.total_ms >= 0.0);
}
