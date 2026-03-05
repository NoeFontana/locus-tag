use locus_core::bench_api::*;
use locus_core::{PoseEstimationMode, Detector, DetectOptions, TagFamily, ImageView};

#[cfg(feature = "bench-internals")]

mod common;

#[test]
fn test_smoke_icra_dataset() {
    let mut detector = Detector::new();
    let img_data = vec![0u8; 640 * 480];
    let input_view = ImageView::new(&img_data, 640, 480, 640).unwrap();
    
    let _res = detector.detect(&input_view, None, None, PoseEstimationMode::Fast);
    
    // Smoke test: should not crash
}
