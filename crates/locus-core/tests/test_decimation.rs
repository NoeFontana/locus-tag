use locus_core::bench_api::*;
use locus_core::{Detector, DetectorBuilder, TagFamily, ImageView};

#[cfg(feature = "bench-internals")]

#[test]
fn test_decimation_accuracy() {
    let canvas_size = 640;
    let tag_id = 0;
    let tag_size_px = 100;
    let family = TagFamily::AprilTag36h11;

    #[cfg(feature = "bench-internals")]
    {
        let (data, gt_corners) = generate_synthetic_test_image(
            family,
            tag_id,
            tag_size_px,
            canvas_size,
            0.0,
        );
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

        // 1. Detect with decimation 1
        let mut detector = DetectorBuilder::new()
            .with_family(family)
            .with_decimation(1)
            .build();
        
        let res = detector.detect_with_stats(&img);
        assert!(!res.detections.is_empty());
        let err1 = compute_corner_error(&res.detections[0].corners, &gt_corners);

        // 2. Detect with decimation 2
        let mut detector2 = DetectorBuilder::new()
            .with_family(family)
            .with_decimation(2)
            .build();
        
        let res2 = detector2.detect_with_stats(&img);
        assert!(!res2.detections.is_empty());
        let err2 = compute_corner_error(&res2.detections[0].corners, &gt_corners);

        println!("Error D1: {}, Error D2: {}", err1, err2);
        // Decimation should maintain reasonable sub-pixel accuracy
        assert!(err2 < 1.5);
    }
}
