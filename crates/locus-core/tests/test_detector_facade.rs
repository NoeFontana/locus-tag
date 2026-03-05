use locus_core::{Detector, DetectorBuilder, TagFamily, ImageView};

#[cfg(feature = "bench-internals")]
use locus_core::bench_api as internals;

#[test]
fn test_detector_builder_basic() {
    let detector = DetectorBuilder::new()
        .with_decimation(2)
        .with_family(TagFamily::AprilTag36h11)
        .with_threads(4)
        .build();

    assert_eq!(detector.config().decimation, 2);
    assert_eq!(detector.config().nthreads, 4);
}

#[test]
fn test_detector_new_default() {
    let mut detector = Detector::new();
    assert_eq!(detector.config().decimation, 1);
    
    // Test detection on empty image
    let data = vec![0u8; 100 * 100];
    let img = ImageView::new(&data, 100, 100, 100).unwrap();
    let detections = detector.detect(&img);
    assert!(detections.is_empty());
}

#[test]
fn test_detector_multiple_families() {
    let mut detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .with_family(TagFamily::ArUco4x4_50)
        .build();

    let canvas_size = 200;
    // Generate AprilTag
    #[cfg(feature = "bench-internals")]
    let (data, _) = internals::generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        0,
        50,
        canvas_size,
        0.0,
    );
    #[cfg(not(feature = "bench-internals"))]
    let (data, _) = (vec![0u8; canvas_size * canvas_size], [[0.0; 2]; 4]);

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();
    let detections = detector.detect(&img);
    assert_eq!(detections.len(), if cfg!(feature = "bench-internals") { 1 } else { 0 });

    // Generate ArUco
    #[cfg(feature = "bench-internals")]
    let (data2, _) = internals::generate_synthetic_test_image(
        TagFamily::ArUco4x4_50,
        5,
        50,
        canvas_size,
        0.0,
    );
    #[cfg(not(feature = "bench-internals"))]
    let (data2, _) = (vec![0u8; canvas_size * canvas_size], [[0.0; 2]; 4]);

    let img2 = ImageView::new(&data2, canvas_size, canvas_size, canvas_size).unwrap();
    let detections2 = detector.detect(&img2);
    assert_eq!(detections2.len(), if cfg!(feature = "bench-internals") { 1 } else { 0 });
}

#[test]
fn test_detector_decimation() {
    let mut detector = DetectorBuilder::new()
        .with_decimation(2)
        .build();

    let canvas_size = 200;
    #[cfg(feature = "bench-internals")]
    let (data, _) = internals::generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        0,
        80, // Large enough to survive decimation
        canvas_size,
        0.0,
    );
    #[cfg(not(feature = "bench-internals"))]
    let (data, _) = (vec![0u8; canvas_size * canvas_size], [[0.0; 2]; 4]);

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();
    let detections = detector.detect(&img);
    assert_eq!(detections.len(), if cfg!(feature = "bench-internals") { 1 } else { 0 });
}
