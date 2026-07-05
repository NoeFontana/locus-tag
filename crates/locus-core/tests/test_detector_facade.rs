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
use locus_core::{Detector, DetectorBuilder, ImageView, TagFamily};

#[cfg(feature = "bench-internals")]
use locus_core::bench_api::*;

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
    let detections = detector
        .detect(&img, None, None, false)
        .expect("detection failed");
    assert!(detections.is_empty());
}

#[test]
fn test_detector_multiple_families() {
    let mut detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .with_family(TagFamily::ArUco4x4_50)
        .build();

    let canvas_size = 200;
    #[cfg(feature = "bench-internals")]
    {
        // Generate AprilTag
        let (data, _) =
            generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, 50, canvas_size, 0.0);
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();
        let detections = detector
            .detect(&img, None, None, false)
            .expect("detection failed");
        assert_eq!(detections.len(), 1);
        assert_eq!(detections.ids[0], 0);

        // Generate ArUco
        let (data2, _) =
            generate_synthetic_test_image(TagFamily::ArUco4x4_50, 5, 50, canvas_size, 0.0);
        let img2 = ImageView::new(&data2, canvas_size, canvas_size, canvas_size).unwrap();
        let detections2 = detector
            .detect(&img2, None, None, false)
            .expect("detection failed");
        assert_eq!(detections2.len(), 1);
        assert_eq!(detections2.ids[0], 5);
    }
}

/// Max per-corner Euclidean difference (px) between two owned detection sets,
/// matched by index. Returns `f64::INFINITY` if the sets differ in length.
#[cfg(feature = "bench-internals")]
fn max_corner_diff(a: &[locus_core::Detection], b: &[locus_core::Detection]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut worst = 0.0f64;
    for (da, db) in a.iter().zip(b.iter()) {
        for k in 0..4 {
            let d =
                (da.corners[k][0] - db.corners[k][0]).hypot(da.corners[k][1] - db.corners[k][1]);
            worst = worst.max(d);
        }
    }
    worst
}

#[cfg(feature = "bench-internals")]
#[test]
fn test_detection_deterministic_across_runs_and_threads() {
    use std::sync::Arc;

    let canvas_size = 200;
    let (data, _) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, 60, canvas_size, 0.0);

    let detect_once = |bytes: &[u8]| -> Vec<locus_core::Detection> {
        let mut detector = DetectorBuilder::new()
            .with_family(TagFamily::AprilTag36h11)
            .build();
        let img = ImageView::new(bytes, canvas_size, canvas_size, canvas_size).unwrap();
        detector
            .detect(&img, None, None, false)
            .expect("detection failed")
            .reassemble_owned()
    };

    let reference = detect_once(&data);
    assert_eq!(
        reference.len(),
        1,
        "fixture must produce exactly one detection to make the determinism check meaningful"
    );

    // A real regression in ordering/reduction shows up as a >>1e-9 px shift.
    // We use a tolerance rather than bit-exact equality because rayon reduction
    // order can legitimately jitter results at the ~1e-13 px level.
    const DETERMINISM_TOL_PX: f64 = 1e-9;

    // 1. Run-to-run: repeated fresh detectors on the same bytes must agree.
    for _ in 0..8 {
        let got = detect_once(&data);
        let diff = max_corner_diff(&got, &reference);
        assert!(
            diff < DETERMINISM_TOL_PX,
            "run-to-run detection is non-deterministic: max corner diff {diff} px"
        );
    }

    // 2. Cross-thread: N threads detecting the same image concurrently, each on
    //    its own detector, must all agree with the single-threaded reference.
    let shared = Arc::new(data.clone());
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let bytes = Arc::clone(&shared);
            std::thread::spawn(move || {
                let mut detector = DetectorBuilder::new()
                    .with_family(TagFamily::AprilTag36h11)
                    .build();
                let img = ImageView::new(&bytes, canvas_size, canvas_size, canvas_size).unwrap();
                detector
                    .detect(&img, None, None, false)
                    .expect("detection failed")
                    .reassemble_owned()
            })
        })
        .collect();
    for handle in handles {
        let got = handle.join().expect("detection thread panicked");
        let diff = max_corner_diff(&got, &reference);
        assert!(
            diff < DETERMINISM_TOL_PX,
            "cross-thread detection is non-deterministic: max corner diff {diff} px"
        );
    }
}

#[cfg(feature = "bench-internals")]
#[test]
fn test_detection_at_image_edge_is_robust() {
    // A tag pushed hard against the image border stresses the SIMD sampler at
    // the buffer boundary (out-of-bounds gather risk) and the edge/corner
    // refinement near the frame edge. Whether or not the tag survives the
    // clipped quiet zone, detection must never panic and must never emit
    // non-finite or wildly out-of-frame corners.
    let family = TagFamily::AprilTag36h11;
    let canvas = 120;

    for &tag_px in &[80usize, 96, 110] {
        // Near-edge placement: only a few pixels of margin, so the outer quad
        // sits within ~5 px of the image border on two sides.
        let (data, _) = generate_synthetic_test_image(family, 0, tag_px, canvas, 0.0);
        let img = ImageView::new(&data, canvas, canvas, canvas).unwrap();

        let mut detector = DetectorBuilder::new().with_family(family).build();
        let detections = detector
            .detect(&img, None, None, false)
            .expect("detection must not error at the image edge")
            .reassemble_owned();

        for det in &detections {
            for k in 0..4 {
                let (x, y) = (det.corners[k][0], det.corners[k][1]);
                assert!(
                    x.is_finite() && y.is_finite(),
                    "tag {tag_px}px: non-finite corner {k} = ({x}, {y}) at image edge"
                );
                // Corners may land marginally outside the frame due to sub-pixel
                // refinement, but never absurdly so (that signals an overflow or
                // a sign error in the boundary handling).
                assert!(
                    x > -2.0 && x < canvas as f64 + 2.0 && y > -2.0 && y < canvas as f64 + 2.0,
                    "tag {tag_px}px: corner {k} = ({x}, {y}) is implausibly far outside \
                     the {canvas}x{canvas} frame"
                );
            }
        }
    }
}

#[test]
fn test_detector_decimation() {
    let mut detector = DetectorBuilder::new().with_decimation(2).build();

    let canvas_size = 200;
    #[cfg(feature = "bench-internals")]
    {
        let (data, _) = generate_synthetic_test_image(
            TagFamily::AprilTag36h11,
            0,
            80, // Large enough to survive decimation
            canvas_size,
            0.0,
        );
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();
        let detections = detector
            .detect(&img, None, None, false)
            .expect("detection failed");
        assert_eq!(detections.len(), 1);
        assert_eq!(detections.ids[0], 0);
    }
}
