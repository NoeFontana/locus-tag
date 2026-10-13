#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]
use locus_core::bench_api::generate_synthetic_test_image;
use locus_core::{Detector, ImageView, TagFamily};

#[test]
fn test_capture_invalid_quads() {
    let canvas_size = 200;
    let (mut data, _gt_corners) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 42, 100, canvas_size, 0.0);

    // Corrupt bits to fail decode but stay a valid quad
    for y in 80..120 {
        for x in 80..120 {
            data[y * canvas_size + x] = 255;
        }
    }

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("valid image");
    let mut detector = Detector::builder()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let batch = detector
        .detect(&img, None, None, true)
        .expect("detection failed");

    assert_eq!(batch.len(), 0);
    assert!(!batch.rejected_corners.is_empty());
}

#[test]
fn test_subpixel_jitter_telemetry() {
    let canvas_size = 200;
    let (data, _gt_corners) = generate_synthetic_test_image(
        TagFamily::AprilTag36h11,
        42,
        100,
        canvas_size,
        5.0, // Noisy
    );

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("valid image");
    let mut detector = Detector::builder()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let batch = detector
        .detect(&img, None, None, true)
        .expect("detection failed");

    assert!(!batch.is_empty());
    let telemetry = batch.telemetry.expect("telemetry should be present");
    assert!(!telemetry.subpixel_jitter_ptr.is_null());
}

#[test]
fn test_failed_decode_telemetry() {
    let canvas_size = 200;
    let (mut data, _gt_corners) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 42, 100, canvas_size, 0.0);

    for y in 80..120 {
        for x in 80..120 {
            data[y * canvas_size + x] = 255;
        }
    }

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("valid image");
    let mut detector = Detector::builder()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let batch = detector
        .detect(&img, None, None, true)
        .expect("detection failed");

    assert_eq!(batch.len(), 0);
    // The corrupted marker's failed decode reports its bit errors. Without blob-shape gates
    // its inner components reach the decoder too, so the count is not fixed.
    assert!(!batch.rejected_corners.is_empty());
    assert!(batch.rejected_error_rates.iter().any(|&e| e > 0.0));
}

#[test]
fn test_reprojection_error_telemetry() {
    let canvas_size = 200;
    let (data, _gt_corners) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 42, 100, canvas_size, 0.0);

    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).expect("valid image");
    let mut detector = Detector::builder()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let intrinsics = locus_core::CameraIntrinsics::new(100.0, 100.0, 100.0, 100.0);
    let tag_size = 0.16;

    let batch = detector
        .detect(&img, Some(&intrinsics), Some(tag_size), true)
        .expect("detection failed");

    assert!(!batch.is_empty());
    let telemetry = batch.telemetry.expect("telemetry should be present");
    assert!(!telemetry.reprojection_errors_ptr.is_null());

    // The telemetry is the residual of the pose the detector emitted, so on a clean synthetic
    // tag it is sub-pixel. Projecting object points whose origin is not the pose's (the tag's
    // corner rather than its centre) inflates it by about half the tag's diagonal.
    #[expect(unsafe_code, reason = "reads the arena-backed telemetry slice")]
    // SAFETY: the detector allocated `num_reprojection` contiguous `f32`s at this non-null,
    // aligned pointer in its frame arena; `batch` (and with it the arena) is live, and nothing
    // writes through it while this shared slice exists.
    let errors = unsafe {
        std::slice::from_raw_parts(
            telemetry.reprojection_errors_ptr,
            telemetry.num_reprojection,
        )
    };
    assert_eq!(errors.len(), batch.len());
    let posed = errors.iter().filter(|e| !e.is_nan()).count();
    assert!(posed > 0, "no detection carried a pose: {errors:?}");
    for &e in errors.iter().filter(|e| !e.is_nan()) {
        assert!(e < 1.0, "reprojection RMSE {e} px on a clean synthetic tag");
    }
}
