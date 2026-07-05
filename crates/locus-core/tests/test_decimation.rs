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
use locus_core::bench_api::*;
use locus_core::{DetectorBuilder, ImageView, TagFamily};

#[test]
fn test_decimation_accuracy() {
    let canvas_size = 640;
    let tag_id = 0;
    let tag_size_px = 100;
    let family = TagFamily::AprilTag36h11;

    let (data, gt_corners) =
        generate_synthetic_test_image(family, tag_id, tag_size_px, canvas_size, 0.0);
    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

    // 1. Detect with decimation 1
    let mut detector = DetectorBuilder::new()
        .with_family(family)
        .with_decimation(1)
        .build();

    let detections = detector
        .detect(&img, None, None, false)
        .expect("detection failed");
    assert!(!detections.is_empty());
    let corners1 = [
        [
            f64::from(detections.corners[0][0].x),
            f64::from(detections.corners[0][0].y),
        ],
        [
            f64::from(detections.corners[0][1].x),
            f64::from(detections.corners[0][1].y),
        ],
        [
            f64::from(detections.corners[0][2].x),
            f64::from(detections.corners[0][2].y),
        ],
        [
            f64::from(detections.corners[0][3].x),
            f64::from(detections.corners[0][3].y),
        ],
    ];
    let err1 = compute_corner_error(&corners1, &gt_corners);

    // 2. Detect with decimation 2
    let mut detector2 = DetectorBuilder::new()
        .with_family(family)
        .with_decimation(2)
        .build();

    let detections2 = detector2
        .detect(&img, None, None, false)
        .expect("detection failed");
    assert!(!detections2.is_empty());
    let corners2 = [
        [
            f64::from(detections2.corners[0][0].x),
            f64::from(detections2.corners[0][0].y),
        ],
        [
            f64::from(detections2.corners[0][1].x),
            f64::from(detections2.corners[0][1].y),
        ],
        [
            f64::from(detections2.corners[0][2].x),
            f64::from(detections2.corners[0][2].y),
        ],
        [
            f64::from(detections2.corners[0][3].x),
            f64::from(detections2.corners[0][3].y),
        ],
    ];
    let err2 = compute_corner_error(&corners2, &gt_corners);

    println!("Error D1: {err1}, Error D2: {err2}");

    // Full-resolution detection of a clean, noise-free synthetic tag must be
    // tightly sub-pixel. Measured ~4e-4 px; gate at 0.01 px (~25x margin) —
    // roughly 150x tighter than the previous `err2 < 1.5` smoke bound, which
    // left `err1` computed-but-never-asserted.
    const ACCURACY_GATE_PX: f64 = 0.01;
    assert!(
        err1 < ACCURACY_GATE_PX,
        "decimation-1 corner error {err1} exceeds {ACCURACY_GATE_PX} px"
    );
    assert!(
        err2 < ACCURACY_GATE_PX,
        "decimation-2 corner error {err2} exceeds {ACCURACY_GATE_PX} px"
    );

    // Decimation invariance: the center-aware decimation mapping (the +0.5
    // pixel rule) must map decimated coordinates back to the full-resolution
    // grid without a systematic shift, so the refined corners at decimation 2
    // must coincide with decimation 1. This is the observable form of the +0.5
    // rule — a broken mapping shows up here as a ~0.5 px offset.
    for k in 0..4 {
        let dx = (corners1[k][0] - corners2[k][0]).abs();
        let dy = (corners1[k][1] - corners2[k][1]).abs();
        assert!(
            dx < ACCURACY_GATE_PX && dy < ACCURACY_GATE_PX,
            "corner {k}: decimation-1 {:?} and decimation-2 {:?} diverge by ({dx}, {dy}) px \
             — center-aware decimation mapping (+0.5 rule) may be broken",
            corners1[k],
            corners2[k],
        );
    }
}
