//! Integration test for ArUco 4x4_50 tag detection.
//!
//! This test validates the complete detection pipeline using synthetic ArUco tags,
//! matching the conditions used in Python benchmarks (`tests/test_config.py`).

use locus_core::Detector;
use locus_core::config::TagFamily;

#[test]
fn test_aruco_4x4_50_detection() {
    const TAG_ID: u16 = 0;
    const TAG_SIZE: usize = 96;
    const CANVAS_SIZE: usize = 400;
    const FAMILY: TagFamily = TagFamily::ArUco4x4_50;

    let (data, _) =
        locus_core::test_utils::generate_test_image(FAMILY, TAG_ID, TAG_SIZE, CANVAS_SIZE, 0.0);
    let img =
        locus_core::image::ImageView::new(&data, CANVAS_SIZE, CANVAS_SIZE, CANVAS_SIZE).unwrap();

    let mut detector = Detector::new();
    detector.set_families(&[FAMILY]);
    let results = detector.detect(&img);

    assert_eq!(results.len(), 1, "Should detect exactly 1 ArUco tag");
    assert_eq!(results[0].id, u32::from(TAG_ID), "Detected ID should match");
}

#[test]
fn test_aruco_multiple_ids() {
    const CANVAS_SIZE: usize = 400;
    const TAG_SIZE: usize = 96;
    const FAMILY: TagFamily = TagFamily::ArUco4x4_50;

    let mut detector = Detector::new();
    detector.set_families(&[FAMILY]);

    for tag_id in [0u16, 1, 5, 10, 25, 49] {
        let (data, _) =
            locus_core::test_utils::generate_test_image(FAMILY, tag_id, TAG_SIZE, CANVAS_SIZE, 0.0);
        let img = locus_core::image::ImageView::new(&data, CANVAS_SIZE, CANVAS_SIZE, CANVAS_SIZE)
            .unwrap();
        let results = detector.detect(&img);

        assert_eq!(results.len(), 1, "Should detect tag ID {tag_id}");
        assert_eq!(
            results[0].id,
            u32::from(tag_id),
            "ID mismatch for tag {tag_id}"
        );
        assert_eq!(
            results[0].hamming, 0,
            "Hamming distance should be 0 for noiseless tag"
        );
    }
}
