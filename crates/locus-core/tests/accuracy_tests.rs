use bumpalo::Bump;
use locus_core::Detector;
use locus_core::image::ImageView;
use locus_core::test_utils::{compute_corner_error, generate_synthetic_tag};

#[test]
fn test_accuracy_synthetic() {
    let width = 640;
    let height = 480;

    // Test with multiple positions and sizes
    let test_cases = [(100, 100, 120), (300, 50, 80), (200, 250, 150)];

    for (tag_id, (x, y, size)) in test_cases.iter().enumerate() {
        let tag_id = tag_id as u32;
        let (data, gt_corners) = generate_synthetic_tag(width, height, tag_id, *x, *y, *size);
        let img = ImageView::new(&data, width, height, width).unwrap();

        // We use detect() which uses the optimized pipeline
        let detections = Detector::new().detect(&img);

        let mut final_corners = None;

        if detections.is_empty() {
            // Debug: check labels and candidates manually
            let local_arena = Bump::new();
            let binarized = local_arena.alloc_slice_fill_copy(width * height, 0u8);
            locus_core::threshold::ThresholdEngine::new().apply_threshold(
                &img,
                &locus_core::threshold::ThresholdEngine::new().compute_tile_stats(&img),
                binarized,
            );
            let res = locus_core::segmentation::label_components_with_stats(
                &local_arena,
                binarized,
                width,
                height,
            );
            let candidates = locus_core::quad::extract_quads_fast(&local_arena, &img, &res);
            println!(
                "Case {}: Found {} quad candidates (no decoded tags)",
                tag_id,
                candidates.len()
            );
            if !candidates.is_empty() {
                final_corners = Some(candidates[0].corners);
            }
        } else {
            final_corners = Some(detections[0].corners);
        }

        assert!(
            final_corners.is_some(),
            "Failed to find any quad at case {}",
            tag_id
        );
        let corners = final_corners.unwrap();

        let err = compute_corner_error(corners, gt_corners);
        println!("Case {}: Corner Error = {} px", tag_id, err);

        // Target sub-pixel accuracy: < 0.5px for these synthetic cases
        assert!(
            err < 0.5,
            "Corner error too high: {} px at case {}",
            err,
            tag_id
        );
    }
}
