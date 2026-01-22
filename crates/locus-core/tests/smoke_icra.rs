mod common;

use locus_core::image::ImageView;
use locus_core::{DetectOptions, Detector, config::TagFamily};
use std::path::PathBuf;

fn run_single_image_test(subfolder: &str, filename: &str, min_recall: f64, max_rmse: f64) {
    let dataset_root =
        common::resolve_dataset_root().expect("ICRA2020 dataset not found. Set LOCUS_DATASET_DIR.");

    let img_path = dataset_root
        .join(subfolder)
        .join("pure_tags_images")
        .join(filename);
    if !img_path.exists() {
        panic!("Test image not found: {:?}", img_path);
    }

    let ground_truth_map = common::load_ground_truth(&dataset_root, subfolder)
        .expect("No tags.csv found");

    let gt = ground_truth_map
        .get(filename)
        .expect("Image not in ground truth");

    // Open image
    let img = image::open(&img_path)
        .expect("Failed to load image")
        .into_luma8();
    let (width, height) = img.dimensions();
    let input_view = ImageView::new(
        img.as_raw(),
        width as usize,
        height as usize,
        width as usize,
    )
    .expect("Failed to create ImageView");

    let mut detector = Detector::new();
    let options = DetectOptions {
        families: vec![TagFamily::AprilTag36h11],
        ..Default::default()
    };

    let detections = detector.detect_with_options(&input_view, &options);

    // Calculate metrics
    let mut total_rmse = 0.0;
    let mut match_count = 0;
    let mut matched_gt_ids = std::collections::HashSet::new();

    for det in &detections {
        if let Some(gt_corners) = gt.corners.get(&det.id) {
            // Calculate center of GT tag
            let mut gt_cx = 0.0;
            let mut gt_cy = 0.0;
            for p in gt_corners {
                gt_cx += p[0];
                gt_cy += p[1];
            }
            gt_cx /= 4.0;
            gt_cy /= 4.0;

            // Only match if center is within a reasonable distance (e.g., 20 pixels)
            let dist_sq = (det.center[0] - gt_cx).powi(2) + (det.center[1] - gt_cy).powi(2);
            if dist_sq < 20.0 * 20.0 {
                // Map library standard (CW: TL, TR, BR, BL) to ICRA 2020 convention (TR, TL, BL, BR)
                let reordered_det = [
                    det.corners[1], // ICRA Idx 0 <- Lib 1 (TR)
                    det.corners[0], // ICRA Idx 1 <- Lib 0 (TL)
                    det.corners[3], // ICRA Idx 2 <- Lib 3 (BL)
                    det.corners[2], // ICRA Idx 3 <- Lib 2 (BR)
                ];
                let rmse = locus_core::test_utils::compute_rmse(&reordered_det, gt_corners);
                println!("  Tag {}: RMSE={:.4} px", det.id, rmse);
                total_rmse += rmse;
                match_count += 1;
                matched_gt_ids.insert(det.id);
            } else {
                println!(
                    "  Tag {}: Ignored distant detection (dist={:.1}px)",
                    det.id,
                    dist_sq.sqrt()
                );
            }
        }
    }
    let recall = matched_gt_ids.len() as f64 / gt.tag_ids.len() as f64;
    let avg_rmse = if match_count > 0 {
        total_rmse / match_count as f64
    } else {
        0.0
    };

    println!(
        "Image {}: Recall={:.2}%, RMSE={:.4} px",
        filename,
        recall * 100.0,
        avg_rmse
    );

    assert!(
        recall >= min_recall,
        "Recall {:.2}% below target {:.2}% for {}",
        recall * 100.0,
        min_recall * 100.0,
        filename
    );
    if recall > 0.0 {
        assert!(
            avg_rmse <= max_rmse,
            "RMSE {:.4} px above target {:.4} px for {}",
            avg_rmse,
            max_rmse,
            filename
        );
    }
}

#[test]
fn test_anchor_0001_far() {
    // Current baseline recall ~1.3%
    run_single_image_test("forward", "0001.png", 0.01, 0.3);
}

#[test]
fn test_anchor_0012_mid() {
    // Current baseline recall ~40%
    run_single_image_test("forward", "0012.png", 0.40, 0.4);
}

#[test]
fn test_anchor_0022_approach() {
    // Current baseline recall ~98%, RMSE 0.38
    run_single_image_test("forward", "0022.png", 0.95, 0.4);
}

#[test]
fn test_anchor_0030_close() {
    // Current baseline recall 100%, RMSE 0.27
    run_single_image_test("forward", "0030.png", 1.0, 0.3);
}

#[test]
fn test_anchor_0040_very_close() {
    // Current baseline recall 100%, RMSE 0.22
    run_single_image_test("forward", "0040.png", 1.0, 0.3);
}
