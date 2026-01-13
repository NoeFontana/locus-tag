//! ICRA 2020 Benchmark Regression Tests
//!
//! These tests verify detection accuracy against the ICRA 2020 AprilTag dataset.
//! They serve as non-regression tests to ensure detector improvements don't
//! introduce regressions.

use locus_core::config::TagFamily;
use locus_core::{DetectOptions, Detector};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct GroundTruth {
    #[allow(dead_code)]
    image: String,
    #[allow(dead_code)]
    description: String,
    expected_min_recall: f64,
    expected_max_corner_error_px: f64,
    tags: Vec<TagGT>,
}

#[derive(Debug, Deserialize)]
struct TagGT {
    tag_id: u32,
    corners: [[f64; 2]; 4],
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/icra2020")
}

struct OwnedImage {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

fn load_image(name: &str) -> OwnedImage {
    let path = fixtures_dir().join(name);
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("Failed to load image {}: {}", path.display(), e))
        .into_luma8();
    let (width, height) = img.dimensions();
    OwnedImage {
        data: img.into_raw(),
        width: width as usize,
        height: height as usize,
    }
}

fn load_ground_truth(name: &str) -> GroundTruth {
    let path = fixtures_dir().join(name);
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to load GT {}: {}", path.display(), e));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse GT {}: {}", path.display(), e))
}

/// Compute RMSE corner error between detected and ground truth corners
fn corner_error(det_corners: &[[f64; 2]; 4], gt_corners: &[[f64; 2]; 4]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..4 {
        let dx = det_corners[i][0] - gt_corners[i][0];
        let dy = det_corners[i][1] - gt_corners[i][1];
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / 4.0).sqrt()
}

#[test]
fn test_icra2020_forward_0037() {
    let gt = load_ground_truth("0037.json");
    let img = load_image("0037.png");

    let options = DetectOptions {
        families: vec![TagFamily::AprilTag36h11],
        ..Default::default()
    };

    let mut detector = Detector::new();
    let img_view =
        locus_core::image::ImageView::new(&img.data, img.width, img.height, img.width).unwrap();
    let detections = detector.detect_with_options(&img_view, &options);

    // Build lookup maps
    let gt_ids: HashSet<u32> = gt.tags.iter().map(|t| t.tag_id).collect();
    let det_ids: HashSet<u32> = detections.iter().map(|d| d.id).collect();

    // Calculate metrics
    let matched: HashSet<u32> = gt_ids.intersection(&det_ids).copied().collect();
    let missed: HashSet<u32> = gt_ids.difference(&det_ids).copied().collect();
    let false_positives: HashSet<u32> = det_ids.difference(&gt_ids).copied().collect();

    let recall = matched.len() as f64 / gt_ids.len() as f64;

    println!("=== ICRA2020 Forward 0037 Results ===");
    println!("Ground Truth Tags: {}", gt_ids.len());
    println!("Detected Tags: {}", det_ids.len());
    println!("Matched: {}", matched.len());
    println!("Missed: {} ({:?})", missed.len(), missed);
    println!(
        "False Positives: {} ({:?})",
        false_positives.len(),
        false_positives
    );
    println!("Recall: {:.2}%", recall * 100.0);

    // Calculate corner errors for matched tags
    let mut corner_errors = Vec::new();
    for det in &detections {
        if let Some(gt_tag) = gt.tags.iter().find(|t| t.tag_id == det.id) {
            let det_corners: [[f64; 2]; 4] = [
                [det.corners[0][0], det.corners[0][1]],
                [det.corners[1][0], det.corners[1][1]],
                [det.corners[2][0], det.corners[2][1]],
                [det.corners[3][0], det.corners[3][1]],
            ];
            let err = corner_error(&det_corners, &gt_tag.corners);
            corner_errors.push(err);
        }
    }

    let avg_corner_error = if corner_errors.is_empty() {
        0.0
    } else {
        corner_errors.iter().sum::<f64>() / corner_errors.len() as f64
    };

    println!("Average Corner Error: {:.2}px", avg_corner_error);

    // Assertions - these are the non-regression thresholds
    assert!(
        recall >= gt.expected_min_recall,
        "Recall {:.2}% below threshold {:.2}%",
        recall * 100.0,
        gt.expected_min_recall * 100.0
    );

    assert!(
        avg_corner_error <= gt.expected_max_corner_error_px,
        "Corner error {:.2}px exceeds threshold {:.2}px",
        avg_corner_error,
        gt.expected_max_corner_error_px
    );

    // No false positives expected on clean images
    assert!(
        false_positives.is_empty(),
        "Unexpected false positives: {:?}",
        false_positives
    );
}
