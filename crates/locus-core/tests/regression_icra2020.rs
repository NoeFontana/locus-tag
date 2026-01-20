//! ICRA 2020 Regression Test Harness
//!
//! This integration test verifies the detector's accuracy against the standard ICRA 2020 dataset.
//! It is designed to be a strict "Golden Master" regression test.
//!
//! # Configuration & Parametrization
//!
//! The test behavior is controlled by the environment:
//!
//! - **`LOCUS_DATASET_DIR`**: (Optional) Path to the root of the ICRA 2020 dataset containing `tags.csv`.
//!   - If unset, checks `tests/data/icra2020`.
//!   - If dataset is missing, the test **SKIPS** (passes) to ensure CI compatibility.
//!
//! # Usage
//!
//! To run all tests:
//! ```bash
//! cargo test --test regression_icra2020 --release
//! ```
//!
//! To run a specific subset (e.g. only rotation):
//! ```bash
//! cargo test --test regression_icra2020 test_regression_icra2020_rotation --release
//! ```

#![allow(missing_docs)]

use locus_core::{DetectOptions, Detector, config::TagFamily};
use locus_core::image::ImageView;
// use rayon::prelude::*; // Removed as we use sequential processing for reliable timing
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

mod common;

#[derive(Serialize, Default, Clone)]
struct PipelineMetrics {
    threshold_ms: f64,
    segmentation_ms: f64,
    quad_extraction_ms: f64,
    decoding_ms: f64,
    total_ms: f64,
    num_candidates: usize,
    num_detections: usize,
}

impl From<locus_core::PipelineStats> for PipelineMetrics {
    fn from(stats: locus_core::PipelineStats) -> Self {
        Self {
            threshold_ms: stats.threshold_ms,
            segmentation_ms: stats.segmentation_ms,
            quad_extraction_ms: stats.quad_extraction_ms,
            decoding_ms: stats.decoding_ms,
            total_ms: stats.total_ms,
            num_candidates: stats.num_candidates,
            num_detections: stats.num_detections,
        }
    }
}

#[derive(Serialize)]
struct ImageMetrics {
    recall: f64,
    avg_rmse: f64,
    stats: PipelineMetrics,
    detected_ids: BTreeSet<u32>,
}

#[derive(Serialize)]
struct RegressionReport {
    // Sorted list of entries ensures deterministic snapshots
    entries: BTreeMap<String, ImageMetrics>,
    // Aggregate Metrics
    mean_stats: PipelineMetrics,
    mean_recall: f64,
    mean_rmse: f64,
}

fn calculate_rmse(det_corners: [[f64; 2]; 4], gt_corners: [[f64; 2]; 4]) -> f64 {
    locus_core::test_utils::compute_rmse(&det_corners, &gt_corners)
}

fn run_dataset_regression(subfolder: &str, use_checkerboard: bool) {
    // 1. Resolve Dataset
    let dataset_root = match common::resolve_dataset_root() {
        Some(path) => path,
        None => {
            println!("SKIPPING: ICRA2020 dataset not found. Set LOCUS_DATASET_DIR to run regression tests.");
            return;
        }
    };

    // We try to find the actual image directory. 
    let candidates = [
        dataset_root.join(subfolder).join("pure_tags_images"),
        dataset_root.join(subfolder),
    ];
    
    let search_dir = candidates.iter().find(|p| p.exists())
        .map(|p| p.clone());

    let search_dir = match search_dir {
        Some(d) => d,
        None => {
             println!("SKIPPING: Sub-dataset {:?} not found in {:?}.", subfolder, dataset_root);
             return;
        }
    };

    println!("Running regression tests on sub-dataset: {:?}", search_dir);

    // 2. Load Ground Truth
    let local_csv_dir = dataset_root.join(subfolder);
    let ground_truth = if let Some(gt) = common::load_ground_truth(&local_csv_dir) {
        println!("Loading local Ground Truth from {:?}", local_csv_dir);
        gt
    } else if let Some(gt) = common::load_ground_truth(&dataset_root) {
        println!("Loading Master Ground Truth from {:?}", dataset_root);
        gt
    } else {
        println!("SKIPPING: No tags.csv found in {:?} or {:?}.", local_csv_dir, dataset_root);
        return;
    };
    println!("Loaded Ground Truth for {} images.", ground_truth.len());

    // 3. Collect Images (Walker)
    let mut image_paths = Vec::new();
    let walker = walkdir::WalkDir::new(&search_dir).into_iter();
    
    for entry in walker.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "png" || e == "jpg") {
            // Only test images that are in our Ground Truth
            let fname = path.file_name().unwrap().to_string_lossy().to_string();
            if ground_truth.contains_key(&fname) {
                image_paths.push(path.to_path_buf());
            }
        }
    }

    if image_paths.is_empty() {
        println!("WARNING: No images found in {:?} that match Ground Truth. Skipping.", search_dir);
        return;
    }

    println!("Found {} images for testing.", image_paths.len());

    // 4. Sequential Detection (for reliable timing)
    let results: Vec<(String, ImageMetrics)> = image_paths.iter().map(|image_path: &std::path::PathBuf| {
        let filename = image_path.file_name().unwrap().to_string_lossy().to_string();
        let gt = ground_truth.get(&filename).unwrap(); // Invariant: filtered in step 3
        
        let mut config = locus_core::DetectorConfig::default();
        if use_checkerboard {
            config.segmentation_connectivity = locus_core::config::SegmentationConnectivity::Four;
        }
        
        let mut detector = Detector::with_config(config);
        let options = DetectOptions {
            families: vec![TagFamily::AprilTag36h11], 
            ..Default::default()
        };
        
        // Open image
        let img = match image::open(image_path) {
            Ok(i) => i.into_luma8(),
            Err(e) => {
                eprintln!("Failed to load {}: {}", filename, e);
                return (filename, ImageMetrics {
                    recall: 0.0,
                    avg_rmse: 0.0,
                    stats: PipelineMetrics::default(),
                    detected_ids: BTreeSet::new(),
                });
            }
        };
        
        let (width, height) = img.dimensions();
        let input_view = ImageView::new(img.as_raw(), width as usize, height as usize, width as usize)
            .expect("Failed to create ImageView");

        // DETECTION & TIMING
        let (detections, stats) = detector.detect_with_stats_and_options(&input_view, &options);
        
        // METRICS CALCULATION
        let mut total_rmse = 0.0;
        let mut match_count = 0;
        let mut matched_gt_ids = BTreeSet::new();
        
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

                // Only match if center is within 20 pixels
                let dist_sq = (det.center[0] - gt_cx).powi(2) + (det.center[1] - gt_cy).powi(2);
                if dist_sq < 20.0 * 20.0 {
                    // Map library standard (CW: TL, TR, BR, BL) to ICRA 2020 convention (TR, TL, BL, BR)
                    let reordered_det = [
                        det.corners[1], // ICRA Idx 0 <- Lib 1 (TR)
                        det.corners[0], // ICRA Idx 1 <- Lib 0 (TL)
                        det.corners[3], // ICRA Idx 2 <- Lib 3 (BL)
                        det.corners[2], // ICRA Idx 3 <- Lib 2 (BR)
                    ];
                    total_rmse += calculate_rmse(reordered_det, *gt_corners);
                    match_count += 1;
                    matched_gt_ids.insert(det.id);
                }
            }
        }
        
        let recall = if gt.tag_ids.is_empty() { 1.0 } else { matched_gt_ids.len() as f64 / gt.tag_ids.len() as f64 };
        let avg_rmse = if match_count > 0 { total_rmse / match_count as f64 } else { 0.0 };
        
        (filename, ImageMetrics {
            recall,
            avg_rmse,
            stats: PipelineMetrics::from(stats),
            detected_ids: detections.iter().map(|d| d.id).collect(),
        })
    }).collect();

    // 4. Aggregate Results (Reduce)
    let mut entries = BTreeMap::new();
    let mut sum_recall = 0.0;
    let mut sum_rmse = 0.0;
    let mut sum_stats = PipelineMetrics::default();
    let count = results.len() as f64;

    for (filename, metrics) in results {
        sum_recall += metrics.recall;
        sum_rmse += metrics.avg_rmse;
        
        sum_stats.threshold_ms += metrics.stats.threshold_ms;
        sum_stats.segmentation_ms += metrics.stats.segmentation_ms;
        sum_stats.quad_extraction_ms += metrics.stats.quad_extraction_ms;
        sum_stats.decoding_ms += metrics.stats.decoding_ms;
        sum_stats.total_ms += metrics.stats.total_ms;
        sum_stats.num_candidates += metrics.stats.num_candidates;
        sum_stats.num_detections += metrics.stats.num_detections;
        
        entries.insert(filename, metrics);
    }

    let report = RegressionReport { 
        entries,
        mean_stats: PipelineMetrics {
            threshold_ms: sum_stats.threshold_ms / count,
            segmentation_ms: sum_stats.segmentation_ms / count,
            quad_extraction_ms: sum_stats.quad_extraction_ms / count,
            decoding_ms: sum_stats.decoding_ms / count,
            total_ms: sum_stats.total_ms / count,
            num_candidates: (sum_stats.num_candidates as f64 / count) as usize,
            num_detections: (sum_stats.num_detections as f64 / count) as usize,
        },
        mean_recall: if count > 0.0 { sum_recall / count } else { 0.0 },
        mean_rmse: if count > 0.0 { sum_rmse / count } else { 0.0 },
    };

    // 5. Snapshot Assertion
    // Use the subfolder name as part of the snapshot name to prevent collisions
    let snapshot_name = format!("icra2020_{}_{}", subfolder, if use_checkerboard { "checkerboard" } else { "standard" });

    println!("Report for {}:", snapshot_name);
    println!("  Recall:          {:.2}%", report.mean_recall * 100.0);
    println!("  RMSE:            {:.4} px", report.mean_rmse);
    println!("  Avg Total:       {:.2} ms", report.mean_stats.total_ms);
    println!("  Avg Threshold:   {:.2} ms", report.mean_stats.threshold_ms);
    println!("  Avg Segmentation: {:.2} ms", report.mean_stats.segmentation_ms);
    println!("  Avg Quad Extr:   {:.2} ms", report.mean_stats.quad_extraction_ms);
    println!("  Avg Decoding:    {:.2} ms", report.mean_stats.decoding_ms);
    println!("  Avg Candidates:  {}", report.mean_stats.num_candidates);

    // We redact per-image latency to avoid 400+ lines of jitter in diffs,
    // but we KEEP mean_stats unredacted so performance shifts are visible in PRs.
    insta::assert_yaml_snapshot!(snapshot_name, report, {
        ".entries.*.stats.threshold_ms" => "[latency]",
        ".entries.*.stats.segmentation_ms" => "[latency]",
        ".entries.*.stats.quad_extraction_ms" => "[latency]",
        ".entries.*.stats.decoding_ms" => "[latency]",
        ".entries.*.stats.total_ms" => "[latency]"
    });
}


macro_rules! define_regression_test {
    ($name:ident, $subfolder:expr, $checkerboard:expr) => {
        #[test]
        fn $name() {
            run_dataset_regression($subfolder, $checkerboard);
        }
    };
}

// Standard Datasets (8-way connectivity)
define_regression_test!(test_regression_icra2020_forward, "forward", false);
define_regression_test!(test_regression_icra2020_rotation, "rotation", false);
define_regression_test!(test_regression_icra2020_random, "random", false);
define_regression_test!(test_regression_icra2020_circle, "circle", false);

// Checkerboard Datasets (4-way connectivity) - Run on same folders, but with 4-way mode
define_regression_test!(test_regression_icra2020_checkerboard_forward, "forward", true);
define_regression_test!(test_regression_icra2020_checkerboard_rotation, "rotation", true);
define_regression_test!(test_regression_icra2020_checkerboard_random, "random", true);
define_regression_test!(test_regression_icra2020_checkerboard_circle, "circle", true);

// ============================================================================
// FIXTURE-BASED SMOKE TEST (runs without full dataset)
// ============================================================================

/// Quick smoke test using local fixture (0037.png from ICRA2020 forward sequence).
/// This always runs regardless of whether the full ICRA2020 dataset is installed.
#[test]
fn test_fixture_forward_0037() {
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;
    use serde::Deserialize;
    
    #[derive(Debug, Deserialize)]
    struct FixtureGT {
        #[allow(dead_code)]
        image: String,
        expected_min_recall: f64,
        expected_max_corner_error_px: f64,
        tags: Vec<FixtureTag>,
    }
    
    #[derive(Debug, Deserialize)]
    struct FixtureTag {
        tag_id: u32,
        corners: [[f64; 2]; 4],
    }
    
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/icra2020");
    let gt_path = fixtures_dir.join("0037.json");
    let img_path = fixtures_dir.join("0037.png");
    
    // Load ground truth
    let gt_content = fs::read_to_string(&gt_path)
        .unwrap_or_else(|e| panic!("Failed to load fixture GT: {}", e));
    let gt: FixtureGT = serde_json::from_str(&gt_content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture GT: {}", e));
    
    // Load image
    let img = image::open(&img_path)
        .unwrap_or_else(|e| panic!("Failed to load fixture image: {}", e))
        .into_luma8();
    let (width, height) = img.dimensions();
    let img_view = ImageView::new(img.as_raw(), width as usize, height as usize, width as usize)
        .expect("Failed to create ImageView");
    
    // Detect
    let mut detector = Detector::new();
    let options = DetectOptions {
        families: vec![TagFamily::AprilTag36h11],
        ..Default::default()
    };
    let detections = detector.detect_with_options(&img_view, &options);
    
    // Calculate metrics
    let gt_ids: HashSet<u32> = gt.tags.iter().map(|t| t.tag_id).collect();
    let det_ids: HashSet<u32> = detections.iter().map(|d| d.id).collect();
    let matched: HashSet<u32> = gt_ids.intersection(&det_ids).copied().collect();
    let recall = matched.len() as f64 / gt_ids.len() as f64;
    
    // Calculate corner errors for matched tags
    let mut total_rmse = 0.0;
    let mut match_count = 0;
    for det in &detections {
        if let Some(gt_tag) = gt.tags.iter().find(|t| t.tag_id == det.id) {
            // Map to ICRA convention for comparison
            let reordered_det = [
                det.corners[1], // ICRA Idx 0 <- Lib 1 (TR)
                det.corners[0], // ICRA Idx 1 <- Lib 0 (TL)
                det.corners[3], // ICRA Idx 2 <- Lib 3 (BL)
                det.corners[2], // ICRA Idx 3 <- Lib 2 (BR)
            ];
            total_rmse += calculate_rmse(reordered_det, gt_tag.corners);
            match_count += 1;
        }
    }
    let avg_rmse = if match_count > 0 { total_rmse / match_count as f64 } else { 0.0 };
    
    // Assertions
    assert!(
        recall >= gt.expected_min_recall,
        "Recall {:.2}% below threshold {:.2}%",
        recall * 100.0,
        gt.expected_min_recall * 100.0
    );
    assert!(
        avg_rmse <= gt.expected_max_corner_error_px,
        "Corner error {:.2}px exceeds threshold {:.2}px",
        avg_rmse,
        gt.expected_max_corner_error_px
    );
}
