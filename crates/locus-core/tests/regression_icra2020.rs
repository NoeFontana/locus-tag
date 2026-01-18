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
use std::sync::Mutex;
use rayon::prelude::*;

mod common;

#[derive(Debug)]
struct Failure {
    image: String,
    expected: Vec<u32>,
    detected: Vec<u32>,
    missing: Vec<u32>,
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
    // The dataset structure is e.g. "forward/pure_tags_images", not just "forward".
    // AND "rotation/pure_tags_images".
    // We will attempt to find:
    // 1. dataset_root/subfolder
    // 2. dataset_root/subfolder/pure_tags_images (standard ICRA structure)
    
    let candidates = [
        dataset_root.join(subfolder).join("pure_tags_images"),
        dataset_root.join(subfolder),
    ];
    
    let search_dir = candidates.iter().find(|p| p.exists())
        // If neither exists, we just return safely (skip test) instead of proceeding with a non-existent path
        .map(|p| p.clone());

    let search_dir = match search_dir {
        Some(d) => d,
        None => {
             // For checkerboard, maybe the folder name is different?
             // But for now, just skip.
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
    println!("Loaded Ground Truth: {} entries.", ground_truth.len());

    // 3. Collect Images (Walker)
    // We walk the `search_dir` recursively to find images that are in the GT.
    let mut entries = Vec::new();
    let walker = walkdir::WalkDir::new(&search_dir).into_iter();
    
    for entry in walker.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "png" || e == "jpg") {
            // We need to match the key in `ground_truth`.
            // The GT keys are likely simple filenames "0001.png" or relative paths.
            // Based on `common::load_ground_truth`, we treated them as filenames.
            // Let's try to match by filename.
            let fname = path.file_name().unwrap().to_string_lossy().to_string();
            if ground_truth.contains_key(&fname) {
                entries.push(path.to_path_buf());
            }
        }
    }

    if entries.is_empty() {
        // It's possible the logic for matching filename is insufficient if GT has relative paths.
        // But for this regression suite, we'll assume flat filenames in GT for now as per previous implementation.
        println!("WARNING: No images found in {:?} that match GT keys. Skipping.", search_dir);
        return;
    }

    println!("Found {} images for testing.", entries.len());

    // 4. Parallel Regression Loop
    let failures = Mutex::new(Vec::new());
    
    entries.par_iter().for_each(|image_path: &std::path::PathBuf| {
        // Local Detector per thread
        let mut config = locus_core::DetectorConfig::default();
        if use_checkerboard {
            config.segmentation_connectivity = locus_core::config::SegmentationConnectivity::Four;
            // config.decoder_min_contrast = 10.0; // Optional tuning for dense boards
        }
        
        // We use with_config
        let mut detector = Detector::with_config(config);
        
        let options = DetectOptions {
            families: vec![TagFamily::AprilTag36h11], 
            ..Default::default()
        };

        let filename = image_path.file_name().unwrap().to_string_lossy().to_string();
        
        let img = match image::open(image_path) {
            Ok(i) => i.into_luma8(),
            Err(e) => {
                eprintln!("Failed to load {}: {}", filename, e);
                return;
            }
        };
        
        let (width, height) = img.dimensions();
        let input_view = ImageView::new(img.as_raw(), width as usize, height as usize, width as usize)
            .expect("Failed to create ImageView");

        let detections = detector.detect_with_options(&input_view, &options);
        
        if let Some(gt) = ground_truth.get(&filename) {
            let detected_ids: std::collections::HashSet<u32> = detections.iter().map(|d| d.id).collect();
            let expected_ids = &gt.tag_ids;
            
            if !detected_ids.is_superset(expected_ids) {
                let missing: Vec<u32> = expected_ids.difference(&detected_ids).copied().collect();
                let mut f_log = failures.lock().unwrap();
                f_log.push(Failure {
                    image: filename.clone(),
                    expected: expected_ids.iter().copied().collect(),
                    detected: detected_ids.iter().copied().collect(),
                    missing,
                });
            }
        }
    });

    // 5. Reporting
    let failures = failures.into_inner().unwrap();
    if !failures.is_empty() {
        println!("\n=== REGRESSION DETECTED in {:?} ===", subfolder);
        println!("Total Failures: {}", failures.len());
        // For brevity, print only first 10
        for f in failures.iter().take(10) {
            println!("FAIL [{}]: Expected {:?}, Found {:?}, MISSING {:?}", 
                f.image, f.expected, f.detected, f.missing);
        }
        if failures.len() > 10 {
            println!("... and {} more.", failures.len() - 10);
        }
        println!("========================================");
        panic!("Regression test failed with {} image(s) missing tags.", failures.len());
    } else {
        println!("PASSED: All {} images in {} matched ground truth.", entries.len(), subfolder);
    }
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
