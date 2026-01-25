//! Unified Regression Test Harness
//!
//! Evaluates the detector against:
//! 1. "Fixtures" (Committed representative images) - Runs in CI, guarantees baseline functionality.
//! 2. "ICRA 2020" (External dataset) - Runs if present, supports sampling for speed.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::unnecessary_debug_formatting,
    clippy::similar_names,
    clippy::trivially_copy_pass_by_ref,
    clippy::needless_pass_by_value,
    clippy::items_after_statements
)]

use locus_core::image::ImageView;
use locus_core::{DetectOptions, Detector, DetectorConfig, PipelineStats, config::TagFamily};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::path::PathBuf;

mod common;

// ============================================================================
// Metrics & Reporting
// ============================================================================

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

impl From<PipelineStats> for PipelineMetrics {
    fn from(stats: PipelineStats) -> Self {
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

fn serialize_rmse<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // Round to 4 decimal places for stability
    let rounded = (value * 10000.0).round() / 10000.0;
    serializer.serialize_f64(rounded)
}

#[derive(Serialize)]
struct ImageMetrics {
    recall: f64,
    #[serde(serialize_with = "serialize_rmse")]
    avg_rmse: f64,
    stats: PipelineMetrics,
    missed_ids: BTreeSet<u32>,
    extra_ids: BTreeSet<u32>,
}

#[derive(Serialize)]
struct RegressionReport {
    summary: SummaryMetrics,
    entries: BTreeMap<String, ImageMetrics>,
}

#[derive(Serialize)]
struct Offender {
    filename: String,
    missed: usize,
    extra: usize,
    #[serde(serialize_with = "serialize_rmse")]
    rmse: f64,
}

#[derive(Serialize)]
struct SummaryMetrics {
    dataset_size: usize,
    mean_recall: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_rmse: f64,
    mean_total_ms: f64,
    worst_offenders: Vec<Offender>,
}

// ============================================================================
// Evaluation Engine
// ============================================================================

/// Ground Truth for a single image
struct GroundTruth {
    tags: HashMap<u32, [[f64; 2]; 4]>,
}

type DatasetItem = (String, Vec<u8>, usize, usize, GroundTruth);

/// Runs the detection pipeline on a stream of images and produces a snapshot-able report.
fn evaluate_dataset(
    snapshot_name: &str,
    items: impl Iterator<Item = DatasetItem>,
    detector_config: DetectorConfig,
    options: DetectOptions,
    icra_corner_ordering: bool,
) {
    let mut detector = Detector::with_config(detector_config);
    let mut results = BTreeMap::new();

    // Aggregators
    let mut total_recall = 0.0;
    let mut total_rmse = 0.0;
    let mut total_time = 0.0;
    let mut count = 0;

    for (filename, data, width, height, gt) in items {
        let img = ImageView::new(&data, width, height, width).expect("valid image");

        let (detections, stats) = detector.detect_with_stats_and_options(&img, &options);

        // --- Metrics Calculation ---
        let mut image_rmse_sum = 0.0;
        let mut match_count = 0;
        let mut found_ids = BTreeSet::new();

        for det in &detections {
            if let Some(gt_corners) = gt.tags.get(&det.id) {
                // Approximate center check to resolve ambiguities in rare multi-tag cases
                let gt_cx: f64 = gt_corners.iter().map(|p| p[0]).sum::<f64>() / 4.0;
                let gt_cy: f64 = gt_corners.iter().map(|p| p[1]).sum::<f64>() / 4.0;
                let dist_sq = (det.center[0] - gt_cx).powi(2) + (det.center[1] - gt_cy).powi(2);

                if dist_sq < 50.0 * 50.0 {
                    let det_corners = if icra_corner_ordering {
                        [
                            det.corners[1],
                            det.corners[0],
                            det.corners[3],
                            det.corners[2],
                        ]
                    } else {
                        det.corners
                    };

                    image_rmse_sum +=
                        locus_core::test_utils::compute_rmse(&det_corners, gt_corners);
                    match_count += 1;
                    found_ids.insert(det.id);
                }
            }
        }

        let recall = if gt.tags.is_empty() {
            1.0
        } else {
            found_ids.len() as f64 / gt.tags.len() as f64
        };
        let avg_rmse = if match_count > 0 {
            image_rmse_sum / f64::from(match_count)
        } else {
            0.0
        };

        total_recall += recall;
        total_rmse += avg_rmse;
        total_time += stats.total_ms;
        count += 1;

        // Compute Missed and Extra sets
        let mut missed_ids = BTreeSet::new();
        for &id in gt.tags.keys() {
            if !found_ids.contains(&id) {
                missed_ids.insert(id);
            }
        }

        let mut extra_ids = BTreeSet::new();
        for det in &detections {
            if !found_ids.contains(&det.id) {
                extra_ids.insert(det.id);
            }
        }

        results.insert(
            filename.clone(),
            ImageMetrics {
                recall,
                avg_rmse,
                stats: PipelineMetrics::from(stats),
                missed_ids: missed_ids.clone(),
                extra_ids: extra_ids.clone(),
            },
        );
    }

    if count == 0 {
        println!("WARNING: Dataset {snapshot_name} yielded no images.");
        return;
    }

    // Identify Worst Offenders
    // Criteria: Missed any tags, or Extra tags, or RMSE > 1.0
    let mut offenders: Vec<Offender> = results
        .iter()
        .filter_map(|(fname, m)| {
            if !m.missed_ids.is_empty() || !m.extra_ids.is_empty() || m.avg_rmse > 1.0 {
                Some(Offender {
                    filename: fname.clone(),
                    missed: m.missed_ids.len(),
                    extra: m.extra_ids.len(),
                    rmse: m.avg_rmse,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by "badness": Missed (desc), then RMSE (desc)
    offenders.sort_by(|a, b| {
        b.missed
            .cmp(&a.missed)
            .then_with(|| b.extra.cmp(&a.extra))
            .then_with(|| {
                b.rmse
                    .partial_cmp(&a.rmse)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Take top 5
    let worst_offenders: Vec<Offender> = offenders.into_iter().take(5).collect();

    let report = RegressionReport {
        summary: SummaryMetrics {
            dataset_size: count,
            mean_recall: total_recall / count as f64,
            mean_rmse: total_rmse / count as f64,
            mean_total_ms: total_time / count as f64,
            worst_offenders,
        },
        entries: results,
    };

    println!("=== {snapshot_name} Results ===");
    println!("  Images: {count}");
    println!("  Recall: {:.2}%", report.summary.mean_recall * 100.0);
    println!("  RMSE:   {:.4} px", report.summary.mean_rmse);

    // Snapshot logic: Redact latency for stability, but keep Summary unredacted for quick checks
    insta::assert_yaml_snapshot!(snapshot_name, report, {
        ".entries.*.stats.threshold_ms" => "[latency]",
        ".entries.*.stats.segmentation_ms" => "[latency]",
        ".entries.*.stats.quad_extraction_ms" => "[latency]",
        ".entries.*.stats.decoding_ms" => "[latency]",
        ".entries.*.stats.total_ms" => "[latency]"
    });
}

// ============================================================================
// Data Providers
// ============================================================================

trait DatasetProvider {
    fn name(&self) -> &str;
    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_>;
}

/// Provides images from `tests/fixtures/icra2020`.
/// This is the "Gold Standard" subset that MUST pass in CI.
struct FixtureProvider {
    fixtures_dir: PathBuf,
}

impl FixtureProvider {
    fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/icra2020");
        Self { fixtures_dir: root }
    }
}

impl DatasetProvider for FixtureProvider {
    fn name(&self) -> &'static str {
        "fixtures"
    }

    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_> {
        // Find pairs of .png and .json
        let walker = walkdir::WalkDir::new(&self.fixtures_dir).sort_by_file_name();

        let iter = walker
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter_map(move |entry| {
                let path = entry.path();
                if path.extension()? != "png" {
                    return None;
                }

                let json_path = path.with_extension("json");
                if !json_path.exists() {
                    return None;
                }

                let img = image::open(path).ok()?.into_luma8();
                let (w, h) = img.dimensions();

                // Load local JSON GT format
                #[derive(Deserialize)]
                struct FixtureJson {
                    tags: Vec<FixtureTag>,
                }
                #[derive(Deserialize)]
                struct FixtureTag {
                    tag_id: u32,
                    corners: [[f64; 2]; 4],
                }

                let json_str = std::fs::read_to_string(&json_path).ok()?;
                let fixture_data: FixtureJson = serde_json::from_str(&json_str).ok()?;

                let mut tags = HashMap::new();
                for t in fixture_data.tags {
                    tags.insert(t.tag_id, t.corners);
                }

                Some((
                    path.file_name()?.to_string_lossy().to_string(),
                    img.into_raw(),
                    w as usize,
                    h as usize,
                    GroundTruth { tags },
                ))
            });

        Box::new(iter)
    }
}

/// Provides images from the external large ICRA dataset.
/// Supports sampling (e.g., every Nth image) for faster local runs.
struct IcraProvider {
    name: String,
    image_paths: Vec<PathBuf>,
    gt: HashMap<String, common::ImageGroundTruth>,
    sample_rate: usize,
}

impl IcraProvider {
    fn new(subfolder: &str, sample_rate: usize) -> Option<Self> {
        let root = common::resolve_dataset_root()?;

        // 1. Load Ground Truth (Context-aware loading)
        let gt_map = common::load_ground_truth(&root, subfolder)?;

        // 2. Locate Image Directory
        // Handle both flat and nested "pure_tags_images" structures gracefully
        let candidates = [
            root.join(subfolder).join("pure_tags_images"),
            root.join(subfolder),
        ];
        let img_dir = candidates.iter().find(|p| p.is_dir())?;

        let mut paths: Vec<_> = walkdir::WalkDir::new(img_dir)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .map(|e| e.path().to_path_buf())
            .filter(|p| p.extension().is_some_and(|e| e == "png" || e == "jpg"))
            .filter(|p| gt_map.contains_key(&p.file_name().unwrap().to_string_lossy().to_string()))
            .collect();

        paths.sort();

        Some(Self {
            name: format!("icra_{subfolder}"),
            image_paths: paths,
            gt: gt_map,
            sample_rate,
        })
    }
}

impl DatasetProvider for IcraProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_> {
        let iter = self
            .image_paths
            .iter()
            .step_by(self.sample_rate)
            .map(move |path| {
                let fname = path.file_name().unwrap().to_string_lossy().to_string();
                // We use expect here because if the file existed during scan, it should load.
                // Failure to load specific images in regression suite is a failure.
                let img = image::open(path)
                    .expect("load regression image")
                    .into_luma8();
                let (w, h) = img.dimensions();

                let icra_gt = self.gt.get(&fname).unwrap();
                let gt = GroundTruth {
                    tags: icra_gt.corners.clone(),
                };

                (fname, img.into_raw(), w as usize, h as usize, gt)
            });
        Box::new(iter)
    }
}

// ============================================================================
// Test Runners
// ============================================================================

/// Always runs on committed fixtures.
#[test]
fn regression_fixtures() {
    let provider = FixtureProvider::new();
    evaluate_dataset(
        "fixtures",
        provider.iter(),
        DetectorConfig::default(),
        DetectOptions {
            families: vec![TagFamily::AprilTag36h11],
            ..Default::default()
        },
        true, // Fixtures currently match ICRA ordering format (from 0037.json)
    );
}

#[test]
fn regression_icra_forward() {
    let sample_rate = if env::var("FULL").is_ok() { 1 } else { 10 };

    if let Some(provider) = IcraProvider::new("forward", sample_rate) {
        let snapshot = format!("{}_sample{}", provider.name(), sample_rate);
        evaluate_dataset(
            &snapshot,
            provider.iter(),
            DetectorConfig::builder()
                .refinement_mode(locus_core::config::CornerRefinementMode::Erf)
                .build(),
            DetectOptions {
                families: vec![TagFamily::AprilTag36h11],
                ..Default::default()
            },
            true, // ICRA uses TR, TL, BL, BR ordering
        );
    } else {
        println!("SKIPPING: ICRA2020 'forward' dataset not found.");
    }
}

#[test]
fn regression_icra_rotation() {
    let sample_rate = if env::var("FULL").is_ok() { 1 } else { 10 };

    if let Some(provider) = IcraProvider::new("rotation", sample_rate) {
        let snapshot = format!("{}_sample{}", provider.name(), sample_rate);
        evaluate_dataset(
            &snapshot,
            provider.iter(),
            DetectorConfig::default(),
            DetectOptions {
                families: vec![TagFamily::AprilTag36h11],
                ..Default::default()
            },
            true,
        );
    }
}
