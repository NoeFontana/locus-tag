//! Render-Tag Hub Regression Suite
//!
//! Evaluates the detector against datasets synchronized from the Hugging Face Hub.
//! These datasets are generated using the `render-tag` pipeline and provide
//! high-fidelity synthetic benchmarks with ground truth.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::unnecessary_debug_formatting,
    clippy::similar_names,
    clippy::trivially_copy_pass_by_ref,
    clippy::needless_pass_by_value,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use
)]

use locus_core::image::ImageView;
use locus_core::{DetectOptions, Detector, DetectorConfig, PipelineStats, config::TagFamily};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

// We need to re-implement or reference the evaluation engine components.
// For simplicity and independence from icra2020.rs, we re-implement the necessary parts.

// ============================================================================
// Configuration Presets
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub enum ConfigPreset {
    /// Optimized for isolated tags on plain backgrounds.
    PlainBoard,
}

impl ConfigPreset {
    pub fn detector_config(self) -> DetectorConfig {
        match self {
            Self::PlainBoard => DetectorConfig::builder()
                .refinement_mode(locus_core::config::CornerRefinementMode::Erf)
                .build(),
        }
    }
}

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
#[derive(Clone)]
pub struct GroundTruth {
    pub tags: HashMap<u32, [[f64; 2]; 4]>,
}

type DatasetItem = (String, Vec<u8>, usize, usize, GroundTruth);

pub trait DatasetProvider {
    fn name(&self) -> &str;
    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_>;
}

/// Unified harness for running regression tests.
pub struct RegressionHarness {
    snapshot_name: String,
    config: DetectorConfig,
    options: DetectOptions,
}

impl RegressionHarness {
    pub fn new(snapshot_name: impl Into<String>) -> Self {
        Self {
            snapshot_name: snapshot_name.into(),
            config: DetectorConfig::default(),
            options: DetectOptions::default(),
        }
    }

    pub fn with_preset(mut self, preset: ConfigPreset) -> Self {
        self.config = preset.detector_config();
        self
    }

    pub fn with_families(mut self, families: Vec<TagFamily>) -> Self {
        self.options.families = families;
        self
    }

    pub fn run(self, provider: impl DatasetProvider) {
        let mut detector = Detector::with_config(self.config);
        let mut results = BTreeMap::new();

        // Aggregators
        let mut total_recall = 0.0;
        let mut total_rmse = 0.0;
        let mut total_time = 0.0;
        let mut count = 0;

        for (filename, data, width, height, gt) in provider.iter() {
            let img = ImageView::new(&data, width, height, width).expect("valid image");
            let (detections, stats) = detector.detect_with_stats_and_options(&img, &self.options);

            // --- Metrics Calculation ---
            let mut image_rmse_sum = 0.0;
            let mut match_count = 0;
            let mut found_ids = BTreeSet::new();

            for det in &detections {
                if let Some(gt_corners) = gt.tags.get(&det.id) {
                    let gt_cx: f64 = gt_corners.iter().map(|p| p[0]).sum::<f64>() / 4.0;
                    let gt_cy: f64 = gt_corners.iter().map(|p| p[1]).sum::<f64>() / 4.0;
                    let dist_sq = (det.center[0] - gt_cx).powi(2) + (det.center[1] - gt_cy).powi(2);

                    if dist_sq < 50.0 * 50.0 {
                        image_rmse_sum +=
                            locus_core::test_utils::compute_rmse(&det.corners, gt_corners);
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
                    missed_ids,
                    extra_ids,
                },
            );
        }

        if count == 0 {
            println!("WARNING: Dataset {} yielded no images.", self.snapshot_name);
            return;
        }

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

        let report = RegressionReport {
            summary: SummaryMetrics {
                dataset_size: count,
                mean_recall: total_recall / count as f64,
                mean_rmse: total_rmse / count as f64,
                mean_total_ms: total_time / count as f64,
                worst_offenders: offenders.into_iter().take(5).collect(),
            },
        };

        println!("=== {} Results ===", self.snapshot_name);
        println!("  Images: {count}");
        println!("  Recall: {:.2}%", report.summary.mean_recall * 100.0);
        println!("  RMSE:   {:.4} px", report.summary.mean_rmse);
        println!("  Latency: {:.4} ms", report.summary.mean_total_ms);

        insta::assert_yaml_snapshot!(self.snapshot_name, report, {
            ".summary.mean_total_ms" => "[DURATION]"
        });
    }
}

// ============================================================================
// Hub Data Provider
// ============================================================================

struct HubProvider {
    name: String,
    base_dir: PathBuf,
    gt_map: HashMap<String, GroundTruth>,
    image_names: Vec<String>,
}

#[derive(Deserialize)]
struct HubEntry {
    image_filename: String,
    tag_id: u32,
    corners: [[f64; 2]; 4],
}

impl HubProvider {
    fn new(dataset_dir: &std::path::Path) -> Option<Self> {
        let jsonl_path = dataset_dir.join("annotations.jsonl");
        let images_dir = dataset_dir.join("images");

        if !jsonl_path.exists() || !images_dir.exists() {
            return None;
        }

        let file = std::fs::File::open(&jsonl_path).ok()?;
        let reader = std::io::BufReader::new(file);

        let mut gt_map: HashMap<String, GroundTruth> = HashMap::new();

        use std::io::BufRead;
        for line in reader.lines().map_while(Result::ok) {
            let entry: HubEntry = serde_json::from_str(&line).ok()?;

            gt_map
                .entry(entry.image_filename)
                .or_insert_with(|| GroundTruth {
                    tags: HashMap::new(),
                })
                .tags
                .insert(entry.tag_id, entry.corners);
        }

        let mut image_names: Vec<_> = gt_map.keys().cloned().collect();
        image_names.sort();

        Some(Self {
            name: dataset_dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            base_dir: dataset_dir.to_path_buf(),
            gt_map,
            image_names,
        })
    }
}

impl DatasetProvider for HubProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_> {
        let base_dir = self.base_dir.clone();
        let iter = self.image_names.iter().map(move |fname| {
            let img_path = base_dir.join("images").join(fname);
            let img = image::open(&img_path).expect("load hub image").into_luma8();
            let (w, h) = img.dimensions();

            let gt = self.gt_map.get(fname).unwrap().clone();

            (fname.clone(), img.into_raw(), w as usize, h as usize, gt)
        });
        Box::new(iter)
    }
}

// ============================================================================
// Test Runners
// ============================================================================

fn run_hub_test(config_name: &str, family: TagFamily) {
    if let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") {
        let root = PathBuf::from(hub_dir);
        let dataset_path = root.join(config_name);

        if !dataset_path.exists() {
            println!("Dataset not found in cache: {config_name}. Skipping.");
            return;
        }

        if let Some(provider) = HubProvider::new(&dataset_path) {
            let snapshot = format!("hub_{}", provider.name());
            RegressionHarness::new(snapshot)
                .with_preset(ConfigPreset::PlainBoard)
                .with_families(vec![family])
                .run(provider);
        }
    } else {
        println!("Skipping hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
    }
}

#[test]
fn regression_hub_tag16h5() {
    run_hub_test("single_tag_locus_v1_tag16h5", TagFamily::Aruco16h5);
}

#[test]
fn regression_hub_tag36h11() {
    run_hub_test("single_tag_locus_v1_tag36h11", TagFamily::Aruco36h11);
}
