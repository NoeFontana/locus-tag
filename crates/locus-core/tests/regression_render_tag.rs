#![allow(
    missing_docs,
    dead_code,
    clippy::unwrap_used,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::type_complexity,
    clippy::unnecessary_debug_formatting,
    clippy::trivially_copy_pass_by_ref,
    clippy::needless_pass_by_value,
    clippy::missing_panics_doc
)]

use locus_core::image::ImageView;
use locus_core::{DetectOptions, Detector, DetectorConfig, config::TagFamily};
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
    total_ms: f64,
    num_detections: usize,
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

            let start = std::time::Instant::now();
            let detections = detector.detect(
                &img,
                self.options.intrinsics.as_ref(),
                self.options.tag_size,
                self.options.pose_estimation_mode,
            );
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;

            // --- Metrics Calculation ---
            let mut image_rmse_sum = 0.0;
            let mut match_count = 0;
            let mut found_ids = BTreeSet::new();

            for i in 0..detections.len() {
                let det_id = detections.ids[i];
                let det_corners_f32 = detections.corners[i];
                let det_corners_f64 = [
                    [
                        f64::from(det_corners_f32[0].x),
                        f64::from(det_corners_f32[0].y),
                    ],
                    [
                        f64::from(det_corners_f32[1].x),
                        f64::from(det_corners_f32[1].y),
                    ],
                    [
                        f64::from(det_corners_f32[2].x),
                        f64::from(det_corners_f32[2].y),
                    ],
                    [
                        f64::from(det_corners_f32[3].x),
                        f64::from(det_corners_f32[3].y),
                    ],
                ];
                let det_center = [
                    (det_corners_f64[0][0]
                        + det_corners_f64[1][0]
                        + det_corners_f64[2][0]
                        + det_corners_f64[3][0])
                        / 4.0,
                    (det_corners_f64[0][1]
                        + det_corners_f64[1][1]
                        + det_corners_f64[2][1]
                        + det_corners_f64[3][1])
                        / 4.0,
                ];

                if let Some(gt_corners) = gt.tags.get(&det_id) {
                    let g_cx: f64 = gt_corners.iter().map(|p| p[0]).sum::<f64>() / 4.0;
                    let g_cy: f64 = gt_corners.iter().map(|p| p[1]).sum::<f64>() / 4.0;
                    let dist_sq = (det_center[0] - g_cx).powi(2) + (det_center[1] - g_cy).powi(2);

                    if dist_sq < 50.0 * 50.0 {
                        image_rmse_sum +=
                            locus_core::test_utils::compute_rmse(&det_corners_f64, gt_corners);
                        match_count += 1;
                        found_ids.insert(det_id);
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
            total_time += total_ms;
            count += 1;

            let mut missed_ids = BTreeSet::new();
            for &id in gt.tags.keys() {
                if !found_ids.contains(&id) {
                    missed_ids.insert(id);
                }
            }

            let mut extra_ids = BTreeSet::new();
            for i in 0..detections.len() {
                let det_id = detections.ids[i];
                if !found_ids.contains(&det_id) {
                    extra_ids.insert(det_id);
                }
            }

            results.insert(
                filename.clone(),
                ImageMetrics {
                    recall,
                    avg_rmse,
                    stats: PipelineMetrics {
                        total_ms,
                        num_detections: detections.len(),
                    },
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
fn regression_hub_tag36h11() {
    run_hub_test("single_tag_locus_v1_tag36h11", TagFamily::AprilTag36h11);
}
