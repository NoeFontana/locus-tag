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
use locus_core::{DetectOptions, Detector, DetectorConfig, ImageView, TagFamily};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

mod common;

// ============================================================================
// Configuration Presets
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub enum ConfigPreset {
    /// Optimized for isolated tags on plain backgrounds.
    PlainBoard,
    /// Optimized for touching tags in checkerboard patterns.
    Checkerboard,
}

impl ConfigPreset {
    pub fn detector_config(self) -> DetectorConfig {
        match self {
            Self::PlainBoard => DetectorConfig::builder()
                .refinement_mode(locus_core::config::CornerRefinementMode::Erf)
                .build(),
            Self::Checkerboard => DetectorConfig::builder()
                .segmentation_connectivity(locus_core::config::SegmentationConnectivity::Four)
                .decoder_min_contrast(10.0)
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
pub struct GroundTruth {
    pub tags: HashMap<u32, [[f64; 2]; 4]>,
}

type DatasetItem = (String, Vec<u8>, usize, usize, GroundTruth);

/// Unified harness for running regression tests.
pub struct RegressionHarness {
    snapshot_name: String,
    config: DetectorConfig,
    options: DetectOptions,
    icra_corner_ordering: bool,
}

impl RegressionHarness {
    pub fn new(snapshot_name: impl Into<String>) -> Self {
        Self {
            snapshot_name: snapshot_name.into(),
            config: DetectorConfig::default(),
            options: DetectOptions::default(),
            icra_corner_ordering: true,
        }
    }

    pub fn with_preset(mut self, preset: ConfigPreset) -> Self {
        self.config = preset.detector_config();
        self
    }

    pub fn with_decode_mode(mut self, mode: locus_core::config::DecodeMode) -> Self {
        self.config.decode_mode = mode;
        self
    }

    pub fn with_families(mut self, families: Vec<TagFamily>) -> Self {
        self.options.families = families;
        self
    }

    pub fn run(self, provider: impl DatasetProvider) {
        if cfg!(debug_assertions) {
            panic!("regression_icra2020 test should always be ran in release mode. Please use `cargo test --release`.");
        }

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
                    let gt_cx: f64 = gt_corners.iter().map(|p| p[0]).sum::<f64>() / 4.0;
                    let gt_cy: f64 = gt_corners.iter().map(|p| p[1]).sum::<f64>() / 4.0;
                    let dist_sq = (det_center[0] - gt_cx).powi(2) + (det_center[1] - gt_cy).powi(2);

                    if dist_sq < 50.0 * 50.0 {
                        let det_corners = if self.icra_corner_ordering {
                            [
                                det_corners_f64[1],
                                det_corners_f64[0],
                                det_corners_f64[3],
                                det_corners_f64[2],
                            ]
                        } else {
                            det_corners_f64
                        };

                        image_rmse_sum +=
                            locus_core::test_utils::compute_rmse(&det_corners, gt_corners);
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
                    println!(
                        "FILE {fname} missed: {:?}, extra: {:?}",
                        m.missed_ids, m.extra_ids
                    );
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
// Data Providers
// ============================================================================

pub trait DatasetProvider {
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
}

impl IcraProvider {
    fn new(subfolder: &str, img_subfolder: Option<&str>) -> Option<Self> {
        let root = common::resolve_dataset_root()?;

        // 1. Load Ground Truth (Context-aware loading)
        let gt_map = common::load_ground_truth(&root, subfolder)?;

        // 2. Locate Image Directory
        // Handle both flat and nested "pure_tags_images" structures gracefully
        let mut candidates = Vec::new();
        if let Some(sub) = img_subfolder {
            candidates.push(root.join(subfolder).join(sub));
        }
        candidates.push(root.join(subfolder).join("pure_tags_images"));
        candidates.push(root.join(subfolder));

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
            name: format!("icra_{}_{}", subfolder, img_subfolder.unwrap_or("default")),
            image_paths: paths,
            gt: gt_map,
        })
    }
}

impl DatasetProvider for IcraProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_> {
        let iter = self.image_paths.iter().map(move |path| {
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

macro_rules! test_icra {
    ($name:ident, $subfolder:expr, $img_subfolder:expr, $preset:ident, $family:expr) => {
        #[test]
        fn $name() {
            let _guard = common::telemetry::init(stringify!($name));
            if let Some(provider) = IcraProvider::new($subfolder, $img_subfolder) {
                let snapshot = provider.name().to_string();
                RegressionHarness::new(snapshot)
                    .with_preset(ConfigPreset::$preset)
                    .with_families(vec![$family])
                    .run(provider);
            }
        }
    };
    (IGNORED $name:ident, $subfolder:expr, $img_subfolder:expr, $preset:ident, $family:expr) => {
        #[test]
        fn $name() {
            let _guard = common::telemetry::init(stringify!($name));
            if std::env::var("LOCUS_EXTENDED_REGRESSION").is_err() {
                println!(
                    "Skipping heavy test {}. Set LOCUS_EXTENDED_REGRESSION=1 to run.",
                    stringify!($name)
                );
                return;
            }
            if let Some(provider) = IcraProvider::new($subfolder, $img_subfolder) {
                let snapshot = provider.name().to_string();
                RegressionHarness::new(snapshot)
                    .with_preset(ConfigPreset::$preset)
                    .with_families(vec![$family])
                    .run(provider);
            }
        }
    };
    (SOFT $name:ident, $subfolder:expr, $img_subfolder:expr, $preset:ident, $family:expr) => {
        #[test]
        fn $name() {
            let _guard = common::telemetry::init(stringify!($name));
            if let Some(provider) = IcraProvider::new($subfolder, $img_subfolder) {
                let snapshot = format!("{}_soft", provider.name());
                RegressionHarness::new(snapshot)
                    .with_preset(ConfigPreset::$preset)
                    .with_families(vec![$family])
                    .with_decode_mode(locus_core::config::DecodeMode::Soft)
                    .run(provider);
            }
        }
    };
}

#[test]
fn regression_fixtures() {
    let _guard = common::telemetry::init("regression_fixtures");
    let provider = FixtureProvider::new();
    RegressionHarness::new("fixtures")
        .with_preset(ConfigPreset::PlainBoard)
        .with_families(vec![TagFamily::AprilTag36h11])
        .run(provider);
}

test_icra!(
    regression_icra_forward,
    "forward",
    Some("pure_tags_images"),
    PlainBoard,
    TagFamily::AprilTag36h11
);
test_icra!(
    SOFT regression_icra_forward_soft,
    "forward",
    Some("pure_tags_images"),
    PlainBoard,
    TagFamily::AprilTag36h11
);
test_icra!(
    regression_icra_forward_checkerboard,
    "forward",
    Some("checkerboard_corners_images"),
    Checkerboard,
    TagFamily::AprilTag36h11
);

// Lengthy tests (Ignored by default)
test_icra!(
    IGNORED regression_icra_circle,
    "circle",
    Some("pure_tags_images"),
    PlainBoard,
    TagFamily::AprilTag36h11
);
test_icra!(
    IGNORED regression_icra_circle_checkerboard,
    "circle",
    Some("checkerboard_corners_images"),
    Checkerboard,
    TagFamily::AprilTag36h11
);
test_icra!(
    IGNORED regression_icra_random,
    "random",
    Some("pure_tags_images"),
    PlainBoard,
    TagFamily::AprilTag36h11
);
test_icra!(
    IGNORED regression_icra_random_checkerboard,
    "random",
    Some("checkerboard_corners_images"),
    Checkerboard,
    TagFamily::AprilTag36h11
);
test_icra!(
    IGNORED regression_icra_rotation,
    "rotation",
    Some("pure_tags_images"),
    PlainBoard,
    TagFamily::AprilTag36h11
);
