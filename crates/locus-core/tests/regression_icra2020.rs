#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::panic,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::type_complexity,
    clippy::unnecessary_debug_formatting,
    clippy::unwrap_used,
    dead_code,
    missing_docs,
    unused_imports
)]
//! ICRA 2020 Dataset Regression Tests.
use locus_core::{DetectOptions, DetectorConfig, TagFamily};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

mod common;

/// Packed checkerboards need a relaxed `quad.min_edge_score` and
/// `decoder.min_contrast` on top of the `grid` profile.
const ICRA_GRID_JSON: &str = include_str!("fixtures/icra_grid.json");

/// ICRA 2020 community-benchmark fixtures — Soft decode + tuned gates
/// for `forward/pure_tags_images`. The ICRA 2020 fiducial dataset is a
/// long-standing research benchmark used in the AprilTag / fiducial-
/// marker literature; these fixtures let researchers reproduce
/// literature-comparable numbers on it. NOT shipped profiles: Soft
/// decode trades precision for recall (+20.2 pp on this dataset's
/// imaging characteristics, +26 false positives per image) and would
/// collapse on data with PSF (real cameras, Blender-rendered hub
/// data) per `lessons.md` §3.2 / §5.5 / §7. Used only by the
/// `regression_icra_forward_synthetic_*` sweep variants below.
const ICRA_SYNTHETIC_SOFT_JSON: &str = include_str!("fixtures/icra_synthetic_soft.json");
const ICRA_SYNTHETIC_MIN_AREA_JSON: &str = include_str!("fixtures/icra_synthetic_min_area.json");
const ICRA_SYNTHETIC_TILE_SIZE_JSON: &str = include_str!("fixtures/icra_synthetic_tile_size.json");
const ICRA_SYNTHETIC_AGGRESSIVE_JSON: &str =
    include_str!("fixtures/icra_synthetic_aggressive.json");

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
}

impl RegressionHarness {
    pub fn new(snapshot_name: impl Into<String>) -> Self {
        Self {
            snapshot_name: snapshot_name.into(),
            config: DetectorConfig::default(),
            options: DetectOptions::default(),
        }
    }

    pub fn with_profile(mut self, name: &str) -> Self {
        self.config = DetectorConfig::from_profile(name);
        self
    }

    pub fn with_profile_json(mut self, json: &str) -> Self {
        self.config =
            DetectorConfig::from_profile_json(json).expect("embedded test fixture must parse");
        self
    }

    pub fn with_decode_mode(mut self, mode: locus_core::config::DecodeMode) -> Self {
        self.config.decode_mode = mode;
        self
    }

    pub fn with_refinement_mode(mut self, mode: locus_core::config::CornerRefinementMode) -> Self {
        self.config.refinement_mode = mode;
        self
    }

    pub fn with_families(mut self, families: Vec<TagFamily>) -> Self {
        self.options.families = families;
        self
    }

    pub fn with_moments_culling(mut self, max_elongation: f64, min_density: f64) -> Self {
        self.config.quad_max_elongation = max_elongation;
        self.config.quad_min_density = min_density;
        self
    }

    pub fn with_quad_extraction_mode(
        mut self,
        mode: locus_core::config::QuadExtractionMode,
    ) -> Self {
        self.config.quad_extraction_mode = mode;
        self
    }

    pub fn run(self, provider: impl DatasetProvider) {
        #[cfg(debug_assertions)]
        {
            let _ = (self, provider);
            panic!(
                "regression_icra2020 test should always be ran in release mode. Please use `cargo test --release`."
            );
        }

        #[cfg(not(debug_assertions))]
        {
            use locus_core::{Detector, ImageView};
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
                let detections = detector
                    .detect(
                        &img,
                        self.options.intrinsics.as_ref(),
                        self.options.tag_size,
                        self.options.pose_estimation_mode,
                        false,
                    )
                    .expect("detection failed");
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
                        let dist_sq =
                            (det_center[0] - gt_cx).powi(2) + (det_center[1] - gt_cy).powi(2);

                        if dist_sq < 50.0 * 50.0 {
                            // ICRA 2020 Parity: The ICRA dataset ground truth follows the UMich/CCW convention:
                            // [0:BL, 1:BR, 2:TR, 3:TL].
                            // Locus follows Modern OpenCV/CW: [TL, TR, BR, BL].
                            //
                            // Based on empirical observation of 0037.png:
                            // DET Corner 0 is at BR (matches GT 1)
                            // DET Corner 1 is at BL (matches GT 0)
                            // DET Corner 2 is at TL (matches GT 3)
                            // DET Corner 3 is at TR (matches GT 2)
                            //
                            // This corresponds to a 180-degree bit-rotation offset in the dataset convention.
                            let gt_locus_aligned = [
                                gt_corners[1], // GT BR -> DET 0
                                gt_corners[0], // GT BL -> DET 1
                                gt_corners[3], // GT TL -> DET 2
                                gt_corners[2], // GT TR -> DET 3
                            ];

                            image_rmse_sum += locus_core::bench_api::compute_rmse(
                                &det_corners_f64,
                                &gt_locus_aligned,
                            );
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
        let env_set = std::env::var("LOCUS_ICRA_DATASET_DIR").is_ok();

        // 1. Load Ground Truth (Context-aware loading)
        let gt_map = match common::load_ground_truth(&root, subfolder) {
            Some(gt) => gt,
            None if env_set => panic!(
                "ground truth not found under '{}' for subfolder '{}' \
                 (LOCUS_ICRA_DATASET_DIR was set — refusing to silent-skip)",
                root.display(),
                subfolder
            ),
            None => return None,
        };

        // 2. Locate Image Directory
        // Handle both flat and nested "pure_tags_images" structures gracefully
        let mut candidates = Vec::new();
        if let Some(sub) = img_subfolder {
            candidates.push(root.join(subfolder).join(sub));
        }
        candidates.push(root.join(subfolder).join("pure_tags_images"));
        candidates.push(root.join(subfolder));

        let img_dir = match candidates.iter().find(|p| p.is_dir()) {
            Some(p) => p,
            None if env_set => panic!(
                "no image directory found for subfolder '{}' under '{}' \
                 (LOCUS_ICRA_DATASET_DIR was set — refusing to silent-skip)",
                subfolder,
                root.display()
            ),
            None => return None,
        };

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

/// Applies either a shipped profile (`"standard"` / `"high_accuracy"`) or the
/// inlined ICRA-grid fixture (`"icra_grid"`) to a harness.
fn apply_profile(harness: RegressionHarness, profile: &str) -> RegressionHarness {
    match profile {
        "icra_grid" => harness.with_profile_json(ICRA_GRID_JSON),
        other => harness.with_profile(other),
    }
}

macro_rules! test_icra {
    ($name:ident, $subfolder:expr, $img_subfolder:expr, $profile:expr, $family:expr) => {
        #[test]
        fn $name() {
            let _guard = common::telemetry::init(stringify!($name));
            if let Some(provider) = IcraProvider::new($subfolder, $img_subfolder) {
                let snapshot = provider.name().to_string();
                let harness = RegressionHarness::new(snapshot);
                apply_profile(harness, $profile)
                    .with_families(vec![$family])
                    .run(provider);
            }
        }
    };
    (IGNORED $name:ident, $subfolder:expr, $img_subfolder:expr, $profile:expr, $family:expr) => {
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
                let harness = RegressionHarness::new(snapshot);
                apply_profile(harness, $profile)
                    .with_families(vec![$family])
                    .run(provider);
            }
        }
    };
    (SOFT $name:ident, $subfolder:expr, $img_subfolder:expr, $profile:expr, $family:expr) => {
        #[test]
        fn $name() {
            let _guard = common::telemetry::init(stringify!($name));
            if let Some(provider) = IcraProvider::new($subfolder, $img_subfolder) {
                let snapshot = format!("{}_soft", provider.name());
                let harness = RegressionHarness::new(snapshot);
                apply_profile(harness, $profile)
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
        .with_profile("standard")
        .with_families(vec![TagFamily::AprilTag36h11])
        .run(provider);
}

test_icra!(
    regression_icra_forward,
    "forward",
    Some("pure_tags_images"),
    "standard",
    TagFamily::AprilTag36h11
);
test_icra!(
    SOFT regression_icra_forward_soft,
    "forward",
    Some("pure_tags_images"),
    "standard",
    TagFamily::AprilTag36h11
);
test_icra!(
    regression_icra_forward_checkerboard,
    "forward",
    Some("checkerboard_corners_images"),
    "icra_grid",
    TagFamily::AprilTag36h11
);

#[test]
fn regression_icra_forward_gwlf() {
    let _guard = common::telemetry::init("regression_icra_forward_gwlf");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_gwlf".to_string();
        RegressionHarness::new(snapshot)
            .with_profile("standard")
            .with_families(vec![TagFamily::AprilTag36h11])
            // Override profile with GWLF
            .with_refinement_mode(locus_core::config::CornerRefinementMode::Gwlf)
            .run(provider);
    }
}
// Lengthy tests (Ignored by default)
test_icra!(
    IGNORED regression_icra_circle,
    "circle",
    Some("pure_tags_images"),
    "standard",
    TagFamily::AprilTag36h11
);
test_icra!(
    IGNORED regression_icra_circle_checkerboard,
    "circle",
    Some("checkerboard_corners_images"),
    "icra_grid",
    TagFamily::AprilTag36h11
);
test_icra!(
    IGNORED regression_icra_random,
    "random",
    Some("pure_tags_images"),
    "standard",
    TagFamily::AprilTag36h11
);
test_icra!(
    IGNORED regression_icra_random_checkerboard,
    "random",
    Some("checkerboard_corners_images"),
    "icra_grid",
    TagFamily::AprilTag36h11
);
test_icra!(
    IGNORED regression_icra_rotation,
    "rotation",
    Some("pure_tags_images"),
    "standard",
    TagFamily::AprilTag36h11
);

// ── New algorithm tuning variants ────────────────────────────────────────────

#[test]
fn regression_icra_forward_moments_culling() {
    let _guard = common::telemetry::init("regression_icra_forward_moments_culling");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_pure_default_moments_culling".to_string();
        RegressionHarness::new(snapshot)
            .with_profile("standard")
            .with_families(vec![TagFamily::AprilTag36h11])
            .with_moments_culling(15.0, 0.15)
            .run(provider);
    }
}

#[test]
fn regression_icra_forward_edlines() {
    let _guard = common::telemetry::init("regression_icra_forward_edlines");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_pure_default_edlines".to_string();
        RegressionHarness::new(snapshot)
            .with_profile("standard")
            .with_families(vec![TagFamily::AprilTag36h11])
            .with_quad_extraction_mode(locus_core::config::QuadExtractionMode::EdLines)
            .run(provider);
    }
}

#[test]
fn regression_icra_forward_edlines_moments() {
    let _guard = common::telemetry::init("regression_icra_forward_edlines_moments");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_pure_default_edlines_moments".to_string();
        RegressionHarness::new(snapshot)
            .with_profile("standard")
            .with_families(vec![TagFamily::AprilTag36h11])
            .with_quad_extraction_mode(locus_core::config::QuadExtractionMode::EdLines)
            .with_moments_culling(15.0, 0.15)
            .run(provider);
    }
}

// ── HighAccuracy (EdLines GN + covariance propagation + Weighted LM) ───────

#[test]
fn regression_icra_forward_highaccuracy() {
    let _guard = common::telemetry::init("regression_icra_forward_highaccuracy");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_pure_default_highaccuracy".to_string();
        RegressionHarness::new(snapshot)
            .with_profile("high_accuracy")
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(provider);
    }
}

// ── Standard (default profile, ContourRdp + Hard) ─────────────────────────────

#[test]
fn regression_icra_forward_standard() {
    let _guard = common::telemetry::init("regression_icra_forward_standard");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_pure_default_standard".to_string();
        RegressionHarness::new(snapshot)
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(provider);
    }
}

// ── MaxRecallAdaptive (PPB-routed: ContourRdp+Erf low / EdLines+None high) ────

#[test]
fn regression_icra_forward_max_recall_adaptive() {
    let _guard = common::telemetry::init("regression_icra_forward_max_recall_adaptive");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_pure_max_recall_adaptive".to_string();
        RegressionHarness::new(snapshot)
            .with_profile("max_recall_adaptive")
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(provider);
    }
}

// ── Grid (icra_grid profile) ──────────────────────────────────────────────────

#[test]
fn regression_icra_forward_grid() {
    let _guard = common::telemetry::init("regression_icra_forward_grid");
    if let Some(provider) = IcraProvider::new("forward", Some("checkerboard_corners_images")) {
        let snapshot = "icra_forward_checkerboard_grid".to_string();
        RegressionHarness::new(snapshot)
            .with_profile_json(ICRA_GRID_JSON)
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(provider);
    }
}

// ── ICRA community-benchmark sweep ────────────────────────────────────────────
//
// Published recall on `forward/pure_tags_images` for community comparison
// against the original ICRA 2020 paper's numbers. NOT a production
// configuration — Soft decode + relaxed gates trade precision for recall on
// the crude-renderer ICRA forward dataset and would collapse on real-camera
// data (lessons.md §3.2 / §5.5). Each variant snapshots its own recall so
// future maintainers can see what each tuning lever buys.

#[test]
fn regression_icra_forward_synthetic_soft() {
    let _guard = common::telemetry::init("regression_icra_forward_synthetic_soft");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_synthetic_soft".to_string();
        RegressionHarness::new(snapshot)
            .with_profile_json(ICRA_SYNTHETIC_SOFT_JSON)
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(provider);
    }
}

#[test]
fn regression_icra_forward_synthetic_min_area() {
    let _guard = common::telemetry::init("regression_icra_forward_synthetic_min_area");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_synthetic_min_area".to_string();
        RegressionHarness::new(snapshot)
            .with_profile_json(ICRA_SYNTHETIC_MIN_AREA_JSON)
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(provider);
    }
}

#[test]
fn regression_icra_forward_synthetic_tile_size() {
    let _guard = common::telemetry::init("regression_icra_forward_synthetic_tile_size");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_synthetic_tile_size".to_string();
        RegressionHarness::new(snapshot)
            .with_profile_json(ICRA_SYNTHETIC_TILE_SIZE_JSON)
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(provider);
    }
}

#[test]
fn regression_icra_forward_synthetic_aggressive() {
    let _guard = common::telemetry::init("regression_icra_forward_synthetic_aggressive");
    if let Some(provider) = IcraProvider::new("forward", Some("pure_tags_images")) {
        let snapshot = "icra_forward_synthetic_aggressive".to_string();
        RegressionHarness::new(snapshot)
            .with_profile_json(ICRA_SYNTHETIC_AGGRESSIVE_JSON)
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(provider);
    }
}
