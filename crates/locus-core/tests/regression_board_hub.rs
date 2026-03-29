//! Regression tests for board-level pose estimation using hub datasets.
//!
//! Requires the `bench-internals` feature and the `LOCUS_HUB_DATASET_DIR`
//! environment variable (or the dataset to be present at the default relative path).
//!
//! Run with:
//! ```
//! LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
//!   cargo nextest run --release --features bench-internals -E 'test(regression_board)'
//! ```
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
    clippy::missing_panics_doc,
    clippy::cast_precision_loss
)]

use locus_core::{
    CameraIntrinsics, DetectOptions, Detector, DetectorConfig, ImageView, PoseEstimationMode,
    TagFamily,
    board::{BoardConfig, BoardEstimator},
};
use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// Configuration Presets
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub enum ConfigPreset {
    /// Production default: balanced recall/precision.
    PlainBoard,
    /// SOTA metrology: GWLF corner refinement + weighted LM.
    SotaMetrology,
}

impl ConfigPreset {
    pub fn detector_config(self) -> DetectorConfig {
        match self {
            Self::PlainBoard => DetectorConfig::production_default(),
            Self::SotaMetrology => DetectorConfig::sota_metrology_default(),
        }
    }
}

// ============================================================================
// Ground Truth Schema (rich_truth.json)
// ============================================================================

#[derive(Deserialize, Clone)]
struct RichTruthEntry {
    image_id: String,
    tag_id: i32,
    record_type: String,
    position: [f64; 3],
    rotation_quaternion: [f64; 4], // [w, x, y, z]
    #[serde(default)]
    k_matrix: Vec<Vec<f64>>,
    #[serde(default)]
    board_definition: Option<BoardDefinitionEntry>,
}

#[derive(Deserialize, Clone)]
struct BoardDefinitionEntry {
    #[serde(rename = "type")]
    board_type: String,
    rows: usize,
    cols: usize,
    square_size_mm: f64,
    marker_size_mm: f64,
}

#[derive(Clone)]
struct BoardImageEntry {
    filename: String,
    board_pose: BoardPoseEntry,
    visible_tag_ids: Vec<u32>,
}

#[derive(Clone)]
struct BoardPoseEntry {
    rotation_quaternion: [f64; 4], // [w, x, y, z]
    translation: [f64; 3],
}

/// Top-level aggregation from rich_truth.json.
struct BoardTruth {
    board_config: BoardConfigEntry,
    camera_intrinsics: CameraIntrinsics,
    images: Vec<BoardImageEntry>,
}

#[derive(Clone)]
struct BoardConfigEntry {
    board_type: String,
    rows: usize,
    cols: usize,
    square_length_m: f64,
    marker_length_m: f64,
}

// ============================================================================
// Metrics
// ============================================================================

#[allow(clippy::trivially_copy_pass_by_ref)]
fn serialize_rmse<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let rounded = (value * 10000.0).round() / 10000.0;
    serializer.serialize_f64(rounded)
}

#[derive(Serialize, Default)]
struct BoardSummaryMetrics {
    dataset_size: usize,
    frames_with_board: usize,
    #[serde(serialize_with = "serialize_rmse")]
    mean_translation_error_m: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p50_translation_error_m: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p90_translation_error_m: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_rotation_error_deg: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p50_rotation_error_deg: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p90_rotation_error_deg: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_translation_std_m: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_rotation_std_deg: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_tag_coverage: f64,
    mean_total_ms: f64,
    frames_no_estimate: usize,
}

#[derive(Serialize)]
struct BoardRegressionReport {
    summary: BoardSummaryMetrics,
}

fn calculate_percentile(values: &mut [f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (percentile * (values.len() - 1) as f64).round() as usize;
    values[idx]
}

// ============================================================================
// Dataset Provider
// ============================================================================

struct BoardHubProvider {
    name: String,
    base_dir: PathBuf,
    truth: BoardTruth,
}

impl BoardHubProvider {
    fn new(dataset_dir: &std::path::Path) -> Option<Self> {
        let rich_truth_path = dataset_dir.join("rich_truth.json");
        let images_dir = dataset_dir.join("images");

        if !images_dir.exists() || !rich_truth_path.exists() {
            return None;
        }

        let file = std::fs::File::open(&rich_truth_path).ok()?;
        let raw_entries: Vec<RichTruthEntry> = serde_json::from_reader(file).ok()?;

        if raw_entries.is_empty() {
            return None;
        }

        // 1. Group tags by image and extract global board config/intrinsics
        let mut board_config_entry = None;
        let mut intrinsics = None;
        let mut image_map: std::collections::BTreeMap<String, (BoardPoseEntry, Vec<u32>)> =
            std::collections::BTreeMap::new();

        for entry in raw_entries {
            if entry.record_type == "BOARD" {
                if board_config_entry.is_none() {
                    if let Some(ref def) = entry.board_definition {
                        board_config_entry = Some(BoardConfigEntry {
                            board_type: def.board_type.clone(),
                            rows: def.rows,
                            cols: def.cols,
                            square_length_m: def.square_size_mm / 1000.0,
                            marker_length_m: def.marker_size_mm / 1000.0,
                        });
                    }
                }
                if intrinsics.is_none() && entry.k_matrix.len() >= 2 {
                    intrinsics = Some(CameraIntrinsics::new(
                        entry.k_matrix[0][0],
                        entry.k_matrix[1][1],
                        entry.k_matrix[0][2],
                        entry.k_matrix[1][2],
                    ));
                }

                let filename = if entry.image_id.ends_with(".png") {
                    entry.image_id.clone()
                } else {
                    format!("{}.png", entry.image_id)
                };

                let pose = BoardPoseEntry {
                    rotation_quaternion: entry.rotation_quaternion,
                    translation: entry.position,
                };

                let img_data = image_map
                    .entry(filename)
                    .or_insert((pose.clone(), Vec::new()));
                // Update pose if we found the BOARD record (sometimes TAG records come first)
                img_data.0 = pose;
            } else if entry.record_type == "TAG" {
                let filename = if entry.image_id.ends_with(".png") {
                    entry.image_id.clone()
                } else {
                    format!("{}.png", entry.image_id)
                };

                // Use identity pose as placeholder until BOARD record is found
                let placeholder_pose = BoardPoseEntry {
                    rotation_quaternion: [1.0, 0.0, 0.0, 0.0],
                    translation: [0.0, 0.0, 0.0],
                };

                let img_data = image_map
                    .entry(filename)
                    .or_insert((placeholder_pose, Vec::new()));
                img_data.1.push(entry.tag_id as u32);
            }
        }

        let board_config = board_config_entry?;
        let camera_intrinsics = intrinsics?;

        let images: Vec<BoardImageEntry> = image_map
            .into_iter()
            .map(|(filename, (board_pose, visible_tag_ids))| BoardImageEntry {
                filename,
                board_pose,
                visible_tag_ids,
            })
            .collect();

        Some(Self {
            name: dataset_dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            base_dir: dataset_dir.to_path_buf(),
            truth: BoardTruth {
                board_config,
                camera_intrinsics,
                images,
            },
        })
    }

    #[allow(clippy::panic)]
    fn board_config(&self) -> BoardConfig {
        let bc = &self.truth.board_config;
        if bc.board_type.contains("charuco") {
            BoardConfig::new_charuco(bc.rows, bc.cols, bc.square_length_m, bc.marker_length_m)
        } else if bc.board_type.contains("april") {
            // Assume 20% spacing for AprilGrid if not specified, or match the constructor logic
            // In BoardConfig::new_aprilgrid, step = marker + spacing.
            // Here we have square_length_m (step) and marker_length_m.
            let spacing = bc.square_length_m - bc.marker_length_m;
            BoardConfig::new_aprilgrid(bc.rows, bc.cols, spacing, bc.marker_length_m)
        } else {
            panic!("Unknown board type: {}", bc.board_type);
        }
    }

    fn intrinsics(&self) -> CameraIntrinsics {
        self.truth.camera_intrinsics
    }
}

// ============================================================================
// Harness
// ============================================================================

struct BoardRegressionHarness {
    snapshot_name: String,
    config: DetectorConfig,
    options: DetectOptions,
}

impl BoardRegressionHarness {
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

    pub fn run(self, provider: &BoardHubProvider) {
        let board_config = provider.board_config();
        let intrinsics = provider.intrinsics();
        let estimator = BoardEstimator::new(board_config);

        let mut detector = Detector::with_config(self.config);
        if !self.options.families.is_empty() {
            detector.set_families(&self.options.families);
        }

        let mut translation_errors: Vec<f64> = Vec::new();
        let mut rotation_errors_deg: Vec<f64> = Vec::new();
        let mut translation_stds: Vec<f64> = Vec::new();
        let mut rotation_stds_deg: Vec<f64> = Vec::new();
        let mut total_time = 0.0;
        let mut frames_with_board = 0;
        let mut frames_no_estimate = 0;
        let mut total_coverage = 0.0;
        let dataset_size = provider.truth.images.len();

        for entry in &provider.truth.images {
            let img_path = provider.base_dir.join("images").join(&entry.filename);
            if !img_path.exists() {
                continue;
            }

            let img = image::open(&img_path)
                .expect("load board image")
                .into_luma8();
            let (w, h) = img.dimensions();
            let raw = img.into_raw();
            let img_view =
                ImageView::new(&raw, w as usize, h as usize, w as usize).expect("valid image");

            let start = std::time::Instant::now();
            detector
                .detect(
                    &img_view,
                    Some(&intrinsics),
                    None,
                    PoseEstimationMode::Accurate,
                    false,
                )
                .expect("detection failed");
            let detect_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Clone the batch after detection (requires bench-internals feature)
            #[cfg(feature = "bench-internals")]
            let batch = detector.bench_api_get_batch_cloned();
            #[cfg(not(feature = "bench-internals"))]
            compile_error!("regression_board_hub requires the bench-internals feature");

            // Count detected tags that match the visible set
            let visible_set: std::collections::HashSet<u32> =
                entry.visible_tag_ids.iter().copied().collect();
            let detected_valid = batch
                .status_mask
                .iter()
                .zip(batch.ids.iter())
                .filter(|&(state, id)| {
                    *state == locus_core::batch::CandidateState::Valid && visible_set.contains(id)
                })
                .count();
            let coverage = if visible_set.is_empty() {
                1.0
            } else {
                detected_valid as f64 / visible_set.len() as f64
            };
            total_coverage += coverage;

            let board_start = std::time::Instant::now();
            let board_pose = estimator.estimate(&batch, &intrinsics);
            let board_ms = board_start.elapsed().as_secs_f64() * 1000.0;
            total_time += detect_ms + board_ms;

            if let Some(est_board) = board_pose {
                frames_with_board += 1;

                // Ground truth pose
                let bp = &entry.board_pose;
                let q_gt = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    bp.rotation_quaternion[0], // w
                    bp.rotation_quaternion[1], // x
                    bp.rotation_quaternion[2], // y
                    bp.rotation_quaternion[3], // z
                ));
                let t_gt = Vector3::new(bp.translation[0], bp.translation[1], bp.translation[2]);

                let est_pose = &est_board.pose;
                let t_err = (est_pose.translation - t_gt).norm();
                let q_est = UnitQuaternion::from_matrix(&est_pose.rotation);
                let r_err_deg = q_est.angle_to(&q_gt).to_degrees();

                translation_errors.push(t_err);
                rotation_errors_deg.push(r_err_deg);

                // Extract standard deviations from the diagonal of the 6x6 covariance matrix
                // [tx, ty, tz, rx, ry, rz]
                let cov = &est_board.covariance;
                let t_std = (cov[(0, 0)].max(0.0) + cov[(1, 1)].max(0.0) + cov[(2, 2)].max(0.0))
                    .sqrt();
                let r_std_deg = (cov[(3, 3)].max(0.0) + cov[(4, 4)].max(0.0) + cov[(5, 5)].max(0.0))
                    .sqrt()
                    .to_degrees();

                translation_stds.push(t_std);
                rotation_stds_deg.push(r_std_deg);
            } else {
                frames_no_estimate += 1;
            }
        }

        if dataset_size == 0 {
            println!("WARNING: Dataset {} yielded no images.", self.snapshot_name);
            return;
        }

        let count = translation_errors.len();
        let mean_t_err = if count > 0 {
            translation_errors.iter().sum::<f64>() / count as f64
        } else {
            0.0
        };
        let mean_r_err = if count > 0 {
            rotation_errors_deg.iter().sum::<f64>() / count as f64
        } else {
            0.0
        };
        let mean_t_std = if count > 0 {
            translation_stds.iter().sum::<f64>() / count as f64
        } else {
            0.0
        };
        let mean_r_std = if count > 0 {
            rotation_stds_deg.iter().sum::<f64>() / count as f64
        } else {
            0.0
        };

        let summary = BoardSummaryMetrics {
            dataset_size,
            frames_with_board,
            mean_translation_error_m: mean_t_err,
            p50_translation_error_m: calculate_percentile(&mut translation_errors, 0.5),
            p90_translation_error_m: calculate_percentile(&mut translation_errors, 0.9),
            mean_rotation_error_deg: mean_r_err,
            p50_rotation_error_deg: calculate_percentile(&mut rotation_errors_deg, 0.5),
            p90_rotation_error_deg: calculate_percentile(&mut rotation_errors_deg, 0.9),
            mean_translation_std_m: mean_t_std,
            mean_rotation_std_deg: mean_r_std,
            mean_tag_coverage: if dataset_size > 0 {
                total_coverage / dataset_size as f64
            } else {
                0.0
            },
            mean_total_ms: total_time / dataset_size as f64,
            frames_no_estimate,
        };

        println!("=== {} ===", self.snapshot_name);
        println!(
            "  Frames: {dataset_size} (board estimated: {frames_with_board}, no estimate: {frames_no_estimate})"
        );
        println!("  Tag coverage: {:.1}%", summary.mean_tag_coverage * 100.0);
        println!("  Trans P50: {:.4} m (std: {:.4} m)", summary.p50_translation_error_m, summary.mean_translation_std_m);
        println!("  Trans P90: {:.4} m", summary.p90_translation_error_m);
        println!("  Rot P50:   {:.4} deg (std: {:.4} deg)", summary.p50_rotation_error_deg, summary.mean_rotation_std_deg);
        println!("  Rot P90:   {:.4} deg", summary.p90_rotation_error_deg);
        println!("  Latency:   {:.2} ms/frame", summary.mean_total_ms);

        let report = BoardRegressionReport { summary };

        insta::assert_yaml_snapshot!(self.snapshot_name, report, {
            ".summary.mean_total_ms" => "[DURATION]"
        });
    }
}

// ============================================================================
// Path Resolution
// ============================================================================

/// Resolves the hub dataset root, anchored to CARGO_MANIFEST_DIR.
fn resolve_hub_root(hub_dir: &str) -> PathBuf {
    let path = PathBuf::from(hub_dir);
    if path.is_absolute() {
        return path;
    }
    if path.is_dir() {
        return std::fs::canonicalize(&path).unwrap_or(path);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let from_workspace = manifest.join("../../").join(&path);
    if from_workspace.is_dir() {
        return from_workspace;
    }
    path
}

fn try_load_board_provider(dataset_name: &str) -> Option<BoardHubProvider> {
    // Try explicit env var first
    if let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") {
        let root = resolve_hub_root(&hub_dir);
        let dataset_path = root.join(dataset_name);
        if let Some(provider) = BoardHubProvider::new(&dataset_path) {
            return Some(provider);
        }
    }
    // Fallback to relative path from workspace
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fallback = manifest
        .join("../../tests/data/hub_cache")
        .join(dataset_name);
    BoardHubProvider::new(&fallback)
}

// ============================================================================
// Test Functions
// ============================================================================

/// Board regression: ChArUco board, production config (tag36h11).
#[test]
fn regression_board_charuco_plain() {
    let dataset_name = "charuco_board_tag36h11_1280x720";
    let Some(provider) = try_load_board_provider(dataset_name) else {
        println!("Skipping board regression: dataset '{dataset_name}' not found.");
        println!(
            "Set LOCUS_HUB_DATASET_DIR or place dataset at tests/data/hub_cache/{dataset_name}"
        );
        return;
    };
    BoardRegressionHarness::new("board_charuco_plain")
        .with_preset(ConfigPreset::PlainBoard)
        .with_families(vec![TagFamily::AprilTag36h11])
        .run(&provider);
}

/// Board regression: ChArUco board, SOTA metrology config (tag36h11).
#[test]
fn regression_board_charuco_sota() {
    let dataset_name = "charuco_board_tag36h11_1280x720";
    let Some(provider) = try_load_board_provider(dataset_name) else {
        println!("Skipping board regression: dataset '{dataset_name}' not found.");
        println!(
            "Set LOCUS_HUB_DATASET_DIR or place dataset at tests/data/hub_cache/{dataset_name}"
        );
        return;
    };
    BoardRegressionHarness::new("board_charuco_sota")
        .with_preset(ConfigPreset::SotaMetrology)
        .with_families(vec![TagFamily::AprilTag36h11])
        .run(&provider);
}

/// Board regression: ChArUco board v1 golden, production config (tag36h11).
#[test]
fn regression_board_charuco_v1_golden_plain() {
    let dataset_name = "charuco_golden_v1";
    let Some(provider) = try_load_board_provider(dataset_name) else {
        println!("Skipping board regression: dataset '{dataset_name}' not found.");
        println!(
            "Set LOCUS_HUB_DATASET_DIR or place dataset at tests/data/hub_cache/{dataset_name}"
        );
        return;
    };
    BoardRegressionHarness::new("board_charuco_v1_golden_plain")
        .with_preset(ConfigPreset::PlainBoard)
        .with_families(vec![TagFamily::ArUco6x6_250])
        .run(&provider);
}

/// Board regression: ChArUco board v1 golden, SOTA metrology config (tag36h11).
#[test]
fn regression_board_charuco_v1_golden_sota() {
    let dataset_name = "charuco_golden_v1";
    let Some(provider) = try_load_board_provider(dataset_name) else {
        println!("Skipping board regression: dataset '{dataset_name}' not found.");
        println!(
            "Set LOCUS_HUB_DATASET_DIR or place dataset at tests/data/hub_cache/{dataset_name}"
        );
        return;
    };
    BoardRegressionHarness::new("board_charuco_v1_golden_sota")
        .with_preset(ConfigPreset::SotaMetrology)
        .with_families(vec![TagFamily::ArUco6x6_250])
        .run(&provider);
}
