//! Regression tests for board-level pose estimation using hub datasets.
#![allow(
    dead_code,
    missing_docs,
    clippy::unwrap_used,
    clippy::collapsible_if,
    clippy::too_many_lines,
    clippy::cast_sign_loss
)]

use locus_core::{
    CameraIntrinsics, Detector, DetectorConfig, PoseEstimationMode, TagFamily,
    board::{AprilGridTopology, BoardEstimator, CharucoTopology},
};
use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ============================================================================
// Data Provider
// ============================================================================

#[derive(Deserialize, Clone)]
struct RichTruthEntry {
    image_id: String,
    #[serde(rename = "tag_id")]
    tag_id_raw: i32,
    record_type: String,
    position: [f64; 3],
    rotation_quaternion: [f64; 4], // [w, x, y, z]
    #[serde(default)]
    k_matrix: Vec<Vec<f64>>,
    #[serde(default)]
    board_definition: Option<BoardDefinitionEntry>,
}

impl RichTruthEntry {
    fn tag_id(&self) -> u32 {
        self.tag_id_raw as u32
    }
}

#[derive(Deserialize, Clone)]
pub struct BoardDefinitionEntry {
    #[serde(rename = "type")]
    pub board_type: String,
    pub rows: usize,
    pub cols: usize,
    pub square_size_mm: f64,
    pub marker_size_mm: f64,
}

#[derive(Clone)]
pub struct BoardImageEntry {
    pub filename: String,
    pub board_pose: BoardPoseEntry,
    pub visible_tag_ids: Vec<u32>,
}

#[derive(Clone)]
pub struct BoardPoseEntry {
    pub rotation_quaternion: [f64; 4], // [w, x, y, z]
    pub translation: [f64; 3],
}

pub struct BoardConfigEntry {
    pub board_type: String,
    pub rows: usize,
    pub cols: usize,
    pub square_length_m: f64,
    pub marker_length_m: f64,
}

pub struct BoardHubProvider {
    pub base_dir: PathBuf,
    /// Marker geometry used by [`BoardEstimator`].
    ///
    /// For ChAruco boards this is built from the marker table only (no saddle
    /// refinement); for AprilGrid boards it is the full topology.
    pub board_config: Arc<AprilGridTopology>,
    pub camera_intrinsics: CameraIntrinsics,
    pub images: Vec<BoardImageEntry>,
}

impl BoardHubProvider {
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn new(dataset_dir: &Path) -> Option<Self> {
        let rich_truth_path = dataset_dir.join("rich_truth.json");
        if !rich_truth_path.exists() {
            return None;
        }

        let file = std::fs::File::open(&rich_truth_path).ok()?;
        let raw_entries: Vec<RichTruthEntry> = serde_json::from_reader(file).ok()?;

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

                let filename = if entry.image_id.to_lowercase().ends_with(".png") {
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
                img_data.0 = pose;
            } else if entry.record_type == "TAG" {
                let filename = if entry.image_id.to_lowercase().ends_with(".png") {
                    entry.image_id.clone()
                } else {
                    format!("{}.png", entry.image_id)
                };

                let placeholder_pose = BoardPoseEntry {
                    rotation_quaternion: [1.0, 0.0, 0.0, 0.0],
                    translation: [0.0, 0.0, 0.0],
                };

                let img_data = image_map
                    .entry(filename)
                    .or_insert((placeholder_pose, Vec::new()));
                img_data.1.push(entry.tag_id());
            }
        }

        let bce = board_config_entry?;
        let board_config = Arc::new(if bce.board_type.contains("charuco") {
            // Build ChAruco marker table, then adapt it for tag-only BoardEstimator use.
            let topo = CharucoTopology::new(
                bce.rows,
                bce.cols,
                bce.square_length_m,
                bce.marker_length_m,
                usize::MAX,
            )
            .expect("valid charuco topology from dataset");
            AprilGridTopology::from_obj_points(
                topo.rows,
                topo.cols,
                topo.marker_length,
                topo.obj_points,
            )
        } else {
            let spacing = bce.square_length_m - bce.marker_length_m;
            AprilGridTopology::new(bce.rows, bce.cols, spacing, bce.marker_length_m, usize::MAX)
                .expect("valid aprilgrid topology from dataset")
        });

        let camera_intrinsics = intrinsics?;

        let images: Vec<BoardImageEntry> = image_map
            .into_iter()
            .map(
                |(filename, (board_pose, visible_tag_ids))| BoardImageEntry {
                    filename,
                    board_pose,
                    visible_tag_ids,
                },
            )
            .collect();

        Some(Self {
            base_dir: dataset_dir.to_path_buf(),
            board_config,
            camera_intrinsics,
            images,
        })
    }
}

// ============================================================================
// Metrics
// ============================================================================

#[derive(Serialize, Default)]
struct BoardSummaryMetrics {
    dataset_size: usize,
    frames_with_board: usize,
    mean_board_translation_error_m: f64,
    p50_board_translation_error_m: f64,
    mean_board_rotation_error_deg: f64,
    p50_board_rotation_error_deg: f64,
    mean_board_translation_std_m: f64,
    mean_board_rotation_std_deg: f64,
    mean_tag_coverage: f64,
    mean_total_ms: f64,
    frames_no_estimate: usize,
}

#[derive(Serialize)]
struct BoardRegressionReport {
    summary: BoardSummaryMetrics,
}

fn calculate_percentile(values: &mut [f64], percentile: f64) -> f64 {
    let mut clean_values: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if clean_values.is_empty() {
        return 0.0;
    }
    clean_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (percentile * (clean_values.len() - 1) as f64).round() as usize;
    clean_values[idx]
}

// ============================================================================
// Harness
// ============================================================================

struct BoardRegressionHarness {
    snapshot_name: String,
    config: DetectorConfig,
    families: Vec<TagFamily>,
}

impl BoardRegressionHarness {
    pub fn new(snapshot_name: impl Into<String>) -> Self {
        Self {
            snapshot_name: snapshot_name.into(),
            config: DetectorConfig::production_default(),
            families: Vec::new(),
        }
    }

    pub fn with_families(mut self, families: Vec<TagFamily>) -> Self {
        self.families = families;
        self
    }

    pub fn run(self, provider: &BoardHubProvider) {
        let mut estimator = BoardEstimator::new(Arc::clone(&provider.board_config));
        let mut detector = Detector::with_config(self.config);
        if !self.families.is_empty() {
            detector.set_families(&self.families);
        }

        let mut t_errors = Vec::new();
        let mut r_errors = Vec::new();
        let mut t_stds = Vec::new();
        let mut r_stds = Vec::new();
        let mut frames_with_board = 0;
        let mut frames_no_estimate = 0;
        let mut total_coverage = 0.0;
        let mut total_time = 0.0;

        for (i, entry) in provider.images.iter().enumerate() {
            if i % 20 == 0 {
                println!("  Processing frame {}/{}...", i, provider.images.len());
            }
            let img_path = provider.base_dir.join("images").join(&entry.filename);
            if !img_path.exists() {
                continue;
            }

            let img = image::open(img_path).unwrap().into_luma8();
            let (w, h) = img.dimensions();
            let raw = img.into_raw();
            let img_view =
                locus_core::ImageView::new(&raw, w as usize, h as usize, w as usize).unwrap();

            let start = std::time::Instant::now();
            let _ = detector
                .detect(
                    &img_view,
                    Some(&provider.camera_intrinsics),
                    Some(provider.board_config.marker_length),
                    PoseEstimationMode::Accurate,
                    false,
                )
                .unwrap();
            let detect_ms = start.elapsed().as_secs_f64() * 1000.0;

            #[cfg(feature = "bench-internals")]
            {
                let batch_cloned = detector.bench_api_get_batch_cloned();

                // Tag coverage calculation
                let visible_set: std::collections::HashSet<u32> =
                    entry.visible_tag_ids.iter().copied().collect();
                let detected_valid = batch_cloned
                    .status_mask
                    .iter()
                    .zip(batch_cloned.ids.iter())
                    .filter(|&(state, id)| {
                        *state == locus_core::batch::CandidateState::Valid
                            && visible_set.contains(id)
                    })
                    .count();
                total_coverage += if visible_set.is_empty() {
                    1.0
                } else {
                    detected_valid as f64 / visible_set.len() as f64
                };

                let board_start = std::time::Instant::now();
                let board_pose = estimator.estimate(&batch_cloned, &provider.camera_intrinsics);
                let board_ms = board_start.elapsed().as_secs_f64() * 1000.0;
                total_time += detect_ms + board_ms;

                if let Some(est) = board_pose {
                    frames_with_board += 1;

                    let gt_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        entry.board_pose.rotation_quaternion[0],
                        entry.board_pose.rotation_quaternion[1],
                        entry.board_pose.rotation_quaternion[2],
                        entry.board_pose.rotation_quaternion[3],
                    ));
                    let gt_t = Vector3::new(
                        entry.board_pose.translation[0],
                        entry.board_pose.translation[1],
                        entry.board_pose.translation[2],
                    );

                    t_errors.push((est.pose.translation - gt_t).norm());
                    r_errors.push(
                        UnitQuaternion::from_matrix(&est.pose.rotation)
                            .angle_to(&gt_q)
                            .to_degrees(),
                    );

                    t_stds.push(
                        (est.covariance[(0, 0)].max(0.0)
                            + est.covariance[(1, 1)].max(0.0)
                            + est.covariance[(2, 2)].max(0.0))
                        .sqrt(),
                    );
                    r_stds.push(
                        (est.covariance[(3, 3)].max(0.0)
                            + est.covariance[(4, 4)].max(0.0)
                            + est.covariance[(5, 5)].max(0.0))
                        .sqrt()
                        .to_degrees(),
                    );
                } else {
                    frames_no_estimate += 1;
                }
            }
        }

        let count = t_errors.len() as f64;
        let summary = BoardSummaryMetrics {
            dataset_size: provider.images.len(),
            frames_with_board,
            mean_board_translation_error_m: if count > 0.0 {
                t_errors.iter().sum::<f64>() / count
            } else {
                0.0
            },
            p50_board_translation_error_m: calculate_percentile(&mut t_errors, 0.5),
            mean_board_rotation_error_deg: if count > 0.0 {
                r_errors.iter().sum::<f64>() / count
            } else {
                0.0
            },
            p50_board_rotation_error_deg: calculate_percentile(&mut r_errors, 0.5),
            mean_board_translation_std_m: if count > 0.0 {
                t_stds.iter().sum::<f64>() / count
            } else {
                0.0
            },
            mean_board_rotation_std_deg: if count > 0.0 {
                r_stds.iter().sum::<f64>() / count
            } else {
                0.0
            },
            mean_tag_coverage: total_coverage / (provider.images.len() as f64).max(1.0),
            mean_total_ms: total_time / (provider.images.len() as f64).max(1.0),
            frames_no_estimate,
        };

        let report = BoardRegressionReport { summary };
        insta::assert_yaml_snapshot!(self.snapshot_name, report, {
            ".summary.mean_total_ms" => "[DURATION]"
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_charuco_golden_v1_metadata() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dataset_path = manifest_dir.join("../../tests/data/hub_cache/charuco_golden_v1");

        if dataset_path.exists() {
            let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
            assert_eq!(provider.board_config.rows, 6);
            assert_eq!(provider.board_config.cols, 6);
            assert!(!provider.images.is_empty());
            println!(
                "Loaded {} images from charuco golden dataset",
                provider.images.len()
            );
        }
    }

    #[test]
    fn test_load_aprilgrid_golden_v1_metadata() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dataset_path = manifest_dir.join("../../tests/data/hub_cache/aprilgrid_golden_v1");

        if dataset_path.exists() {
            let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
            // AprilGrid Golden v1 is 6x6 tags
            assert_eq!(provider.board_config.rows, 6);
            assert_eq!(provider.board_config.cols, 6);
            assert!(!provider.images.is_empty());
            println!(
                "Loaded {} images from aprilgrid golden dataset",
                provider.images.len()
            );
        }
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_board_charuco_v1_golden() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dataset_path = manifest_dir.join("../../tests/data/hub_cache/charuco_golden_v1");

        if dataset_path.exists() {
            let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
            BoardRegressionHarness::new("board_charuco_v1_golden")
                .with_families(vec![TagFamily::ArUco6x6_250])
                .run(&provider);
        }
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_board_aprilgrid_v1_golden() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dataset_path = manifest_dir.join("../../tests/data/hub_cache/aprilgrid_golden_v1");

        if dataset_path.exists() {
            let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
            BoardRegressionHarness::new("board_aprilgrid_v1_golden")
                .with_families(vec![TagFamily::AprilTag36h11])
                .run(&provider);
        }
    }
}
