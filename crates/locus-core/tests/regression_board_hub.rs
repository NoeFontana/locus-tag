//! Data provider and parsing utilities for board-level hub datasets.
#![allow(dead_code, missing_docs, clippy::collapsible_if)]

use locus_core::{CameraIntrinsics, board::BoardConfig};
use serde::Deserialize;
use std::path::{Path, PathBuf};

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
    pub board_config: BoardConfig,
    pub camera_intrinsics: CameraIntrinsics,
    pub images: Vec<BoardImageEntry>,
}

impl BoardHubProvider {
    #[must_use]
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
                #[allow(clippy::cast_sign_loss)]
                img_data.1.push(entry.tag_id as u32);
            }
        }

        let bce = board_config_entry?;
        let board_config = if bce.board_type.contains("charuco") {
            BoardConfig::new_charuco(bce.rows, bce.cols, bce.square_length_m, bce.marker_length_m)
        } else {
            let spacing = bce.square_length_m - bce.marker_length_m;
            BoardConfig::new_aprilgrid(bce.rows, bce.cols, spacing, bce.marker_length_m)
        };

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
}
