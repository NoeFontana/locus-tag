#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::collapsible_if,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::panic,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! Regression tests for board-level pose estimation using hub datasets.

use locus_core::{
    CameraIntrinsics, Detector, DetectorConfig, TagFamily,
    board::{AprilGridTopology, BoardEstimator, CharucoTopology},
    pose::quat_from_so3,
};
use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod common;

// ============================================================================
// Data Provider
// ============================================================================

/// Top-level wrapper introduced with rich_truth.json v2.0.
///
/// The legacy on-disk format was a bare `[RichTruthEntry, ...]`; v2.0 wraps
/// it in `{"version", "evaluation_context", "records": [...]}`. The harness
/// only cares about `records`.
#[derive(Deserialize)]
struct RichTruthFile {
    #[serde(default)]
    records: Vec<RichTruthEntry>,
}

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
    /// ChArUco-specific topology, populated only when the dataset is a ChArUco
    /// board. Used by `CharucoRefiner` (the saddle-refinement path), which
    /// needs the full `CharucoTopology` (saddle table + marker table), not the
    /// `AprilGridTopology` adaptation stored in `board_config`.
    pub charuco_config: Option<Arc<CharucoTopology>>,
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
        // v2.0 wraps records in {"version", "evaluation_context", "records": [...]}.
        // Fall back to the legacy bare-array form for older fixtures.
        let raw_entries: Vec<RichTruthEntry> = if let Ok(wrapped) =
            serde_json::from_reader::<_, RichTruthFile>(std::io::BufReader::new(&file))
        {
            wrapped.records
        } else {
            let file = std::fs::File::open(&rich_truth_path).ok()?;
            serde_json::from_reader(file).ok()?
        };

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
        let (board_config, charuco_config) = if bce.board_type.contains("charuco") {
            // Build ChAruco marker table, then adapt it for tag-only BoardEstimator use.
            let topo = CharucoTopology::new(
                bce.rows,
                bce.cols,
                bce.square_length_m,
                bce.marker_length_m,
                usize::MAX,
            )
            .expect("valid charuco topology from dataset");
            let charuco_arc = Arc::new(topo);
            let board_arc = Arc::new(AprilGridTopology::from_obj_points(
                charuco_arc.rows,
                charuco_arc.cols,
                charuco_arc.marker_length,
                charuco_arc.obj_points.clone(),
            ));
            (board_arc, Some(charuco_arc))
        } else {
            let spacing = bce.square_length_m - bce.marker_length_m;
            let board_arc = Arc::new(
                AprilGridTopology::new(
                    bce.rows,
                    bce.cols,
                    spacing,
                    bce.marker_length_m,
                    usize::MAX,
                )
                .expect("valid aprilgrid topology from dataset"),
            );
            (board_arc, None)
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
            charuco_config,
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
    p95_board_translation_error_m: f64,
    p99_board_translation_error_m: f64,
    mean_board_rotation_error_deg: f64,
    p50_board_rotation_error_deg: f64,
    p95_board_rotation_error_deg: f64,
    p99_board_rotation_error_deg: f64,
    mean_board_translation_std_m: f64,
    mean_board_rotation_std_deg: f64,
    mean_tag_coverage: f64,
    mean_total_ms: f64,
    frames_no_estimate: usize,
    // Frames where the board pose was estimated but the LM Hessian was
    // singular (covariance is the `Matrix6::from_element(NAN)` sentinel
    // from `refine_aw_lm`). Excluded from std metrics; included in
    // error metrics (the pose itself is still a best-effort estimate).
    frames_singular_covariance: usize,
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

/// Optional corner-refinement strategy applied to the cloned `DetectionBatch`
/// before it is passed to `BoardEstimator`. The default `None` reproduces the
/// historical harness behaviour. `ForstnerSaddle` replaces every detector
/// corner with the direct Förstner saddle solution over a structure-tensor
/// window. It is kept as a research anchor: applying Förstner to ChArUco
/// saddles regresses pose by 4.5× on `board_charuco_v1_golden_forstner` vs
/// the homography-projection-only path that `CharucoRefiner` now ships.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RefinementStrategy {
    #[default]
    None,
    ForstnerSaddle,
}

struct BoardRegressionHarness {
    snapshot_name: String,
    config: DetectorConfig,
    families: Vec<TagFamily>,
    refinement: RefinementStrategy,
}

impl BoardRegressionHarness {
    pub fn new(snapshot_name: impl Into<String>) -> Self {
        Self {
            snapshot_name: snapshot_name.into(),
            config: DetectorConfig::from_profile("standard"),
            families: Vec::new(),
            refinement: RefinementStrategy::default(),
        }
    }

    pub fn with_profile(mut self, profile: &str) -> Self {
        self.config = DetectorConfig::from_profile(profile);
        self
    }

    pub fn with_families(mut self, families: Vec<TagFamily>) -> Self {
        self.families = families;
        self
    }

    pub fn with_refinement(mut self, refinement: RefinementStrategy) -> Self {
        self.refinement = refinement;
        self
    }

    pub fn run(self, provider: &BoardHubProvider) {
        let mut estimator = BoardEstimator::new(Arc::clone(&provider.board_config));
        let outlier_drop_d2_threshold = self.config.outlier_drop_d2_threshold;
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
        let mut frames_singular_covariance = 0;
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
                    false,
                )
                .unwrap();
            let detect_ms = start.elapsed().as_secs_f64() * 1000.0;

            #[cfg(feature = "bench-internals")]
            {
                let mut batch_cloned = detector.bench_api_get_batch_cloned();

                // Tag coverage calculation
                let visible_set: std::collections::HashSet<u32> =
                    entry.visible_tag_ids.iter().copied().collect();
                let detected_valid = batch_cloned
                    .status_mask
                    .iter()
                    .zip(batch_cloned.ids.iter())
                    .filter(|&(state, id)| {
                        *state == locus_core::bench_api::CandidateState::Valid
                            && visible_set.contains(id)
                    })
                    .count();
                total_coverage += if visible_set.is_empty() {
                    1.0
                } else {
                    detected_valid as f64 / visible_set.len() as f64
                };

                // Optional pre-pose refinement: mutate the cloned batch's
                // corners with the chosen strategy. The detector's batch is
                // never touched (we cloned it above).
                if self.refinement == RefinementStrategy::ForstnerSaddle {
                    apply_forstner_to_batch(&mut batch_cloned, &img_view, 3);
                }

                let board_start = std::time::Instant::now();
                let v = batch_cloned.partition(batch_cloned.capacity());
                let board_pose = estimator.estimate(
                    &batch_cloned.view(v),
                    &provider.camera_intrinsics,
                    outlier_drop_d2_threshold,
                );
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
                        quat_from_so3(est.pose.rotation)
                            .angle_to(&gt_q)
                            .to_degrees(),
                    );

                    // NaN-sentinel guard: a singular `JᵀWJ` makes
                    // `refine_aw_lm` emit a NaN-filled covariance.
                    // `f64::NAN.max(0.0) == 0` silently coerces NaN to 0,
                    // so an isfinite check is the only correct gate.
                    // Exclude such frames from std metrics but keep the
                    // pose in error metrics (it is still an estimate).
                    let std_finite = (0..6).all(|i| est.covariance[(i, i)].is_finite());
                    if std_finite {
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
                        frames_singular_covariance += 1;
                    }
                } else {
                    frames_no_estimate += 1;
                }
            }
        }

        let count = t_errors.len() as f64;
        let std_count = t_stds.len() as f64;
        let summary = BoardSummaryMetrics {
            dataset_size: provider.images.len(),
            frames_with_board,
            mean_board_translation_error_m: if count > 0.0 {
                t_errors.iter().sum::<f64>() / count
            } else {
                0.0
            },
            p50_board_translation_error_m: calculate_percentile(&mut t_errors, 0.5),
            p95_board_translation_error_m: calculate_percentile(&mut t_errors, 0.95),
            p99_board_translation_error_m: calculate_percentile(&mut t_errors, 0.99),
            mean_board_rotation_error_deg: if count > 0.0 {
                r_errors.iter().sum::<f64>() / count
            } else {
                0.0
            },
            p50_board_rotation_error_deg: calculate_percentile(&mut r_errors, 0.5),
            p95_board_rotation_error_deg: calculate_percentile(&mut r_errors, 0.95),
            p99_board_rotation_error_deg: calculate_percentile(&mut r_errors, 0.99),
            mean_board_translation_std_m: if std_count > 0.0 {
                t_stds.iter().sum::<f64>() / std_count
            } else {
                0.0
            },
            mean_board_rotation_std_deg: if std_count > 0.0 {
                r_stds.iter().sum::<f64>() / std_count
            } else {
                0.0
            },
            mean_tag_coverage: total_coverage / (provider.images.len() as f64).max(1.0),
            mean_total_ms: total_time / (provider.images.len() as f64).max(1.0),
            frames_no_estimate,
            frames_singular_covariance,
        };

        let report = BoardRegressionReport { summary };
        insta::assert_yaml_snapshot!(self.snapshot_name, report, {
            ".summary.mean_total_ms" => "[DURATION]"
        });
    }
}

/// Direct Förstner saddle solve. Builds the 2×2 normal equations
///   S · p = b,    S = Σ ∇I_i ∇I_iᵀ,   b = Σ ∇I_i ∇I_iᵀ · p_i
/// over a window of `radius` integer pixels around `(cx, cy)`. Returns the
/// analytic minimiser of the gradient-projection residual, or `None` when S
/// is singular (flat window) or the window touches an image boundary.
///
/// Research anchor for the saddle-refinement question. The Newton-on-
/// surrogate-Hessian formulation that `charuco.rs` previously shipped was
/// empirically inert (step magnitude ~1e-5 px below the downstream f32
/// cast); it was deleted in favour of the homography-projection-only
/// path. Applying this Förstner solver instead regresses pose by 4.5× on
/// `regression_board_hub::board_charuco_v1_golden_forstner` — kept as a
/// guard against future "let's add Newton/Förstner back" proposals. See
/// `memory/project_refine_saddle_noop.md` for the full record.
#[cfg(feature = "bench-internals")]
fn forstner_saddle(
    img: &locus_core::ImageView,
    cx: f64,
    cy: f64,
    radius: isize,
) -> Option<(f64, f64)> {
    let cxi = cx.round() as isize;
    let cyi = cy.round() as isize;
    let w = img.width as isize;
    let h = img.height as isize;
    let stride = img.stride as isize;

    let x_start = (cxi - radius).max(1);
    let x_end = (cxi + radius).min(w - 2);
    let y_start = (cyi - radius).max(1);
    let y_end = (cyi + radius).min(h - 2);
    if x_end < x_start || y_end < y_start {
        return None;
    }

    let mut s00 = 0.0_f64;
    let mut s01 = 0.0_f64;
    let mut s11 = 0.0_f64;
    let mut b0 = 0.0_f64;
    let mut b1 = 0.0_f64;

    for py_i in y_start..=y_end {
        for px_i in x_start..=x_end {
            let row_above = (py_i - 1) * stride;
            let row_mid = py_i * stride;
            let row_below = (py_i + 1) * stride;
            let p00 = i32::from(img.data[(row_above + (px_i - 1)) as usize]);
            let p01 = i32::from(img.data[(row_above + px_i) as usize]);
            let p02 = i32::from(img.data[(row_above + (px_i + 1)) as usize]);
            let p10 = i32::from(img.data[(row_mid + (px_i - 1)) as usize]);
            let p12 = i32::from(img.data[(row_mid + (px_i + 1)) as usize]);
            let p20 = i32::from(img.data[(row_below + (px_i - 1)) as usize]);
            let p21 = i32::from(img.data[(row_below + px_i) as usize]);
            let p22 = i32::from(img.data[(row_below + (px_i + 1)) as usize]);

            let gx = f64::from((p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20));
            let gy = f64::from((p22 + 2 * p21 + p20) - (p02 + 2 * p01 + p00));

            let gxx = gx * gx;
            let gxy = gx * gy;
            let gyy = gy * gy;
            let pxf = px_i as f64;
            let pyf = py_i as f64;

            s00 += gxx;
            s01 += gxy;
            s11 += gyy;
            b0 += gxx * pxf + gxy * pyf;
            b1 += gxy * pxf + gyy * pyf;
        }
    }

    let det = s00 * s11 - s01 * s01;
    if det.abs() < 1e-6 {
        return None;
    }
    let inv_det = 1.0 / det;
    let px_out = (s11 * b0 - s01 * b1) * inv_det;
    let py_out = (-s01 * b0 + s00 * b1) * inv_det;
    Some((px_out, py_out))
}

/// Replace each valid corner in the cloned batch with its Förstner saddle.
/// Drops the refinement if the solve fails or if the post-refinement position
/// drifts more than `MAX_DRIFT_PX` from the detector seed (defensive guard
/// against catastrophic Newton jumps; conservative at 5 px).
#[cfg(feature = "bench-internals")]
fn apply_forstner_to_batch(
    batch: &mut locus_core::bench_api::DetectionBatch,
    img: &locus_core::ImageView,
    radius: isize,
) {
    const MAX_DRIFT_PX: f64 = 5.0;
    let cap = batch.capacity();
    for i in 0..cap {
        if batch.status_mask[i] != locus_core::bench_api::CandidateState::Valid {
            continue;
        }
        for k in 0..4 {
            let seed = batch.corners[i][k];
            let (sx, sy) = (seed.x as f64, seed.y as f64);
            if let Some((rx, ry)) = forstner_saddle(img, sx, sy, radius) {
                let d2 = (rx - sx).powi(2) + (ry - sy).powi(2);
                if d2 <= MAX_DRIFT_PX * MAX_DRIFT_PX {
                    batch.corners[i][k] = locus_core::bench_api::Point2f {
                        x: rx as f32,
                        y: ry as f32,
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_charuco_golden_v1_metadata() {
        let dataset_path = common::resolve_hub_root("charuco_golden_v1_1920x1080");

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
        let dataset_path = common::resolve_hub_root("aprilgrid_golden_v1_1920x1080");

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

    fn resolve_hub_dataset(config_name: &str) -> Option<std::path::PathBuf> {
        let hub_dir = std::env::var("LOCUS_HUB_DATASET_DIR")
            .unwrap_or_else(|_| "tests/data/hub_cache".to_string());
        let dataset_path = common::resolve_hub_root(&hub_dir).join(config_name);
        if dataset_path.exists() {
            Some(dataset_path)
        } else {
            println!("Skipping {config_name}: dataset not found at {dataset_path:?}");
            None
        }
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_board_charuco_v1_golden() {
        let Some(dataset_path) = resolve_hub_dataset("charuco_golden_v1_1920x1080") else {
            return;
        };
        let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
        BoardRegressionHarness::new("board_charuco_v1_golden")
            .with_families(vec![TagFamily::ArUco6x6_250])
            .run(&provider);
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_board_aprilgrid_v1_golden() {
        let Some(dataset_path) = resolve_hub_dataset("aprilgrid_golden_v1_1920x1080") else {
            return;
        };
        let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
        BoardRegressionHarness::new("board_aprilgrid_v1_golden")
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(&provider);
    }

    // ── Förstner saddle refinement A/B ──────────────────────────────────────
    //
    // These tests share the harness with the baseline above but apply direct
    // Förstner saddle refinement to detector corners before pose estimation.
    // The resulting snapshots can be diff'd against the baseline to quantify
    // whether saddle refinement helps or hurts on each board geometry.
    //
    // Expectation, based on the AprilGrid spike in
    // `tests/aprilgrid_saddle_spike.rs`:
    //   * `board_aprilgrid_v1_golden_forstner` — rotation regresses (bit-pattern
    //     PSF leakage biases per-corner positions outward by ~1 px in a
    //     k-index-dependent direction; shared linking-square corners get
    //     inconsistent placements).
    //   * `board_charuco_v1_golden_forstner` — open question; ChArUco corners
    //     are AprilTag corners on a chessboard background, so the bit-pattern
    //     asymmetry still applies (the bit pattern is inside the marker, on
    //     the chessboard's white square).
    //
    // The snapshots are anchors for the AprilGrid refinement plan / the
    // `project_refine_saddle_noop.md` finding.

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_board_charuco_v1_golden_forstner() {
        let Some(dataset_path) = resolve_hub_dataset("charuco_golden_v1_1920x1080") else {
            return;
        };
        let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
        BoardRegressionHarness::new("board_charuco_v1_golden_forstner")
            .with_families(vec![TagFamily::ArUco6x6_250])
            .with_refinement(RefinementStrategy::ForstnerSaddle)
            .run(&provider);
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_board_aprilgrid_v1_golden_forstner() {
        let Some(dataset_path) = resolve_hub_dataset("aprilgrid_golden_v1_1920x1080") else {
            return;
        };
        let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
        BoardRegressionHarness::new("board_aprilgrid_v1_golden_forstner")
            .with_families(vec![TagFamily::AprilTag36h11])
            .with_refinement(RefinementStrategy::ForstnerSaddle)
            .run(&provider);
    }

    // ── CharucoRefiner integration test ─────────────────────────────────────
    //
    // Until 2026-05-16 there was no integration coverage for `CharucoRefiner`
    // (the saddle-refinement pose path exposed to Python users via the
    // `locus_tag.CharucoRefiner` wrapper). The existing `regression_board_*`
    // tests above exercise `BoardEstimator` (the tag-corner path) — even on
    // the ChArUco dataset.
    //
    // `CharucoRefiner` now ships the homography-projection-only saddle path
    // (the iterative Newton refinement was deleted after `regression_board_hub::
    // board_charuco_v1_golden_forstner` falsified the principled Förstner
    // replacement at 4.5× pose regression; see `memory/project_refine_saddle_noop.md`).
    // This test anchors the production behaviour so silently-broken changes
    // to the saddle path are caught against a known baseline.

    /// Per-frame diagnostic for the CharucoRefiner. Prints accepted-saddle
    /// counts, rejection telemetry, and the homography-vs-detected-corner
    /// geometry for the first few frames. Used to root-cause the catastrophic
    /// snapshot numbers; run with `--ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic; run explicitly with --ignored --nocapture"]
    #[cfg(feature = "bench-internals")]
    fn diagnose_charuco_refiner() {
        use locus_core::bench_api::{CharucoBatch, CharucoRefiner};

        let Some(dataset_path) = resolve_hub_dataset("charuco_golden_v1_1920x1080") else {
            return;
        };
        let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
        let charuco_topo = provider.charuco_config.clone().expect("charuco_config");

        let mut detector =
            locus_core::Detector::with_config(DetectorConfig::from_profile("standard"));
        detector.set_families(&[TagFamily::ArUco6x6_250]);

        let mut refiner = CharucoRefiner::new((*charuco_topo).clone());
        let mut out_batch: CharucoBatch = refiner.new_batch_with_telemetry();

        println!(
            "\n=== Diagnose CharucoRefiner ===\nboard: {}x{}, square={:.3}m marker={:.3}m, num_saddles={}",
            charuco_topo.rows,
            charuco_topo.cols,
            charuco_topo.square_length,
            charuco_topo.marker_length,
            charuco_topo.saddle_points.len(),
        );

        for (i, entry) in provider.images.iter().enumerate().take(5) {
            let img_path = provider.base_dir.join("images").join(&entry.filename);
            if !img_path.exists() {
                continue;
            }
            let img = image::open(&img_path).unwrap().into_luma8();
            let (w, h) = img.dimensions();
            let raw = img.into_raw();
            let img_view =
                locus_core::ImageView::new(&raw, w as usize, h as usize, w as usize).unwrap();

            let _ = detector
                .detect(
                    &img_view,
                    Some(&provider.camera_intrinsics),
                    Some(provider.board_config.marker_length),
                    false,
                )
                .unwrap();
            let mut batch_cloned = detector.bench_api_get_batch_cloned();
            let n_detections = batch_cloned
                .status_mask
                .iter()
                .filter(|s| **s == locus_core::bench_api::CandidateState::Valid)
                .count();

            // Dump first 3 detected tags: ID, corners, homography (row-major
            // view), and what the homography maps canonical [(-1,-1)..(1,1)]
            // to. If the homography is being applied transposed elsewhere,
            // this will show whether the stored matrix is correct.
            let mut shown = 0;
            for i_det in 0..batch_cloned.capacity() {
                if batch_cloned.status_mask[i_det] != locus_core::bench_api::CandidateState::Valid {
                    continue;
                }
                if shown >= 3 {
                    break;
                }
                let cs = &batch_cloned.corners[i_det];
                let h = &batch_cloned.homographies[i_det].data;
                println!(
                    "    tag_id={}, corners=[({:.1},{:.1}),({:.1},{:.1}),({:.1},{:.1}),({:.1},{:.1})]",
                    batch_cloned.ids[i_det],
                    cs[0].x,
                    cs[0].y,
                    cs[1].x,
                    cs[1].y,
                    cs[2].x,
                    cs[2].y,
                    cs[3].x,
                    cs[3].y
                );
                println!(
                    "      H data (raw, 9 elems): [{:.3},{:.3},{:.3}, {:.3},{:.3},{:.3}, {:.3},{:.3},{:.3}]",
                    h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]
                );
                // Project each canonical corner via apply_homography_col_major-style read
                let project = |u: f64, v: f64| -> (f64, f64) {
                    let xh = h[0] as f64 * u + h[3] as f64 * v + h[6] as f64;
                    let yh = h[1] as f64 * u + h[4] as f64 * v + h[7] as f64;
                    let wh = h[2] as f64 * u + h[5] as f64 * v + h[8] as f64;
                    (xh / wh, yh / wh)
                };
                let (p_tl_x, p_tl_y) = project(-1.0, -1.0);
                let (p_br_x, p_br_y) = project(1.0, 1.0);
                println!(
                    "      H proj: (-1,-1)→({:.1},{:.1}) (+1,+1)→({:.1},{:.1})",
                    p_tl_x, p_tl_y, p_br_x, p_br_y
                );
                shown += 1;
            }

            let v = batch_cloned.partition(batch_cloned.capacity());
            refiner.estimate_with_telemetry(
                &batch_cloned.view(v),
                &img_view,
                &provider.camera_intrinsics,
                &mut out_batch,
                0.0,
            );

            println!("\nframe {i} ({}):", entry.filename);
            println!("  detections (Valid): {n_detections}");
            println!("  saddles accepted:   {}", out_batch.count);
            if let Some(telem) = &out_batch.telemetry {
                println!("  saddles rejected:   {}", telem.count);
                for j in 0..telem.count.min(8) {
                    let p = telem.rejected_predictions[j];
                    let det = telem.rejected_determinants[j];
                    println!(
                        "    rejection {j}: predicted=({:.2},{:.2}) det={:.3e}",
                        p.x, p.y, det
                    );
                }
            }
            // Print first accepted saddles
            for j in 0..out_batch.count.min(5) {
                let sid = out_batch.saddle_ids[j];
                let img_p = out_batch.saddle_image_pts[j];
                let obj_p = out_batch.saddle_obj_pts[j];
                println!(
                    "    accepted {j}: saddle_id={sid}, img=({:.2},{:.2}), obj=({:.3},{:.3},{:.3})",
                    img_p.x, img_p.y, obj_p[0], obj_p[1], obj_p[2]
                );
            }
            if let Some(pose) = &out_batch.board_pose {
                let t = pose.pose.translation;
                let q = quat_from_so3(pose.pose.rotation);
                let gt_t = nalgebra::Vector3::new(
                    entry.board_pose.translation[0],
                    entry.board_pose.translation[1],
                    entry.board_pose.translation[2],
                );
                let gt_q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    entry.board_pose.rotation_quaternion[0],
                    entry.board_pose.rotation_quaternion[1],
                    entry.board_pose.rotation_quaternion[2],
                    entry.board_pose.rotation_quaternion[3],
                ));
                println!(
                    "  pose: t=({:.4},{:.4},{:.4}), gt_t=({:.4},{:.4},{:.4})",
                    t[0], t[1], t[2], gt_t[0], gt_t[1], gt_t[2]
                );
                println!(
                    "  t_err={:.5}m, r_err={:.4}deg",
                    (t - gt_t).norm(),
                    q.angle_to(&gt_q).to_degrees()
                );
            } else {
                println!("  pose: NONE");
            }
        }
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_charuco_refiner_v1_golden() {
        use locus_core::bench_api::{CharucoBatch, CharucoRefiner};

        let Some(dataset_path) = resolve_hub_dataset("charuco_golden_v1_1920x1080") else {
            return;
        };
        let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
        let Some(charuco_topo) = provider.charuco_config.clone() else {
            panic!("charuco_golden_v1 should populate provider.charuco_config");
        };

        let mut detector =
            locus_core::Detector::with_config(DetectorConfig::from_profile("standard"));
        detector.set_families(&[TagFamily::ArUco6x6_250]);

        let mut refiner = CharucoRefiner::new((*charuco_topo).clone());
        let mut out_batch: CharucoBatch = refiner.new_batch();

        let mut t_errors = Vec::new();
        let mut r_errors = Vec::new();
        let mut t_stds = Vec::new();
        let mut r_stds = Vec::new();
        let mut frames_with_board = 0;
        let mut frames_no_estimate = 0;
        let mut frames_singular_covariance = 0;
        let mut total_coverage = 0.0;
        let mut total_time = 0.0;

        for (i, entry) in provider.images.iter().enumerate() {
            if i % 30 == 0 {
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
                    false,
                )
                .unwrap();
            let detect_ms = start.elapsed().as_secs_f64() * 1000.0;

            let mut batch_cloned = detector.bench_api_get_batch_cloned();

            let visible_set: std::collections::HashSet<u32> =
                entry.visible_tag_ids.iter().copied().collect();
            let detected_valid = batch_cloned
                .status_mask
                .iter()
                .zip(batch_cloned.ids.iter())
                .filter(|&(state, id)| {
                    *state == locus_core::bench_api::CandidateState::Valid
                        && visible_set.contains(id)
                })
                .count();
            total_coverage += if visible_set.is_empty() {
                1.0
            } else {
                detected_valid as f64 / visible_set.len() as f64
            };

            let refine_start = std::time::Instant::now();
            let v = batch_cloned.partition(batch_cloned.capacity());
            refiner.estimate(
                &batch_cloned.view(v),
                &img_view,
                &provider.camera_intrinsics,
                &mut out_batch,
                0.0,
            );
            let refine_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
            total_time += detect_ms + refine_ms;

            if let Some(ref est) = out_batch.board_pose {
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
                    quat_from_so3(est.pose.rotation)
                        .angle_to(&gt_q)
                        .to_degrees(),
                );

                // NaN-sentinel guard (see charuco run for rationale):
                // `f64::NAN.max(0.0) == 0` silently coerces NaN cov to 0;
                // the explicit isfinite check excludes singular-Hessian
                // frames from std metrics without poisoning the pose
                // error statistics.
                let std_finite = (0..6).all(|i| est.covariance[(i, i)].is_finite());
                if std_finite {
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
                    frames_singular_covariance += 1;
                }
            } else {
                frames_no_estimate += 1;
            }
        }

        let count = t_errors.len() as f64;
        let std_count = t_stds.len() as f64;
        let summary = BoardSummaryMetrics {
            dataset_size: provider.images.len(),
            frames_with_board,
            mean_board_translation_error_m: if count > 0.0 {
                t_errors.iter().sum::<f64>() / count
            } else {
                0.0
            },
            p50_board_translation_error_m: calculate_percentile(&mut t_errors, 0.5),
            p95_board_translation_error_m: calculate_percentile(&mut t_errors, 0.95),
            p99_board_translation_error_m: calculate_percentile(&mut t_errors, 0.99),
            mean_board_rotation_error_deg: if count > 0.0 {
                r_errors.iter().sum::<f64>() / count
            } else {
                0.0
            },
            p50_board_rotation_error_deg: calculate_percentile(&mut r_errors, 0.5),
            p95_board_rotation_error_deg: calculate_percentile(&mut r_errors, 0.95),
            p99_board_rotation_error_deg: calculate_percentile(&mut r_errors, 0.99),
            mean_board_translation_std_m: if std_count > 0.0 {
                t_stds.iter().sum::<f64>() / std_count
            } else {
                0.0
            },
            mean_board_rotation_std_deg: if std_count > 0.0 {
                r_stds.iter().sum::<f64>() / std_count
            } else {
                0.0
            },
            mean_tag_coverage: total_coverage / (provider.images.len() as f64).max(1.0),
            mean_total_ms: total_time / (provider.images.len() as f64).max(1.0),
            frames_no_estimate,
            frames_singular_covariance,
        };

        let report = BoardRegressionReport { summary };
        insta::assert_yaml_snapshot!("charuco_refiner_v1_golden", report, {
            ".summary.mean_total_ms" => "[DURATION]"
        });
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_board_aprilgrid_v1_golden_high_accuracy() {
        let Some(dataset_path) = resolve_hub_dataset("aprilgrid_golden_v1_1920x1080") else {
            return;
        };
        let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
        BoardRegressionHarness::new("board_aprilgrid_v1_golden_high_accuracy")
            .with_profile("high_accuracy")
            .with_families(vec![TagFamily::AprilTag36h11])
            .run(&provider);
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn regression_board_charuco_v1_golden_high_accuracy() {
        let Some(dataset_path) = resolve_hub_dataset("charuco_golden_v1_1920x1080") else {
            return;
        };
        let provider = BoardHubProvider::new(&dataset_path).expect("failed to load provider");
        BoardRegressionHarness::new("board_charuco_v1_golden_high_accuracy")
            .with_profile("high_accuracy")
            .with_families(vec![TagFamily::ArUco6x6_250])
            .run(&provider);
    }
}
