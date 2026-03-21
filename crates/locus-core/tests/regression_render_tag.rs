//! Regression tests for rendered tags from the hub.
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

use locus_core::{
    CameraIntrinsics, DetectOptions, Detector, DetectorConfig, ImageView, Pose, PoseEstimationMode,
    TagFamily,
};
use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

mod common;

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
            Self::PlainBoard => DetectorConfig::production_default(),
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
    precision: f64,
    #[serde(serialize_with = "serialize_rmse")]
    avg_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    reprojection_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_hamming: f64,
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
    mean_precision: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_reprojection_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p50_translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p90_translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p99_translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p50_rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p90_rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    p99_rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    mean_hamming: f64,
    mean_total_ms: f64,
    geometry_violations: usize,
    worst_offenders: Vec<Offender>,
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
// Evaluation Engine
// ============================================================================

/// Ground Truth for a single image
#[derive(Clone)]
pub struct GroundTruth {
    pub tags: HashMap<u32, [[f64; 2]; 4]>,
    pub poses: HashMap<u32, Pose>,
    pub intrinsics: Option<CameraIntrinsics>,
    pub tag_size: Option<f64>,
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

    pub fn with_options(mut self, options: DetectOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_refinement_mode(mut self, mode: locus_core::config::CornerRefinementMode) -> Self {
        self.config.refinement_mode = mode;
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
        let mut detector = Detector::with_config(self.config);
        if !self.options.families.is_empty() {
            detector.set_families(&self.options.families);
        }
        let mut results = BTreeMap::new();

        // Aggregators
        let mut total_recall = 0.0;
        let mut total_precision = 0.0;
        let mut total_rmse = 0.0;
        let mut total_repro_rmse = 0.0;
        let mut total_hamming = 0.0;
        let mut total_time = 0.0;
        let mut count = 0;

        // Individual errors for percentiles
        let mut translation_errors = Vec::new();
        let mut rotation_errors = Vec::new();
        let mut geometry_violations = 0;

        for (filename, data, width, height, gt) in provider.iter() {
            let img = ImageView::new(&data, width, height, width).expect("valid image");

            let intrinsics = gt.intrinsics.or(self.options.intrinsics);
            let tag_size = gt.tag_size.or(self.options.tag_size);

            let start = std::time::Instant::now();
            let detections = detector
                .detect(
                    &img,
                    intrinsics.as_ref(),
                    tag_size,
                    self.options.pose_estimation_mode,
                    false,
                )
                .expect("detection failed");
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;

            // --- Metrics Calculation ---
            let mut image_rmse_sum = 0.0;
            let mut image_repro_rmse_sum = 0.0;
            let mut image_translation_error_sum = 0.0;
            let mut image_rotation_error_sum = 0.0;
            let mut image_hamming_sum = 0.0;
            let mut match_count = 0;
            let mut pose_match_count = 0;
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

                    if dist_sq < 100.0 * 100.0 {
                        // Strict Index-to-Index Comparison (No circular shifting)
                        let mut rmse_sq = 0.0;
                        for k in 0..4 {
                            rmse_sq += (det_corners_f64[k][0] - gt_corners[k][0]).powi(2)
                                + (det_corners_f64[k][1] - gt_corners[k][1]).powi(2);
                        }
                        image_rmse_sum += (rmse_sq / 4.0).sqrt();

                        image_hamming_sum += f64::from(detections.error_rates[i]);
                        match_count += 1;
                        found_ids.insert(det_id);

                        // --- Pose Metrics ---
                        let det_pose_data = detections.poses[i].data;
                        if det_pose_data[2] > 0.0 {
                            let det_t = Vector3::new(
                                f64::from(det_pose_data[0]),
                                f64::from(det_pose_data[1]),
                                f64::from(det_pose_data[2]),
                            );
                            let det_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                                f64::from(det_pose_data[6]), // w
                                f64::from(det_pose_data[3]), // x
                                f64::from(det_pose_data[4]), // y
                                f64::from(det_pose_data[5]), // z
                            ));

                            if let (Some(gt_pose), Some(intr), Some(size)) =
                                (gt.poses.get(&det_id), intrinsics, tag_size)
                            {
                                // --- Rigid Translation Shift (Center to Top-Left) ---
                                // Generator: Origin at center.
                                // Detector: Origin at Top-Left.
                                // Local offset in Object Frame (Y-Down, Z-In): TL is [-s/2, -s/2, 0]
                                let s_half = size * 0.5;
                                let v_offset_obj = Vector3::new(-s_half, -s_half, 0.0);

                                // Transform offset to Camera Frame: t_TL = t_center + R_gt * v_offset_obj
                                let t_gt_tl = gt_pose.translation + gt_pose.rotation * v_offset_obj;

                                // --- Geometry Safeguard Gate ---
                                // 1. Z-Polarity: Tag must remain in front of the camera.
                                // 2. Z-Depth Consistency: Planar shift cannot exceed half-diagonal (~0.707 * size).
                                // We allow some epsilon for slightly non-orthonormal matrices if they exist.
                                let z_delta = (t_gt_tl.z - gt_pose.translation.z).abs();
                                let max_physically_possible_z_shift = size * 0.75; // ~0.707 * size + epsilon

                                if t_gt_tl.z <= 0.0 || z_delta > max_physically_possible_z_shift {
                                    println!(
                                        "CRITICAL GEOMETRY VIOLATION: Image: {}, Tag ID: {}, Center Z: {:.4}, TL Z: {:.4}, Delta: {:.4}, Limit: {:.4}",
                                        filename,
                                        det_id,
                                        gt_pose.translation.z,
                                        t_gt_tl.z,
                                        z_delta,
                                        max_physically_possible_z_shift
                                    );
                                    geometry_violations += 1;
                                    continue; // Skip this pose comparison as GT is corrupt/invalid.
                                }

                                let q_gt = UnitQuaternion::from_matrix(&gt_pose.rotation);
                                let r_err = det_q.angle_to(&q_gt);
                                let t_err = (det_t - t_gt_tl).norm();

                                image_translation_error_sum += t_err;
                                image_rotation_error_sum += r_err;
                                pose_match_count += 1;

                                translation_errors.push(t_err);
                                rotation_errors.push(r_err.to_degrees());

                                // --- Reprojection Error (vs Ground Truth Corners) ---
                                let s = size;
                                // Canonical Object Frame: Origin at Top-Left, +X Right, +Y Down
                                let model_corners = [
                                    Vector3::new(0.0, 0.0, 0.0), // 0: TL
                                    Vector3::new(s, 0.0, 0.0),   // 1: TR
                                    Vector3::new(s, s, 0.0),     // 2: BR
                                    Vector3::new(0.0, s, 0.0),   // 3: BL
                                ];

                                // Estimated pose in detector frame
                                let est_pose =
                                    Pose::new(det_q.to_rotation_matrix().into_inner(), det_t);

                                // Strict element-wise reprojection RMSE
                                let mut repro_rmse_sq = 0.0;
                                for k in 0..4 {
                                    let proj = est_pose.project(&model_corners[k], &intr);
                                    repro_rmse_sq += (proj[0] - gt_corners[k][0]).powi(2)
                                        + (proj[1] - gt_corners[k][1]).powi(2);
                                }
                                image_repro_rmse_sum += (repro_rmse_sq / 4.0).sqrt();
                            }
                        }
                    }
                }
            }

            let recall = if gt.tags.is_empty() {
                1.0
            } else {
                found_ids.len() as f64 / gt.tags.len() as f64
            };
            let precision = if detections.is_empty() {
                1.0
            } else {
                f64::from(match_count) / detections.len() as f64
            };
            let avg_rmse = if match_count > 0 {
                image_rmse_sum / f64::from(match_count)
            } else {
                0.0
            };
            let avg_repro_rmse = if pose_match_count > 0 {
                image_repro_rmse_sum / f64::from(pose_match_count)
            } else {
                0.0
            };
            let avg_translation_error = if pose_match_count > 0 {
                image_translation_error_sum / f64::from(pose_match_count)
            } else {
                0.0
            };
            let avg_rotation_error = if pose_match_count > 0 {
                image_rotation_error_sum.to_degrees() / f64::from(pose_match_count)
            } else {
                0.0
            };
            let avg_hamming = if match_count > 0 {
                image_hamming_sum / f64::from(match_count)
            } else {
                0.0
            };

            total_recall += recall;
            total_precision += precision;
            total_rmse += avg_rmse;
            total_repro_rmse += avg_repro_rmse;
            total_hamming += avg_hamming;
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
                    precision,
                    avg_rmse,
                    reprojection_rmse: avg_repro_rmse,
                    translation_error: avg_translation_error,
                    rotation_error: avg_rotation_error,
                    mean_hamming: avg_hamming,
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
                mean_precision: total_precision / count as f64,
                mean_rmse: total_rmse / count as f64,
                mean_reprojection_rmse: total_repro_rmse / count as f64,
                p50_translation_error: calculate_percentile(&mut translation_errors, 0.5),
                p90_translation_error: calculate_percentile(&mut translation_errors, 0.9),
                p99_translation_error: calculate_percentile(&mut translation_errors, 0.99),
                p50_rotation_error: calculate_percentile(&mut rotation_errors, 0.5),
                p90_rotation_error: calculate_percentile(&mut rotation_errors, 0.9),
                p99_rotation_error: calculate_percentile(&mut rotation_errors, 0.99),
                mean_hamming: total_hamming / count as f64,
                mean_total_ms: total_time / count as f64,
                geometry_violations,
                worst_offenders: offenders.into_iter().take(5).collect(),
            },
        };

        println!("=== {} Results ===", self.snapshot_name);
        println!("  Images: {count}");
        println!("  Recall: {:.2}%", report.summary.mean_recall * 100.0);
        println!("  Precision: {:.2}%", report.summary.mean_precision * 100.0);
        println!(
            "  Geometry Violations: {}",
            report.summary.geometry_violations
        );
        println!("  RMSE:   {:.4} px", report.summary.mean_rmse);
        println!(
            "  Repro RMSE: {:.4} px",
            report.summary.mean_reprojection_rmse
        );
        println!("  Trans P50: {:.4} m", report.summary.p50_translation_error);
        println!("  Rot P50: {:.4} deg", report.summary.p50_rotation_error);
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
    #[serde(alias = "image_id")]
    image_filename: String,
    tag_id: u32,
    corners: [[f64; 2]; 4],
    position: [f64; 3],
    rotation_quaternion: [f64; 4], // [w, x, y, z]
    /// Per-detection intrinsics (optional)
    k_matrix: Option<[[f64; 3]; 3]>,
    /// Per-detection tag size (optional)
    tag_size_mm: Option<f64>,
}

impl HubProvider {
    fn new(dataset_dir: &std::path::Path) -> Option<Self> {
        let rich_path = dataset_dir.join("rich_truth.json");
        let _jsonl_path = dataset_dir.join("annotations.jsonl");
        let images_dir = dataset_dir.join("images");

        if !images_dir.exists() {
            return None;
        }

        let mut gt_map: HashMap<String, GroundTruth> = HashMap::new();

        // Load all metadata from rich_truth.json as requested.
        if rich_path.exists() {
            let file = std::fs::File::open(&rich_path).ok()?;
            let entries: Vec<HubEntry> = serde_json::from_reader(file).ok()?;

            for entry in entries {
                let image_name = &entry.image_filename;

                let fname = if std::path::Path::new(image_name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
                {
                    image_name.clone()
                } else {
                    format!("{image_name}.png")
                };

                let gt = gt_map.entry(fname.clone()).or_insert_with(|| GroundTruth {
                    tags: HashMap::new(),
                    poses: HashMap::new(),
                    intrinsics: None,
                    tag_size: None,
                });

                gt.tags.insert(entry.tag_id, entry.corners);

                if gt.intrinsics.is_none()
                    && let Some(k) = entry.k_matrix
                {
                    gt.intrinsics = Some(CameraIntrinsics::new(k[0][0], k[1][1], k[0][2], k[1][2]));
                }

                if gt.tag_size.is_none()
                    && let Some(size_mm) = entry.tag_size_mm
                {
                    gt.tag_size = Some(size_mm / 1000.0);
                }

                let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    entry.rotation_quaternion[0], // w
                    entry.rotation_quaternion[1], // x
                    entry.rotation_quaternion[2], // y
                    entry.rotation_quaternion[3], // z
                ))
                .to_rotation_matrix();

                let pose = Pose {
                    rotation: *rotation.matrix(),
                    translation: Vector3::new(
                        entry.position[0],
                        entry.position[1],
                        entry.position[2],
                    ),
                };

                gt.poses.insert(entry.tag_id, pose);
            }
        } else {
            return None;
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
        let iter = self.image_names.iter().filter_map(move |fname| {
            let img_path = base_dir.join("images").join(fname);
            if !img_path.exists() {
                return None;
            }
            let img = image::open(&img_path).expect("load hub image").into_luma8();
            let (w, h) = img.dimensions();

            let gt = self.gt_map.get(fname).unwrap().clone();

            Some((fname.clone(), img.into_raw(), w as usize, h as usize, gt))
        });
        Box::new(iter)
    }
}

// ============================================================================
// Test Runners
// ============================================================================

/// Resolves the hub dataset root directory, mirroring the `common::resolve_dataset_root`
/// pattern: relative paths are anchored to `CARGO_MANIFEST_DIR` (compile-time absolute),
/// not the process CWD (which cargo sets to the package root, not the workspace root).
fn resolve_hub_root(hub_dir: &str) -> PathBuf {
    let path = PathBuf::from(hub_dir);
    if path.is_absolute() {
        return path;
    }
    // Relative: first try directly (works when an absolute env var was given
    // or when CWD happens to be the workspace root).
    if path.is_dir() {
        return std::fs::canonicalize(&path).unwrap_or(path);
    }
    // Resolve from the crate manifest dir so it works regardless of CWD.
    // CARGO_MANIFEST_DIR for locus-core = <workspace>/crates/locus-core.
    // Joining "../../" reaches the workspace root.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let from_workspace = manifest.join("../../").join(&path);
    if from_workspace.is_dir() {
        return from_workspace;
    }
    path // fallback — will fail on `exists()` with a clear skip message
}

fn run_hub_test(
    config_name: &str,
    family: TagFamily,
    mode: PoseEstimationMode,
    refinement: Option<locus_core::config::CornerRefinementMode>,
) {
    if let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") {
        let root = resolve_hub_root(&hub_dir);
        let dataset_path = root.join(config_name);

        if !dataset_path.exists() {
            println!("Dataset not found in cache: {config_name}. Skipping.");
            return;
        }

        if let Some(provider) = HubProvider::new(&dataset_path) {
            let mut options = DetectOptions::default();
            let metadata_path = dataset_path.join("provenance.json");
            let rich_path = dataset_path.join("rich_truth.json");

            if metadata_path.exists() {
                let metadata_str = std::fs::read_to_string(metadata_path).unwrap();
                let meta: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();

                if let Some(intrinsics) = meta.get("camera_intrinsics") {
                    let fx = intrinsics["fx"].as_f64().unwrap();
                    let fy = intrinsics["fy"].as_f64().unwrap();
                    let cx = intrinsics["cx"].as_f64().unwrap();
                    let cy = intrinsics["cy"].as_f64().unwrap();
                    options.intrinsics = Some(CameraIntrinsics::new(fx, fy, cx, cy));
                }

                if let Some(tag_size_mm) = meta.get("tag_size_mm") {
                    // New API: tag_size_mm is the physical edge length of the black border.
                    options.tag_size = Some(tag_size_mm.as_f64().unwrap() / 1000.0);
                }
            }

            // Fallback to rich_truth.json if intrinsics or tag_size are still missing
            if (options.intrinsics.is_none() || options.tag_size.is_none()) && rich_path.exists() {
                let file = std::fs::File::open(&rich_path).unwrap();
                let entries: Vec<HubEntry> = serde_json::from_reader(file).unwrap();
                if let Some(first) = entries.first() {
                    if options.intrinsics.is_none()
                        && let Some(k) = first.k_matrix
                    {
                        options.intrinsics =
                            Some(CameraIntrinsics::new(k[0][0], k[1][1], k[0][2], k[1][2]));
                    }
                    if options.tag_size.is_none()
                        && let Some(size_mm) = first.tag_size_mm
                    {
                        options.tag_size = Some(size_mm / 1000.0);
                    }
                }
            }

            options.pose_estimation_mode = mode;

            let mode_suffix = match mode {
                PoseEstimationMode::Fast => "_fast",
                PoseEstimationMode::Accurate => "",
            };

            let refinement_suffix =
                if let Some(locus_core::config::CornerRefinementMode::Gwlf) = refinement {
                    "_gwlf"
                } else {
                    ""
                };

            let snapshot = format!(
                "hub_{}{}{}",
                provider.name(),
                mode_suffix,
                refinement_suffix
            );
            let mut harness = RegressionHarness::new(snapshot)
                .with_preset(ConfigPreset::PlainBoard)
                .with_families(vec![family])
                .with_options(options);

            if let Some(r) = refinement {
                harness = harness.with_refinement_mode(r);
            }

            harness.run(provider);
        }
    } else {
        println!("Skipping hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
    }
}

fn run_hub_test_tuned(
    config_name: &str,
    family: TagFamily,
    mode: PoseEstimationMode,
    snapshot_suffix: &str,
    max_elongation: f64,
    min_density: f64,
    quad_mode: locus_core::config::QuadExtractionMode,
) {
    run_hub_test_tuned_r(
        config_name,
        family,
        mode,
        snapshot_suffix,
        max_elongation,
        min_density,
        quad_mode,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_hub_test_tuned_r(
    config_name: &str,
    family: TagFamily,
    mode: PoseEstimationMode,
    snapshot_suffix: &str,
    max_elongation: f64,
    min_density: f64,
    quad_mode: locus_core::config::QuadExtractionMode,
    refinement: Option<locus_core::config::CornerRefinementMode>,
) {
    if let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") {
        let root = resolve_hub_root(&hub_dir);
        let dataset_path = root.join(config_name);

        if !dataset_path.exists() {
            println!("Dataset not found in cache: {config_name}. Skipping.");
            return;
        }

        if let Some(provider) = HubProvider::new(&dataset_path) {
            let mut options = DetectOptions::default();
            let metadata_path = dataset_path.join("provenance.json");
            let rich_path = dataset_path.join("rich_truth.json");

            if metadata_path.exists() {
                let metadata_str = std::fs::read_to_string(metadata_path).unwrap();
                let meta: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();
                if let Some(intrinsics) = meta.get("camera_intrinsics") {
                    let fx = intrinsics["fx"].as_f64().unwrap();
                    let fy = intrinsics["fy"].as_f64().unwrap();
                    let cx = intrinsics["cx"].as_f64().unwrap();
                    let cy = intrinsics["cy"].as_f64().unwrap();
                    options.intrinsics = Some(CameraIntrinsics::new(fx, fy, cx, cy));
                }
                if let Some(tag_size_mm) = meta.get("tag_size_mm") {
                    options.tag_size = Some(tag_size_mm.as_f64().unwrap() / 1000.0);
                }
            }
            if (options.intrinsics.is_none() || options.tag_size.is_none()) && rich_path.exists() {
                let file = std::fs::File::open(&rich_path).unwrap();
                let entries: Vec<HubEntry> = serde_json::from_reader(file).unwrap();
                if let Some(first) = entries.first() {
                    if options.intrinsics.is_none()
                        && let Some(k) = first.k_matrix
                    {
                        options.intrinsics =
                            Some(CameraIntrinsics::new(k[0][0], k[1][1], k[0][2], k[1][2]));
                    }
                    if options.tag_size.is_none()
                        && let Some(size_mm) = first.tag_size_mm
                    {
                        options.tag_size = Some(size_mm / 1000.0);
                    }
                }
            }

            options.pose_estimation_mode = mode;
            let mode_suffix = match mode {
                PoseEstimationMode::Fast => "_fast",
                PoseEstimationMode::Accurate => "",
            };
            let snapshot = format!("hub_{}{}{}", provider.name(), mode_suffix, snapshot_suffix);
            let mut harness = RegressionHarness::new(snapshot)
                .with_preset(ConfigPreset::PlainBoard)
                .with_families(vec![family])
                .with_options(options)
                .with_moments_culling(max_elongation, min_density)
                .with_quad_extraction_mode(quad_mode);
            if let Some(r) = refinement {
                harness = harness.with_refinement_mode(r);
            }
            harness.run(provider);
        }
    } else {
        println!("Skipping hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
    }
}

// ── Accurate mode (Structure Tensor + Weighted LM) ───────────────────────────

#[test]
fn regression_hub_tag36h11_640x480() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_640x480");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_640x480",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        None,
    );
}

#[test]
fn regression_hub_tag36h11_640x480_gwlf() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_640x480_gwlf");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_640x480",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        Some(locus_core::config::CornerRefinementMode::Gwlf),
    );
}

#[test]
fn regression_hub_tag36h11_720p() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        None,
    );
}

#[test]
fn regression_hub_tag36h11_720p_gwlf() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_gwlf");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        Some(locus_core::config::CornerRefinementMode::Gwlf),
    );
}

#[test]
fn regression_hub_tag36h11_1080p() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        None,
    );
}

#[test]
fn regression_hub_tag36h11_1080p_gwlf() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_gwlf");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        Some(locus_core::config::CornerRefinementMode::Gwlf),
    );
}

#[test]
fn regression_hub_tag36h11_2160p() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        None,
    );
}

#[test]
fn regression_hub_tag36h11_2160p_gwlf() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p_gwlf");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        Some(locus_core::config::CornerRefinementMode::Gwlf),
    );
}

// ── Fast mode (Trust-Region LM + Huber M-Estimator) ──────────────────────────

#[test]
fn regression_hub_fast_tag36h11_640x480() {
    let _guard = common::telemetry::init("regression_hub_fast_tag36h11_640x480");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_640x480",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Fast,
        None,
    );
}

#[test]
fn regression_hub_fast_tag36h11_720p() {
    let _guard = common::telemetry::init("regression_hub_fast_tag36h11_720p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Fast,
        None,
    );
}

#[test]
fn regression_hub_fast_tag36h11_1080p() {
    let _guard = common::telemetry::init("regression_hub_fast_tag36h11_1080p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Fast,
        None,
    );
}

#[test]
fn regression_hub_fast_tag36h11_2160p() {
    let _guard = common::telemetry::init("regression_hub_fast_tag36h11_2160p");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Fast,
        None,
    );
}

// ── Algorithm tuning variants (moments culling + EDLines) ────────────────────

#[test]
fn regression_hub_tag36h11_720p_moments_culling() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_moments_culling");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_moments_culling",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::ContourRdp,
    );
}

#[test]
fn regression_hub_tag36h11_720p_edlines() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_edlines");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_720p_edlines_moments() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_edlines_moments");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_moments",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_720p_edlines_none() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_edlines_none");
    run_hub_test_tuned_r(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_none",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
        Some(locus_core::config::CornerRefinementMode::None),
    );
}

#[test]
fn regression_hub_tag36h11_720p_edlines_gwlf() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_720p_edlines_gwlf");
    run_hub_test_tuned_r(
        "single_tag_locus_v1_tag36h11_1280x720",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_gwlf",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
        Some(locus_core::config::CornerRefinementMode::Gwlf),
    );
}

#[test]
fn regression_hub_tag36h11_1080p_moments_culling() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_moments_culling");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_moments_culling",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::ContourRdp,
    );
}

#[test]
fn regression_hub_tag36h11_1080p_edlines() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_edlines");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_1080p_edlines_moments() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_1080p_edlines_moments");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_1920x1080",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_moments",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_2160p_moments_culling() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p_moments_culling");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_moments_culling",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::ContourRdp,
    );
}

#[test]
fn regression_hub_tag36h11_2160p_edlines() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p_edlines");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines",
        0.0,
        0.0,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}

#[test]
fn regression_hub_tag36h11_2160p_edlines_moments() {
    let _guard = common::telemetry::init("regression_hub_tag36h11_2160p_edlines_moments");
    run_hub_test_tuned(
        "single_tag_locus_v1_tag36h11_3840x2160",
        TagFamily::AprilTag36h11,
        PoseEstimationMode::Accurate,
        "_edlines_moments",
        15.0,
        0.15,
        locus_core::config::QuadExtractionMode::EdLines,
    );
}
