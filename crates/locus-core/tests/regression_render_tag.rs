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
    CameraIntrinsics, DetectOptions, Detector, DetectorConfig, ImageView, Pose, TagFamily,
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
            Self::PlainBoard => DetectorConfig::builder()
                .refinement_mode(locus_core::config::CornerRefinementMode::Erf)
                .decoder_min_contrast(15.0)
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

    pub fn run(self, provider: impl DatasetProvider) {
        let mut detector = Detector::with_config(self.config);
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

        println!("INTRINSICS: {:?}", self.options.intrinsics);

        for (filename, data, width, height, gt) in provider.iter() {
            let img = ImageView::new(&data, width, height, width).expect("valid image");

            let intrinsics = gt.intrinsics.or(self.options.intrinsics);
            let tag_size = gt.tag_size.or(self.options.tag_size);

            let start = std::time::Instant::now();
            let detections = detector.detect(
                &img,
                intrinsics.as_ref(),
                tag_size,
                self.options.pose_estimation_mode,
                false,
            );
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

                    if dist_sq < 50.0 * 50.0 {
                        // Use the test_utils version which tries all 4 rotations for best match
                        image_rmse_sum += locus_core::test_utils::compute_corner_error(
                            &det_corners_f64,
                            gt_corners,
                        );
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

                            if let (Some(gt_pose), Some(intr), Some(size)) = (gt.poses.get(&det_id), intrinsics, tag_size) {
                                let t_err = (det_t - gt_pose.translation).norm();

                                // Align ground truth rotation with detector convention (180 deg flip about X)
                                let q_correction = UnitQuaternion::from_axis_angle(
                                    &Vector3::x_axis(),
                                    std::f64::consts::PI,
                                );
                                let q_gt_aligned =
                                    UnitQuaternion::from_matrix(&gt_pose.rotation) * q_correction;

                                // Detectors often have 90-degree ambiguities. We check all 4 rotations about Z.
                                let mut min_r_err = det_q.angle_to(&q_gt_aligned);
                                let mut best_rot_idx = 0;
                                for k in 1..4 {
                                    let q_rot = UnitQuaternion::from_axis_angle(
                                        &Vector3::z_axis(),
                                        f64::from(k) * std::f64::consts::FRAC_PI_2,
                                    );
                                    let r_err = det_q.angle_to(&(q_gt_aligned * q_rot));
                                    if r_err < min_r_err {
                                        min_r_err = r_err;
                                        best_rot_idx = k;
                                    }
                                }

                                image_translation_error_sum += t_err;
                                image_rotation_error_sum += min_r_err;
                                pose_match_count += 1;

                                translation_errors.push(t_err);
                                rotation_errors.push(min_r_err.to_degrees());

                                // --- Reprojection Error (vs Ground Truth Corners) ---
                                let s = size * 0.5;
                                // 3D model corners in the tag frame (centered)
                                // Standard ordering: TL, TR, BR, BL
                                let model_corners = [
                                    Vector3::new(-s, -s, 0.0),
                                    Vector3::new(s, -s, 0.0),
                                    Vector3::new(s, s, 0.0),
                                    Vector3::new(-s, s, 0.0),
                                ];
                                
                                // Estimated pose in detector frame
                                let est_pose = Pose::new(det_q.to_rotation_matrix().into_inner(), det_t);
                                
                                // To reproject accurately, we need to handle the 90-degree rotation ambiguity
                                // by rotating the model corners to match the detected corner ordering.
                                let mut reprojected_corners = [[0.0, 0.0]; 4];
                                for k in 0..4 {
                                    // Apply the inverse of the ambiguity rotation to the model points
                                    let q_rot_inv = UnitQuaternion::from_axis_angle(
                                        &Vector3::z_axis(),
                                        -f64::from(best_rot_idx) * std::f64::consts::FRAC_PI_2,
                                    );
                                    let p_rotated = q_rot_inv * model_corners[k];
                                    reprojected_corners[k] = est_pose.project(&p_rotated, &intr);
                                }
                                
                                image_repro_rmse_sum += locus_core::test_utils::compute_corner_error(&reprojected_corners, gt_corners);
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
                match_count as f64 / detections.len() as f64
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
                worst_offenders: offenders.into_iter().take(5).collect(),
            },
        };

        println!("=== {} Results ===", self.snapshot_name);
        println!("  Images: {count}");
        println!("  Recall: {:.2}%", report.summary.mean_recall * 100.0);
        println!("  Precision: {:.2}%", report.summary.mean_precision * 100.0);
        println!("  RMSE:   {:.4} px", report.summary.mean_rmse);
        println!("  Repro RMSE: {:.4} px", report.summary.mean_reprojection_rmse);
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
    rotation_quaternion: [f64; 4], // [x, y, z, w]
    /// Per-detection intrinsics (optional)
    k_matrix: Option<[[f64; 3]; 3]>,
    /// Per-detection tag size (optional)
    tag_size_mm: Option<f64>,
}

impl HubProvider {
    fn new(dataset_dir: &std::path::Path) -> Option<Self> {
        let rich_path = dataset_dir.join("rich_truth.json");
        let jsonl_path = dataset_dir.join("annotations.jsonl");
        let images_dir = dataset_dir.join("images");

        if !images_dir.exists() {
            return None;
        }

        let mut gt_map: HashMap<String, GroundTruth> = HashMap::new();
        let mut entries = Vec::new();

        if rich_path.exists() {
            let file = std::fs::File::open(&rich_path).ok()?;
            entries = serde_json::from_reader(file).ok()?;
        } else if jsonl_path.exists() {
            let file = std::fs::File::open(&jsonl_path).ok()?;
            let reader = std::io::BufReader::new(file);
            use std::io::BufRead;
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(entry) = serde_json::from_str::<HubEntry>(&line) {
                    entries.push(entry);
                }
            }
        } else {
            return None;
        }

        for entry in entries {
            // Normalize filename: rich_truth uses image_id, annotations uses image_filename
            let fname = if std::path::Path::new(&entry.image_filename)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
            {
                entry.image_filename.clone()
            } else {
                format!("{}.png", entry.image_filename)
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
                entry.rotation_quaternion[3], // w
                entry.rotation_quaternion[0], // x (i)
                entry.rotation_quaternion[1], // y (j)
                entry.rotation_quaternion[2], // z (k)
            ))
            .to_rotation_matrix();

            let pose = Pose {
                rotation: *rotation.matrix(),
                translation: Vector3::new(entry.position[0], entry.position[1], entry.position[2]),
            };

            gt_map
                .get_mut(&fname)
                .unwrap()
                .poses
                .insert(entry.tag_id, pose);
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
        let root = std::fs::canonicalize(PathBuf::from(hub_dir.clone()))
            .unwrap_or_else(|_| PathBuf::from(hub_dir));
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

            options.pose_estimation_mode = locus_core::PoseEstimationMode::Fast;

            let snapshot = format!("hub_{}", provider.name());
            RegressionHarness::new(snapshot)
                .with_preset(ConfigPreset::PlainBoard)
                .with_families(vec![family])
                .with_options(options)
                .run(provider);
        }
    } else {
        println!("Skipping hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
    }
}

#[test]
#[ignore = "Metadata gaps in HF dataset"]
fn regression_hub_tag36h11() {
    let _guard = common::telemetry::init("regression_hub_tag36h11");
    run_hub_test(
        "single_tag_locus_v1_tag36h11_640x480",
        TagFamily::AprilTag36h11,
    );
}

#[test]
fn regression_hub_std41h12() {
    run_hub_test(
        "single_tag_locus_v1_std41h12_640x480",
        TagFamily::AprilTag41h12,
    );
}

#[test]
fn regression_hub_std41h12_720p() {
    run_hub_test(
        "single_tag_locus_v1_std41h12_1280x720",
        TagFamily::AprilTag41h12,
    );
}

#[test]
fn regression_hub_std41h12_1080p() {
    run_hub_test(
        "single_tag_locus_v1_std41h12_1920x1080",
        TagFamily::AprilTag41h12,
    );
}
