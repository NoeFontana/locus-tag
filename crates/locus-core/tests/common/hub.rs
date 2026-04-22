#![allow(clippy::collapsible_if, clippy::needless_pass_by_value)]
//! Shared hub dataset infrastructure reused by multiple regression test files.

use locus_core::{
    CameraIntrinsics, DetectOptions, Detector, DetectorConfig, ImageView, Pose, PoseEstimationMode,
    TagFamily,
    config::{CornerRefinementMode, QuadExtractionMode},
};
use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

// ============================================================================
// Metrics & Reporting
// ============================================================================

#[derive(Serialize, Default, Clone)]
pub struct PipelineMetrics {
    pub total_ms: f64,
    pub num_detections: usize,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
pub fn serialize_rmse<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let rounded = (value * 10000.0).round() / 10000.0;
    serializer.serialize_f64(rounded)
}

#[derive(Serialize)]
pub struct ImageMetrics {
    pub recall: f64,
    pub precision: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub avg_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub reprojection_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub mean_hamming: f64,
    pub stats: PipelineMetrics,
    pub missed_ids: BTreeSet<u32>,
    pub extra_ids: BTreeSet<u32>,
}

#[derive(Serialize)]
pub struct RegressionReport {
    pub summary: SummaryMetrics,
}

#[derive(Serialize)]
pub struct Offender {
    pub filename: String,
    pub missed: usize,
    pub extra: usize,
    #[serde(serialize_with = "serialize_rmse")]
    pub rmse: f64,
}

#[derive(Serialize)]
pub struct SummaryMetrics {
    pub dataset_size: usize,
    pub mean_recall: f64,
    pub mean_precision: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub mean_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub mean_reprojection_rmse: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub p50_translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub p90_translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub p99_translation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub p50_rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub p90_rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub p99_rotation_error: f64,
    #[serde(serialize_with = "serialize_rmse")]
    pub mean_hamming: f64,
    pub mean_total_ms: f64,
    pub worst_offenders: Vec<Offender>,
}

/// Sort once, return (p50, p90, p99).
pub fn calculate_percentiles(values: &mut [f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let last = (values.len() - 1) as f64;
    let idx = |p: f64| (p * last).round() as usize;
    (values[idx(0.5)], values[idx(0.9)], values[idx(0.99)])
}

// ============================================================================
// Evaluation Engine
// ============================================================================

/// Ground truth for a single image.
#[derive(Clone)]
pub struct GroundTruth {
    pub tags: BTreeMap<u32, [[f64; 2]; 4]>,
    pub poses: BTreeMap<u32, Pose>,
    pub intrinsics: Option<CameraIntrinsics>,
    pub tag_size: Option<f64>,
}

pub type DatasetItem = (String, Vec<u8>, usize, usize, GroundTruth);

pub trait DatasetProvider {
    fn name(&self) -> &str;
    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_>;
}

/// Unified harness for running regression tests.
pub struct RegressionHarness {
    pub snapshot_name: String,
    pub config: DetectorConfig,
    pub options: DetectOptions,
}

impl RegressionHarness {
    pub fn new(snapshot_name: impl Into<String>) -> Self {
        Self {
            snapshot_name: snapshot_name.into(),
            config: DetectorConfig::default(),
            options: DetectOptions::default(),
        }
    }

    /// Load one of the three shipped profiles by name: `"standard"`, `"grid"`,
    /// or `"high_accuracy"`. Panics on unknown names (closed set).
    pub fn with_profile(mut self, name: &str) -> Self {
        self.config = DetectorConfig::from_profile(name);
        self
    }

    /// Load a custom profile from a JSON string (e.g. an `include_str!`'d
    /// fixture file). Panics on parse failure, same policy as `from_profile`.
    pub fn with_profile_json(mut self, json: &str) -> Self {
        self.config =
            DetectorConfig::from_profile_json(json).expect("embedded test fixture must parse");
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

    pub fn with_extraction_policy(
        mut self,
        policy: locus_core::config::QuadExtractionPolicy,
    ) -> Self {
        self.config.quad_extraction_policy = policy;
        self
    }

    pub fn with_roi_rescue_enabled(mut self, enabled: bool) -> Self {
        self.config.roi_rescue.enabled = enabled;
        self
    }

    pub fn with_decode_mode(mut self, mode: locus_core::config::DecodeMode) -> Self {
        self.config.decode_mode = mode;
        self
    }

    pub fn run(self, provider: impl DatasetProvider) {
        let mut detector = Detector::with_config(self.config);
        if !self.options.families.is_empty() {
            detector.set_families(&self.options.families);
        }
        let mut results = BTreeMap::new();

        let mut total_recall = 0.0;
        let mut total_precision = 0.0;
        let mut total_rmse = 0.0;
        let mut total_repro_rmse = 0.0;
        let mut total_hamming = 0.0;
        let mut total_time = 0.0;
        let mut count = 0;

        let mut translation_errors = Vec::new();
        let mut rotation_errors = Vec::new();

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
                        let mut rmse_sq = 0.0;
                        for k in 0..4 {
                            rmse_sq += (det_corners_f64[k][0] - gt_corners[k][0]).powi(2)
                                + (det_corners_f64[k][1] - gt_corners[k][1]).powi(2);
                        }
                        image_rmse_sum += (rmse_sq / 4.0).sqrt();

                        image_hamming_sum += f64::from(detections.error_rates[i]);
                        match_count += 1;
                        found_ids.insert(det_id);

                        let det_pose_data = detections.poses[i].data;
                        if det_pose_data[2] > 0.0 {
                            let det_t = Vector3::new(
                                f64::from(det_pose_data[0]),
                                f64::from(det_pose_data[1]),
                                f64::from(det_pose_data[2]),
                            );
                            let det_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                                f64::from(det_pose_data[6]),
                                f64::from(det_pose_data[3]),
                                f64::from(det_pose_data[4]),
                                f64::from(det_pose_data[5]),
                            ));

                            if let (Some(gt_pose), Some(intr), Some(size)) =
                                (gt.poses.get(&det_id), intrinsics, tag_size)
                            {
                                let q_gt = UnitQuaternion::from_matrix(&gt_pose.rotation);
                                let r_err = det_q.angle_to(&q_gt);
                                let t_err = (det_t - gt_pose.translation).norm();

                                image_translation_error_sum += t_err;
                                image_rotation_error_sum += r_err;
                                pose_match_count += 1;

                                translation_errors.push(t_err);
                                rotation_errors.push(r_err.to_degrees());

                                let h = size * 0.5;
                                let model_corners = [
                                    Vector3::new(-h, -h, 0.0),
                                    Vector3::new(h, -h, 0.0),
                                    Vector3::new(h, h, 0.0),
                                    Vector3::new(-h, h, 0.0),
                                ];

                                let est_pose =
                                    Pose::new(det_q.to_rotation_matrix().into_inner(), det_t);

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

            let missed_ids: BTreeSet<u32> = gt
                .tags
                .keys()
                .filter(|id| !found_ids.contains(*id))
                .copied()
                .collect();

            let extra_ids: BTreeSet<u32> = (0..detections.len())
                .map(|i| detections.ids[i])
                .filter(|id| !found_ids.contains(id))
                .collect();

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

        let (p50_t, p90_t, p99_t) = calculate_percentiles(&mut translation_errors);
        let (p50_r, p90_r, p99_r) = calculate_percentiles(&mut rotation_errors);

        let report = RegressionReport {
            summary: SummaryMetrics {
                dataset_size: count,
                mean_recall: total_recall / count as f64,
                mean_precision: total_precision / count as f64,
                mean_rmse: total_rmse / count as f64,
                mean_reprojection_rmse: total_repro_rmse / count as f64,
                p50_translation_error: p50_t,
                p90_translation_error: p90_t,
                p99_translation_error: p99_t,
                p50_rotation_error: p50_r,
                p90_rotation_error: p90_r,
                p99_rotation_error: p99_r,
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

pub struct HubProvider {
    pub name: String,
    pub base_dir: PathBuf,
    pub gt_map: BTreeMap<String, GroundTruth>,
}

#[derive(Deserialize)]
pub struct HubEntry {
    #[serde(alias = "image_id")]
    pub image_filename: String,
    pub tag_id: u32,
    pub corners: [[f64; 2]; 4],
    pub position: [f64; 3],
    pub rotation_quaternion: [f64; 4], // [w, x, y, z]
    pub k_matrix: Option<[[f64; 3]; 3]>,
    pub tag_size_mm: Option<f64>,
    /// Lens distortion model: "brown_conrady" or "kannala_brandt"
    pub distortion_model: Option<String>,
    /// Distortion coefficients matching the model
    #[serde(alias = "distortion_coeffs")]
    pub dist_coeffs: Option<Vec<f64>>,
}

/// `rich_truth.json` format dispatcher.
///
/// v1 datasets write a flat JSON array. v2 datasets (distortion suites onward)
/// wrap entries in `{ "version": …, "records": [...] }` and mix per-tag
/// records with per-board records (`record_type = "BOARD"`, `tag_id = -1`,
/// 1-point `corners`) that don't match the per-tag `HubEntry` shape.
#[derive(Deserialize)]
#[serde(untagged)]
enum RichTruthEnvelope {
    V2 { records: Vec<serde_json::Value> },
    V1(Vec<serde_json::Value>),
}

/// Load `rich_truth.json` entries, transparently handling v1 and v2 schemas.
/// Non-tag records (v2 `record_type = "BOARD"`, etc.) are filtered out.
pub fn load_rich_truth_entries(path: &Path) -> Option<Vec<HubEntry>> {
    let file = std::fs::File::open(path).ok()?;
    let raw = match serde_json::from_reader::<_, RichTruthEnvelope>(file).ok()? {
        RichTruthEnvelope::V1(v) | RichTruthEnvelope::V2 { records: v } => v,
    };
    raw.into_iter()
        .filter(|r| {
            r.get("record_type")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|t| t == "TAG")
        })
        .map(serde_json::from_value)
        .collect::<Result<Vec<_>, _>>()
        .ok()
}

/// Build `CameraIntrinsics` from a 3×3 K-matrix, optionally with distortion.
pub fn build_intrinsics(
    k: [[f64; 3]; 3],
    distortion_model: Option<&str>,
    dist_coeffs: Option<&[f64]>,
) -> CameraIntrinsics {
    let (fx, fy, cx, cy) = (k[0][0], k[1][1], k[0][2], k[1][2]);
    match (distortion_model, dist_coeffs) {
        #[cfg(feature = "non_rectified")]
        (Some("brown_conrady"), Some(c)) if c.len() >= 5 => {
            CameraIntrinsics::with_brown_conrady(fx, fy, cx, cy, c[0], c[1], c[2], c[3], c[4])
        },
        #[cfg(feature = "non_rectified")]
        (Some("kannala_brandt"), Some(c)) if c.len() >= 4 => {
            CameraIntrinsics::with_kannala_brandt(fx, fy, cx, cy, c[0], c[1], c[2], c[3])
        },
        _ => CameraIntrinsics::new(fx, fy, cx, cy),
    }
}

/// Load `DetectOptions` (intrinsics + tag size) for a dataset directory.
///
/// Reads `provenance.json` first; falls back to the first `rich_truth.json` entry.
/// Distortion coefficients from `rich_truth.json` are applied automatically.
pub fn load_detect_options(dataset_path: &Path) -> DetectOptions {
    let mut options = DetectOptions::default();

    if let Ok(s) = std::fs::read_to_string(dataset_path.join("provenance.json")) {
        if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(intr) = meta.get("camera_intrinsics") {
                if let (Some(fx), Some(fy), Some(cx), Some(cy)) = (
                    intr["fx"].as_f64(),
                    intr["fy"].as_f64(),
                    intr["cx"].as_f64(),
                    intr["cy"].as_f64(),
                ) {
                    options.intrinsics = Some(CameraIntrinsics::new(fx, fy, cx, cy));
                }
            }
            if let Some(sz) = meta.get("tag_size_mm").and_then(serde_json::Value::as_f64) {
                options.tag_size = Some(sz / 1000.0);
            }
        }
    }

    if options.intrinsics.is_none() || options.tag_size.is_none() {
        if let Some(entries) = load_rich_truth_entries(&dataset_path.join("rich_truth.json")) {
            if let Some(first) = entries.first() {
                if options.intrinsics.is_none() {
                    if let Some(k) = first.k_matrix {
                        options.intrinsics = Some(build_intrinsics(
                            k,
                            first.distortion_model.as_deref(),
                            first.dist_coeffs.as_deref(),
                        ));
                    }
                }
                if options.tag_size.is_none() {
                    if let Some(sz) = first.tag_size_mm {
                        options.tag_size = Some(sz / 1000.0);
                    }
                }
            }
        }
    }

    options
}

impl HubProvider {
    pub fn new(dataset_dir: &Path) -> Option<Self> {
        let images_dir = dataset_dir.join("images");
        if !images_dir.exists() {
            return None;
        }

        let entries = load_rich_truth_entries(&dataset_dir.join("rich_truth.json"))?;

        let mut gt_map: BTreeMap<String, GroundTruth> = BTreeMap::new();

        for entry in entries {
            let fname = {
                let name = &entry.image_filename;
                if std::path::Path::new(name)
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("png"))
                {
                    name.clone()
                } else {
                    format!("{name}.png")
                }
            };

            let gt = gt_map.entry(fname).or_insert_with(|| GroundTruth {
                tags: BTreeMap::new(),
                poses: BTreeMap::new(),
                intrinsics: None,
                tag_size: None,
            });

            gt.tags.insert(entry.tag_id, entry.corners);

            if gt.intrinsics.is_none()
                && let Some(k) = entry.k_matrix
            {
                gt.intrinsics = Some(build_intrinsics(
                    k,
                    entry.distortion_model.as_deref(),
                    entry.dist_coeffs.as_deref(),
                ));
            }

            if gt.tag_size.is_none()
                && let Some(size_mm) = entry.tag_size_mm
            {
                gt.tag_size = Some(size_mm / 1000.0);
            }

            let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                entry.rotation_quaternion[0],
                entry.rotation_quaternion[1],
                entry.rotation_quaternion[2],
                entry.rotation_quaternion[3],
            ))
            .to_rotation_matrix();

            gt.poses.insert(
                entry.tag_id,
                Pose {
                    rotation: *rotation.matrix(),
                    translation: Vector3::new(
                        entry.position[0],
                        entry.position[1],
                        entry.position[2],
                    ),
                },
            );
        }

        Some(Self {
            name: dataset_dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            base_dir: dataset_dir.to_path_buf(),
            gt_map,
        })
    }

    /// Intrinsics from the first image's ground truth, used as dataset-level fallback.
    pub fn fallback_intrinsics(&self) -> Option<CameraIntrinsics> {
        self.gt_map.values().next()?.intrinsics
    }

    /// Tag size from the first image's ground truth, used as dataset-level fallback.
    pub fn fallback_tag_size(&self) -> Option<f64> {
        self.gt_map.values().next()?.tag_size
    }
}

// ============================================================================
// Render-tag test runner
// ============================================================================

/// Optional knobs for [`run_render_tag_test`]. Defaults reproduce the
/// Accurate-mode `"standard"`-profile baseline.
pub struct RenderTagOpts {
    pub mode: PoseEstimationMode,
    pub profile: Option<&'static str>,
    /// Inline JSON profile override (e.g. `include_str!`'d fixture). When set,
    /// takes precedence over `profile` — the harness calls `with_profile_json`
    /// instead of `with_profile`. Used by per-subset tuning variants that
    /// override specific knobs without shipping a new named profile.
    pub profile_json: Option<&'static str>,
    pub snapshot_suffix: &'static str,
    pub refinement: Option<CornerRefinementMode>,
    pub quad_mode: Option<QuadExtractionMode>,
    pub moments_culling: Option<(f64, f64)>,
    pub roi_rescue_enabled: bool,
}

impl Default for RenderTagOpts {
    fn default() -> Self {
        Self {
            mode: PoseEstimationMode::Accurate,
            profile: None,
            profile_json: None,
            snapshot_suffix: "",
            refinement: None,
            quad_mode: None,
            moments_culling: None,
            roi_rescue_enabled: false,
        }
    }
}

/// Run a render-tag regression against a hub dataset.
///
/// Skips gracefully when `LOCUS_HUB_DATASET_DIR` is unset or the dataset is
/// missing from the cache. Snapshot name is
/// `hub_{provider.name}{mode_suffix}{snapshot_suffix}`, matching the historic
/// naming convention for the render-tag suite.
pub fn run_render_tag_test(config_name: &str, family: TagFamily, opts: RenderTagOpts) {
    let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") else {
        println!("Skipping hub tests. Set LOCUS_HUB_DATASET_DIR to run.");
        return;
    };

    let dataset_path = super::resolve_hub_root(&hub_dir).join(config_name);
    let Some(provider) = HubProvider::new(&dataset_path) else {
        println!("Dataset not in cache: {config_name}. Skipping.");
        return;
    };

    let mut options = load_detect_options(&dataset_path);
    options.pose_estimation_mode = opts.mode;
    options.families = vec![family];

    let mode_suffix = match opts.mode {
        PoseEstimationMode::Fast => "_fast",
        PoseEstimationMode::Accurate => "",
    };
    let snapshot = format!(
        "hub_{}{}{}",
        provider.name, mode_suffix, opts.snapshot_suffix
    );

    let harness = RegressionHarness::new(snapshot);
    let harness = match opts.profile_json {
        Some(json) => harness.with_profile_json(json),
        None => harness.with_profile(opts.profile.unwrap_or("standard")),
    };
    let mut harness = harness.with_options(options);
    if let Some(r) = opts.refinement {
        harness = harness.with_refinement_mode(r);
    }
    if let Some(q) = opts.quad_mode {
        harness = harness.with_quad_extraction_mode(q);
    }
    if let Some((max_e, min_d)) = opts.moments_culling {
        harness = harness.with_moments_culling(max_e, min_d);
    }
    if opts.roi_rescue_enabled {
        harness = harness.with_roi_rescue_enabled(true);
    }
    harness.run(provider);
}

impl DatasetProvider for HubProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_> {
        let base_dir = self.base_dir.clone();
        let iter = self.gt_map.iter().filter_map(move |(fname, gt)| {
            let img_path = base_dir.join("images").join(fname);
            let img = image::open(&img_path).ok()?.into_luma8();
            let (w, h) = img.dimensions();
            Some((
                fname.clone(),
                img.into_raw(),
                w as usize,
                h as usize,
                gt.clone(),
            ))
        });
        Box::new(iter)
    }
}
