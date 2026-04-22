#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! Differential tests: verify that opt-in profiles do not silently lose
//! detections vs. `standard`.
//!
//! These tests are gated behind `LOCUS_DIFF_ADAPTIVE=1` — they act as
//! Phase 2 exit criteria for the `max_recall_adaptive` profile and
//! Phase 3 exit criteria for ROI rescue. Running them locally requires
//! `LOCUS_ICRA_DATASET_DIR` for the richer ICRA corpus; without it,
//! only the small `tests/fixtures/icra2020` corpus is checked.

use locus_core::{DetectOptions, Detector, DetectorConfig, ImageView, TagFamily};
use std::collections::HashMap;
use std::path::PathBuf;

mod common;

const MAX_CORNER_DEVIATION_PX: f64 = 0.5;

struct Sample {
    filename: String,
    image: Vec<u8>,
    width: usize,
    height: usize,
    /// Ground-truth tag IDs keyed by u32. Used only by the rescue test
    /// to gate extras; the adaptive test accepts extras unconditionally.
    gt_ids: std::collections::HashSet<u32>,
}

fn load_samples() -> Vec<Sample> {
    let mut samples = Vec::new();

    let fixtures_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/icra2020");
    append_fixture_samples(&fixtures_dir, &mut samples);

    if let Ok(dataset_dir) = std::env::var("LOCUS_ICRA_DATASET_DIR") {
        let forward = PathBuf::from(&dataset_dir)
            .join("forward")
            .join("pure_tags_images");
        if forward.is_dir()
            && let Some(gt_map) =
                common::load_ground_truth(&PathBuf::from(&dataset_dir), "forward")
        {
            append_icra_samples(&forward, &gt_map, &mut samples);
        }
    }

    samples
}

fn append_fixture_samples(dir: &std::path::Path, out: &mut Vec<Sample>) {
    if !dir.is_dir() {
        return;
    }
    let walker = walkdir::WalkDir::new(dir).sort_by_file_name();
    for entry in walker.into_iter().flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("png") {
            continue;
        }
        let json_path = path.with_extension("json");
        if !json_path.exists() {
            continue;
        }
        let Ok(img) = image::open(path) else { continue };
        let img = img.into_luma8();
        let (w, h) = img.dimensions();

        #[derive(serde::Deserialize)]
        struct FixtureJson {
            tags: Vec<FixtureTag>,
        }
        #[derive(serde::Deserialize)]
        struct FixtureTag {
            tag_id: u32,
        }

        let json_str = std::fs::read_to_string(&json_path).unwrap();
        let fixture: FixtureJson = serde_json::from_str(&json_str).unwrap();
        let gt_ids = fixture.tags.into_iter().map(|t| t.tag_id).collect();

        out.push(Sample {
            filename: path.file_name().unwrap().to_string_lossy().into_owned(),
            image: img.into_raw(),
            width: w as usize,
            height: h as usize,
            gt_ids,
        });
    }
}

fn append_icra_samples(
    img_dir: &std::path::Path,
    gt_map: &HashMap<String, common::ImageGroundTruth>,
    out: &mut Vec<Sample>,
) {
    let walker = walkdir::WalkDir::new(img_dir).sort_by_file_name();
    for entry in walker.into_iter().flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("png") {
            continue;
        }
        let fname = path.file_name().unwrap().to_string_lossy().into_owned();
        let Some(gt) = gt_map.get(&fname) else {
            continue;
        };
        let Ok(img) = image::open(path) else { continue };
        let img = img.into_luma8();
        let (w, h) = img.dimensions();

        out.push(Sample {
            filename: fname,
            image: img.into_raw(),
            width: w as usize,
            height: h as usize,
            gt_ids: gt.tag_ids.clone(),
        });
    }
}

fn detect_by_id(
    detector: &mut Detector,
    sample: &Sample,
    options: &DetectOptions,
) -> HashMap<u32, [[f64; 2]; 4]> {
    let img = ImageView::new(&sample.image, sample.width, sample.height, sample.width)
        .expect("valid image");
    let detections = detector
        .detect(
            &img,
            options.intrinsics.as_ref(),
            options.tag_size,
            options.pose_estimation_mode,
            false,
        )
        .expect("detection failed");

    let mut map = HashMap::new();
    for i in 0..detections.len() {
        let corners = detections.corners[i];
        let corners_f64 = [
            [f64::from(corners[0].x), f64::from(corners[0].y)],
            [f64::from(corners[1].x), f64::from(corners[1].y)],
            [f64::from(corners[2].x), f64::from(corners[2].y)],
            [f64::from(corners[3].x), f64::from(corners[3].y)],
        ];
        map.insert(detections.ids[i], corners_f64);
    }
    map
}

fn max_corner_deviation(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> f64 {
    (0..4)
        .map(|k| {
            ((a[k][0] - b[k][0]).powi(2) + (a[k][1] - b[k][1]).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max)
}

fn run_diff(
    baseline_config: DetectorConfig,
    candidate_config: DetectorConfig,
    allow_extras_not_in_gt: bool,
    label: &str,
) {
    if std::env::var("LOCUS_DIFF_ADAPTIVE").is_err() {
        println!(
            "Skipping diff_adaptive::{label}. Set LOCUS_DIFF_ADAPTIVE=1 to run."
        );
        return;
    }

    let samples = load_samples();
    assert!(
        !samples.is_empty(),
        "diff_adaptive corpus is empty; missing fixtures/icra2020"
    );

    let mut baseline = Detector::with_config(baseline_config);
    baseline.set_families(&[TagFamily::AprilTag36h11]);
    let mut candidate = Detector::with_config(candidate_config);
    candidate.set_families(&[TagFamily::AprilTag36h11]);

    let options = DetectOptions::default();
    let mut missing_tags: Vec<String> = Vec::new();
    let mut corner_drift: Vec<String> = Vec::new();
    let mut illegal_extras: Vec<String> = Vec::new();

    for sample in &samples {
        let base_map = detect_by_id(&mut baseline, sample, &options);
        let cand_map = detect_by_id(&mut candidate, sample, &options);

        for (id, base_corners) in &base_map {
            match cand_map.get(id) {
                None => missing_tags.push(format!("{}: missing id={id}", sample.filename)),
                Some(cand_corners) => {
                    let drift = max_corner_deviation(base_corners, cand_corners);
                    if drift > MAX_CORNER_DEVIATION_PX {
                        corner_drift.push(format!(
                            "{}: id={id} drift={drift:.3}px",
                            sample.filename
                        ));
                    }
                }
            }
        }

        if !allow_extras_not_in_gt {
            for id in cand_map.keys() {
                if !base_map.contains_key(id) && !sample.gt_ids.contains(id) {
                    illegal_extras
                        .push(format!("{}: extra id={id} not in GT", sample.filename));
                }
            }
        }
    }

    let mut failures = Vec::new();
    if !missing_tags.is_empty() {
        failures.push(format!(
            "{} tag(s) missing on candidate profile:\n  {}",
            missing_tags.len(),
            missing_tags.join("\n  ")
        ));
    }
    if !corner_drift.is_empty() {
        failures.push(format!(
            "{} tag(s) exceed {MAX_CORNER_DEVIATION_PX}px corner deviation:\n  {}",
            corner_drift.len(),
            corner_drift.join("\n  ")
        ));
    }
    if !illegal_extras.is_empty() {
        failures.push(format!(
            "{} extra tag(s) not supported by GT:\n  {}",
            illegal_extras.len(),
            illegal_extras.join("\n  ")
        ));
    }
    assert!(failures.is_empty(), "diff_adaptive::{label} failed:\n{}", failures.join("\n"));
}

#[test]
fn diff_adaptive_vs_standard() {
    run_diff(
        DetectorConfig::from_profile("standard"),
        DetectorConfig::from_profile("max_recall_adaptive"),
        /* allow_extras_not_in_gt */ true,
        "adaptive_vs_standard",
    );
}

#[test]
fn diff_rescue_vs_standard() {
    let mut rescue_config = DetectorConfig::from_profile("standard");
    rescue_config.roi_rescue.enabled = true;
    run_diff(
        DetectorConfig::from_profile("standard"),
        rescue_config,
        /* allow_extras_not_in_gt */ false,
        "rescue_vs_standard",
    );
}
