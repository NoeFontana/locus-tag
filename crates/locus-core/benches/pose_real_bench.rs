#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::unwrap_used,
    missing_docs
)]
//! Tag pose estimation on real detections.
//!
//! `pose_bench` refines synthetic axis-aligned squares, which the LM accepts in
//! one step. This bench captures the detected corners of every frame of the hub
//! `locus_v1_tag36h11_1920x1080` set once (not timed), then times only the pose
//! stage over all of them: IPPE, branch selection, the LM and its covariance.
//!
//! ```bash
//! cargo bench -p locus-core --bench pose_real_bench --features bench-internals
//! ```

use divan::{Bencher, black_box};
use locus_core::bench_api::{DetectionBatch, refine_poses_soa_with_config};
use locus_core::pose::estimate_tag_pose_with_config;
use locus_core::{CameraIntrinsics, Detector, DetectorConfig, ImageView, TagFamily};
use utils::{hub_dir, rich_truth_records};

mod utils;

const DATASET: &str = "locus_v1_tag36h11_1920x1080";
/// tag36h11's grid dimension, the `edge_refine_dim` the detector derives.
const TAG36H11_DIM: usize = 6;

fn main() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
    divan::Divan::from_args().threads([1]).run_benches();
}

struct Frame {
    luma: Vec<u8>,
    width: usize,
    height: usize,
    batch: Box<DetectionBatch>,
    valid: usize,
}

struct Scene {
    intrinsics: CameraIntrinsics,
    tag_size: f64,
    frames: Vec<Frame>,
}

impl Frame {
    fn view(&self) -> ImageView<'_> {
        ImageView::new(&self.luma, self.width, self.height, self.width).unwrap()
    }

    fn corners(&self) -> impl Iterator<Item = [[f64; 2]; 4]> + '_ {
        self.batch.corners[..self.valid]
            .iter()
            .map(|c| c.map(|p| [f64::from(p.x), f64::from(p.y)]))
    }
}

/// Every frame of the dataset, detected once with the `standard` profile.
fn scene() -> Scene {
    #[derive(serde::Deserialize)]
    struct Record {
        image_id: String,
        tag_size_mm: f64,
        k_matrix: Vec<Vec<f64>>,
    }

    let dir = hub_dir().join(DATASET);
    let records: Vec<Record> = rich_truth_records(DATASET);
    let first = &records[0];
    let k = &first.k_matrix;
    let intrinsics = CameraIntrinsics::new(k[0][0], k[1][1], k[0][2], k[1][2]);
    let tag_size = first.tag_size_mm / 1000.0;

    let mut ids: Vec<&str> = records.iter().map(|r| r.image_id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();

    let mut detector = Detector::with_config(DetectorConfig::from_profile("standard"));
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let frames = ids
        .iter()
        .map(|id| {
            let luma = image::open(dir.join("images").join(format!("{id}.png")))
                .unwrap()
                .to_luma8();
            let (w, h) = luma.dimensions();
            let luma = luma.into_raw();
            let (width, height) = (w as usize, h as usize);
            let view = ImageView::new(&luma, width, height, width).unwrap();
            let _ = detector
                .detect(&view, Some(&intrinsics), Some(tag_size), false)
                .unwrap();
            let mut batch = detector.bench_api_get_batch_cloned();
            let valid = batch.partition(batch.capacity());
            Frame {
                luma,
                width,
                height,
                batch,
                valid,
            }
        })
        .collect();
    Scene {
        intrinsics,
        tag_size,
        frames,
    }
}

fn estimate_all(bencher: Bencher, profile: &str, with_image: bool) {
    let s = scene();
    let config = DetectorConfig::from_profile(profile);
    bencher.bench_local(|| {
        for frame in &s.frames {
            let view = frame.view();
            let img = with_image.then_some(&view);
            for corners in frame.corners() {
                black_box(estimate_tag_pose_with_config(
                    &s.intrinsics,
                    black_box(&corners),
                    s.tag_size,
                    img,
                    &config,
                    None,
                ));
            }
        }
    });
}

/// The structure-tensor-weighted LM of the `standard` profile, per tag.
#[divan::bench]
fn estimate_weighted(bencher: Bencher) {
    estimate_all(bencher, "standard", true);
}

/// The unweighted LM (no image), per tag.
#[divan::bench]
fn estimate_unweighted(bencher: Bencher) {
    estimate_all(bencher, "standard", false);
}

fn refine_soa(bencher: Bencher, profile: &str, edge_refine_dim: Option<usize>) {
    let mut s = scene();
    let config = DetectorConfig::from_profile(profile);
    bencher.bench_local(|| {
        for frame in &mut s.frames {
            let view = ImageView::new(&frame.luma, frame.width, frame.height, frame.width).unwrap();
            refine_poses_soa_with_config(
                &mut frame.batch,
                frame.valid,
                &s.intrinsics,
                s.tag_size,
                Some(&view),
                &config,
                edge_refine_dim,
            );
        }
    });
}

/// The batch path the detector runs with the `standard` profile, write-back included, per frame.
#[divan::bench]
fn refine_soa_standard(bencher: Bencher) {
    refine_soa(bencher, "standard", None);
}

/// The batch path with `high_accuracy`'s model-edge refinement, per frame.
#[divan::bench]
fn refine_soa_high_accuracy(bencher: Bencher) {
    refine_soa(bencher, "high_accuracy", Some(TAG36H11_DIM));
}
