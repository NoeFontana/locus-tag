#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
mod utils;

use divan::bench;
use locus_core::PoseEstimationMode;
use locus_core::bench_api::{CameraIntrinsics, refine_poses_soa, refine_poses_soa_with_config};
use locus_core::config::DetectorConfig;
use utils::BenchDataset;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench(args = [10, 50])]
fn bench_pose_fast(bencher: divan::Bencher, &v: &usize) {
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;
    let mut batch = BenchDataset::generate_bench_batch(v, 0);

    bencher.bench_local(move || {
        refine_poses_soa(
            &mut batch,
            v,
            &intrinsics,
            tag_size,
            None,
            PoseEstimationMode::Fast,
        );
    });
}

#[bench(args = [10, 50])]
fn bench_pose_accurate(bencher: divan::Bencher, &v: &usize) {
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;
    let dataset = BenchDataset::icra_forward_0();
    let img = locus_core::ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let mut batch = BenchDataset::generate_bench_batch(v, 0);

    bencher.bench_local(move || {
        refine_poses_soa(
            &mut batch,
            v,
            &intrinsics,
            tag_size,
            Some(&img),
            PoseEstimationMode::Accurate,
        );
    });
}

// ---------------------------------------------------------------------------
// Pose-consistency gate latency matrix.
//
// Four cells:
//   - pinhole         × gate disabled (baseline; identical to bench_pose_fast)
//   - pinhole         × gate enabled at fpr = 1e-3
//   - Brown-Conrady   × gate disabled
//   - Brown-Conrady   × gate enabled at fpr = 1e-3
//
// The Brown-Conrady leg additionally exercises `project_with_distortion`
// inside the gate; the pinhole leg measures the pure χ²-test overhead.
// ---------------------------------------------------------------------------

#[cfg(feature = "non_rectified")]
fn brown_conrady_intrinsics() -> CameraIntrinsics {
    // Mild distortion typical of a commodity webcam.
    CameraIntrinsics::with_brown_conrady(800.0, 800.0, 400.0, 300.0, -0.12, 0.05, 0.0, 0.0, 0.0)
}

fn config_with_fpr(fpr: f64) -> DetectorConfig {
    DetectorConfig {
        pose_consistency_fpr: fpr,
        ..DetectorConfig::default()
    }
}

#[bench(args = [10, 50])]
fn bench_pose_gate_pinhole_off(bencher: divan::Bencher, &v: &usize) {
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;
    let cfg = config_with_fpr(0.0);
    let mut batch = BenchDataset::generate_bench_batch(v, 0);

    bencher.bench_local(move || {
        refine_poses_soa_with_config(
            &mut batch,
            v,
            &intrinsics,
            tag_size,
            None,
            PoseEstimationMode::Fast,
            &cfg,
        );
    });
}

#[bench(args = [10, 50])]
fn bench_pose_gate_pinhole_on(bencher: divan::Bencher, &v: &usize) {
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;
    let cfg = config_with_fpr(1.0e-3);
    let mut batch = BenchDataset::generate_bench_batch(v, 0);

    bencher.bench_local(move || {
        refine_poses_soa_with_config(
            &mut batch,
            v,
            &intrinsics,
            tag_size,
            None,
            PoseEstimationMode::Fast,
            &cfg,
        );
    });
}

#[cfg(feature = "non_rectified")]
#[bench(args = [10, 50])]
fn bench_pose_gate_brown_conrady_off(bencher: divan::Bencher, &v: &usize) {
    let intrinsics = brown_conrady_intrinsics();
    let tag_size = 0.16;
    let cfg = config_with_fpr(0.0);
    let mut batch = BenchDataset::generate_bench_batch(v, 0);

    bencher.bench_local(move || {
        refine_poses_soa_with_config(
            &mut batch,
            v,
            &intrinsics,
            tag_size,
            None,
            PoseEstimationMode::Fast,
            &cfg,
        );
    });
}

#[cfg(feature = "non_rectified")]
#[bench(args = [10, 50])]
fn bench_pose_gate_brown_conrady_on(bencher: divan::Bencher, &v: &usize) {
    let intrinsics = brown_conrady_intrinsics();
    let tag_size = 0.16;
    let cfg = config_with_fpr(1.0e-3);
    let mut batch = BenchDataset::generate_bench_batch(v, 0);

    bencher.bench_local(move || {
        refine_poses_soa_with_config(
            &mut batch,
            v,
            &intrinsics,
            tag_size,
            None,
            PoseEstimationMode::Fast,
            &cfg,
        );
    });
}
