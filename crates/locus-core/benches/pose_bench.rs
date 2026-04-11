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
use locus_core::bench_api::{CameraIntrinsics, refine_poses_soa};
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
