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
    clippy::return_self_not_must_use
)]

mod utils;

use divan::bench;
use locus_core::bench_api::{CameraIntrinsics, refine_poses_soa};
use locus_core::PoseEstimationMode;
use utils::BenchDataset;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench]
fn bench_pose_estimation_soa_realistic(bencher: divan::Bencher) {
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;
    
    // 50 valid tags to refine
    let mut batch = BenchDataset::generate_bench_batch(50, 0);
    let v = 50;

    bencher.bench_local(move || {
        refine_poses_soa(
            &mut batch,
            v,
            &intrinsics,
            tag_size,
            None, // No refinement img
            PoseEstimationMode::Fast,
        );
    });
}
