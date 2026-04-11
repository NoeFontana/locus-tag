#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
mod utils;

use divan::bench;
use locus_core::ImageView;
use locus_core::bench_api::{decode_batch_soa, family_to_decoder};
use utils::BenchDataset;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench]
fn bench_decoding_soa_realistic(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();

    // 50 valid tags, 200 false positives
    let mut batch = BenchDataset::generate_bench_batch(50, 200);
    let n = 250;

    let config = locus_core::DetectorConfig::default();
    let decoders = vec![family_to_decoder(locus_core::TagFamily::AprilTag36h11)];

    // Compute homographies once outside the loop
    locus_core::bench_api::compute_homographies_soa(
        &batch.corners[0..n],
        &batch.status_mask[0..n],
        &mut batch.homographies[0..n],
    );

    bencher.bench_local(move || {
        decode_batch_soa(&mut batch, n, &img, &decoders, &config);
    });
}
