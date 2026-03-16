//! Real-world decoding benchmarks.
#![allow(missing_docs, clippy::unwrap_used)]
mod utils;

use divan::bench;
use locus_core::ImageView;
use locus_core::bench_api::{decode_batch_soa, family_to_decoder};
use utils::BenchDataset;

fn main() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench]
fn bench_decoding_only_real(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();

    // Use a real detector to get real quads from the image
    let config = locus_core::DetectorConfig::builder()
        .refinement_mode(locus_core::config::CornerRefinementMode::None)
        .build();
    let mut detector = locus_core::Detector::with_config(config);

    // Warm up to get the batch populated with candidates
    let _ = detector
        .detect(
            &img,
            None,
            None,
            locus_core::PoseEstimationMode::Fast,
            false,
        )
        .unwrap();

    // Capture the state
    let mut batch = detector.bench_api_get_batch_cloned();
    let n = 1024; // Use full batch capacity for bench
    let decoders = vec![family_to_decoder(locus_core::TagFamily::AprilTag36h11)];

    // Ensure homographies are computed
    locus_core::bench_api::compute_homographies_soa(
        &batch.corners[0..n],
        &batch.status_mask[0..n],
        &mut batch.homographies[0..n],
    );

    bencher.bench_local(move || {
        decode_batch_soa(&mut batch, n, &img, &decoders, &config);
    });
}
