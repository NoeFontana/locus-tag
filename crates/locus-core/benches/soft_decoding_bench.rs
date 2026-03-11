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
use divan::bench;
use locus_core::bench_api::AprilTag36h11;
use locus_core::strategy::{DecodingStrategy, SoftStrategy};

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench]
fn bench_soft_decoding_200_candidates(bencher: divan::Bencher) {
    let decoder = AprilTag36h11;
    let dict = locus_core::dictionaries::get_dictionary(locus_core::TagFamily::AprilTag36h11);
    // Create a SoftCode from a known match (ID 42)
    let orig_code = dict.get_code(42).unwrap();

    // Simulate LLRs: matching bits have +50/-50, mismatching bit has -50/+50
    let mut intensities = [100.0f64; 64];
    let thresholds = [150.0f64; 64];
    for (i, (intensity, _threshold)) in intensities
        .iter_mut()
        .zip(thresholds.iter())
        .enumerate()
        .take(36)
    {
        if (orig_code >> i) & 1 == 1 {
            *intensity = 200.0; // Positive LLR (match)
        } else {
            *intensity = 100.0; // Negative LLR (match)
        }
    }

    // Flip bit 0 to force it to miss the hard-decoding fast path (hamming=0)
    intensities[0] = if (orig_code & 1) == 1 { 100.0 } else { 200.0 };

    let code = SoftStrategy::from_intensities(&intensities[..36], &thresholds[..36]);

    // Simulate 200 candidates
    bencher.bench_local(move || {
        let mut count = 0;
        for _ in 0..200 {
            if let Some((id, _, _)) = SoftStrategy::decode(&code, &decoder, 2)
                && id == 42
            {
                count += 1;
            }
        }
        divan::black_box(count);
    });
}
#[bench]
fn bench_soft_decoding_36h11_200_candidates(bencher: divan::Bencher) {
    use locus_core::bench_api::AprilTag36h11;
    let decoder = AprilTag36h11;
    let dict = locus_core::dictionaries::get_dictionary(locus_core::TagFamily::AprilTag36h11);
    // id=100 is a valid ID for 36h11
    let code_val = dict.get_code(100).unwrap();

    let mut intensities = [100.0f64; 64];
    let thresholds = [150.0f64; 64];
    for (i, (intensity, _threshold)) in intensities
        .iter_mut()
        .zip(thresholds.iter())
        .enumerate()
        .take(36)
    {
        if (code_val >> i) & 1 == 1 {
            *intensity = 200.0;
        } else {
            *intensity = 100.0;
        }
    }

    // Flip bit 0
    intensities[0] = if (code_val & 1) == 1 { 100.0 } else { 200.0 };

    let code = SoftStrategy::from_intensities(&intensities[..36], &thresholds[..36]);

    bencher.bench_local(move || {
        let mut count = 0;
        for _ in 0..200 {
            if let Some((id, _, _)) = SoftStrategy::decode(&code, &decoder, 2)
                && id == 100
            {
                count += 1;
            }
        }
        divan::black_box(count);
    });
}
