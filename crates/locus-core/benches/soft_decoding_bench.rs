#![allow(missing_docs, clippy::unwrap_used)]
use divan::bench;
use locus_core::decoder::AprilTag36h11;
use locus_core::strategy::{DecodingStrategy, SoftStrategy};

fn main() {
    divan::main();
}

#[bench]
fn bench_soft_decoding_200_candidates(bencher: divan::Bencher) {
    let decoder = AprilTag36h11;
    // Create a SoftCode from a known match (ID 42)
    let orig_code = locus_core::dictionaries::APRILTAG_36H11
        .get_code(42)
        .unwrap();

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
fn bench_soft_decoding_41h12_200_candidates(bencher: divan::Bencher) {
    use locus_core::decoder::AprilTag41h12;
    let decoder = AprilTag41h12;
    // id=100 is a valid ID for 41h12
    let orig_code = locus_core::dictionaries::APRILTAG_41H12
        .get_code(100)
        .unwrap();

    let mut intensities = [100.0f64; 64];
    let thresholds = [150.0f64; 64];
    for (i, (intensity, _threshold)) in intensities
        .iter_mut()
        .zip(thresholds.iter())
        .enumerate()
        .take(41)
    {
        if (orig_code >> i) & 1 == 1 {
            *intensity = 200.0;
        } else {
            *intensity = 100.0;
        }
    }

    // Flip bit 0
    intensities[0] = if (orig_code & 1) == 1 { 100.0 } else { 200.0 };

    let code = SoftStrategy::from_intensities(&intensities[..41], &thresholds[..41]);

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
