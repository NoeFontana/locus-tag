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
//! Benchmarks for TagDictionary decoding performance.
//!
//! Run with `cargo bench --bench dictionary_bench`.

use divan::Bencher;
use locus_core::TagFamily;
use locus_core::bench_api::get_dictionary;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[divan::bench]
fn bench_36h11_exact(bencher: Bencher) {
    let dict = get_dictionary(TagFamily::AprilTag36h11);
    // Pick a code from the middle of the dictionary
    let code_idx = 300usize;
    let code = dict.get_code(code_idx as u16).unwrap();

    bencher.bench_local(move || {
        // Exact match (max_hamming = 0) should hit the HashMap
        divan::black_box(dict.decode(code, 0))
    });
}

#[divan::bench]
fn bench_36h11_exact_rotated(bencher: Bencher) {
    let dict = get_dictionary(TagFamily::AprilTag36h11);
    let code_idx = 300usize;
    // Get the 90-degree rotated code directly from the dictionary
    let code = dict.codes[code_idx * 4 + 1];

    bencher.bench_local(move || {
        // Rotated match means it will miss the first HashMap lookup but hit subsequent ones
        // TagDictionary tries 4 rotations for exact match
        divan::black_box(dict.decode(code, 0))
    });
}

#[divan::bench]
fn bench_36h11_hamming_1(bencher: Bencher) {
    let dict = get_dictionary(TagFamily::AprilTag36h11);
    let code_idx = 300usize;
    let code = dict.get_code(code_idx as u16).unwrap();
    // Flip 1 bit (at pos 0)
    let noisy = code ^ 1;

    bencher.bench_local(move || {
        // Hamming 1 means linear scan of all 587 codes * 4 rotations
        divan::black_box(dict.decode(noisy, 1))
    });
}

#[divan::bench]
fn bench_36h11_hamming_2(bencher: Bencher) {
    let dict = get_dictionary(TagFamily::AprilTag36h11);
    let code_idx = 300usize;
    let code = dict.get_code(code_idx as u16).unwrap();
    // Flip 2 bits
    let noisy = code ^ 3;

    bencher.bench_local(move || divan::black_box(dict.decode(noisy, 2)));
}

#[divan::bench]
fn bench_aruco_exact(bencher: Bencher) {
    let dict = get_dictionary(TagFamily::ArUco4x4_50);
    let code_idx = 25usize;
    let code = dict.get_code(code_idx as u16).unwrap();

    bencher.bench_local(move || divan::black_box(dict.decode(code, 0)));
}

#[divan::bench]
fn bench_aruco_hamming_1(bencher: Bencher) {
    let dict = get_dictionary(TagFamily::ArUco4x4_50);
    let code_idx = 25usize;
    let code = dict.get_code(code_idx as u16).unwrap();
    // Flip 1 bit
    let noisy = code ^ 1;

    bencher.bench_local(move || {
        // Smaller dictionary (50 codes) -> faster linear scan
        divan::black_box(dict.decode(noisy, 1))
    });
}

#[divan::bench]
fn bench_rejection(bencher: Bencher) {
    let dict = get_dictionary(TagFamily::AprilTag36h11);
    // Random bits unlikely to be a valid tag
    let noise = 0xAAAA_AAAA_AAAA_AAAA;

    bencher.bench_local(move || divan::black_box(dict.decode(noise, 1)));
}
