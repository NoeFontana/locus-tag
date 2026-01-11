#![allow(clippy::unwrap_used)]
//! Benchmarks for TagDictionary decoding performance.
//!
//! Run with `cargo bench --bench dictionary_bench`.

use divan::Bencher;
use locus_core::dictionaries::{APRILTAG_36H11, ARUCO_4X4_50};

fn main() {
    divan::main();
}

#[divan::bench]
fn bench_36h11_exact(bencher: Bencher) {
    // Pick a code from the middle of the dictionary
    let code_idx = 300;
    let code = APRILTAG_36H11.get_code(code_idx).unwrap();

    bencher.bench_local(move || {
        // Exact match (max_hamming = 0) should hit the HashMap
        divan::black_box(APRILTAG_36H11.decode(code, 0))
    });
}

#[divan::bench]
fn bench_36h11_exact_rotated(bencher: Bencher) {
    let code_idx = 300;
    let code = APRILTAG_36H11.get_code(code_idx).unwrap();
    // Rotate it once
    let code = locus_core::dictionaries::rotate90(code, 6);

    bencher.bench_local(move || {
        // Rotated match means it will miss the first HashMap lookup but hit subsequent ones
        // TagDictionary tries 4 rotations for exact match
        divan::black_box(APRILTAG_36H11.decode(code, 0))
    });
}

#[divan::bench]
fn bench_36h11_hamming_1(bencher: Bencher) {
    let code_idx = 300;
    let code = APRILTAG_36H11.get_code(code_idx).unwrap();
    // Flip 1 bit (at pos 0)
    let noisy = code ^ 1;

    bencher.bench_local(move || {
        // Hamming 1 means linear scan of all 587 codes * 4 rotations
        divan::black_box(APRILTAG_36H11.decode(noisy, 1))
    });
}

#[divan::bench]
fn bench_36h11_hamming_2(bencher: Bencher) {
    let code_idx = 300;
    let code = APRILTAG_36H11.get_code(code_idx).unwrap();
    // Flip 2 bits
    let noisy = code ^ 3;

    bencher.bench_local(move || divan::black_box(APRILTAG_36H11.decode(noisy, 2)));
}

#[divan::bench]
fn bench_aruco_exact(bencher: Bencher) {
    let code_idx = 25;
    let code = ARUCO_4X4_50.get_code(code_idx).unwrap();

    bencher.bench_local(move || divan::black_box(ARUCO_4X4_50.decode(code, 0)));
}

#[divan::bench]
fn bench_aruco_hamming_1(bencher: Bencher) {
    let code_idx = 25;
    let code = ARUCO_4X4_50.get_code(code_idx).unwrap();
    // Flip 1 bit
    let noisy = code ^ 1;

    bencher.bench_local(move || {
        // Smaller dictionary (50 codes) -> faster linear scan
        divan::black_box(ARUCO_4X4_50.decode(noisy, 1))
    });
}

#[divan::bench]
fn bench_rejection(bencher: Bencher) {
    // Random bits unlikely to be a valid tag
    let noise = 0xAAAA_AAAA_AAAA_AAAA;

    bencher.bench_local(move || divan::black_box(APRILTAG_36H11.decode(noise, 1)));
}
