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
use locus_core::bench_api::*;

/// Apply a `Matrix3x3` (column-major, f32 storage) to a 2D point, returning the
/// perspective-divided image coordinates computed in f64.
fn apply_h(h: &Matrix3x3, x: f64, y: f64) -> [f64; 2] {
    let m = |r: usize, c: usize| f64::from(h.data[c * 3 + r]);
    let px = m(0, 0) * x + m(0, 1) * y + m(0, 2);
    let py = m(1, 0) * x + m(1, 1) * y + m(1, 2);
    let pw = m(2, 0) * x + m(2, 1) * y + m(2, 2);
    [px / pw, py / pw]
}

#[test]
fn test_homography_soa_empty() {
    let corners = vec![];
    let mut homographies = vec![];
    let status_mask = vec![];
    compute_homographies_soa(&corners, &status_mask, &mut homographies);

    // Zero-length boundary contract: no panic, output slice stays empty.
    assert!(homographies.is_empty());
}

#[test]
fn test_homography_soa_single() {
    // Canonical square vertices, in the order the pass maps them from.
    const CANON: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let quad = [
        Point2f { x: 0.0, y: 0.0 },
        Point2f { x: 10.0, y: 0.0 },
        Point2f { x: 10.0, y: 10.0 },
        Point2f { x: 0.0, y: 10.0 },
    ];
    let corners = vec![quad];
    let mut homographies = vec![Matrix3x3::default()];
    let status_mask = vec![CandidateState::Active];
    compute_homographies_soa(&corners, &status_mask, &mut homographies);

    // Real correctness check: the computed homography must map each canonical
    // square vertex onto the corresponding image quad corner. Tolerance is
    // f32-storage-limited (the 9 matrix entries are cast to f32 in the SoA
    // column), so ~1e-3 px is the tightest reliable bound here, still ~1e4x
    // stronger than the previous `data[0] != 0.0` smoke assertion.
    for (canon, corner) in CANON.iter().zip(quad.iter()) {
        let mapped = apply_h(&homographies[0], canon[0], canon[1]);
        assert!(
            (mapped[0] - f64::from(corner.x)).abs() < 1e-3
                && (mapped[1] - f64::from(corner.y)).abs() < 1e-3,
            "canonical {canon:?} mapped to {mapped:?}, expected ({}, {})",
            corner.x,
            corner.y,
        );
    }
}

#[test]
fn test_homography_soa_skip_inactive() {
    let corners = vec![[
        Point2f { x: 0.0, y: 0.0 },
        Point2f { x: 10.0, y: 0.0 },
        Point2f { x: 10.0, y: 10.0 },
        Point2f { x: 0.0, y: 10.0 },
    ]];
    let mut homographies = vec![Matrix3x3::default()];
    let status_mask = vec![CandidateState::FailedDecode];
    compute_homographies_soa(&corners, &status_mask, &mut homographies);

    // Homography should be zero
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(homographies[0].data[0], 0.0);
    }
}
