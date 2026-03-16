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
use locus_core::bench_api::*;

#[test]
fn test_homography_soa_empty() {
    let corners = vec![];
    let mut homographies = vec![];
    let status_mask = vec![];
    compute_homographies_soa(&corners, &status_mask, &mut homographies);
}

#[test]
fn test_homography_soa_single() {
    let corners = vec![[
        Point2f { x: 0.0, y: 0.0 },
        Point2f { x: 10.0, y: 0.0 },
        Point2f { x: 10.0, y: 10.0 },
        Point2f { x: 0.0, y: 10.0 },
    ]];
    let mut homographies = vec![Matrix3x3::default()];
    let status_mask = vec![CandidateState::Active];
    compute_homographies_soa(&corners, &status_mask, &mut homographies);
    
    // Non-zero homography
    assert!(homographies[0].data[0] != 0.0);
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
    assert_eq!(homographies[0].data[0], 0.0);
}
