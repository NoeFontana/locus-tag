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
    compute_homographies_soa(&corners, &mut homographies);
}
