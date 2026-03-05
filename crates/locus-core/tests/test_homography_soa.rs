use locus_core::bench_api::*;

#[test]
fn test_homography_soa_empty() {
    let mut corners = vec![];
    let mut homographies = vec![];
    compute_homographies_soa(&corners, &mut homographies);
}
