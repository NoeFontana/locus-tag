//! Tests for the SoA Homography Computation.

use locus_core::{DetectionBatch, Point2f, Matrix3x3};
use locus_core::decoder::compute_homographies_soa;

#[test]
fn test_compute_homographies_soa_interface() {
    let mut batch = DetectionBatch::new();
    
    // Setup a simple square quad at index 0
    batch.corners[0] = Point2f { x: 0.0, y: 0.0 };
    batch.corners[1] = Point2f { x: 10.0, y: 0.0 };
    batch.corners[2] = Point2f { x: 10.0, y: 10.0 };
    batch.corners[3] = Point2f { x: 0.0, y: 10.0 };
    
    let n = 1;
    
    // This should fail to compile because compute_homographies_soa is not defined yet.
    compute_homographies_soa(&batch.corners[0..n*4], &mut batch.homographies[0..n]);
    
    // Check if homography was populated (not all zeros)
    let h = batch.homographies[0];
    let sum: f32 = h.data.iter().sum();
    assert!(sum != 0.0);
}
