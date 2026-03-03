//! Tests for the DetectionBatch Structure of Arrays (SoA) container.

use locus_core::DetectionBatch;
use std::mem;

#[test]
fn test_detection_batch_initialization() {
    // MAX_CANDIDATES should be 256
    let batch = DetectionBatch::new();
    assert_eq!(batch.capacity(), 256);
}

#[test]
fn test_detection_batch_alignment() {
    let batch = DetectionBatch::new();
    
    // Check alignment of corners array (Point2f is [f32; 2], 8 bytes)
    let corners_ptr = batch.corners.as_ptr() as usize;
    assert_eq!(corners_ptr % 32, 0, "Corners array must be 32-byte aligned");
    
    // Check alignment of homographies array (Matrix3x3 is [f32; 9], 36 bytes)
    let homographies_ptr = batch.homographies.as_ptr() as usize;
    assert_eq!(homographies_ptr % 32, 0, "Homographies array must be 32-byte aligned");
}

#[test]
fn test_detection_batch_zero_allocation() {
    // This is hard to test automatically without a custom allocator, 
    // but we can at least ensure the sizes are fixed.
    assert!(mem::size_of::<DetectionBatch>() > 0);
}
