//! Tests for the SoA Partitioning.

use locus_core::{DetectionBatch, CandidateState};
use locus_core::batch::partition_batch_soa;

#[test]
fn test_partition_batch_soa_interface() {
    let mut batch = DetectionBatch::new();
    
    // Setup: 0: Failed, 1: Valid, 2: Failed, 3: Valid
    batch.status_mask[0] = CandidateState::FailedDecode;
    batch.status_mask[1] = CandidateState::Valid;
    batch.status_mask[2] = CandidateState::FailedDecode;
    batch.status_mask[3] = CandidateState::Valid;
    
    // Tag payloads to verify they move with the state
    batch.payloads[1] = 111;
    batch.payloads[3] = 333;
    
    let n = 4;
    
    // This should fail to compile because partition_batch_soa is not defined yet.
    let v = partition_batch_soa(&mut batch, n);
    
    assert_eq!(v, 2);
    assert_eq!(batch.status_mask[0], CandidateState::Valid);
    assert_eq!(batch.status_mask[1], CandidateState::Valid);
    assert_eq!(batch.payloads[0], 111);
    assert_eq!(batch.payloads[1], 333);
}
