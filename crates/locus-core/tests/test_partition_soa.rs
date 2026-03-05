use locus_core::bench_api::*;

#[test]
fn test_partition_basic() {
    let mut batch = DetectionBatch::new();
    batch.status_mask[0] = CandidateState::Valid;
    batch.status_mask[1] = CandidateState::FailedDecode;
    batch.status_mask[2] = CandidateState::Valid;
    
    batch.payloads[0] = 111;
    batch.payloads[1] = 222;
    batch.payloads[2] = 333;

    let v = partition_batch_soa(&mut batch, 3);
    assert_eq!(v, 2);
    assert_eq!(batch.payloads[0], 111);
    assert_eq!(batch.payloads[1], 333);
}
