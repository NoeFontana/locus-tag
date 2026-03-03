# Implementation Plan: Detection Pipeline SoA Migration

## Phase 1: DetectionBatch Core Architecture [checkpoint: 5760461]
### Task: DetectionBatch Struct Implementation
- [x] **Task: Write Failing Tests: DetectionBatch Invariants** faf2f58
    - [ ] Create `tests/test_detection_batch.rs` to verify zero-allocation initialization and SIMD alignment.
    - [ ] Assert `DetectionBatch` can be instantiated with `MAX_CANDIDATES = 256`.
- [x] **Task: Implement DetectionBatch and Memory Layout** faf2f58
    - [ ] Define `DetectionBatch` struct with aligned parallel arrays for `corners`, `homographies`, `payloads`, `error_rates`, `poses`, and `status_mask`.
    - [ ] Ensure 32-byte alignment for SIMD-accessible fields.
- [x] **Task: Phase Completion Verification and Checkpointing** 5760461
    - [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Refactor Quad Extraction
### Task: Quad Extraction SoA Refactor
- [ ] **Task: Write Failing Tests: Quad Extraction Slices**
    - [ ] Define tests that provide empty arrays and expect `N` quads to be written correctly.
- [ ] **Task: Implement Sequential Write-Only Quad Extractor**
    - [ ] Refactor `extract_quads` to take mutable slices and populate `corners` and `status_mask`.
    - [ ] Return count `N`.
- [ ] **Task: Phase Completion Verification and Checkpointing**
    - [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Isolated Math Passes (Homography & Decoding)
### Task: Homography Pass Refactor
- [ ] **Task: Write Failing Tests: Homography Computation**
    - [ ] Verify that homographies are correctly computed from corners in a pure-function loop.
- [ ] **Task: Implement Parallel Homography Pass**
    - [ ] Use `rayon` to parallelize homography computation across `[0..N]`.
### Task: SIMD Decoding Pass Refactor
- [ ] **Task: Write Failing Tests: SIMD Decoding**
    - [ ] Verify bit extraction into `payloads` and `error_rates`.
- [ ] **Task: Implement SIMD-Accelerated Decoding Pass**
    - [ ] Refactor decoder to use `DetectionBatch` slices and SIMD-optimized sampling.
- [ ] **Task: Phase Completion Verification and Checkpointing**
    - [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Pose Refinement & FFI Boundary
### Task: Pose Refinement Partitioning
- [ ] **Task: Write Failing Tests: Pose Refinement Partitioning**
    - [ ] Ensure valid candidates are partitioned to the front `[0..V]` before pose computation.
- [ ] **Task: Implement Partitioned Pose Solver**
    - [ ] Refactor LM solver to iterate over valid indices in the batch.
### Task: FFI Late Reassembly
- [ ] **Task: Write Failing Tests: Detection Object Compatibility**
    - [ ] Verify the PyO3 wrapper returns `List[Detection]` from the SoA batch.
- [ ] **Task: Implement Python Object Reassembly Loop**
    - [ ] Add the late reassembly loop in the `pyo3` binding logic.
- [ ] **Task: Phase Completion Verification and Checkpointing**
    - [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)

## Phase 5: Final Performance Validation
### Task: Performance Audit & Baseline Verification
- [ ] **Task: Benchmarking and Cache Miss Audit**
    - [ ] Run `locus_bench` and compare against the baseline to verify improvements.
- [ ] **Task: Final Phase Completion Verification and Checkpointing**
    - [ ] Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)
