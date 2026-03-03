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

## Phase 2: Refactor Quad Extraction [checkpoint: 5cd2b27]
### Task: Quad Extraction SoA Refactor
- [x] **Task: Write Failing Tests: Quad Extraction Slices** bb5654d
    - [ ] Define tests that provide empty arrays and expect `N` quads to be written correctly.
- [x] **Task: Implement Sequential Write-Only Quad Extractor** bb5654d
    - [ ] Refactor `extract_quads` to take mutable slices and populate `corners` and `status_mask`.
    - [ ] Return count `N`.
- [x] **Task: Phase Completion Verification and Checkpointing** 5cd2b27
    - [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Isolated Math Passes (Homography & Decoding) [checkpoint: 90f1b0a]
### Task: Homography Pass Refactor
- [x] **Task: Write Failing Tests: Homography Computation** 6fce68a
    - [ ] Verify that homographies are correctly computed from corners in a pure-function loop.
- [x] **Task: Implement Parallel Homography Pass** 6fce68a
    - [ ] Use `rayon` to parallelize homography computation across `[0..N]`.
### Task: SIMD Decoding Pass Refactor
- [x] **Task: Write Failing Tests: SIMD Decoding** 6998cda
    - [ ] Verify bit extraction into `payloads` and `error_rates`.
- [x] **Task: Implement SIMD-Accelerated Decoding Pass** 6998cda
    - [ ] Refactor decoder to use `DetectionBatch` slices and SIMD-optimized sampling.
- [x] **Task: Phase Completion Verification and Checkpointing** 90f1b0a
    - [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Pose Refinement & FFI Boundary [checkpoint: 39cfd3b]
### Task: Pose Refinement Partitioning
- [x] **Task: Write Failing Tests: Pose Refinement Partitioning** 60977d3
    - [ ] Ensure valid candidates are partitioned to the front `[0..V]` before pose computation.
- [x] **Task: Implement Partitioned Pose Solver** 60977d3
    - [ ] Refactor LM solver to iterate over valid indices in the batch.
### Task: FFI Late Reassembly
- [x] **Task: Write Failing Tests: Detection Object Compatibility** 60977d3
    - [ ] Verify the PyO3 wrapper returns `List[Detection]` from the SoA batch.
- [x] **Task: Implement Python Object Reassembly Loop** 60977d3
    - [ ] Add the late reassembly loop in the `pyo3` binding logic.
- [x] **Task: Phase Completion Verification and Checkpointing** 39cfd3b
    - [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)

## Phase 5: Final Performance Validation [checkpoint: ea4fbf0]
### Task: Performance Audit & Baseline Verification
- [x] **Task: Benchmarking and Cache Miss Audit** ea4fbf0
    - [ ] Run `locus_bench` and compare against the baseline to verify improvements.
- [x] **Task: Final Phase Completion Verification and Checkpointing** ea4fbf0
    - [ ] Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)
