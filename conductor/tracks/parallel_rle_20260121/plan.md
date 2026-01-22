# Implementation Plan - Parallelize RLE Pass in Segmentation

## Phase 1: Baseline & TDD Setup
- [x] Task: Establish Performance Baseline
    - [x] Run `cargo test --release --test regression_icra2020` and record current segmentation latency metrics. [baseline: fixtures=198ms, icra_forward=120ms]
- [x] Task: Create RLE Correctness Test
    - [x] Create `crates/locus-core/tests/segmentation_parallel_test.rs`.
    - [x] Write a test that compares the output of `label_components_with_stats` (sequential) against a new parallel-capable version (or just ensuring the current one passes before we change it).
    - [x] Focus on complex patterns and large images to ensure edge cases in run merging are handled.

## Phase 2: Implementation [checkpoint: 9529d1f]
- [x] Task: Implement Parallel RLE Extraction
    - [x] Modify `crates/locus-core/src/segmentation.rs`:
        - [x] Identify the sequential RLE loop in `label_components_with_stats`.
        - [x] Replace with a parallel iterator (e.g., `binary.par_chunks(width)`).
        - [x] Implement local collection of runs per task/thread.
        - [x] Flatten/Merge the local collections into the main run buffer.
    - [x] Ensure the implementation adheres to "Zero-Allocation" principles by using the `Bump` arena for local task results if possible, or efficient vector reuse.
- [x] Task: Verify Compilation & Basic Tests
    - [x] Run `cargo check`.
    - [x] Run `crates/locus-core/tests/segmentation_parallel_test.rs`.
- [x] Task: Conductor - User Manual Verification 'Implementation' (Protocol in workflow.md)

## Phase 3: Verification & Performance
- [x] Task: Regression Testing
    - [x] Run `cargo test --release --test regression_icra2020` and verify Recall/RMSE are unchanged.
- [x] Task: Performance Evaluation
    - [x] Compare `regression_icra2020.rs` latency metrics with Phase 1 baseline.
    - [x] Verify speedup on 4K resolutions. [Speedup measured: ~13% on noisy 4K]
- [x] Task: Conductor - User Manual Verification 'Verification' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Verification' (Protocol in workflow.md)
