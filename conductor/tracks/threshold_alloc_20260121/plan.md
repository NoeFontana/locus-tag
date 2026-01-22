# Implementation Plan - Elimination of Hot-Path Allocations

## Phase 1: Benchmark & Baseline
- [x] Task: Baseline Performance Measurement
    - [x] Run `cargo test --release --test regression_icra2020` to capture current metrics (Recall, RMSE, Latency).
    - [x] Save the output/report for comparison. [baseline-metrics-recorded]

## Phase 2: Refactoring [checkpoint: 68169e5]
- [x] Task: TDD - Refactor Setup
    - [x] Create a dedicated unit test in a new test file `crates/locus-core/tests/threshold_alloc_test.rs` that specifically exercises the `apply_threshold` method on a large synthetic image.
    - [x] Ensure this test passes with the current implementation.
- [x] Task: Implement `for_each_init`
    - [x] Modify `crates/locus-core/src/threshold.rs`:
        - [x] Locate the `apply_threshold` parallel loop.
        - [x] Refactor to use `par_chunks_mut(...).enumerate().for_each_init(...)`.
        - [x] Init closure: Allocate `row_thresholds` and `row_valid` vectors with `Vec::with_capacity(width)`.
        - [x] Loop closure: `clear()` and `resize()` (or unsafe `set_len` + write if valid) the reused buffers. **Constraint:** Ensure correctness (zero-init if algorithm relies on it).
    - [x] Verify the code compiles.
- [x] Task: Conductor - User Manual Verification 'Refactoring' (Protocol in workflow.md)

## Phase 3: Verification
- [x] Task: Verify Correctness
    - [x] Run `cargo test --release --test threshold_alloc_test` to ensure the logic still holds.
    - [x] Run `cargo test --release --test regression_icra2020` to verify no regressions in Recall/RMSE.
- [x] Task: Verify Performance
    - [x] Compare new `regression_icra2020` latency stats against the baseline.
    - [x] (Optional) Run `cargo bench` if specific micro-benchmarks exist for thresholding.
- [x] Task: Cleanup
    - [x] Remove the temporary `threshold_alloc_test.rs` if it duplicates existing coverage, or promote it if valuable.
- [x] Task: Conductor - User Manual Verification 'Verification' (Protocol in workflow.md)
