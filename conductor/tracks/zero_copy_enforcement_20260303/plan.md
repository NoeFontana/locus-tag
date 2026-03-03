# Implementation Plan: Zero-Copy Pointer Enforcement (Project 1)

## Phase 1: Performance Baseline & Test Environment [checkpoint: d2f22fd]
- [x] Task: Establish performance baseline for contiguous vs. non-contiguous arrays. 4feee69
    - [x] Run `python tests/test_ffi_overhead.py` to confirm the baseline for contiguous arrays.
    - [x] Create a benchmark script `scripts/bench_ingestion_penalty.py` to measure the overhead for non-contiguous arrays (the 110ms bottleneck).
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md) d2f22fd

## Phase 2: Red Phase - Implement Failing Tests
- [x] Task: Modify `tests/test_non_contiguous.py` to expect `ValueError` for non-contiguous arrays. 3202274
    - [x] Change `test_non_contiguous_ingestion` to use `pytest.raises(ValueError)`.
    - [x] Verify that these tests fail when run against the current "auto-copy" implementation.
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Green Phase - Implement Strict Zero-Copy Enforcement
- [ ] Task: Refactor `crates/locus-py/src/lib.rs` to enforce zero-copy.
    - [ ] Remove the `ImageInput::Owned` variant from the `ImageInput` enum.
    - [ ] Update `prepare_image_input` to throw a `ValueError` with the specified message if `stride_x != 1`.
    - [ ] Update high-performance `Detector` methods and legacy `detect_tags` functions to handle the change.
- [ ] Task: Verify the implementation with tests.
    - [ ] Run `pytest tests/test_non_contiguous.py` and confirm all tests pass.
    - [ ] Run `python tests/test_ffi_overhead.py` and confirm the overhead for contiguous arrays remains < 0.1ms.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Final Verification & Checkpointing
- [ ] Task: Perform final performance audit and code cleanup.
    - [ ] Run the new `scripts/bench_ingestion_penalty.py` and confirm that non-contiguous arrays are blocked.
    - [ ] Review `crates/locus-py/src/lib.rs` for any remaining implicit copies.
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
