# Implementation Plan: Asynchronous GIL Release (Track: async_gil_release)

## Phase 1: Legacy API Removal & Cleanup
- [ ] Task: Remove legacy functions (`detect_tags`, `detect_tags_with_stats`, `dummy_detect`, `debug_threshold`, `debug_segmentation`) from `crates/locus-py/src/lib.rs`.
- [ ] Task: Update existing Python tests (`tests/test_python_api.py`, `tests/test_non_contiguous.py`, etc.) to remove all references to legacy functions.
- [ ] Task: Rebuild and verify that `locus` only exports the `Detector` and supporting classes.
- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Core Purity Audit & Metadata Snapshotting
- [ ] Task: Perform a purity audit on `locus-core` to ensure no `PyO3` types or Python objects are held in the hot path.
- [ ] Task: Refactor the `locus-py` wrapper to ensure all `DetectOptions` and `DetectorConfig` values are copied to native Rust structs *before* the GIL is released.
- [ ] Task: Verify that `prepare_image_input` (zero-copy view) is correctly separated from the GIL-free execution block.
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: GIL Release Implementation (Green Phase)
- [ ] Task: Implement `pyo3::Python::allow_threads` in `Detector.detect()`.
- [ ] Task: Implement `pyo3::Python::allow_threads` in `Detector.detect_with_stats()`.
- [ ] Task: Implement `pyo3::Python::allow_threads` in `Detector.detect_full()`.
- [ ] Task: Implement `pyo3::Python::allow_threads` in `Detector.extract_candidates()`.
- [ ] Task: Verify thread-safe `rerun` visualization logging from within the GIL-free background thread.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Concurrency Testing & Performance Validation
- [ ] Task: Create `tests/test_concurrency.py` to simulate simultaneous execution of Locus and heavy NumPy computations.
- [ ] Task: Implement a trace script to visualize and confirm overlapping execution periods.
- [ ] Task: Run performance benchmarks to confirm zero overhead (< 5µs) for single-threaded detection.
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)

## Phase 5: Final Verification & Checkpointing
- [ ] Task: Final code audit for any remaining `PyObject` access in the `allow_threads` block.
- [ ] Task: Update API documentation (if any) to reflect the removal of legacy functions.
- [ ] Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)
