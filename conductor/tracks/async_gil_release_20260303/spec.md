# Specification: Asynchronous GIL Release (Track: async_gil_release)

## Overview
Locus aims to minimize total system latency in robotics pipelines. A major bottleneck is the blocking nature of the Python Global Interpreter Lock (GIL). This track implements asynchronous GIL release for the high-performance `Detector` methods and removes the legacy, function-based API to streamline the library.

## Functional Requirements
1. **Legacy API Deprecation & Removal:** Permanently remove the function-based Python API (`detect_tags`, `detect_tags_with_stats`, `dummy_detect`, `debug_threshold`, and `debug_segmentation`) to ensure all ingestion flows follow the high-performance `Detector` class.
2. **Purity Audit:** Verify that `locus-core` contains no `PyO3` objects, references, or dependencies. The core detection pipeline must operate exclusively on standard Rust primitives.
3. **Metadata Snapshotting:** Before entering the GIL-free boundary, extract all necessary metadata from Python objects (e.g., camera intrinsics, tag family configurations) and copy them into lightweight, native Rust structs.
4. **Thread-Release Boundary:** Wrap the actual `detector.detect()` execution block inside a `pyo3::Python::allow_threads` closure. This signals to the Python interpreter that it is free to resume executing other Python code (like running a neural network inference on the previous frame) while Rust crunches the current frame.
5. **Thread-Safe Visualization:** Ensure `rerun` logging calls within the detection pipeline remain thread-safe and functional while the GIL is released.
6. **Reacquisition and Serialization:** Once the Rust hot-loop finishes finding the 6D poses, the thread must wait to reacquire the GIL. Once locked, safely translate the native Rust output structs into Python dataclasses or NumPy arrays to return to the user.

## Non-Functional Requirements
- **Latency Invariance:** The overhead of releasing and re-acquiring the GIL should be negligible (< 5µs).
- **Architecture Simplicity:** By removing legacy entry points, the `locus-py` binding layer becomes more maintainable and focused on the `Detector` object model.

## Acceptance Criteria
- [ ] Legacy functions (`detect_tags`, etc.) are removed from the `locus` Python module and no longer accessible.
- [ ] Concurrent Python tasks (e.g., `numpy.dot` or a dummy CPU-heavy loop) execute in parallel with `Detector.detect()` as verified by a concurrency trace.
- [ ] All remaining `Detector` methods (`detect`, `detect_with_stats`, `detect_full`) function correctly while releasing the GIL.
- [ ] `rerun` visualizations are correctly emitted and visible in the viewer from the background thread.
- [ ] All existing Python tests that used legacy functions are updated or removed.

## Out of Scope
- Refactoring `locus-core`'s internal parallelism (`rayon` integration).
