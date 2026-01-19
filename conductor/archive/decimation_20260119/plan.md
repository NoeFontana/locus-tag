# Implementation Plan: Decimation Strategy

This plan outlines the steps to implement a configurable decimation strategy in Locus to improve performance on high-resolution streams while maintaining sub-pixel accuracy.

## Phase 1: Core Rust Implementation (locus-core)

In this phase, we will update the internal `locus-core` detection pipeline to support a decimation factor.

- [x] Task: Update internal image representation and thresholding to handle decimation [fc1413d]
    - [x] Write Tests: Create unit tests in `crates/locus-core/src/threshold.rs` to verify thresholding on decimated inputs.
    - [x] Implement: Modify the adaptive thresholding logic to skip pixels based on the decimation factor $.
- [x] Task: Update segmentation and quad detection for decimated coordinate space [dfd6f5a]
    - [x] Write Tests: Create tests in `crates/locus-core/src/segmentation.rs` to verify component extraction on decimated grids.
    - [x] Implement: Adjust Union-Find and Quad extraction to operate on the reduced coordinate space.
- [x] Task: Implement high-resolution sub-pixel refinement [dfd6f5a]
    - [x] Write Tests: Create a test case where a decimated quad is refined against the original high-res gradient.
    - [x] Implement: Scale the decimated quad corners by $ and pass them to the existing gradient-based refinement engine using the original image.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Rust Implementation' [8995cb5]

## Phase 2: Python Bindings & API (locus-py)

In this phase, we will expose the `decimation` parameter to the Python API and ensure seamless integration.

- [x] Task: Expose `decimation` in PyO3 bindings [8995cb5]
    - [x] Write Tests: Add a test case in `tests/test_python_api.py` that passes a `decimation` argument.
    - [x] Implement: Update `crates/locus-py/src/lib.rs` to accept an optional `decimation` integer in `detect_tags`.
- [x] Task: Update Python configuration and defaults [8995cb5]
    - [x] Write Tests: Verify that the default decimation is 1.
    - [x] Implement: Update `locus/_config.py` (if applicable) to include decimation settings.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Python Bindings & API' [8995cb5]

## Phase 3: Validation & Benchmarking

In this final phase, we will verify the performance and accuracy gains.

- [x] Task: Performance benchmarking (Rust) [8995cb5]
    - [x] Write Tests: Create or update a Criterion benchmark in `crates/locus-core/benches/` to compare latency of =1$ vs =2$ on 1080p images.
    - [x] Implement: Run `cargo bench` and document the speedup in the preprocessing phases.
- [x] Task: Accuracy validation (RMSE, Rust) [8995cb5]
    - [x] Write Tests: Use `test_decimation.rs` to compare RMSE and Recall between =1$ and =2$.
    - [x] Implement: Verify that RMSE and recall degradation is minimal (<1%) as specified in the acceptance criteria.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Validation & Benchmarking' [8995cb5]
