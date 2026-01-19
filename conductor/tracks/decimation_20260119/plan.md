# Implementation Plan: Decimation Strategy

This plan outlines the steps to implement a configurable decimation strategy in Locus to improve performance on high-resolution streams while maintaining sub-pixel accuracy.

## Phase 1: Core Rust Implementation (locus-core)

In this phase, we will update the internal `locus-core` detection pipeline to support a decimation factor.

- [ ] Task: Update internal image representation and thresholding to handle decimation
    - [ ] Write Tests: Create unit tests in `crates/locus-core/src/threshold.rs` to verify thresholding on decimated inputs.
    - [ ] Implement: Modify the adaptive thresholding logic to skip pixels based on the decimation factor $.
- [ ] Task: Update segmentation and quad detection for decimated coordinate space
    - [ ] Write Tests: Create tests in `crates/locus-core/src/segmentation.rs` to verify component extraction on decimated grids.
    - [ ] Implement: Adjust Union-Find and Quad extraction to operate on the reduced coordinate space.
- [ ] Task: Implement high-resolution sub-pixel refinement
    - [ ] Write Tests: Create a test case where a decimated quad is refined against the original high-res gradient.
    - [ ] Implement: Scale the decimated quad corners by $ and pass them to the existing gradient-based refinement engine using the original image.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Core Rust Implementation' (Protocol in workflow.md)

## Phase 2: Python Bindings & API (locus-py)

In this phase, we will expose the `decimation` parameter to the Python API and ensure seamless integration.

- [ ] Task: Expose `decimation` in PyO3 bindings
    - [ ] Write Tests: Add a test case in `tests/test_python_api.py` that passes a `decimation` argument.
    - [ ] Implement: Update `crates/locus-py/src/lib.rs` to accept an optional `decimation` integer in `detect_tags`.
- [ ] Task: Update Python configuration and defaults
    - [ ] Write Tests: Verify that the default decimation is 1.
    - [ ] Implement: Update `locus/_config.py` (if applicable) to include decimation settings.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Python Bindings & API' (Protocol in workflow.md)

## Phase 3: Validation & Benchmarking

In this final phase, we will verify the performance and accuracy gains.

- [ ] Task: Performance benchmarking (Rust)
    - [ ] Write Tests: Create or update a Criterion benchmark in `crates/locus-core/benches/` to compare latency of =1$ vs =2$ on 1080p images.
    - [ ] Implement: Run `cargo bench` and document the speedup in the preprocessing phases.
- [ ] Task: Accuracy validation (RMSE)
    - [ ] Write Tests: Use `tests/evaluate_forward_performance.py` to compare RMSE and Recall between =1$ and =2$.
    - [ ] Implement: Verify that RMSE degradation is minimal as specified in the acceptance criteria.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Validation & Benchmarking' (Protocol in workflow.md)
