# Implementation Plan: Gradient-Weighted Line Fitting (GWLF)

## Phase 1: Configuration and API Interface
- [x] Task: TDD - Configuration Toggle (4f0ee8e)
    - [ ] Add `RefinementMethod` enum to Rust core (ERF vs GWLF).
    - [ ] Add `refinement_method` field to `LocusConfig`.
    - [ ] Implement toggle support in Python bindings (`locus-py`).
- [ ] Task: Conductor - User Manual Verification 'Configuration and API Interface' (Protocol in workflow.md)

## Phase 2: Core Mathematical Components
- [x] Task: TDD - Moment Accumulation (88ec500)
    - [x] Write failing test for gradient-weighted moment accumulation on synthetic edge data.
    - [x] Implement single-pass, cache-coherent accumulation loop (`sum_w`, `sum_wx`, etc.).
- [x] Task: TDD - Analytic Eigendecomposition (88ec500)
    - [x] Write failing test for 2x2 covariance solver using known matrices.
    - [x] Implement analytic solution for smallest eigenvalue and normal vector.
- [ ] Task: Conductor - User Manual Verification 'Core Mathematical Components' (Protocol in workflow.md)

## Phase 3: Edge Fitting and Corner Intersection
- [x] Task: TDD - Line Intersection (354b1f5)
    - [x] Write failing test for homogeneous line intersection (`l1 x l2`).
    - [x] Implement projective-to-Cartesian coordinate conversion.
- [x] Task: TDD - Sanity Gate and Robustness (354b1f5)
    - [x] Write failing test for 3.0-pixel fallback logic.
    - [x] Implement distance-based fallback to coarse corner coordinates.
- [ ] Task: Conductor - User Manual Verification 'Edge Fitting and Corner Intersection' (Protocol in workflow.md)

## Phase 4: Benchmarking and Verification
- [x] Task: TDD - Telemetry and Monitoring (b56bcb7)
    - [x] Implement `Fallback Frequency` counter and `Refinement Delta` reporting.
- [x] Task: Performance and Accuracy Validation (5956d0e)
    - [x] Run benchmarks to verify latency is < 1.0ms per quad.
    - [x] Run regression tests on ICRA 2020 to verify RMSE < 0.8px.
- [x] Task: Conductor - User Manual Verification 'Benchmarking and Verification' (Protocol in workflow.md) (0a8a2f2)
