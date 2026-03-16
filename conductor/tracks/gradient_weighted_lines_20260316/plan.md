# Implementation Plan: Gradient-Weighted Line Fitting (GWLF)

## Phase 1: Configuration and API Interface
- [x] Task: TDD - Configuration Toggle (4f0ee8e)
    - [ ] Add `RefinementMethod` enum to Rust core (ERF vs GWLF).
    - [ ] Add `refinement_method` field to `LocusConfig`.
    - [ ] Implement toggle support in Python bindings (`locus-py`).
- [ ] Task: Conductor - User Manual Verification 'Configuration and API Interface' (Protocol in workflow.md)

## Phase 2: Core Mathematical Components
- [ ] Task: TDD - Moment Accumulation
    - [ ] Write failing test for gradient-weighted moment accumulation on synthetic edge data.
    - [ ] Implement single-pass, cache-coherent accumulation loop (`sum_w`, `sum_wx`, etc.).
- [ ] Task: TDD - Analytic Eigendecomposition
    - [ ] Write failing test for 2x2 covariance solver using known matrices.
    - [ ] Implement analytic solution for smallest eigenvalue and normal vector.
- [ ] Task: Conductor - User Manual Verification 'Core Mathematical Components' (Protocol in workflow.md)

## Phase 3: Edge Fitting and Corner Intersection
- [ ] Task: TDD - Line Intersection
    - [ ] Write failing test for homogeneous line intersection (`l1 x l2`).
    - [ ] Implement projective-to-Cartesian coordinate conversion.
- [ ] Task: TDD - Sanity Gate and Robustness
    - [ ] Write failing test for 3.0-pixel fallback logic.
    - [ ] Implement distance-based fallback to coarse corner coordinates.
- [ ] Task: Conductor - User Manual Verification 'Edge Fitting and Corner Intersection' (Protocol in workflow.md)

## Phase 4: Benchmarking and Verification
- [ ] Task: TDD - Telemetry and Monitoring
    - [ ] Implement `Fallback Frequency` counter and `Refinement Delta` reporting.
- [ ] Task: Performance and Accuracy Validation
    - [ ] Run benchmarks to verify latency is < 1.0ms per quad.
    - [ ] Run regression tests on ICRA 2020 to verify RMSE < 0.8px.
- [ ] Task: Conductor - User Manual Verification 'Benchmarking and Verification' (Protocol in workflow.md)
