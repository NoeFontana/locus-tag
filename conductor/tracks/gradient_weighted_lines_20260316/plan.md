# Implementation Plan: GWLF Refinement

**Track ID:** `gradient_weighted_lines_20260316`

## Phase 1: Configuration and API Interface
- [ ] Task: TDD - Add `Gwlf` to `CornerRefinementMode`
    - [ ] Add `Gwlf` variant to `CornerRefinementMode` in `crates/locus-core/src/config.rs`.
    - [ ] Update `DetectorConfigBuilder` and `DetectorBuilder` to handle the new mode.
- [ ] Task: TDD - Python Bindings for GWLF
    - [ ] Update `crates/locus-py/src/lib.rs` to expose `Gwlf` to Python.
- [ ] Task: Conductor - User Manual Verification 'Configuration and API Interface' (Protocol in workflow.md)

## Phase 2: Core Mathematical Components
- [ ] Task: TDD - Moment Accumulator and PCA
    - [ ] Implement `MomentAccumulator` in `crates/locus-core/src/gwlf.rs`.
    - [ ] Implement 2x2 symmetric eigendecomposition for PCA.
- [ ] Task: TDD - Homogeneous Line and Intersection
    - [ ] Implement `HomogeneousLine` with cross-product intersection.
- [ ] Task: Conductor - User Manual Verification 'Core Mathematical Components' (Protocol in workflow.md)

## Phase 3: Edge Fitting and Corner Intersection
- [ ] Task: TDD - Implement `refine_quad_gwlf`
    - [ ] Implement the main refinement loop in `crates/locus-core/src/gwlf.rs`.
    - [ ] Sample image gradients along quad edges and fit lines.
    - [ ] Compute corner intersections and Cartesian projections.
- [ ] Task: TDD - Integrate into Detection Pipeline
    - [ ] Call `refine_quad_gwlf` from `Detector::detect` when GWLF mode is enabled.
- [ ] Task: TDD - Telemetry and Monitoring
    - [ ] Implement `Fallback Frequency` counter and `Refinement Delta` reporting.
- [ ] Task: Conductor - User Manual Verification 'Edge Fitting and Corner Intersection' (Protocol in workflow.md)

## Phase 4: Benchmarking and Verification
- [ ] Task: Performance and Accuracy Validation
    - [ ] Run benchmarks to verify latency is < 1.0ms per quad.
    - [ ] Run regression tests on ICRA 2020 to verify RMSE < 0.8px.
- [ ] Task: Conductor - User Manual Verification 'Benchmarking and Verification' (Protocol in workflow.md)
