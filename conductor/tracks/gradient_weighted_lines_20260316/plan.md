# Implementation Plan: GWLF Refinement

**Track ID:** `gradient_weighted_lines_20260316`

## Phase 1: Configuration and API Interface
- [x] Task: TDD - Add `Gwlf` to `CornerRefinementMode` (a6bb999)
    - [x] Add `Gwlf` variant to `CornerRefinementMode` in `crates/locus-core/src/config.rs`.
    - [x] Update `DetectorConfigBuilder` and `DetectorBuilder` to handle the new mode.
- [x] Task: TDD - Python Bindings for GWLF (a6bb999)
    - [x] Update `crates/locus-py/src/lib.rs` to expose `Gwlf` to Python.
- [x] Task: Conductor - User Manual Verification 'Configuration and API Interface' (Protocol in workflow.md) (a6bb999)

## Phase 2: Core Mathematical Components
- [x] Task: TDD - Moment Accumulator and PCA (5956d0e)
    - [x] Implement `MomentAccumulator` in `crates/locus-core/src/gwlf.rs`.
    - [x] Implement 2x2 symmetric eigendecomposition for PCA.
- [x] Task: TDD - Homogeneous Line and Intersection (5956d0e)
    - [x] Implement `HomogeneousLine` with cross-product intersection.
- [x] Task: Conductor - User Manual Verification 'Core Mathematical Components' (Protocol in workflow.md) (5956d0e)

## Phase 3: Edge Fitting and Corner Intersection
- [x] Task: TDD - Implement `refine_quad_gwlf` (5956d0e)
    - [x] Implement the main refinement loop in `crates/locus-core/src/gwlf.rs`.
    - [x] Sample image gradients along quad edges and fit lines.
    - [x] Compute corner intersections and Cartesian projections.
- [x] Task: TDD - Integrate into Detection Pipeline (5956d0e)
    - [x] Call `refine_quad_gwlf` from `Detector::detect` when GWLF mode is enabled.
- [x] Task: TDD - Telemetry and Monitoring (b56bcb7)
    - [x] Implement `Fallback Frequency` counter and `Refinement Delta` reporting.
- [x] Task: Conductor - User Manual Verification 'Edge Fitting and Corner Intersection' (Protocol in workflow.md) (b56bcb7)

## Phase 4: Benchmarking and Verification
- [x] Task: Performance and Accuracy Validation (5956d0e)
    - [x] Run benchmarks to verify latency is < 1.0ms per quad.
    - [x] Run regression tests on ICRA 2020 to verify RMSE < 0.8px.
- [x] Task: Conductor - User Manual Verification 'Benchmarking and Verification' (Protocol in workflow.md) (b56bcb7)
