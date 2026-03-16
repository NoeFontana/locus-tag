# Implementation Plan: Fast-Path Decoding Funnel (SIMD & DDA)

## Phase 1: Foundation (SoA & Contrast Gate)

- [ ] Task: Extend `DetectionBatch` SoA for the funnel status.
    - [ ] Add `is_valid` flag to the Structure-of-Arrays (SoA) if not already present.
    - [ ] Create a `FunnelStatus` enum to track rejection reasons for debugging.
- [ ] Task: Implement the O(1) Edge Contrast Gate.
    - [ ] **Red Phase**: Write failing tests for quad mid-point calculation and contrast sampling.
    - [ ] **Green Phase**: Implement `calculate_edge_midpoints` and `sample_boundary_contrast`.
    - [ ] Integrate into a standalone `apply_funnel_gate` function.
    - [ ] **Refactor**: Optimize intensity fetching from the zero-copy image view.
- [ ] Task: Conductor - User Manual Verification 'Foundation (SoA & Contrast Gate)' (Protocol in workflow.md)

## Phase 2: SIMD & DDA Implementation

- [ ] Task: Implement Homography Digital Differential Analyzer (DDA).
    - [ ] **Red Phase**: Write failing tests for incremental step calculation ($\partial_u, \partial_v$).
    - [ ] **Green Phase**: Implement the DDA state initialization and stepping logic.
    - [ ] Verify DDA precision against standard matrix-based perspective projection.
- [ ] Task: Implement SIMD-Vectorized Bilinear Interpolation (x86_64).
    - [ ] **Red Phase**: Write failing tests for AVX2/FMA3/SSE4.2 interpolation (where hardware permits).
    - [ ] **Green Phase**: Implement the vectorized sampler using intrinsics.
    - [ ] Integrate `_mm256_rcp_ps` with Newton-Raphson for perspective divide.
- [ ] Task: Implement SIMD-Vectorized Bilinear Interpolation (AArch64).
    - [ ] **Red Phase**: Write failing tests for NEON interpolation.
    - [ ] **Green Phase**: Implement the NEON-based sampler using `vrecpeq_f32` (or equivalent).
- [ ] Task: Conductor - User Manual Verification 'SIMD & DDA Implementation' (Protocol in workflow.md)

## Phase 3: Integration & Performance Validation

- [ ] Task: Integrate Fast-Path Funnel into the Decoding Pipeline.
    - [ ] Replace the existing scalar homography-sampling loop with the new DDA-SIMD routine.
    - [ ] Ensure `bumpalo` arena is correctly utilized for the temporary intensity scratchpad.
- [ ] Task: Performance Benchmarking and Verification.
    - [ ] **Verification**: Run `cargo nextest` to ensure zero regressions in detection accuracy.
    - [ ] **Benchmark**: Run micro-benchmarks to measure decoding latency reduction.
    - [ ] Target: <0.5ms per candidate in representative 640x480 images.
- [ ] Task: Conductor - User Manual Verification 'Integration & Performance Validation' (Protocol in workflow.md)
