# Implementation Plan: Fast-Path Decoding Funnel (SIMD & DDA)

## Phase 1: Foundation (SoA & Contrast Gate) [checkpoint: f696d09]

- [x] Task: Extend `DetectionBatch` SoA for the funnel status. b1cd92d
    - [x] Add `is_valid` flag to the Structure-of-Arrays (SoA) if not already present.
    - [x] Create a `FunnelStatus` enum to track rejection reasons for debugging.
- [x] Task: Implement the O(1) Edge Contrast Gate. d3a1df1
    - [x] **Red Phase**: Write failing tests for quad mid-point calculation and contrast sampling.
    - [x] **Green Phase**: Implement `calculate_edge_midpoints` and `sample_boundary_contrast`.
    - [x] Integrate into a standalone `apply_funnel_gate` function.
    - [x] **Refactor**: Optimize intensity fetching from the zero-copy image view.
- [x] Task: Conductor - User Manual Verification 'Foundation (SoA & Contrast Gate)' (Protocol in workflow.md)

## Phase 2: SIMD & DDA Implementation [checkpoint: 0bcf559]

- [x] Task: Implement Homography Digital Differential Analyzer (DDA). 76b4e6a
    - [x] **Red Phase**: Write failing tests for incremental step calculation ($\partial_u, \partial_v$).
    - [x] **Green Phase**: Implement the DDA state initialization and stepping logic.
    - [x] Verify DDA precision against standard matrix-based perspective projection.
- [x] Task: Implement SIMD-Vectorized Bilinear Interpolation (x86_64). d563745
    - [x] **Red Phase**: Write failing tests for AVX2/FMA3/SSE4.2 interpolation (where hardware permits).
    - [x] **Green Phase**: Implement the vectorized sampler using intrinsics.
    - [x] Integrate `_mm256_rcp_ps` with Newton-Raphson for perspective divide.
- [x] Task: Implement SIMD-Vectorized Bilinear Interpolation (AArch64). b850a91
    - [x] **Red Phase**: Write failing tests for NEON interpolation.
    - [x] **Green Phase**: Implement the NEON-based sampler using `vrecpeq_f32` (or equivalent).
- [x] Task: Conductor - User Manual Verification 'SIMD & DDA Implementation' (Protocol in workflow.md)

## Phase 3: Integration & Performance Validation [checkpoint: 70df37d]

- [x] Task: Integrate Fast-Path Funnel into the Decoding Pipeline. 4e24359
    - [x] Replace the existing scalar homography-sampling loop with the new DDA-SIMD routine.
    - [x] Ensure `bumpalo` arena is correctly utilized for the temporary intensity scratchpad.
- [x] Task: Performance Benchmarking and Verification. f2b7363
    - [x] **Verification**: Run `cargo nextest` to ensure zero regressions in detection accuracy.
    - [x] **Benchmark**: Run micro-benchmarks to measure decoding latency reduction.
    - [x] Target: <0.5ms per candidate in representative 640x480 images.
- [x] Task: Conductor - User Manual Verification 'Integration & Performance Validation' (Protocol in workflow.md)

## Phase: Review Fixes
- [x] Task: Apply review suggestions 2e5371a
