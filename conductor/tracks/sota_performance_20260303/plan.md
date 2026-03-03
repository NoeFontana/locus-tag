# Implementation Plan: SOTA Performance Optimization

**Phase 1: Research & Benchmarking (Baseline) [checkpoint: 110e952]**
- [x] Task: Establish performance baseline for current decoder sampling. (cbe7132)
    - [x] Run existing benchmarks in `crates/locus-core/benches/`.
    - [x] Document baseline latency for Hard and Soft decoding strategies.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Research & Benchmarking (Baseline)' (Protocol in workflow.md)

**Phase 2: SIMD Fast-Math & Fixed-Point Kernels [checkpoint: 750c6d2]**
- [x] Task: Implement SIMD reciprocal estimation with Newton-Raphson. (9c01a2d)
    - [x] Create new SIMD module (e.g., `src/simd/math.rs`).
    - [x] Implement `rcp_nr` for AVX2, AVX-512, and NEON using `multiversion`.
    - [x] Add unit tests for mathematical precision (vs. `1.0/x`).
- [x] Task: Implement Fixed-Point (16.16) bilinear interpolation. (9c01a2d)
    - [x] Implement fixed-point weights and accumulation in SIMD.
    - [x] Add unit tests for interpolation accuracy.
- [x] Task: Conductor - User Manual Verification 'Phase 2: SIMD Fast-Math & Fixed-Point Kernels' (Protocol in workflow.md)

**Phase 3: Hybrid ROI Caching [checkpoint: fb78995]**
- [x] Task: Implement AABB extraction and ROI copying. (635cc17)
    - [x] Update `Quad` struct to compute AABB. (Added to `Detection` instead)
    - [x] Implement hybrid buffer allocation (Stack vs. Arena).
    - [x] Implement contiguous ROI copy into cached buffer.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Hybrid ROI Caching' (Protocol in workflow.md)

**Phase 4: Integration & Optimization [checkpoint: f723cbe]**
- [x] Task: Integrate new kernels into `decoder.rs`. (3de4227)
    - [x] Refactor `sample_grid` to use the ROI cache and SIMD kernels.
    - [x] Ensure parity for both `HardStrategy` and `SoftStrategy`.
- [x] Task: Verify performance and correctness. (3de4227)
    - [x] Run full test suite to ensure no regressions.
    - [x] Re-run benchmarks and confirm sub-millisecond target.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Integration & Optimization' (Protocol in workflow.md)

## Phase: Review Fixes
- [x] Task: Apply review suggestions (a083a3d)
