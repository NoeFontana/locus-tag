# Implementation Plan: SOTA Performance Optimization

**Phase 1: Research & Benchmarking (Baseline)**
- [x] Task: Establish performance baseline for current decoder sampling. (cbe7132)
    - [ ] Run existing benchmarks in `crates/locus-core/benches/`.
    - [ ] Document baseline latency for Hard and Soft decoding strategies.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Research & Benchmarking (Baseline)' (Protocol in workflow.md)

**Phase 2: SIMD Fast-Math & Fixed-Point Kernels**
- [ ] Task: Implement SIMD reciprocal estimation with Newton-Raphson.
    - [ ] Create new SIMD module (e.g., `src/simd/math.rs`).
    - [ ] Implement `rcp_nr` for AVX2, AVX-512, and NEON using `multiversion`.
    - [ ] Add unit tests for mathematical precision (vs. `1.0/x`).
- [ ] Task: Implement Fixed-Point (16.16) bilinear interpolation.
    - [ ] Implement fixed-point weights and accumulation in SIMD.
    - [ ] Add unit tests for interpolation accuracy.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: SIMD Fast-Math & Fixed-Point Kernels' (Protocol in workflow.md)

**Phase 3: Hybrid ROI Caching**
- [ ] Task: Implement AABB extraction and ROI copying.
    - [ ] Update `Quad` struct to compute AABB.
    - [ ] Implement hybrid buffer allocation (Stack vs. Arena).
    - [ ] Implement contiguous ROI copy into cached buffer.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Hybrid ROI Caching' (Protocol in workflow.md)

**Phase 4: Integration & Optimization**
- [ ] Task: Integrate new kernels into `decoder.rs`.
    - [ ] Refactor `sample_grid` to use the ROI cache and SIMD kernels.
    - [ ] Ensure parity for both `HardStrategy` and `SoftStrategy`.
- [ ] Task: Verify performance and correctness.
    - [ ] Run full test suite to ensure no regressions.
    - [ ] Re-run benchmarks and confirm sub-millisecond target.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration & Optimization' (Protocol in workflow.md)
