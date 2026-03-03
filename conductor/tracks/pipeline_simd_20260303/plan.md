# Implementation Plan: Deep Pipeline SIMD Optimization

**Phase 1: Research & Micro-Profiling [checkpoint: cfffc8e]**
- [x] Task: Locate exact hot loops in `segmentation.rs`, `quad.rs`, and `decoder.rs` (ERF). (cfffc8e)
    - [x] Run `cargo bench` for individual stages.
    - [x] Use `tracy` or `perf` to identify specific SIMD candidates.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Research & Micro-Profiling' (Protocol in workflow.md)

**Phase 2: SIMD Segmentation (CCL)**
- [ ] Task: Implement SIMD-accelerated scan for threshold-model CCL.
    - [ ] Optimize the row-scan and bit-packing logic.
    - [ ] Implement `multiversion` kernels for Union-Find merging.
- [ ] Task: Verify CCL correctness and performance.
    - [ ] Run existing segmentation tests.
    - [ ] Compare latency vs. current scalar implementation.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: SIMD Segmentation (CCL)' (Protocol in workflow.md)

**Phase 3: SIMD Quad Extraction**
- [ ] Task: Optimize Douglas-Peucker fitting with SIMD.
    - [ ] Implement SIMD point-to-line distance kernel.
    - [ ] Optimize contour tracing memory patterns.
- [ ] Task: Verify quad extraction precision.
    - [ ] Run quad extraction unit tests.
    - [ ] Ensure no regressions in ICRA 2020 recall.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: SIMD Quad Extraction' (Protocol in workflow.md)

**Phase 4: SIMD ERF Refinement**
- [ ] Task: Implement SIMD gradient accumulation in `refine_corners_erf`.
    - [ ] Rewrite the gradient descent hot loop using SIMD Fast-Math.
    - [ ] Use `rcp_nr` for weight normalization where applicable.
- [ ] Task: Verify refinement precision.
    - [ ] Add precision comparison tests (SIMD vs. Scalar).
- [ ] Task: Conductor - User Manual Verification 'Phase 4: SIMD ERF Refinement' (Protocol in workflow.md)

**Phase 5: Final Validation & Integration**
- [ ] Task: Run full E2E evaluation.
    - [ ] Execute ICRA 2020 regression suite and accept new snapshots.
    - [ ] Run final Python benchmarks and update README.md.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Validation & Integration' (Protocol in workflow.md)
