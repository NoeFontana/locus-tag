# Implementation Plan: Deep Pipeline SIMD Optimization

**Phase 1: Research & Micro-Profiling [checkpoint: cfffc8e]**
- [x] Task: Locate exact hot loops in `segmentation.rs`, `quad.rs`, and `decoder.rs` (ERF). (cfffc8e)
    - [x] Run `cargo bench` for individual stages.
    - [x] Use `tracy` or `perf` to identify specific SIMD candidates.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Research & Micro-Profiling' (Protocol in workflow.md)

**Phase 2: SIMD Segmentation (CCL) [checkpoint: 2f13df5]**
- [x] Task: Implement SIMD-accelerated scan for threshold-model CCL. (2f13df5)
    - [x] Optimize the row-scan and bit-packing logic.
    - [x] Implement `multiversion` kernels for Union-Find merging. (Optimized row scan instead)
- [x] Task: Verify CCL correctness and performance. (2f13df5)
    - [x] Run existing segmentation tests.
    - [x] Compare latency vs. current scalar implementation.
- [x] Task: Conductor - User Manual Verification 'Phase 2: SIMD Segmentation (CCL)' (Protocol in workflow.md)

**Phase 3: SIMD Quad Extraction [checkpoint: 2f13df5]**
- [x] Task: Optimize Douglas-Peucker fitting with SIMD. (2f13df5)
    - [x] Implement SIMD point-to-line distance kernel.
    - [x] Optimize contour tracing memory patterns. (Optimized D-P instead)
- [x] Task: Verify quad extraction precision. (2f13df5)
    - [x] Run quad extraction unit tests.
    - [x] Ensure no regressions in ICRA 2020 recall.
- [x] Task: Conductor - User Manual Verification 'Phase 3: SIMD Quad Extraction' (Protocol in workflow.md)

**Phase 4: SIMD ERF Refinement [checkpoint: 001cc07]**
- [x] Task: Implement SIMD gradient accumulation in `refine_corners_erf`. (001cc07)
    - [x] Rewrite the gradient descent hot loop using SIMD Fast-Math.
    - [x] Use `rcp_nr` for weight normalization where applicable.
- [x] Task: Verify refinement precision. (001cc07)
    - [x] Add precision comparison tests (SIMD vs. Scalar).
- [x] Task: Conductor - User Manual Verification 'Phase 4: SIMD ERF Refinement' (Protocol in workflow.md)

**Phase 5: Final Validation & Integration [checkpoint: 8927ab6]**
- [x] Task: Run full E2E evaluation. (8927ab6)
    - [x] Execute ICRA 2020 regression suite and accept new snapshots.
    - [x] Run final Python benchmarks and update README.md.
- [x] Task: Conductor - User Manual Verification 'Phase 5: Final Validation & Integration' (Protocol in workflow.md)
