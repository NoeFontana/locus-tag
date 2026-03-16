# Specification: Fast-Path Decoding Funnel (SIMD & DDA)

## 1. Overview
The decoding stage currently accounts for ~70.6% of the pipeline's latency. This track implements a multi-stage "funnel" to eliminate background artifacts quickly and optimize the bit sampling process for valid candidates using SIMD-accelerated Digital Differential Analyzer (DDA) routines.

## 2. Functional Requirements

### 2.1 O(1) Edge Contrast Rejection Gate
- **Purpose**: Rapidly reject low-contrast quads (shadows, artifacts) before expensive homography estimation.
- **Mechanism**: Sample 8 points (4 inner, 4 outer) around the midpoints of the quad's edges.
- **Threshold**: Automatically derived from the local adaptive threshold variance.
- **Integration**: Standalone stage operating on the `DetectionBatch` Structure-of-Arrays (SoA).

### 2.2 Forward Incremental Sampling (Homography DDA)
- **Purpose**: Eliminate matrix multiplications during bit grid sampling.
- **Mechanism**: Compute linear partial derivatives ($\partial_u, \partial_v$) for perspective projection numerators and denominators.
- **Constraint**: Maintain precision suitable for bit-perfect decoding.

### 2.3 SIMD-Vectorized Bilinear Interpolation
- **Purpose**: Parallelize pixel sampling across SIMD lanes.
- **Support**: Implement backends for AVX2/FMA3, NEON (ARM), and SSE4.2.
- **Math**: Use hardware reciprocal approximations (e.g., `_mm256_rcp_ps`) with Newton-Raphson refinement for the perspective divide.
- **Efficiency**: Utilize Fused Multiply-Add (FMA) for vectorized interpolation.

## 3. Non-Functional Requirements
- **Latency**: Target a >40% reduction in decoding stage time.
- **Memory**: Zero-allocation hot loop using the existing `bumpalo` arena.
- **Execution**: Strictly sequential processing (no Rayon multi-threading for this phase).
- **Safety**: Purely memory-safe Rust implementation.

## 4. Acceptance Criteria
- [ ] Decoding latency reduced from ~0.8ms/candidate to <0.5ms/candidate (representative benchmark).
- [ ] No regressions in detection accuracy on the ICRA 2020 dataset.
- [ ] Unit tests pass for all supported SIMD architectures (where hardware permits).
- [ ] Profiling confirms no new allocations in the `funnel` and `sampling` stages.

## 5. Out of Scope
- Integration of multi-threading (Rayon) for this track.
- Modifications to segmentation or final bit-search (MIH) algorithms.
