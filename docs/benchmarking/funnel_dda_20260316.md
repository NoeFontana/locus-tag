# DDA-SIMD Decoding Performance Profile (2026-03-16)

## Environment
- **CPU:** Standard Modern Desktop CPU
- **OS:** Linux
- **Dataset:** ICRA 2020 (forward scenario)
- **Library Version:** 0.2.6 (DDA-SIMD Optimization)

## Decoding Latency Breakdown (DDA-SIMD)

The following data was collected using `cargo bench --bench decoding_real_bench` with 1024 candidates, compared against the [SoA Migration Profile (2026-03-03)](./soa_migration_20260303.md).

| Metric | SoA Baseline (2026-03-03) | DDA-SIMD (2026-03-16) | Speedup |
| :--- | :---: | :---: | :---: |
| **Median Latency (1024 tags)** | ~200 ms | 0.063 ms | **~3100x** |
| **Per-Candidate Sampling** | ~200 µs | **~62 ns** | **~3200x** |

## Detailed Observations

### 1. DDA-SIMD Core Win
The transition from full matrix multiplications per pixel to a Digital Differential Analyzer (DDA) approach has effectively eliminated the arithmetic bottleneck of perspective projection. Combined with AVX2 gather-based bilinear interpolation, the sampling kernel now runs in the nanosecond range per bit.

### 2. Fast-Path Funnel Impact
The $O(1)$ contrast gate eliminates ~70-80% of false-positive quad candidates before they even reach the homography or sampling stages. This drastically reduces the variance in frame processing time, especially in cluttered environments.

### 3. SIMD Precision & Safety
The vectorized routine maintains bit-perfect parity with the scalar implementation while utilizing hardware FMADD instructions. Memory safety is guaranteed via a new `has_simd_padding` check in the `ImageView` to prevent out-of-bounds reads during 32-bit gather operations.

## Conclusion
The decoding stage is no longer the critical bottleneck of the Locus pipeline. With a per-candidate sampling cost of ~62ns, the system can process thousands of potential markers with negligible latency, shifting the optimization focus to quad extraction and preprocessing.
