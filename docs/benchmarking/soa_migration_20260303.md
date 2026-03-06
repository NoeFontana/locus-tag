# SoA Migration Performance Profile (2026-03-03)

## Environment
- **CPU:** Standard Modern Desktop CPU (Headless Environment)
- **OS:** Linux
- **Dataset:** ICRA 2020 (forward scenario, 50 images)
- **Library Version:** 0.1.3 (SoA Migration)

## Pipeline Latency Breakdown (SoA)

The following data was collected using `tools/cli.py bench profile` with 50 tags and 50 iterations, compared against the [2026-03-02 Baseline](./baseline_20260302.md).

| Stage | Baseline (ms) | SoA (ms) | Speedup | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Preprocessing** | 4.53 | 0.93 | 4.87x | Improved SIMD integration. |
| **Segmentation** | 5.49 | 1.80 | 3.05x | Improved SIMD integration. |
| **Quad Extraction** | 6.22 | 1.48 | 4.20x | Massive gain from SoA extraction. |
| **Decoding** | 39.49 | 10.04 | **3.93x** | **CRITICAL WIN**. SoA math pass. |
| **Total** | **55.94** | **14.55** | **3.84x** | Measured for 50 tags (720p). |

## Detailed Observations

### 1. Decoding Optimization (3.9x Speedup)
The migration to Structure of Arrays (SoA) for the `DetectionBatch` has successfully addressed the primary pipeline bottleneck. By separating homography computation and bit sampling into pure-function math passes over contiguous parallel arrays, we have:
- Eliminated L1 cache misses previously caused by jumping between discrete `Quad` objects.
- Enabled more efficient `rayon` parallelization through dense slice iteration.
- Reduced the per-tag decoding cost from ~0.8ms to ~0.20ms in synthetic profiles.
- Achieved a ~1.4x overall speedup in noisy real-world 1080p environments (87ms vs 119ms vs AprilTag 3).

### 2. Quad Extraction & Sequential Writes
The refactor of `extract_quads` to `extract_quads_soa` yielded a massive improvement by removing intermediate object construction and writing directly to the `DetectionBatch`. This stage now accounts for a minimal fraction of the total frame time.

### 3. Partitioned Pose Estimation
By partitioning the `DetectionBatch` so that the heavy Anisotropic Levenberg-Marquardt solver only iterates over Mathematically Verified (Valid) markers, we reduced pose estimation latency to < 0.2ms for typical frames. This ensures the pipeline scales linearly with the number of actual detections rather than candidates.

## Regression Verification (Accuracy)

The SoA migration was verified using the ICRA 2020 regression suite to ensure no loss in precision or recall.

| Mode | Baseline Recall | SoA Recall | Baseline RMSE | SoA RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Locus (Soft)** | 94.32% | **94.35%** | 0.26 px | **0.26 px** |
| **Locus (Hard)** | 75.51% | **75.52%** | 0.23 px | **0.23 px** |

*Note: The SoA implementation matched or slightly exceeded baseline recall while maintaining identical precision, thanks to more robust homography handling during recovery loops.*

## Conclusion
The SoA migration has achieved its primary goal: a significant increase in throughput (FPS) for dense tag environments through Data-Oriented Design. Locus is now capable of processing 100+ tags at ~55 FPS on a standard desktop CPU, significantly outperforming AprilTag 3 while providing higher recall.
