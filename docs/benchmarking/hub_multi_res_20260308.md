# Hub Multi-Resolution Benchmark (2026-03-08)

This report compares the detection accuracy of Locus across standard resolutions using the `std41h12` dataset.

## Comparative Metrics

| Category | Metric | 640x480 (480p) | 1280x720 (720p) | 1920x1080 (1080p) |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | Recall | 100.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 99.47% |
| | Det. RMSE (px) | 1.3633 | 1.7236 | 1.6399 |
| | Repro. RMSE (px) | 2.8573 | 4.4871 | 4.7629 |
| | Mean Hamming | 0.00 | 0.00 | 0.00 |
| **Translation** | P50 Error (m) | 0.0102 | 0.0081 | 0.0103 |
| | P90 Error (m) | 0.0406 | 0.0465 | 0.0924 |
| | P99 Error (m) | 0.0723 | 0.2090 | 0.2954 |
| **Rotation** | P50 Error (deg) | 6.2725 | 5.1824 | 3.9013 |
| | P90 Error (deg) | 15.8761 | 17.8689 | 21.8403 |
| | P99 Error (deg) | 31.3435 | 28.8244 | 31.7762 |

## Analysis & Insights

### 1. Accuracy vs. Resolution
- **Recall:** Remains perfect (100%) across all tested resolutions, indicating the pipeline is robust to scale changes within these bounds.
- **Precision:** A slight drop at 1080p (99.47%) suggests a small increase in false positives (likely background noise segments passing the initial filters) at higher pixel counts.
- **RMSE:** Detection RMSE is relatively stable, but **Reprojection RMSE** increases with resolution. This is expected as sub-pixel errors in detection map to larger reprojection residuals when focal length and resolution increase.

### 2. Pose Robustness
- **Translation:** The median (P50) translation error is excellent (~1cm) across all resolutions. However, the **P99 error** grows significantly at higher resolutions (up to ~30cm at 1080p). This suggests that while most poses are stable, the "tail" of difficult cases (e.g., steep angles or small tags) becomes more volatile as resolution increases.
- **Rotation:** Rotation P50 actually improves at higher resolutions (6.2° -> 3.9°), likely due to the higher angular resolution provided by more pixels on the tag edges.

## Methodology
- **Data Source:** Hugging Face `NoeFontana/locus-tag-bench` (`std41h12` subsets).
- **Harness:** `locus-core` regression suite with per-image metadata (intrinsics/tag size).
- **Environment:** Linux, AMD EPYC-Milan Processor (Release Profile).
