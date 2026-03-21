# Micro-Benchmark Baseline (2026-03-19)

Latest baseline for core pipeline stages across multiple resolutions after SIMD CCL Fusion and alignment of production defaults.

**Environment:**
- CPU: AMD EPYC-Milan Processor
- OS: Linux
- Build: `--release` (Profile: bench)
- Mode: Single-threaded (`--threads 1`)

## Thresholding (Median Latency)

| Resolution | Latency (ms) | Speedup vs 2026-03-15 |
| :--- | :--- | :--- |
| VGA (640x480) | 0.80 ms | 1.7x |
| 720p (1280x720) | 2.18 ms | 1.9x |
| 1080p (1920x1080) | 4.26 ms | 2.9x |
| 4K (3840x2160) | 16.29 ms | 2.4x |

## Segmentation (Median Latency) - SIMD CCL Fusion

| Resolution | Latency (ms) | Speedup vs 2026-03-15 |
| :--- | :--- | :--- |
| VGA (640x480) | **0.67 ms** | 3.3x |
| 720p (1280x720) | **1.53 ms** | 3.2x |
| 1080p (1920x1080) | **2.95 ms** | 3.1x |
| 4K (3840x2160) | **9.97 ms** | 3.0x |

## Quad Extraction (Median Latency)

| Resolution | Latency (ms) | Speedup vs 2026-03-15 |
| :--- | :--- | :--- |
| VGA (640x480) | 5.87 ms | - |
| 720p (1280x720) | 19.18 ms | - |
| 1080p (1920x1080) | 34.35 ms | 1.4x |
| 4K (3840x2160) | 143.7 ms | 1.4x |

---

*Note: Raw metrics are stored in `docs/engineering/benchmarking/divan_baseline.json`.*
