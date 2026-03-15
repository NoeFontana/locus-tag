# Micro-Benchmark Baseline (2026-03-15)

Latest baseline for core pipeline stages across multiple resolutions using ICRA 2020 frame gradients.

**Environment:**
- CPU: AMD EPYC-Milan Processor
- OS: Linux
- Build: `--release` (Profile: bench)
- Mode: Single-threaded (`--threads 1`)

## Thresholding (Median Latency)

| Resolution | Latency (ms) |
| :--- | :--- |
| VGA (640x480) | 1.33 ms |
| 720p (1280x720) | 4.12 ms |
| 1080p (1920x1080) | 12.21 ms |
| 4K (3840x2160) | 38.87 ms |

## Segmentation (Median Latency)

| Resolution | Latency (ms) |
| :--- | :--- |
| VGA (640x480) | 2.24 ms |
| 720p (1280x720) | 4.90 ms |
| 1080p (1920x1080) | 9.16 ms |
| 4K (3840x2160) | 29.73 ms |

## Quad Extraction (Median Latency)

| Resolution | Latency (ms) |
| :--- | :--- |
| VGA (640x480) | 4.83 ms |
| 720p (1280x720) | 16.68 ms |
| 1080p (1920x1080) | 48.87 ms |
| 4K (3840x2160) | 207.4 ms |

---

*Note: Raw metrics are stored in `docs/benchmarking/divan_baseline.json`.*
