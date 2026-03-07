# Micro-Benchmark Baseline (2026-03-07)

Initial baseline for core pipeline stages across multiple resolutions using ICRA 2020 frame gradients.

**Environment:**
- CPU: [User to fill]
- OS: Linux
- Build: `--release` (Profile: bench)
- Mode: Single-threaded (`--threads 1`)

## Thresholding (Median Latency)

| Resolution | Latency (ms) |
| :--- | :--- |
| VGA (640x480) | 1.16 ms |
| 720p (1280x720) | 3.30 ms |
| 1080p (1920x1080) | 13.49 ms |
| 4K (3840x2160) | 54.13 ms |

## Segmentation (Median Latency)

| Resolution | Latency (ms) |
| :--- | :--- |
| VGA (640x480) | 1.58 ms |
| 720p (1280x720) | 4.98 ms |
| 1080p (1920x1080) | 11.71 ms |
| 4K (3840x2160) | 38.93 ms |

## Quad Extraction (Median Latency)

| Resolution | Latency (ms) |
| :--- | :--- |
| VGA (640x480) | 3.77 ms |
| 720p (1280x720) | 12.69 ms |
| 1080p (1920x1080) | 37.03 ms |
| 4K (3840x2160) | 178.0 ms |

---

*Note: Raw metrics are stored in `docs/benchmarking/divan_baseline.json`.*
