# Locus Release Performance Report (2026-04-18)

Historical snapshot for Locus v0.3.1. Numbers are stale (superseded by
`render_tag_sota_20260713.md` and later reports); kept for the
kernel-speedup trend and as the origin of the PPB investigation now
summarized in [`lessons.md` §4.4](lessons.md).

## Tier 1: End-to-End Performance (Python CLI)

Validated on the **ICRA 2020** dataset and synthetic targets.
Measurements taken with `RAYON_NUM_THREADS=1`.

### Real-World (ICRA 2020 - Scenario: forward)
Measured on all 50 images of the `forward/pure_tags_images` subset.

| Configuration | Recall | RMSE (px) | Latency (ms) |
|---------------|--------|-----------|--------------|
| **Locus (Soft)** | 96.23% | 0.3152 | 151.12 |
| **Locus (Hard)** | 76.87% | 0.2567 | 92.24 |

*Note: Python CLI latency includes significant FFI and result-marshalling overhead. Rust core latencies for the same workload are ~32ms (Hard) and ~42ms (Soft).*

---

## Tier 2: Micro-Benchmarking (Divan)

Fine-grained mathematical kernels measured strictly single-threaded on 1080p images.
Substantial improvements observed compared to the March 2026 baseline.

### Mathematical Kernels (1080p)
| Kernel | Latest Median (ms) | Baseline (2026-03-19) | Speedup |
|--------|--------------------|-----------------------|---------|
| **Thresholding (Apply)** | **1.51 ms** | 4.26 ms | **2.82x** |
| **Segmentation** | **2.57 ms** | 2.95 ms | **1.15x** |
| **Quad Extraction** | **23.79 ms** | 34.35 ms | **1.44x** |

---

## Tier 3: Rust Regression Suite

### ICRA 2020 (HighAccuracy Presets)
| Scenario | Preset | Recall | RMSE (px) | Latency (ms) |
|----------|--------|--------|-----------|--------------|
| **Forward** | Standard (Soft) | 96.23% | 0.3152 | 41.81 |
| **Forward** | Grid | 91.43% | 0.4577 | 58.53 |
| **Forward** | HighAccuracy | 46.31% | 0.7535 | 15.43 |

#### Investigation: HighAccuracy Performance
HighAccuracy: 46.3% recall on ICRA 2020 forward vs. 95.6% on Hub 1080p —
root-caused to PPB (pixels-per-bit); full writeup and the resulting
per-PPB-band recommendation table now live in
[`lessons.md` §4.4](lessons.md).

### Hub Regression (Rendered Tags - 1080p Tag36h11)
| Mode | Recall | Trans P50 (mm) | Latency (ms) |
|------|--------|----------------|--------------|
| **Standard** | 100.0% | 5.3 mm | 21.69 |
| **HighAccuracy** | 95.56% | 0.7 mm | 11.44 |

---

## Technical Updates in this Version
- **Snapshot Stabilization:** All regression snapshots have been updated to established v0.3.1 baselines.
- **Kernel Optimization:** SIMD-accelerated thresholding and segmentation provide significant throughput gains.
- **Metrology Handover:** GN corner covariances now propagate directly to the weighted LM pose solver in HighAccuracy mode.
