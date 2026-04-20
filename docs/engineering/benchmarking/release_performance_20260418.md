# Locus Release Performance Report (2026-04-18)

This report documents the performance of Locus v0.3.1, covering end-to-end latency, mathematical kernel efficiency, and regression accuracy.

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
HighAccuracy exhibits lower recall on ICRA 2020 compared to Hub datasets. 
- **ICRA 2020 (Forward):** 46.3% Recall
- **Hub (1080p):** 95.6% Recall

**Root Cause: Sub-Pixel Bit Density.**
Analysis of the ICRA 2020 dataset reveals that a significant portion of tags are extremely small relative to their bit grid:
- **Forward:** ~18% of tags have < 1.2 pixels per bit (PPB).
- **Circle:** Thousands of tags have < 1.0 PPB (some as low as 0.03 PPB).

The `HighAccuracy` preset is optimized for maximum precision on high-quality images. It uses `EdLines` quad extraction and disables Laplacian sharpening to avoid distorting the Point Spread Function (PSF). 
- **EdLines sensitivity:** Empirical testing shows `EdLines` recall collapses on tags with < 1.5 PPB (reaching < 1% for < 1.2 PPB), whereas the legacy `ContourRdp` algorithm remains robust down to ~1.2 PPB.
- **Sharpening impact:** Disabling sharpening (as in HighAccuracy mode) further degrades recall on small tags by smearing bit boundaries. Enabling sharpening recovers `EdLines` recall to 100% for tags > 2.0 PPB, but it remains ineffective for the "sub-pixel" tags prevalent in ICRA 2020.

For real-world robotics tracking where high recall on distant tags is required, the `standard` profile (Soft Decode + ContourRdp + Sharpening) remains the recommended configuration. The `high_accuracy` profile should be reserved for high-resolution near-field calibration where PPB is > 5.0.

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
