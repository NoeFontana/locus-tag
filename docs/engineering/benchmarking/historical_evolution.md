# Performance Evolution (Mar 2–16, 2026)

This document consolidates the superseded benchmark reports into a single optimization timeline, preserving the key numbers and design decisions that shaped the current architecture.

## 1. Initial Baseline (2026-03-02)

**Dataset:** ICRA 2020 forward, 50 images, 50 tags/frame. **Hardware:** Desktop CPU, Linux.

| Stage | Latency (ms) | % Total |
| :--- | :---: | :---: |
| Preprocessing | 4.53 | 8.1% |
| Segmentation | 5.49 | 9.8% |
| Quad Extraction | 6.22 | 11.1% |
| **Decoding** | **39.49** | **70.6%** |
| Pose Estimation | ~0.21 | <1% |
| **Total** | **55.94** | |

**Key finding:** Decoding consumed 70% of frame time (~0.8 ms/candidate). Homography estimation + bilinear sampling were the dominant costs.

## 2. SoA Migration (2026-03-03)

Migrated from discrete `Quad` objects to the `DetectionBatch` Structure-of-Arrays layout.

| Stage | Before (ms) | After (ms) | Speedup |
| :--- | :---: | :---: | :---: |
| Preprocessing | 4.53 | 0.93 | 4.9x |
| Segmentation | 5.49 | 1.80 | 3.1x |
| Quad Extraction | 6.22 | 1.48 | 4.2x |
| Decoding | 39.49 | 10.04 | 3.9x |
| **Total** | **55.94** | **14.55** | **3.8x** |

**Design decisions:**

- Eliminated L1 cache misses from object hopping via contiguous parallel arrays.
- Enabled `rayon` parallelization on dense slices.
- Per-tag decoding: 0.8 ms → 0.20 ms.
- Partitioned pose estimation — heavy LM solver runs only on Valid markers.

**Accuracy verification:** Recall and RMSE matched pre-SoA baseline (Soft: 94.35% recall, 0.26 px RMSE).

## 3. Micro-Benchmark Baselines (2026-03-07 → 03-15)

**Hardware:** AMD EPYC-Milan, Linux, `--release`, single-threaded.

Two successive baselines captured the pre-SIMD state of per-stage latency:

| Resolution | Thresholding | Segmentation | Quad Extraction |
| :--- | :---: | :---: | :---: |
| VGA | 1.2–1.3 ms | 1.6–2.2 ms | 3.8–4.8 ms |
| 720p | 3.3–4.1 ms | 4.9–5.0 ms | 12.7–16.7 ms |
| 1080p | 12.2–13.5 ms | 9.2–11.7 ms | 37.0–48.9 ms |
| 4K | 38.9–54.1 ms | 29.7–38.9 ms | 178–207 ms |

*Range reflects environmental variance across two measurement sessions; no code changes between them.*

## 4. Hub Multi-Resolution Accuracy (2026-03-08)

**Dataset:** Hugging Face `std41h12` (480p/720p/1080p). **Hardware:** AMD EPYC-Milan.

Validated 100% recall across resolutions. Key observations:

- Reprojection RMSE grows with resolution (2.9 px at 480p → 4.8 px at 1080p) due to focal length scaling.
- Rotation P50 improves at higher resolution (6.3° → 3.9°) — more pixels on edges.
- Translation P50 excellent (~1 cm), but P99 tail grows at 1080p (~30 cm).

## 5. GWLF Introduction (2026-03-12/15)

**Dataset:** Hugging Face `single_tag_locus_v1_tag36h11` (45–50 images/resolution).

Gradient-Weighted Line Fitting replaced ERF as the recommended corner refinement for robotics:

| Metric | ContourRdp+ERF | GWLF | Improvement |
| :--- | :---: | :---: | :---: |
| Det. RMSE (720p) | 0.99 px | **0.71 px** | 28% |
| Rotation P50 (720p) | 2.22° | **0.30°** | **7.4x** |
| Precision | 96.7% | **100%** | |
| Latency (720p) | 15.5 ms | **13.8 ms** | 12% faster |

**Design decisions:**

- GWLF fits infinite lines to edge gradients via Weighted Orthogonal Distance Regression (PCA).
- Corners computed as algebraic line intersections in homogeneous space.
- Bilinear gradient sampling + Adaptive Transversal Windowing ($\pm \max(2, 0.01L)$).
- Also established Fast vs Accurate pose mode comparison — both use Huber IRLS; Accurate adds gain-scheduled Tikhonov regularization.

## 6. DDA-SIMD Decoding (2026-03-16)

Replaced per-pixel matrix multiplications with a Digital Differential Analyzer + AVX2 gather-based bilinear interpolation:

| Metric | SoA Baseline | DDA-SIMD | Speedup |
| :--- | :---: | :---: | :---: |
| Batch (1024 tags) | ~200 ms | **0.063 ms** | **~3100x** |
| Per-candidate | ~200 µs | **~62 ns** | **~3200x** |

**Design decisions:**

- DDA eliminates O(n) matrix multiplications per pixel — uses incremental partial derivatives.
- O(1) contrast gate (Fast-Path Funnel) rejects 70–80% of false-positive candidates.
- `has_simd_padding` safety check prevents out-of-bounds 32-bit gathers.
- **Conclusion:** Decoding is no longer the bottleneck — focus shifted to quad extraction and preprocessing.

## 7. Improvements Regression Verification (2026-03-15)

Verified that a large refactoring branch (`misc/improvements`) introduced **no algorithmic regressions** across all 15 regression tests (9 ICRA + 6 Hub). Changes included: error propagation (`Result`-based API), config validation, thread-local arenas, capacity hints, constant centralization, ~350 lines of dead code removal, and expanded benchmarks.

Per-tag pose latency: ~2.3 µs (Fast), ~4.7 µs (Accurate) at 50 tags.

---

## Optimization Timeline Summary

```
Mar 02  55.94 ms/frame  ─── Initial baseline (decoding = 70%)
   │
Mar 03  14.55 ms/frame  ─── SoA migration (3.8x total)
   │
Mar 08  ── accuracy ──  ─── Hub multi-res: 100% recall validated
   │
Mar 12  ── accuracy ──  ─── GWLF: 7x rotation improvement
   │
Mar 16   0.063 ms/1024  ─── DDA-SIMD: decoding eliminated as bottleneck
   │
Mar 19  ── current ──   ─── SIMD CCL Fusion: 3x segmentation (see current docs)
   │
Mar 21  ── current ──   ─── EdLines GN + SOTA presets (see current docs)
```

## Superseded Reports

The following individual reports were consolidated into this document:

- `baseline_20260302.md` — Initial pipeline profile
- `soa_migration_20260303.md` — SoA migration results
- `baseline_micro_20260307.md` — First micro-benchmark baseline
- `baseline_micro_20260315.md` — Second micro-benchmark baseline
- `hub_multi_res_20260308.md` — Multi-resolution accuracy (std41h12)
- `hub_locus_v1_20260312.md` — GWLF introduction + Fast/Accurate comparison
- `funnel_dda_20260316.md` — DDA-SIMD decoding optimization
- `improvements_regression_20260315.md` — Refactoring regression verification
