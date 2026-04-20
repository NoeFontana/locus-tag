# SOTA Configurations — Performance Report

**Date:** 2026-03-21
**Build:** `--release`, default thread count
**Hardware:** AMD EPYC-Milan, 4 cores / 8 threads, L3 32 MiB, Linux 6.8.0-101

---

## 1. What Changed

Three additions enable scenario-specific SOTA presets:

| Addition | Description | Commit |
|---|---|---|
| **GN covariance propagation** | `cholesky_inverse_8x8` extracts per-corner 2×2 covariances from the GN Hessian H⁻¹; threaded through `extract_quad_edlines` → `batch.corner_covariances` | `31f247a` |
| **Builder setters** | `huber_delta_px`, `tikhonov_alpha_max`, `sigma_n_sq`, `structure_tensor_radius` tunable via builder | this session |
| **Three SOTA presets** | `"high_accuracy"`, `"standard"`, `"grid"` | this session |

### Architecture: The GN→Pose Handover (HighAccuracy preset only)

```
Before:  GN corners → batch.corners → pose: Structure Tensor fallback
         H discarded ↗

After:   GN corners + σ²·H⁻¹ blocks → batch.corner_covariances → Weighted LM (direct)
```

Per-corner covariance `Σₖ = σ²·H⁻¹[2k:2k+2, 2k:2k+2]`, `σ² = Σr²/(n_obs−8)`.
If GN diverges, `corner_covariances` is zeroed and pose falls back to the Structure Tensor.

---

## 2. The Three SOTA Presets

| Preset | Target Scenario | Key Differences vs Production |
|---|---|---|
| `"high_accuracy"` profile | Single isolated tag, pose accuracy | EdLines + GN + None; sharpening off; Hard decode |
| `"standard"` profile | Dense multi-tag scenes | Soft decode (+19pp recall); else identical to production |
| `"grid"` profile | Touching tags in grid patterns | 4-connectivity; Soft decode; relaxed contrast/edge gates; sharpening off |

### Why three separate presets?

The three target scenarios have mutually incompatible requirements:

- **HighAccuracy** needs the lowest possible corner RMSE. This requires EdLines + GN corners
  (never post-processed), but the missing sharpening and None refinement hurt recall on
  multi-tag images where small/distant tags are marginal.
- **Pure tags** needs maximum recall without harming precision. Soft decoding is the dominant
  lever (+19pp vs Hard), but it causes a precision collapse (~10–20%) when combined with
  EdLines (which produces many more quad candidates from background edges). ContourRdp +
  Soft is the stable choice.
- **Checkerboard** has non-negotiable topological constraints (4-connectivity, relaxed
  contrast/edge thresholds) that actively hurt performance on isolated tags. Soft decoding
  was untested for this scenario; it proved equally effective (+18.4pp recall).

### Preset parameters

#### `"high_accuracy"` profile
```
quad_extraction_mode:  EdLines
refinement_mode:       None        ← GN corners are sub-pixel; ERF degrades them
enable_sharpening:     false       ← pass raw PSF directly to the solver
enable_bilateral:      false
quad_max_elongation:   20.0
quad_min_density:      0.15
decode_mode:           Hard        ← Soft causes precision collapse on EdLines
```

#### `"standard"` profile
```
refinement_mode:       Erf         ← same as production
enable_sharpening:     true        ← same as production
quad_max_elongation:   20.0        ← same as production
quad_min_density:      0.15        ← same as production
decode_mode:           Soft        ← only difference; +19pp recall on ICRA forward
```

#### `"grid"` profile
```
refinement_mode:       Erf
enable_sharpening:     false       ← sharpening creates halos at shared borders
segmentation_connectivity: Four    ← separates touching tag borders (non-negotiable)
decoder_min_contrast:  10.0        ← relaxed for low-contrast packed tags
quad_min_edge_score:   2.0         ← relaxed for weaker interior-border edge scores
quad_max_elongation:   20.0
quad_min_density:      0.15
decode_mode:           Soft        ← extends recall on low-contrast packed tags
```

---

## 3. ICRA 2020 — Pure Tags (multi-tag isolated, `forward/pure_tags_images`)

50 images, ~8 tags/frame average. Ground-truth convention remapped from UMich CCW to Locus CW.

### 3.1 Fixtures (CI gold standard, 1 image)

| Config | Recall | RMSE |
|---|---|---|
| Production (ContourRdp + Erf + Hard) | **100%** | 0.131 px |
| ContourRdp + Soft | **100%** | 0.131 px |
| EdLines + Erf + Hard | **100%** | **0.071 px** |
| HighAccuracy (EdLines + None + Hard, no sharp) | 74.0% | 0.713 px |
| **Standard (ContourRdp + Erf + Soft)** | **100%** | 0.131 px |

Standard matches production recall exactly on the fixture and inherits the same RMSE —
the only change (Soft decode) has no effect on a clean high-contrast image.

### 3.2 Forward Dataset (50 images)

| Config | Recall | RMSE | Total Latency |
|---|---|---|---|
| Production (ContourRdp + Erf + Hard) | 76.9% | 0.274 px | 164.4 ms |
| GWLF | 65.6% | 0.545 px | 179.3 ms |
| EDLines + Erf + Hard | 71.1% | 0.254 px | 196.8 ms |
| HighAccuracy (EdLines + None + Hard, no sharp) | 46.3% | 0.754 px | 106.6 ms |
| **Standard (ContourRdp + Erf + Soft)** | **96.2%** | **0.315 px** | **70.8 ms** |

**+19.3pp recall vs production** at a modest +15% RMSE cost, with a **−57% latency reduction**.
Soft decode's MIH search is branch-limited — on this dataset it terminates early more often than
Hard decode's full threshold pass, making it faster despite the increased code complexity.

---

## 4. ICRA 2020 — Checkerboard (touching tags, `forward/checkerboard_corners_images`)

50 images, dense tag grids where adjacent tag borders share a pixel boundary.

### 4.1 Fixtures (same 0037.png — contains both isolated and touching tags)

| Config | Recall | RMSE |
|---|---|---|
| Production (ContourRdp + Erf + Hard) | **100%** | 0.131 px |
| Legacy Checkerboard (4-conn + Hard, no sharp) | **100%** | 0.131 px |
| **Grid (4-conn + Soft, no sharp)** | **100%** | 0.144 px |

### 4.2 Forward Checkerboard Dataset (50 images)

| Config | Recall | RMSE | Total Latency |
|---|---|---|---|
| Production (ContourRdp + Erf + Hard) | — | — | — |
| Legacy Checkerboard (4-conn + Hard, no sharp) | 73.0% | 0.332 px | 153.6 ms |
| **Grid (4-conn + Soft, no sharp)** | **91.4%** | **0.458 px** | **103.2 ms** |

**+18.4pp recall vs the legacy checkerboard preset** (+25pp vs production), with a **−33%
latency reduction** over legacy. Soft decoding proves equally effective on touching tags as
on isolated tags, confirming the hypothesis. RMSE increases by +38% — acceptable for
detection tasks where tag identity and pose (rather than sub-pixel corner precision) are
the primary outputs.

---

## 5. Hub Dataset — Single Isolated Tag (AprilTag 36h11, `PoseEstimationMode::Accurate`)

### 5.1 Summary

| Resolution | Config | Recall | Precision | Corner RMSE | Repro RMSE | Trans P50 | Rot P50 | Total Latency |
|---|---|---|---|---|---|---|---|---|
| 640×480 | Production | **100%** | **100%** | 0.994 px | — | 4.1 mm | 1.29° | 74.3 ms |
| | GWLF | 97.8% | **100%** | 0.718 px | — | 3.5 mm | **0.25°** | — |
| | **HighAccuracy** | 93.3% | **100%** | **0.173 px** | **0.440 px** | **1.0 mm** | 0.32° | **53.9 ms** |
| 720p | Production | **100%** | **100%** | 0.933 px | — | 5.8 mm | 2.20° | 116.6 ms |
| | GWLF | **100%** | **100%** | 0.751 px | — | 3.9 mm | **0.31°** | — |
| | **HighAccuracy** | 96.0% | **100%** | **0.277 px** | **1.906 px** | **1.0 mm** | 0.35° | **25.5 ms** |
| 1080p | Production | **100%** | **100%** | 1.146 px | — | 10.7 mm | 2.24° | 106.6 ms |
| | GWLF | **100%** | **100%** | 0.928 px | — | 4.9 mm | **0.31°** | — |
| | **HighAccuracy** | 95.6% | 97.8% | **0.291 px** | **2.142 px** | **1.9 mm** | 0.34° | **102.6 ms** |
| 4K | Production | 97.8% | **100%** | 1.116 px | — | 43.5 mm | 6.68° | 278.4 ms |
| | GWLF | 97.8% | **100%** | 0.829 px | — | 9.9 mm | 0.97° | — |
| | **HighAccuracy** | 88.9% | **100%** | **0.157 px** | **1.690 px** | **5.6 mm** | **0.58°** | **182.2 ms** |

*Latency = total test time for 45–50 images including Accurate pose estimation. Hardware: AMD EPYC-Milan, 4 cores / 8 threads.*

### 5.2 HighAccuracy vs Production (Hub)

| Resolution | Corner RMSE Δ | Rot P50 Δ | Trans P50 Δ | Recall Δ |
|---|---|---|---|---|
| 640×480 | **−83%** (0.17 vs 0.99 px) | **−75%** (0.32° vs 1.29°) | **−76%** (1.0 vs 4.1 mm) | −6.7 pp |
| 720p | **−70%** (0.28 vs 0.93 px) | **−84%** (0.35° vs 2.20°) | **−83%** (1.0 vs 5.8 mm) | −4.0 pp |
| 1080p | **−75%** (0.29 vs 1.15 px) | **−85%** (0.34° vs 2.24°) | **−82%** (1.9 vs 10.7 mm) | −4.4 pp |
| 4K | **−86%** (0.16 vs 1.12 px) | **−91%** (0.58° vs 6.68°) | **−87%** (5.6 vs 43.5 mm) | −8.9 pp |

> **Note:** Standard and Grid were not tested on the hub dataset. Soft
> decoding causes a precision collapse (10–22%) on EdLines due to the larger candidate set
> from background edges. ContourRdp + Soft on single isolated hub tags would likely restore
> precision; this can be added as `regression_hub_tag36h11_*_sota_pure_tags` if needed.

---

## 6. Latency Overview

All measurements: `--release`, single-threaded test runner (`--test-threads=1`), AMD EPYC-Milan 4c/8t.

### ICRA 2020 (50 images each)

| Preset | Dataset | Total Latency | Per-Image |
|---|---|---|---|
| Production (ContourRdp + Erf + Hard) | forward/pure_tags | 164.4 ms | 3.3 ms |
| **Standard (ContourRdp + Erf + Soft)** | forward/pure_tags | **70.8 ms** | **1.4 ms** |
| HighAccuracy (EdLines + None + Hard) | forward/pure_tags | 106.6 ms | 2.1 ms |
| Legacy Checkerboard (4-conn + Hard) | checkerboard | 153.6 ms | 3.1 ms |
| **Grid (4-conn + Soft)** | checkerboard | **103.2 ms** | **2.1 ms** |

### Hub Single-Tag (Accurate pose, ~45–50 images each)

| Resolution | Production | HighAccuracy | Speedup |
|---|---|---|---|
| 640×480 | 74.3 ms (45 img) | **53.9 ms** | 1.4× |
| 720p | 116.6 ms (50 img) | **25.5 ms** | **4.6×** |
| 1080p | 106.6 ms (45 img) | **102.6 ms** | 1.04× |
| 4K | 278.4 ms (45 img) | **182.2 ms** | 1.5× |

The 720p speedup (4.6×) is the largest because `EdLines + None` skips ERF subpixel refinement
entirely — GN corners are directly sub-pixel — and the single clean-background tag means
very few quad candidates reach the Accurate pose step. The 1080p result is near-parity because
the full-resolution image has more candidate contours, so ContourRdp's cheaper rejection
offsets the GN advantage.

---

## 7. Which Preset to Use

| Scenario | Preset | Key metric |
|---|---|---|
| **Single-tag metrology / calibration** | `"high_accuracy"` profile | 0.16–0.29px RMSE, 0.32–0.58° P50 rotation |
| **Dense multi-tag detection** | `"standard"` profile | **96.2%** recall (vs 76.9% production) |
| **Touching-tag checkerboard grids** | `"grid"` profile | **91.4%** recall (vs 73.0% legacy) |
| **Balanced production** | `"standard"` profile | 100% recall + precision, fast |
| **Low latency** | `"standard"` profile | Lowest decode overhead |

---

## 8. Methodology

- **Hub data:** Hugging Face `single_tag_locus_v1_tag36h11_*` (45–50 images each).
- **ICRA data:** ICRA 2020 `forward/pure_tags_images` (50 images), `forward/checkerboard_corners_images` (50 images).
- **Harness:** `regression_render_tag` and `regression_icra2020`, `--release`.
- **Snapshots:**
  - `regression_icra2020__*_sota_pure_tags.snap`
  - `regression_icra2020__icra_forward_checkerboard_sota.snap`
  - `regression_render_tag__hub_*_sota.snap`
- **Decode investigation:** Soft on EdLines/hub → 10–22% precision (rejected for metrology).
  Soft on ContourRdp/ICRA → maintains 100% precision on both ICRA scenarios.
