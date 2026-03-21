# EDLines + Joint Gauss-Newton — Comparative Performance Report

**Date:** 2026-03-21
**Build:** `--release`, single-threaded benchmarks (`RAYON_NUM_THREADS=1` for latency, default for regression)
**Hardware:** AMD EPYC-Milan, 4 cores / 8 threads, 2 GHz, L3 32 MiB, Linux 6.8.0-101

---

## 1. What Changed

This report covers the two algorithmic additions landed on `feat/edline-2`:

| Phase | Description | Commit |
|---|---|---|
| **EDLines pipeline** (Phases 1–4) | Angular-arc boundary scan → Huber IRLS → micro-ray parabolic sub-pixel → per-edge IRLS re-fit + line intersection | earlier |
| **Phase 5 Joint Gauss-Newton** | 8-DOF joint corner optimiser; stack-allocated 8×8 Cholesky, zero allocations, exclusion-zone filter | `8b1d85f` |

The Gauss-Newton stage replaces independent pairwise line intersection with a joint optimisation that enforces geometric planarity across all four corners simultaneously.

---

## 2. Hub Dataset — Single Isolated Tag (AprilTag 36h11)

All variants use `PoseEstimationMode::Accurate` (Structure Tensor + Weighted LM pose).
Latencies measured on the regression machine above; they include full detection + pose.

### 2.1 640×480 (VGA, 45 images)

| Config | Recall | Corner RMSE | Trans P50 | Rot P50 | Latency |
|---|---|---|---|---|---|
| ContourRdp (default) | **100%** | 0.994 px | 4.1 mm | 1.29° | 40 ms |
| ContourRdp Fast | **100%** | 0.994 px | 4.0 mm | 1.27° | 49 ms |
| GWLF | 97.8% | **0.718 px** | **3.5 mm** | **0.25°** | 35 ms |

### 2.2 720p / 1280×720 (50 images)

| Config | Recall | Corner RMSE | Trans P50 | Rot P50 | Latency |
|---|---|---|---|---|---|
| ContourRdp (default) | **100%** | 0.933 px | 5.8 mm | 2.20° | 60 ms |
| ContourRdp Fast | **100%** | 0.933 px | 5.3 mm | 2.20° | 80 ms |
| Moments Culling | **100%** | 0.933 px | 5.8 mm | 2.20° | 58 ms |
| GWLF | **100%** | 0.751 px | 3.7 mm | **0.31°** | 59 ms |
| **EDLines** | 96% | **0.661 px** | **3.9 mm** | 0.39° | 68 ms |
| **EDLines + Moments** | 96% | **0.661 px** | **3.9 mm** | 0.39° | 61 ms |

EDLines gives the lowest corner RMSE (−29% vs ContourRdp) and excellent rotation accuracy
(0.39° vs 2.20° for default = −82%). GWLF achieves the best absolute rotation (0.31°) at
full recall; EDLines closes to within 0.08° at 4% recall cost.

### 2.3 1080p / 1920×1080 (45 images)

| Config | Recall | Corner RMSE | Trans P50 | Rot P50 | Latency |
|---|---|---|---|---|---|
| ContourRdp (default) | **100%** | 1.146 px | 10.7 mm | 2.24° | 109 ms |
| ContourRdp Fast | **100%** | 1.146 px | 11.8 mm | 2.13° | 117 ms |
| Moments Culling | **100%** | 1.146 px | 10.7 mm | 2.24° | 109 ms |
| GWLF | **100%** | 0.928 px | 4.9 mm | **0.31°** | 108 ms |
| **EDLines** | 97.8% | **0.772 px** | **4.4 mm** | 0.50° | 124 ms |
| **EDLines + Moments** | 97.8% | **0.772 px** | **4.4 mm** | 0.50° | 120 ms |

EDLines: −33% corner RMSE vs ContourRdp default, −17% vs GWLF. Rotation within 0.19° of
GWLF's best. Latency overhead vs ContourRdp: +14% (EDLines) / +10% (EDLines+Moments).

### 2.4 4K / 3840×2160 (45 images)

| Config | Recall | Corner RMSE | Trans P50 | Rot P50 | Latency |
|---|---|---|---|---|---|
| ContourRdp (default) | 97.8% | 1.116 px | 43.5 mm | 6.68° | 136 ms |
| ContourRdp Fast | 97.8% | 1.116 px | 41.4 mm | 6.68° | 178 ms |
| Moments Culling | 97.8% | 1.116 px | 43.5 mm | 6.68° | 208 ms |
| GWLF | 97.8% | **0.829 px** | **9.9 mm** | **0.97°** | 109 ms |
| **EDLines** | 75.6% | **0.522 px** | 12.2 mm | 0.80° | 233 ms |
| **EDLines + Moments** | 82.2% | 0.563 px | 12.2 mm | 0.80° | 231 ms |

At 4K, EDLines achieves the lowest corner RMSE of all configurations (−53% vs ContourRdp)
but at a significant recall penalty (−22% absolute). Root cause: at high resolution the
component bounding boxes are large; boundary scan + IRLS is losing some highly-oblique tags.
Moments culling recovers 6.6 percentage points of recall (82% vs 76%). GWLF is the
recommended mode at 4K — it matches EDLines pose accuracy while maintaining recall.

---

## 3. ICRA 2020 — Multi-Tag Forward Scene (50 images, 8 tags/frame avg)

The ICRA forward dataset is the harder workload: tags at varying ranges, angles, and lighting.
Ground-truth corner convention: UMich CCW [BL,BR,TR,TL] → remapped to Locus CW for comparison.

| Config | Recall | Corner RMSE | Latency |
|---|---|---|---|
| ContourRdp (default, Hard) | **76.9%** | 0.274 px | 141 ms |
| ContourRdp (default, Soft) | **96.2%** | 0.315 px | 152 ms |
| Moments Culling | **76.9%** | 0.274 px | 141 ms |
| GWLF | 65.6% | 0.545 px | 145 ms |
| **EDLines** | 71.8% | **0.248 px** | 154 ms |
| **EDLines + Moments** | 71.8% | **0.248 px** | 149 ms |

Key observations:
- **Soft decoding** is the recall winner for ICRA (+19pp over Hard, correct for noisy scenes).
- EDLines achieves the best corner RMSE of all hard-decode paths (−9% vs ContourRdp).
- EDLines recall trails ContourRdp by 5pp on this multi-tag scene — consistent with the
  ~2–4% recall gap on hub single-tag data.
- GWLF recall drops significantly (−11pp) on the ICRA scene — edge fitting struggles with
  motion blur and steep angles common here; its RMSE increase confirms this.
- Moments culling adds no accuracy benefit on ICRA forward (clutter is already filtered by
  the existing geometry gates).

### 3.1 ICRA Fixtures (CI gold standard, 1 image)

| Config | Recall | Corner RMSE |
|---|---|---|
| ContourRdp (default) | 100% | 0.131 px |
| Moments Culling | 100% | 0.131 px |
| **EDLines** | 100% | **0.071 px** |
| **EDLines + Moments** | 100% | **0.071 px** |

EDLines cuts corner error nearly in half on the clean fixture image (−46%).

---

## 4. Phase 5 Gauss-Newton — Isolated Impact

Comparing Phase 4 only (IRLS re-fit + pairwise intersection) vs Phase 4 + Phase 5 (joint GN)
on the 720p hub dataset:

| Metric | Phase 4 only | Phase 4 + GN | Δ |
|---|---|---|---|
| Recall | 96% | 96% | 0 |
| Corner RMSE | 0.655 px | 0.661 px | +0.9% |
| Reprojection RMSE | 1.184 px | 1.056 px | **−10.8%** |
| Trans P50 | 4.0 mm | 3.9 mm | −2.5% |
| **Rot P50** | **0.64°** | **0.39°** | **−39%** |
| **Rot P90** | **3.17°** | **2.35°** | **−26%** |

And at 1080p:

| Metric | Phase 4 only | Phase 4 + GN | Δ |
|---|---|---|---|
| Corner RMSE | 0.758 px | 0.772 px | +1.8% |
| Reprojection RMSE | 2.739 px | 2.469 px | **−9.9%** |
| Trans P50 | 4.4 mm | 4.4 mm | 0 |
| **Rot P50** | **0.75°** | **0.50°** | **−34%** |
| **Rot P90** | **8.68°** | **4.96°** | **−43%** |

The joint GN trades ~1% corner RMSE for a 34–39% reduction in median rotation error and a
26–43% reduction in 90th-percentile rotation error. This is the expected result: the planarity
constraint yields geometrically consistent corners that map to better poses, even if individual
corner pixel positions move slightly from their independently-optimal locations.

---

## 5. Configuration Guide

| Use Case | Recommended Config | Rationale |
|---|---|---|
| Production robotics (single tag) | **GWLF** | Best recall + rotation accuracy |
| Metrology / calibration | **EDLines + GN** | Lowest corner RMSE |
| High-recall multi-tag scenes | **ContourRdp + Soft** | +20pp recall vs hard decode |
| Cluttered scenes (speed) | **Moments Culling** | Rejects elongated blobs before contour |
| 4K single tag | **GWLF** | EDLines recall loss too large at 4K |
| Fastest path (embedded) | **ContourRdp Fast** | Skips Structure Tensor pose |

---

## 6. Implementation Notes

### Phase 5 Gauss-Newton Algorithm

State θ = [x₀,y₀,x₁,y₁,x₂,y₂,x₃,y₃] (4 corners × 2 coords).

For each sub-pixel observation q on edge k (connecting corners k and k+1):
- Compute projection α = (q−cₖ)·tₖ/Lₖ; discard if α ∉ [0.05, 0.95] (exclusion zone)
- Residual: r = (q−cₖ)·nₖ (signed perpendicular distance)
- Sparse Jacobian: ∂r/∂cₖ = −(1−α)nₖ, ∂r/∂cₖ₊₁ = −αnₖ

Normal equations accumulated in a single pass:
```
H[8×8] += J^T J   (4×4 = 16 adds per observation, no Jacobian materialized)
g[8]   += J^T r
```

Cholesky solver: unrolled LL^T factorization, all 36 lower-triangular scalars on the stack.
Tikhonov regularisation λ=1e-6 prevents degenerate systems when edges lack sub-pixel points.
Convergence: ‖Δθ‖∞ < 0.005 px, max 3 iterations (typically converges in 1).
Fallback: if any corner moves >5 px from Phase-4 estimate, revert (divergence guard).

### Memory & Allocation Profile

EDLines pipeline per-quad allocation budget (arena):
- Phase 1 boundary scan: O(perimeter) BumpVec entries
- Phase 2 IRLS weight vectors: 4 × O(N/4) f64 slices
- Phase 3 sub-pixel points: 4 × O(L/step) f64 pair slices
- Phase 5 GN: zero — all on stack ([f64;64] H, [f64;8] g, [f64;8] Δθ)
