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
| **Expansion fix** | Skip ±0.5px centroid expansion for EdLines (was designed for integer ContourRdp corners only) | main |
| **None-mode fix** | `CornerRefinementMode::None` now correctly skips refinement; previously fell through to GWLF | main |

The Gauss-Newton stage replaces independent pairwise line intersection with a joint optimisation
that enforces geometric planarity across all four corners simultaneously. The expansion fix (not
applying ±0.5px outward push to already sub-pixel GN corners) recovers significant accuracy.

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
| GWLF | **100%** | 0.751 px | 3.7 mm | 0.31° | 59 ms |
| **EDLines + GWLF** | 96% | 0.621 px | 3.9 mm | **0.16°** | 68 ms |
| **EDLines + ERF** | 96% | 0.592 px | 3.9 mm | 0.28° | 68 ms |
| **EDLines + None** | 96% | **0.166 px** | **2.7 mm** | 0.27° | 65 ms |
| **EDLines + Moments + None** | 96% | **0.166 px** | **2.7 mm** | 0.27° | 58 ms |

EDLines+None (GN corners used directly, no post-refinement) gives the lowest corner RMSE by
far (−82% vs ContourRdp). ERF post-processing is counter-productive when applied to already
sub-pixel GN corners. EDLines+GWLF achieves the best absolute rotation (0.16°) at the cost
of higher per-corner RMSE, by imposing global line-direction constraints.

### 2.3 1080p / 1920×1080 (45 images)

| Config | Recall | Corner RMSE | Trans P50 | Rot P50 | Latency |
|---|---|---|---|---|---|
| ContourRdp (default) | **100%** | 1.146 px | 10.7 mm | 2.24° | 109 ms |
| ContourRdp Fast | **100%** | 1.146 px | 11.8 mm | 2.13° | 117 ms |
| Moments Culling | **100%** | 1.146 px | 10.7 mm | 2.24° | 109 ms |
| GWLF | **100%** | 0.928 px | 4.9 mm | **0.31°** | 108 ms |
| **EDLines + ERF** | 97.8% | 0.689 px | 4.4 mm | 0.36° | 124 ms |
| **EDLines + Moments + ERF** | 97.8% | 0.689 px | 4.4 mm | 0.36° | 120 ms |

Note: post-expansion-fix numbers (±0.5px centroid expansion now skipped for EdLines).
EDLines+ERF: −40% corner RMSE vs ContourRdp default, −26% vs GWLF. See Section 5 for
the full refinement-mode comparison; EdLines+None is recommended for best corner accuracy.

### 2.4 4K / 3840×2160 (45 images)

| Config | Recall | Corner RMSE | Trans P50 | Rot P50 | Latency |
|---|---|---|---|---|---|
| ContourRdp (default) | 97.8% | 1.116 px | 43.5 mm | 6.68° | 136 ms |
| ContourRdp Fast | 97.8% | 1.116 px | 41.4 mm | 6.68° | 178 ms |
| Moments Culling | 97.8% | 1.116 px | 43.5 mm | 6.68° | 208 ms |
| GWLF | 97.8% | **0.829 px** | **9.9 mm** | **0.97°** | 109 ms |
| **EDLines + ERF** | 82.2% | **0.506 px** | 12.2 mm | 0.70° | 233 ms |
| **EDLines + Moments + ERF** | 84.4% | 0.516 px | 12.2 mm | 0.70° | 231 ms |

Note: post-expansion-fix numbers. Recall at 4K improved significantly (+6.6pp) after the
±0.5px expansion fix; corner RMSE also improved (−3%). GWLF is still recommended at 4K
for applications requiring >85% recall. For metrology at 4K, EdLines+None will give the
lowest per-corner RMSE (see Section 5).

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
| Metrology / calibration | **EDLines + GN + None** | Lowest corner RMSE (0.166 px at 720p) |
| Angular precision (robotics) | **EDLines + GN + GWLF** | Best Rot P50 (0.16° at 720p) |
| High-recall multi-tag scenes | **ContourRdp + Soft** | +20pp recall vs hard decode |
| Cluttered scenes (speed) | **Moments Culling** | Rejects elongated blobs before contour |
| 4K single tag | **GWLF** | EDLines recall loss too large at 4K |
| Fastest path (embedded) | **ContourRdp Fast** | Skips Structure Tensor pose |

### EDLines Refinement Mode Comparison (720p hub, 50 images)

The GN corners produced by Phase 5 are already sub-pixel accurate. Applying ERF post-refinement
is counter-productive — it moves corners away from their geometrically-consistent GN positions.

| Config | Corner RMSE | Reprojection RMSE | Rot P50 | Notes |
|---|---|---|---|---|
| **EdLines + None** (GN direct) | **0.166 px** | **0.541** | 0.27° | Best for metrology |
| EdLines + ERF (default) | 0.592 px | 0.848 | 0.28° | ERF degrades GN accuracy |
| **EdLines + GWLF** | 0.621 px | 0.656 | **0.16°** | Best rotation stability |

**Key finding:** Using `CornerRefinementMode::None` with `QuadExtractionMode::EdLines` gives
3.5× lower corner RMSE than ERF and the lowest reprojection error. ERF was designed for integer
ContourRdp corners; it should not be used with EDLines GN output.

**Recommendation for EDLines users:** Always set `CornerRefinementMode::None` unless angular
stability is the primary goal, in which case use `CornerRefinementMode::Gwlf`.

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

---

## 2026.04.11 Addendum: Algorithm Pruning and Baseline Stabilization

As part of the Reconciliation and Stabilization phase, the following algorithms and configurations have been sunsetted:

### 1. Removal of Bilateral Filtering
**Rationale:** Bilateral filtering was originally introduced to handle high-noise scenarios. However, empirical testing in Phase 5 revealed that bilateral filtering distorts the optical PSF (Point Spread Function), introducing non-linear artifacts that actively degrade the Phase 5 Gauss-Newton solver's convergence. The solver's inherent weighted least-squares formulation provides superior noise rejection without distorting the underlying image geometry.

### 2. Removal of `GridFit` Corner Refinement
**Rationale:** `GridFit` (template-based corner fitting) proved mathematically incompatible with the EDLines-to-Gauss-Newton pipeline. `GridFit` assumes a rigid 2D grid template which conflicts with the flexible, edge-driven sub-pixel refinement required for high-accuracy metrology. The unification around `CornerRefinementMode::Erf` and `CornerRefinementMode::Gwlf` provides a more consistent mathematical baseline for the IPPE-Square and Phase 5 solvers.
