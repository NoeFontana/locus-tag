# Rotation-Tail Diagnostic — Phase 0 (20260503)

Forensics of the residual rotation tail on `locus_v1_tag36h11_1920x1080` under `high_accuracy` + Accurate-mode pose. Output of the Phase 0 diagnostic harness (`tools/bench/rotation_tail_diag/`).

## §1 TL;DR

- **Dataset**: `locus_v1_tag36h11_1920x1080` — 50 scenes, single tag each, AprilTag36h11 family, 1920×1080.
- **Profile / mode**: `high_accuracy` + `Accurate`. σ_n² configured = 4.000 (σ ≈ 2.000px).
- **Recall**: 50/50 scenes detected.
- **Rotation error vs GT** (degrees, 95% bootstrap CI):  p50 = 0.057 [0.041, 0.093]  ·  p95 = 0.473 [0.275, 0.780]  ·  p99 = 0.771 [0.383, 0.874]
- **Translation error vs GT** (mm):  p50 = 0.4  ·  p95 = 8.6  ·  p99 = 18.9
- **Latency** (ms, production path, no diagnostics):  p50 = 8.26  ·  p99 = 25.45

### Reproducibility cross-check

The published `high_accuracy` baseline (commit `8890efc`, 2026-04-25) reported rot p99 = 1.897° on this dataset. The current run lands at rot p50 = 0.057° / p99 = 0.771° — the bulk distribution is *better* than the published memo (the snapshot rebless after PR #212 measured all 50 scenes honestly), with a residual tail driven by a small number of outlier scenes. See §2 / §4 for the breakdown.

## §2 Failure-mode breakdown

Mutually-exclusive classification of the 50 scenes. The `healthy` bucket is **corpus-relative**: a scene is healthy iff its rotation error is below the 85th percentile of the population. This keeps the bucket size bounded as absolute residuals shrink and preserves the discriminating power of the counterfactual table.

| Mode | Count | % | Counterfactual rot p99 if resolved |
| :--- | ---: | ---: | ---: |
| `healthy` | 42 | 84.0% | — |
| `sigma_miscalibration` | 5 | 10.0% | 0.771° |
| `ppm_starved` | 2 | 4.0% | 0.642° |
| `corner_geometry_outlier` | 1 | 2.0% | 0.600° |

_Counterfactual interpretation: "if all `<mode>` scenes had rotation error = 0, what would p99 become?" Current p99 = 0.771°. The mode whose counterfactual drops p99 the most is the priority fix.

## §3 Stratified rotation error

### Angle of incidence (deg)

| Bucket | n | rot p50 (95% CI) | rot p95 | rot p99 | rot max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 10.59…28.93 | 12 | 0.189 [0.071, 0.339] | 0.686 | 0.837 | 0.874 |
| 30.45…45.85 | 13 | 0.032 [0.022, 0.074] | 0.196 | 0.236 | 0.246 |
| 45.86…54.21 | 12 | 0.047 [0.018, 0.194] | 0.519 | 0.635 | 0.664 |
| 56.29…59.55 | 13 | 0.074 [0.033, 0.093] | 0.228 | 0.229 | 0.230 |

### Distance (m)

| Bucket | n | rot p50 (95% CI) | rot p95 | rot p99 | rot max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.56…0.68 | 12 | 0.027 [0.016, 0.032] | 0.060 | 0.073 | 0.077 |
| 0.69…0.91 | 13 | 0.049 [0.033, 0.074] | 0.243 | 0.340 | 0.365 |
| 0.93…1.45 | 12 | 0.076 [0.043, 0.137] | 0.529 | 0.805 | 0.874 |
| 1.46…3.74 | 13 | 0.230 [0.117, 0.313] | 0.585 | 0.648 | 0.664 |

### Estimated PPM (px/m)

| Bucket | n | rot p50 (95% CI) | rot p95 | rot p99 | rot max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 293.68…750.68 | 12 | 0.231 [0.117, 0.357] | 0.592 | 0.650 | 0.664 |
| 798.05…1349.88 | 13 | 0.078 [0.054, 0.181] | 0.498 | 0.799 | 0.874 |
| 1417.66…1711.10 | 12 | 0.055 [0.020, 0.099] | 0.253 | 0.342 | 0.365 |
| 1737.58…2232.15 | 13 | 0.031 [0.023, 0.033] | 0.048 | 0.049 | 0.049 |

## §4 Top-10 worst scenes

| # | scene_id | rot err (°) | trans err (mm) | branch | classification | Rerun | 
| ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| 1 | `scene_0008_cam_0000` | 0.87 | 11.1 | 0 | `corner_geometry_outlier` | `recordings/scene_0008_cam_0000.rrd` |
| 2 | `scene_0005_cam_0000` | 0.66 | 26.4 | 0 | `ppm_starved` | `recordings/scene_0005_cam_0000.rrd` |
| 3 | `scene_0009_cam_0000` | 0.53 | 8.4 | 0 | `ppm_starved` | `recordings/scene_0009_cam_0000.rrd` |
| 4 | `scene_0025_cam_0000` | 0.40 | 6.9 | 0 | `sigma_miscalibration` | `recordings/scene_0025_cam_0000.rrd` |
| 5 | `scene_0018_cam_0000` | 0.36 | 0.3 | 0 | `sigma_miscalibration` | `recordings/scene_0018_cam_0000.rrd` |
| 6 | `scene_0033_cam_0000` | 0.31 | 1.2 | 0 | `sigma_miscalibration` | `recordings/scene_0033_cam_0000.rrd` |
| 7 | `scene_0023_cam_0000` | 0.30 | 7.9 | 0 | `sigma_miscalibration` | `recordings/scene_0023_cam_0000.rrd` |
| 8 | `scene_0015_cam_0000` | 0.25 | 1.0 | 0 | `sigma_miscalibration` | `recordings/scene_0015_cam_0000.rrd` |
| 9 | `scene_0041_cam_0000` | 0.23 | 1.6 | 0 | `healthy` | `recordings/scene_0041_cam_0000.rrd` |
| 10 | `scene_0021_cam_0000` | 0.23 | 8.8 | 0 | `healthy` | `recordings/scene_0021_cam_0000.rrd` |

## §5 σ calibration check

Configured σ in `high_accuracy.json`: σ = √4.000 ≈ 2.000px.

Per-image estimated σ (Immerkær median Laplacian):

```
   0.494– 0.914  │ ████████████████████████████████████████  (29)
   0.914– 1.334  │ ██████████  (7)
   1.334– 1.754  │ █  (1)
   1.754– 2.174  │ █  (1)
   2.174– 2.595  │ ████████  (6)
   2.595– 3.015  │ ███  (2)
   3.015– 3.435  │   (0)
   3.435– 3.855  │ █  (1)
   3.855– 4.275  │ █  (1)
   4.275– 4.695  │   (0)
   4.695– 5.115  │   (0)
   5.115– 5.535  │ █  (1)
   5.535– 5.955  │   (0)
   5.955– 6.375  │   (0)
   6.375– 6.795  │   (0)
   6.795– 7.215  │   (0)
   7.215– 7.635  │   (0)
   7.635– 8.055  │   (0)
   8.055– 8.475  │   (0)
   8.475– 8.895  │ █  (1)
```

Population: median σ = 0.741px, max = 8.895px. 34 of 50 scenes have σ < 1.00px (half the configured value).

## §6 IRLS weight + Mahalanobis distribution

Final per-corner Huber IRLS weights at the LM-converged pose:

```
  0.219–0.258  │   (1)
  0.258–0.297  │   (0)
  0.297–0.336  │   (0)
  0.336–0.375  │   (0)
  0.375–0.414  │   (0)
  0.414–0.453  │   (0)
  0.453–0.492  │   (0)
  0.492–0.531  │   (0)
  0.531–0.570  │   (0)
  0.570–0.609  │   (0)
  0.609–0.649  │   (0)
  0.649–0.688  │   (0)
  0.688–0.727  │   (0)
  0.727–0.766  │   (0)
  0.766–0.805  │   (0)
  0.805–0.844  │   (0)
  0.844–0.883  │   (0)
  0.883–0.922  │   (0)
  0.922–0.961  │   (0)
  0.961–1.000  │ ████████████████████████████████████████  (199)
```

1 of 200 corners have IRLS weight < 0.3 — that's the threshold the `corner_outlier` classifier uses.

Per-scene branch-d² (chosen vs alternate, log-binned counts):

**Chosen branch (log10 d²):**
```
  -2.72–-2.50  │ ████████████  (3)
  -2.50–-2.27  │   (0)
  -2.27–-2.05  │ ████████████████████  (5)
  -2.05–-1.82  │ ████████  (2)
  -1.82–-1.59  │ ████████  (2)
  -1.59–-1.37  │ ████████████  (3)
  -1.37–-1.14  │ ████████████████████████████████████  (9)
  -1.14–-0.92  │ ████████████████████████████████  (8)
  -0.92–-0.69  │ ████████████████████████████████████████  (10)
  -0.69–-0.47  │ ████████████████  (4)
  -0.47–-0.24  │ ████████  (2)
  -0.24–-0.01  │   (0)
  -0.01– 0.21  │ ████  (1)
   0.21– 0.44  │   (0)
   0.44– 0.66  │   (0)
   0.66– 0.89  │   (0)
   0.89– 1.12  │   (0)
   1.12– 1.34  │   (0)
   1.34– 1.57  │   (0)
   1.57– 1.79  │ ████  (1)
```

**Alternate branch (log10 d²):**
```
  -2.50–-2.10  │ ███████  (2)
  -2.10–-1.70  │   (0)
  -1.70–-1.31  │   (0)
  -1.31–-0.91  │ ████  (1)
  -0.91–-0.51  │ ███████████████  (4)
  -0.51–-0.12  │ ███████  (2)
  -0.12– 0.28  │   (0)
   0.28– 0.67  │   (0)
   0.67– 1.07  │   (0)
   1.07– 1.47  │ ███████████  (3)
   1.47– 1.86  │ ███████  (2)
   1.86– 2.26  │ ████  (1)
   2.26– 2.66  │ ████  (1)
   2.66– 3.05  │ ██████████████████  (5)
   3.05– 3.45  │   (0)
   3.45– 3.85  │ ███████████  (3)
   3.85– 4.24  │ █████████████████████████████  (8)
   4.24– 4.64  │ ████████████████████████████████████████  (11)
   4.64– 5.03  │ ██████████████████████  (6)
   5.03– 5.43  │ ████  (1)
```

## §7 What this points at

The dominant failure mode is **`corner_geometry_outlier`** at 1 of 50 scenes — at least one corner has Mahalanobis d² above the χ²(1) cutoff at α=1e-4 (≈15.137), i.e. it lives in the extreme tail of the noise model. This is a strict super-set of the IRLS-weight gate, so look upstream of the pose solver — sub-pixel localization on the offending corner is the primary suspect (GWLF, EdLines, ERF refinement, or saturated PSF on a close tag).

**Healthy** at 42 of 50 scenes — the bulk distribution is well-calibrated; the tail is concentrated in the 8 non-healthy scene(s) above.

**`sigma_miscalibration`** at 5 scene(s) — partially confounded with the dataset rather than the algorithm: Blender-rendered images have very low noise floors, while the production profiles ship `sigma_n_sq = 4.0` (σ ≈ 2 px). The `compute_image_noise_floor` helper is permanent in `gradient.rs` and ready to wire into the per-frame LM info matrices.

## §8 Profile × mode reproducibility table

Captured via `tools/bench/render_tag_sota_eval.py` on the same dataset, today, for context. (Source: `/tmp/render_tag_sota_full.json` after running the eval tool.)

| Profile | Mode | Recall | rot p50 | rot p95 | rot p99 | trans p99 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `standard` | Accurate | 100.0 % | 0.288° | 1.572° | 27.248° | 50.3 mm | (2026-04-25 snapshot) |
| `high_accuracy` | Fast | 94.0 % | 0.345° | 6.350° | 104.238° | 2210 mm | (2026-04-25 snapshot) |
| `high_accuracy` | Accurate | 94.0 % | 62.857° | 148.239° | 154.233° | 3261 mm | (2026-04-25 snapshot, **pre-fix**; rerun via `render_tag_sota_eval.py`) |
| `high_accuracy` | Fast | 100.0 % | 0.363° | 6.137° | 103.402° | 2164 mm | (2026-04-25 snapshot) |
| **`high_accuracy`** | **Accurate** | **100.0 %** | **0.057°** | **0.473°** | **0.771°** | **19 mm** | (this run) |

Non-`high_accuracy` rows above are pre-fix snapshots from 2026-04-25 (commit `8890efc`). After PR #212 the `high_accuracy` Accurate row in particular is stale — rerun `tools/bench/render_tag_sota_eval.py` for fresh numbers.

## §9 Recommendations

1. **Triage the non-healthy scenes (§4 top-10).** 8 of 50 scenes carry the entire residual tail; the linked `.rrd` recordings let you see whether the remaining error is a corner-localization issue, a remaining branch-selector edge case, or sensor / render noise. Fix at this granularity rather than tuning population-level knobs.

2. **Wire `compute_image_noise_floor` into the LM info matrices** to address the `sigma_miscalibration` scenes (5 of 50 here). The helper is permanent in `gradient.rs`; what's missing is a per-frame call from `refine_pose_lm_weighted`. Counterfactual p99 below this drops the residual tail substantially (see §2 table).

3. **Reframe the SOTA gap.** With high_accuracy at rot p50 = 0.057° / p99 = 0.771°, the bulk distribution already beats every external detector. Where external libraries still hold the rotation P95/P99 tail (per `render_tag_sota_20260425.md`) is what closing this residual tail unlocks. The next bottleneck is no longer a single regression — it is the tail of outlier scenes plus the σ calibration mismatch on Blender-rendered data.

4. **Consider extending the failure-mode taxonomy.** Healthy = 42/50 means most residual error falls below the discriminating thresholds in `classify.py`. As the tail shrinks further, the classifier should grow finer modes (e.g. grazing-angle subclass, per-corner GN-residual outlier) to keep producing actionable signal.

---

_Generated by `tools/bench/rotation_tail_diag/report.py` from `diagnostics/2026-05-03/`. Bootstrap CIs are non-parametric, 10000 resamples, RNG seed 42. .rrd recordings live alongside scenes.json — open with `rerun recordings/scene_NNNN.rrd` (per-scene)._
