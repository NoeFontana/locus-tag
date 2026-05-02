# Rotation-Tail Diagnostic — Phase 0 (20260502)

Forensics of the residual rotation tail on `locus_v1_tag36h11_1920x1080` under `render_tag_hub` + Accurate-mode pose. Output of the Phase 0 diagnostic harness (`tools/bench/rotation_tail_diag/`).

## §1 TL;DR

- **Dataset**: `locus_v1_tag36h11_1920x1080` — 50 scenes, single tag each, AprilTag36h11 family, 1920×1080.
- **Profile / mode**: `render_tag_hub` + `Accurate`. σ_n² configured = 4.000 (σ ≈ 2.000px).
- **Recall**: 50/50 scenes detected.
- **Rotation error vs GT** (degrees, 95% bootstrap CI):  p50 = 60.470 [45.152, 100.276]  ·  p95 = 148.124 [130.570, 154.240]  ·  p99 = 154.212 [144.291, 154.555]
- **Translation error vs GT** (mm):  p50 = 771.9  ·  p95 = 2800.7  ·  p99 = 3248.4
- **Latency** (ms, production path, no diagnostics):  p50 = 12.54  ·  p99 = 37.13

### Reproducibility cross-check

The published `render_tag_hub` baseline (commit `8890efc`, 2026-04-25) reported rot p99 = 1.897° on this exact dataset. The numbers above show **a regression**: the live `render_tag_hub` + Accurate-mode pose path now produces rot p50 ≈ 60° (two orders of magnitude). Reproduced independently via `tools/bench/render_tag_sota_eval.py` — see §7.

## §2 Failure-mode breakdown

Mutually-exclusive classification of the 50 scenes.

| Mode | Count | % | Counterfactual rot p99 if resolved |
| :--- | ---: | ---: | ---: |
| `frame_or_winding` | 27 | 54.0% | 150.840° |
| `healthy` | 15 | 30.0% | — |
| `branch_flip` | 6 | 12.0% | 151.573° |
| `sigma_miscalibration` | 2 | 4.0% | 154.212° |

_Counterfactual interpretation: "if all `<mode>` scenes had rotation error = 0, what would p99 become?" Current p99 = 154.212°. The mode whose counterfactual drops p99 the most is the priority fix.

## §3 Stratified rotation error

### Angle of incidence (deg)

| Bucket | n | rot p50 (95% CI) | rot p95 | rot p99 | rot max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 10.59…28.93 | 12 | 35.862 [0.328, 105.290] | 150.471 | 153.178 | 153.855 |
| 30.45…45.85 | 13 | 100.187 [0.027, 109.010] | 141.501 | 147.075 | 148.469 |
| 45.86…54.21 | 12 | 48.028 [0.051, 82.722] | 133.933 | 150.430 | 154.555 |
| 56.29…59.55 | 13 | 74.250 [62.907, 114.563] | 139.502 | 145.702 | 147.252 |

### Distance (m)

| Bucket | n | rot p50 (95% CI) | rot p95 | rot p99 | rot max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.56…0.68 | 12 | 54.695 [0.027, 118.013] | 140.696 | 146.914 | 148.469 |
| 0.69…0.91 | 13 | 48.619 [0.044, 100.187] | 122.917 | 142.744 | 147.701 |
| 0.93…1.45 | 12 | 78.337 [60.470, 125.709] | 154.170 | 154.478 | 154.555 |
| 1.46…3.74 | 13 | 74.250 [21.981, 117.061] | 135.614 | 144.924 | 147.252 |

### Estimated PPM (px/m)

| Bucket | n | rot p50 (95% CI) | rot p95 | rot p99 | rot max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 293.68…750.68 | 12 | 70.870 [11.258, 114.179] | 136.584 | 145.118 | 147.252 |
| 798.05…1349.88 | 13 | 89.185 [58.084, 121.395] | 154.135 | 154.471 | 154.555 |
| 1417.66…1711.10 | 12 | 54.272 [0.081, 100.276] | 118.968 | 131.263 | 134.336 |
| 1737.58…2232.15 | 13 | 0.046 [0.027, 110.059] | 148.008 | 148.377 | 148.469 |

## §4 Top-10 worst scenes

| # | scene_id | rot err (°) | trans err (mm) | branch | classification | Rerun | 
| ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| 1 | `scene_0040_cam_0000` | 154.55 | 982.1 | 0 | `frame_or_winding` | `recordings/scene_0040_cam_0000.rrd` |
| 2 | `scene_0024_cam_0000` | 153.86 | 1014.9 | 0 | `branch_flip` | `recordings/scene_0024_cam_0000.rrd` |
| 3 | `scene_0047_cam_0000` | 148.47 | 599.3 | 0 | `frame_or_winding` | `recordings/scene_0047_cam_0000.rrd` |
| 4 | `scene_0030_cam_0000` | 147.70 | 807.9 | 0 | `branch_flip` | `recordings/scene_0030_cam_0000.rrd` |
| 5 | `scene_0013_cam_0000` | 147.25 | 2169.7 | 0 | `frame_or_winding` | `recordings/scene_0013_cam_0000.rrd` |
| 6 | `scene_0004_cam_0000` | 136.86 | 932.2 | 0 | `frame_or_winding` | `recordings/scene_0004_cam_0000.rrd` |
| 7 | `scene_0049_cam_0000` | 134.34 | 620.7 | 0 | `frame_or_winding` | `recordings/scene_0049_cam_0000.rrd` |
| 8 | `scene_0021_cam_0000` | 127.86 | 2661.5 | 0 | `branch_flip` | `recordings/scene_0021_cam_0000.rrd` |
| 9 | `scene_0027_cam_0000` | 125.97 | 641.4 | 0 | `frame_or_winding` | `recordings/scene_0027_cam_0000.rrd` |
| 10 | `scene_0016_cam_0000` | 121.40 | 1459.0 | 0 | `frame_or_winding` | `recordings/scene_0016_cam_0000.rrd` |

## §5 σ calibration check

Configured σ in `render_tag_hub.json`: σ = √4.000 ≈ 2.000px.

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
  0.500–0.550  │   (0)
  0.550–0.600  │   (0)
  0.600–0.650  │   (0)
  0.650–0.700  │   (0)
  0.700–0.750  │   (0)
  0.750–0.800  │   (0)
  0.800–0.850  │   (0)
  0.850–0.900  │   (0)
  0.900–0.950  │   (0)
  0.950–1.000  │   (0)
  1.000–1.050  │ ████████████████████████████████████████  (60)
  1.050–1.100  │   (0)
  1.100–1.150  │   (0)
  1.150–1.200  │   (0)
  1.200–1.250  │   (0)
  1.250–1.300  │   (0)
  1.300–1.350  │   (0)
  1.350–1.400  │   (0)
  1.400–1.450  │   (0)
  1.450–1.500  │   (0)
```

0 of 60 corners have IRLS weight < 0.3 — that's the threshold the `corner_outlier` classifier uses.

Per-scene branch-d² (chosen vs alternate, log-binned counts):

**Chosen branch (log10 d²):**
```
  -2.50–-2.29  │ ████████  (2)
  -2.29–-2.09  │ ████  (1)
  -2.09–-1.88  │ ████████████████  (4)
  -1.88–-1.68  │ ████████  (2)
  -1.68–-1.47  │ ████████████  (3)
  -1.47–-1.27  │   (0)
  -1.27–-1.06  │ ████████████████████████  (6)
  -1.06–-0.86  │ ████████████████████████████████████████  (10)
  -0.86–-0.65  │ ████████████████████████████████████  (9)
  -0.65–-0.45  │ ████████████████████████  (6)
  -0.45–-0.24  │ ████████████  (3)
  -0.24–-0.04  │ ████████  (2)
  -0.04– 0.17  │   (0)
   0.17– 0.37  │ ████  (1)
   0.37– 0.58  │   (0)
   0.58– 0.78  │   (0)
   0.78– 0.99  │   (0)
   0.99– 1.19  │   (0)
   1.19– 1.40  │   (0)
   1.40– 1.60  │ ████  (1)
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

The dominant failure mode is **`frame_or_winding`** at 27 of 50 scenes — i.e. *the chosen IPPE branch fits the observed corners well (low aggregate d²) but its rotation against GT is large* (>30°). This is the signature of a coordinate-frame or corner-ordering mismatch upstream of the pose solver, not a pose-refinement issue. Fixing the LM solver alone cannot recover from it; the offending permutation/sign needs to be located in:

1. The corner-extraction stage (EdLines vs ContourRdp produces different corner orderings on this dataset — see the render_tag_sota_eval.py cross-product table in §8).
2. The Accurate-mode-only weighted LM path (`refine_pose_lm_weighted`), which is the only path that regresses; Fast-mode `refine_pose_lm` is healthy on `standard` profile (rot p50 = 0.288°).

**`branch_flip`** at 6 scenes is real but secondary. It will only ever explain a small fraction of the tail while `frame_or_winding` dominates.

**`sigma_miscalibration`** is partially confounded with the dataset rather than the algorithm: Blender-rendered images have very low noise floors, while the production profiles ship `sigma_n_sq = 4.0` (σ ≈ 2 px). Phase 3 of the SOTA plan (per-frame σ estimation) addresses this directly.

## §8 Profile × mode reproducibility table

Captured via `tools/bench/render_tag_sota_eval.py` on the same dataset, today, for context. (Source: `/tmp/render_tag_sota_full.json` after running the eval tool.)

| Profile | Mode | Recall | rot p50 | rot p95 | rot p99 | trans p99 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `standard` | Accurate | 100.0 % | 0.288° | 1.572° | 27.248° | 50.3 mm |
| `high_accuracy` | Fast | 94.0 % | 0.345° | 6.350° | 104.238° | 2210 mm |
| `high_accuracy` | Accurate | 94.0 % | 62.857° | 148.239° | 154.233° | 3261 mm |
| `render_tag_hub` | Fast | 100.0 % | 0.363° | 6.137° | 103.402° | 2164 mm |
| **`render_tag_hub`** | **Accurate** | **100.0 %** | **60.470°** | **148.124°** | **154.212°** | **3248 mm** | (this run) |

**Recovery path: switch the rotation-tail Phase 1–4 work to use `standard` profile first** — that's where the ~28° p99 tail still behaves like a real perception problem. `render_tag_hub` and `high_accuracy` need their Accurate-mode pose regression fixed before they can serve as the SOTA floor.

## §9 Recommendations (reorders Phase 1–4 from the SOTA plan)

1. **Phase 0.1 (new)**: Bisect the Accurate-mode regression on `render_tag_hub` / `high_accuracy`. Most likely culprits, in order:
   - EdLines corner ordering vs ContourRdp; check the four corners' winding direction.
   - Recent `refine_pose_lm_weighted` changes (`8890efc` introduced the Mahalanobis χ² gate).
   - `pose_consistency_fpr = 1e-3` rejecting all geometrically-consistent poses for a frame-flip reason.

2. **Phase 1 (photometric refinement)**: deferred until the regression above is fixed. Photometric refinement cannot recover from a corner-ordering bug.

3. **Phase 2 (branch hardening)**: real but small (12% of scenes). Only worthwhile after Phase 0.1; otherwise hardened branch selection still picks a corner-mis-ordered IPPE candidate.

4. **Phase 3 (per-frame σ estimation)**: the harness already provides `compute_image_noise_floor` (permanent in `gradient.rs`); Phase 3 just needs to wire it into the per-frame LM info matrices. Independently useful regardless of Phase 0.1 outcome.

5. **Phase 4 (deferred)**: revisit after re-running the diagnostic on the fixed `render_tag_hub` / `high_accuracy` paths. Likely the failure-mode population shifts substantially when `frame_or_winding` is resolved.

---

_Generated by `tools/bench/rotation_tail_diag/report.py` from `diagnostics/2026-05-02/`. Bootstrap CIs are non-parametric, 10000 resamples, RNG seed 42. .rrd recordings live alongside scenes.json — open with `rerun recordings/scene_NNNN.rrd` (per-scene)._
