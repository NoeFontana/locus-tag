# Model-based edge pose refinement — beating the rotation gap (2026-07-15, tuned 2026-07-17)

## Summary

An opt-in Accurate-mode pose-refinement stage that aligns a **decoded** tag's
internal bit-grid edges (plus the border) to the image and refines the 6-DoF pose
against them. A corner PnP pose is fixed by only 4 points; a 36h11 tag exposes
**~40 internal high-contrast edges**, an order of magnitude more constraints,
distributed across the interior where they pin orientation far better than the
corners. On the render-tag suite this **cuts rotation p99 47–76 % at every
resolution — well under OpenCV's `apriltag` refinement — while keeping Locus's
best-in-class translation.**

Enabled by `pose.pose_edge_refinement_enabled` (the field default is **false**;
**`high_accuracy` ships it on as of v0.7.0** — see the profile-decision section
below — while `standard`/`grid` leave it off). Requires camera intrinsics + `tag_size`.

> **2026-07-17 update.** Rebased onto the unified pose-LM machinery (#337–#341):
> the estimator now rides the shared body-frame Nielsen trust-region core
> (`nielsen_lm` + `BodyFrameNormalEquations` + distortion-aware
> `projection_and_gradient`) instead of a bespoke left-perturbation LM loop — each
> 1-D edge-normal residual enters the same accumulator as a corner residual under a
> rank-1 `n·nᵀ` weight. Two tuning levers were then added (**robust Huber edge
> loss** + **denser edge sampling**); the results below are the tuned figures, with
> the pre-tuning shipped-#336 column kept for reference.

## Method

Post-decode, per tag (`crates/locus-core/src/model_edge.rs`):

1. **Sample** every candidate cell-boundary + border line in the tag frame (7
   interior points per boundary), project under the current pose, and measure the
   sub-pixel edge by the **50 %-intensity crossing** along each edge normal
   (unbiased for a symmetric PSF). Low-contrast samples (same-colour boundaries)
   are dropped automatically — the image contrast selects the real edges, so no
   bit-pattern reconstruction is needed. Junction, out-of-image, and large-offset
   (spurious) samples are excluded.
2. **`measure → fit`** iterations: hold the measured edge points fixed and refine
   the full 6-DoF pose to them with the shared **Nielsen trust-region LM** under a
   robust **Huber** loss (δ = 0.5 px, so the clean interior edges — not a few
   latched/biased ones — control the fit), then re-measure. This gives an excellent
   **rotation** but weakly constrains depth/scale.
3. **Re-anchor translation** with a translation-only solve against the 4 trusted
   corners under the refined rotation. *Edges → rotation, corners → translation.*
4. A **no-worse gate** rejects untrustworthy refinements (too few edges, or an
   implausible pose jump), falling back to the corner pose. The refined pose is
   re-validated against the pose-consistency χ² gate before it is emitted.

### Tuning levers (2026-07-17)

Both attack the rotation **p99 tail**, ranked on render-tag with hard
translation/RMSE/recall guardrails (corners are untouched by this stage, so mean
corner RMSE is unaffected):

- **Robust Huber edge loss (δ = 0.5 px).** An L2 fit lets a few edges that latch
  onto an adjacent boundary or carry PSF-asymmetry bias drag the rotation; the
  Huber weight `w·n·nᵀ` down-weights them. δ = 0.5 px was the sweep optimum (0.35
  regressed p95, 0.75 was weaker). *1080p rot p99 0.409° → 0.364°.*
- **7 (was 3) samples per cell boundary.** More *independent* edge measurements
  average per-scan noise down, tightening the tail further. Seven is the knee under
  the 64 KB on-stack sample budget (`constraints.md §1`); denser would force a
  per-tag heap allocation on the pose path. *1080p rot p99 0.364° → 0.249°.*

The `measure → fit` count (4) was already converged (6 gave no change) and the
inner LM reuses the vetted `NielsenConfig::POSE`, so no per-geometry calibration.

## Results (render-tag, single-thread, Accurate pose)

**Recall and precision are 100 % / 100 % for both baseline and the enabled stage
at every resolution** — the refinement only sharpens an already-accepted pose (the
no-worse + pose-consistency χ² gates never null one), so it neither drops nor adds
a detection. **Mean corner RMSE is unchanged by construction**: the stage never
writes `batch.corners`; it reads them as fixed input and emits only the pose.

Locus `high_accuracy`, **baseline** (no edge stage) → **tuned** (Huber δ = 0.5 + 7
samples/boundary), full rotation/translation distribution:

| resolution | rot mean (°) | rot p50 (°) | rot p95 (°) | rot p99 (°) | t mean (mm) | t p95 (mm) | t p99 (mm) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 640×480   | 0.155→**0.095** | 0.120→**0.084** | 0.373→**0.201** | 0.508→**0.269** | 1.09→**1.03** | 3.62→4.15 | 8.11→**7.60** |
| 1280×720  | 0.131→**0.064** | 0.058→**0.046** | 0.464→**0.194** | 0.624→**0.262** | 1.56→**1.47** | 8.16→8.26 | 15.97→**14.46** |
| 1920×1080 | 0.122→**0.062** | 0.057→**0.041** | 0.385→**0.180** | 0.600→**0.249** | 1.95→**1.88** | 8.62→9.75 | 18.62→20.10 |
| 3840×2160 | 0.139→**0.063** | 0.055→**0.041** | 0.432→**0.203** | 1.113→**0.267** | 3.81→4.07 | 12.82→17.72 | 39.02→**36.98** |

Rotation p99 for reference — **baseline → L2-edge (pre-tuning) → tuned**:
640 `0.508→0.314→0.269`, 720 `0.624→0.472→0.262`, 1080 `0.600→0.409→0.249`,
2160 `1.113→0.429→0.267`. The tuning (Huber + denser sampling) cuts rot p99 a
further −11 % to −44 % over the L2 stage.

Reading the table:

- **Rotation improves across the *entire* distribution, not just the tail.** Mean
  −39 % to −55 %, p50 roughly halved, p95 roughly halved, p99 −47 % to −76 %. At
  1080p the tuned rot p99 **0.249°** (p95 **0.180°**) is well under OpenCV
  `cv2.aruco` apriltag's **0.376°**, at trans p99 ~20 mm vs apriltag's ~55 mm —
  **apriltag-class-and-better rotation at ~2.7× better translation.**
- **Translation carries a bounded trade** — the edges→rotation / corners→translation
  split. **t mean is flat-to-better** at 640/720/1080 (−0.06 to −0.09 mm) and
  +0.26 mm only at 2160; **t p99 is better** at 640/720/2160 and +1.5 mm at 1080p.
  The regression concentrates in **t p95** at higher resolution (1080p +1.1 mm,
  2160p +4.9 mm) — the 1-2-tag pose-tail where a large rotation correction pulls the
  corner-anchored translation. This is the known, gated trade; it stays inside the
  opt-in stage (shipped profiles are byte-identical) and never touches recall,
  precision, or corner RMSE.
- **Latency** (single-thread, best-effort): +~1 ms/frame at 640/720/1080, roughly
  flat at 2160 (baseline → tuned: 2.4→3.8 / 6.3→7.5 / 14.3→15.3 / 57.4→58.2 ms).
  The stage is a bounded per-tag Gauss-Newton on the opt-in Accurate path.

### Image-space accuracy — 2D corner RMSE & reprojection RMSE

Two RMSEs make the "translation trades but the pose is still more accurate" story
concrete (harness: `tools/bench/model_edge_rmse.py`):

- **2D corner RMSE** (detected vs GT corners, px) — **identical** baseline vs edge
  at every resolution (the detected corners are byte-identical; the stage emits only
  the pose): mean 0.210 / 0.208 / 0.215 / 0.178 px for 640/720/1080/2160.
- **Reprojection RMSE** (model corners reprojected through the *estimated pose* vs
  GT corners, px — the `hub.rs` `mean_reprojection_rmse` quantity) — **improves at
  every resolution and percentile**:

| resolution | reproj mean (px) | reproj p95 (px) | reproj p99 (px) |
| :--- | ---: | ---: | ---: |
| 640×480   | 0.181 → **0.151** | 0.400 → **0.290** | 0.456 → **0.370** |
| 1280×720  | 0.185 → **0.144** | 0.418 → **0.278** | 0.457 → **0.339** |
| 1920×1080 | 0.173 → **0.141** | 0.406 → **0.254** | 0.759 → **0.553** |
| 3840×2160 | 0.158 → **0.132** | 0.395 → **0.254** | 0.411 → **0.393** |

Reprojection RMSE (mean −13…−22 %, p95 −27…−36 %, p99 −4…−27 %) falls everywhere:
in **image space** the refined pose is uniformly *closer to truth*, so the small
metric-translation p95 trade above is a depth/scale artefact of the corner re-anchor,
not a loss of pose accuracy where it is observable. Corner localization is untouched.

### Robustness — degraded-imagery corpora (does the render-tag tuning generalize?)

The tuning constants (Huber δ, sampling density, gates) were fit on the *clean*
render-tag suite. To check they are not render-tag-PSF-specific overfit, the stage
was run on the three degraded-imagery robustness sets (1080p, baseline vs edge):

| dataset | rec/prec | rot mean (°) | rot p99 (°) | t mean (mm) | t p99 (mm) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `high_iso` (sensor noise) | 100/100 both | 0.119 → **0.062** | 0.568 → **0.250** | 1.86 → 1.84 | 18.7 → 19.1 |
| `low_key` (dark/low-key)  | 100/100 both | 0.127 → **0.020** | 0.953 → **0.154** | 7.46 → 7.88 | 51.3 → 51.4 |
| `raw_pipeline`            | 100/100 both | 0.119 → **0.029** | 0.867 → **0.145** | 4.17 → 4.52 | 36.1 → 37.7 |

The rotation win is **larger** on degraded imagery (p99 −56 % to −84 %) than on clean
render-tag, because the corner baseline is noisier there while the ~40 distributed
edges still average that noise down. Crucially **recall and precision stay 100 %/100 %**:
the `CONTRAST_MIN=50` selection gate does *not* silently decline, and the absolute-pixel
Huber δ does *not* mis-clip, on high-ISO / low-key / raw-pipeline inputs — the two
failure modes an absolute-unit constant tuned on one corpus would be most prone to.
(These are still synthetic Blender degradations, not real-camera; the absolute δ and
the reused `NielsenConfig::POSE` absolute `grad_tol`/`damping_floor` remain the natural
first place to make scale-relative if real-camera data ever shows a regression.)

## Why this works where corner-level levers failed

The 2026-07-14 study (`refine_variants_20260714.md`) proved every *corner-level*
lever is trade-bound: GWLF/apriltag-edge, per-corner repair, per-tag switch, and
fusion all trade rotation for translation because they only reshape the same 4
corner observations. This stage instead **adds independent single-frame
information** — the decoded interior pattern's ~40 edges — which over-constrains
rotation without disturbing the corner-anchored translation. No multi-tag or
temporal fusion required.

## Provenance

AMD EPYC-Milan (Zen 3), 1 socket × 4 cores × 2 threads = 8 vCPU (`avx2`, `fma`,
`sse4_2`); `rustc 1.92.0`; `--release`; `RAYON_NUM_THREADS=1`, Locus `threads=1`;
`locus_v1_tag36h11_{640x480,1280x720,1920x1080,3840x2160}` and the robustness sets
`locus_v1_{high_iso,low_key,raw_pipeline}_1920x1080` (50 frames each); pose mode
**Accurate**. Accuracy columns are deterministic (verified identical across repeated
runs); the latency column is a best-effort single-thread wall-time.

## Reproduce

```bash
PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache RAYON_NUM_THREADS=1 \
  uv run --group bench tools/bench/model_edge_eval.py    # pose: recall/precision/rot/trans
PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache RAYON_NUM_THREADS=1 \
  uv run --group bench tools/bench/model_edge_rmse.py    # image-space: 2D corner + reproj RMSE
```

Both harnesses take `MODEL_EDGE_BASE_PROFILE` (default `high_accuracy`; set
`=standard` to measure the gain against the `standard` corner baseline).

## v0.7.0 profile decision — enabled in `high_accuracy`, NOT in `standard`

For the v0.7.0 cut the stage is turned **on in `high_accuracy`** (the numbers above
reproduced on the release hardware — AMD EPYC-Milan, `rustc 1.92.0`, `--release`,
`RAYON_NUM_THREADS=1`; clean render-tag + all three degraded sets, recall/precision
100 %/100 % throughout) and left **off in `standard`**.

**Why `standard` stays off** (measured 2026-07-19, `MODEL_EDGE_BASE_PROFILE=standard`,
Accurate pose). Against `standard`'s own corner baseline (`ContourRdp` + `Erf`, and —
critically — **no pose-consistency χ² gate and no outlier-drop**), edge refinement is
a large win on the *bulk* of the distribution but does **not** clean the tail:

| res  | rot mean (°) base→edge | rot p95 (°) | rot p99 (°) | t p99 (mm) | reproj mean (px) | reproj p99 (px) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 640  | 0.831→**0.261** | 2.556→**0.269** | 4.270→4.270 | 22.7→**20.7** | 0.966→**0.696** | 2.520→**2.044** |
| 720  | 0.934→**0.567** | 1.840→**0.228** | 14.11→**12.99** | 41.3→**39.5** | 0.888→**0.688** | 2.164→**1.791** |
| 1080 | 0.974→**0.594** | 1.508→**0.196** | 14.76→**13.77** | 50.4→50.0 | 0.939→**0.776** | 2.046→**3.030** |
| 2160 | 7.058→**6.740** | 40.16→40.16 | 108.4→108.3 | 120.8→123.8 | 1.096→**0.942** | 4.691→**5.280** |

Recall/precision stay 100 %/100 % and 2D corner RMSE is unchanged (as on
`high_accuracy` — the stage never writes `batch.corners`). But `standard`'s **rotation
p99 tail is essentially untouched** (4–108° gross IPPE-ambiguity outliers the edge fit
cannot recover from a flipped seed), and the **high-resolution reproj/translation p99
tail regresses** (reproj p99 1080 2.05→3.03, 2160 4.69→5.28). The mechanism is the
missing safety net: on `high_accuracy` the refined pose is re-validated against the
pose-consistency **χ²** gate (`pose_consistency_fpr = 1e-3`), so a refinement that
worsens reprojection is rejected — reproj RMSE improves at *every* percentile there. On
`standard` (`pose_consistency_fpr = 0`) there is no such re-validation, so the tail can
drift. Enabling the stage on `standard` would therefore **trade the render-tag tail**,
which the project rules forbid doing silently.

`standard`'s real gap is the absent pose gates, not corner refinement. A fuller
`standard` overhaul (edge refinement **+** χ² consistency gate **+** outlier-drop, the
levers `high_accuracy` already ships) is the candidate that would fix the tail too, but
it changes rejection behaviour and needs its own recall/precision benchmark (ICRA
included) — deferred to a follow-up, not v0.7.0. `standard` stays the byte-compatible
fast default this release.
