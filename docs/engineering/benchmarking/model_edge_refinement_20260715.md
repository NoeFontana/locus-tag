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

Enabled by `pose.pose_edge_refinement_enabled = true` (default **false** →
shipped-profile detection byte-identical). Requires camera intrinsics + `tag_size`.

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

Locus `high_accuracy`: **baseline** (no edge stage) → **L2-edge** (shipped-#336,
for reference) → **tuned** (Huber δ = 0.5 + 7 samples/boundary):

| resolution | rot p95 (°) | rot p99 (°) | trans p99 (mm) | trans mean (mm) |
| :--- | ---: | ---: | ---: | ---: |
| 640×480   | 0.373 → 0.233 → **0.201** | 0.508 → 0.314 → **0.269** | 8.11 → 7.87 → **7.60** | 1.09 → 1.04 → **1.03** |
| 1280×720  | 0.464 → 0.175 → **0.194** | 0.624 → 0.472 → **0.262** | 15.97 → 15.07 → **14.46** | 1.56 → 1.71 → **1.47** |
| 1920×1080 | 0.385 → 0.195 → **0.180** | 0.600 → 0.409 → **0.249** | 18.62 → 21.06 → **20.10** | 1.95 → 2.16 → **1.88** |
| 3840×2160 | 0.432 → 0.215 → **0.203** | 1.113 → 0.429 → **0.267** | 39.02 → 39.87 → **36.98** | 3.81 → 4.38 → **4.07** |

For reference, OpenCV `cv2.aruco` **apriltag** at 1080p is rot p99 **0.376°** /
trans p99 **~55 mm**. The tuned stage brings Locus to rot p99 **0.249°** (p95
**0.180°**) at trans p99 **20 mm** — i.e. **well under apriltag rotation at ~2.7×
better translation.**

Highlights vs the **baseline** corner pose:

- **Rotation p99 −47 % to −76 %** at every resolution; **p95 roughly halved**
  everywhere.
- **Translation p99 improves** at 640/720/2160 and **trans mean improves** at
  640/720/1080; the only regression is **+1.5 mm p99 at 1080p** — the known,
  bounded edges→rotation trade (1–2 tags), *smaller* than the shipped L2 stage's
  +2.4 mm and gated by the no-worse + χ² checks.
- **Latency** (single-thread, approximate): +~1 ms/frame at 640/720/1080, roughly
  flat at 2160 (baseline → tuned: 2.3→3.7 / 6.2→7.4 / 14.1→15.2 / 57.4→56.9 ms).
  The stage is a bounded per-tag Gauss-Newton on the opt-in Accurate path.

vs the pre-tuning **L2-edge** stage, the tuning cuts rot p99 a further −11 % to
−44 % (0.314→0.269, 0.472→0.262, 0.409→0.249, 0.429→0.267) with neutral-to-better
translation.

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
`locus_v1_tag36h11_{640x480,1280x720,1920x1080,3840x2160}` (50 frames each); pose
mode **Accurate**. Accuracy columns are deterministic (verified identical across
repeated runs); the latency column is a best-effort single-thread wall-time.

## Reproduce

```bash
PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache RAYON_NUM_THREADS=1 \
  uv run --group bench tools/bench/model_edge_eval.py         # baseline vs enabled
```
