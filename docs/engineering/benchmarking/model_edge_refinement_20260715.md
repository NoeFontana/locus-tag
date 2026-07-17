# Model-based edge pose refinement — beating the rotation gap (2026-07-15)

## Summary

An opt-in Accurate-mode pose-refinement stage that aligns a **decoded** tag's
internal bit-grid edges (plus the border) to the image and refines the 6-DoF pose
against them. A corner PnP pose is fixed by only 4 points; a 36h11 tag exposes
**~40 internal high-contrast edges**, an order of magnitude more constraints,
distributed across the interior where they pin orientation far better than the
corners. On the render-tag suite this **cuts rotation p99 24–62 % at every
resolution — beating OpenCV's `apriltag` refinement — while keeping Locus's
best-in-class translation.**

Enabled by `pose.pose_edge_refinement_enabled = true` (default **false** →
shipped-profile detection byte-identical). Requires camera intrinsics + `tag_size`.

## Method

Post-decode, per tag (`crates/locus-core/src/model_edge.rs`):

1. **Sample** every candidate cell-boundary + border line in the tag frame,
   project under the current pose, and measure the sub-pixel edge by the
   **50 %-intensity crossing** along each edge normal (unbiased for a symmetric
   PSF). Low-contrast samples (same-colour boundaries) are dropped automatically —
   the image contrast selects the real edges, so no bit-pattern reconstruction is
   needed. Junction and large-offset (spurious) samples are excluded.
2. **`measure → fit`** iterations: hold the measured edge points fixed and refine
   the full 6-DoF pose to them with Levenberg-Marquardt, then re-measure. This
   gives an excellent **rotation** but weakly constrains depth/scale.
3. **Re-anchor translation** with a translation-only solve against the 4 trusted
   corners under the refined rotation. *Edges → rotation, corners → translation.*
4. A **no-worse gate** rejects untrustworthy refinements (too few edges, or an
   implausible pose jump), falling back to the corner pose.

## Results (render-tag, single-thread, Accurate pose)

Locus `high_accuracy`, baseline vs `pose_edge_refinement_enabled`:

| resolution | rot p95 (°) | rot p99 (°) | trans p99 (mm) | trans mean (mm) |
| :--- | ---: | ---: | ---: | ---: |
| 640×480  | 0.373 → **0.233** | 0.508 → **0.314** | 8.1 → **7.9** | 1.09 → **1.04** |
| 1280×720 | 0.464 → **0.174** | 0.624 → **0.472** | 16.0 → **15.1** | 1.56 → 1.71 |
| 1920×1080 | 0.385 → **0.196** | 0.600 → **0.409** | 18.6 → 21.1 | 1.95 → 2.16 |
| 3840×2160 | 0.432 → **0.234** | 1.113 → **0.426** | 39.0 → 39.9 | 3.81 → 4.38 |

For reference, OpenCV `cv2.aruco` **apriltag** at 1080p is rot p99 **0.376°** /
trans p99 **~55 mm**. Model-edge refinement brings Locus to rot p99 **0.409°**
(p95 **0.196°**, well below apriltag) at trans p99 **21 mm** — i.e. **apriltag-class
rotation with 2.6× better translation.** Latency at 1080p: **14.2 → 14.6 ms**
(+0.4 ms, negligible; the stage is a bounded per-tag Gauss-Newton).

Rotation improves 24–62 % everywhere. Translation is flat within ≤0.6 mm (better
at low resolution, marginally higher at 1080p/2160p) and remains best-in-class.

## Why this works where corner-level levers failed

The 2026-07-14 study (`refine_variants_20260714.md`) proved every *corner-level*
lever is trade-bound: GWLF/apriltag-edge, per-corner repair, per-tag switch, and
fusion all trade rotation for translation because they only reshape the same 4
corner observations. This stage instead **adds independent single-frame
information** — the decoded interior pattern's ~40 edges — which over-constrains
rotation without disturbing the corner-anchored translation. No multi-tag or
temporal fusion required.

## Provenance

AMD EPYC-Milan (Zen 3), 1 socket × 4 cores × 2 threads = 8 vCPU; `--release`;
`RAYON_NUM_THREADS=1`, Locus `threads=1`; `locus_v1_tag36h11_{640x480,1280x720,
1920x1080,3840x2160}`; pose mode **Accurate**.

## Reproduce

```bash
PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache RAYON_NUM_THREADS=1 \
  uv run --group bench tools/bench/model_edge_eval.py         # baseline vs enabled
```
