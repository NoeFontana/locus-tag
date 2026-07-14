# Corner-refinement variants vs the rotation-p99 tail (2026-07-14)

## Question

Locus wins translation accuracy against every library but trails OpenCV's
`cv2.aruco` **apriltag** corner refinement on the **rotation p99** of the 1080p
render-tag set. This note records a controlled study of whether an apriltag-style
**edge-line** corner refinement can close that gap without giving up Locus's
best-in-class translation.

| Detector (1080p, single-thread, Accurate pose) | rot p99 (°) | trans p99 (mm) |
| :--- | ---: | ---: |
| Locus `high_accuracy` (shipped, EdLines + None) | **0.600** | **18.6** |
| OpenCV `cv2.aruco` apriltag | 0.376 | ~55 |

The gap to beat is **0.600° → 0.376°** with **no** translation regression.

## Method

`tools/bench/refine_variants_eval.py` runs the shipped `high_accuracy` detector
with only its large-marker (`AdaptivePpb` high-route) refinement swapped, so every
variant shares extraction, decode, and pose. Config-only — the three shipped
profiles are untouched and detection snapshots stay byte-identical. Corner-error
decomposition and oracle probes are in the 2026-07-14 job scratch scripts
(`corner_decomp.py`, `weight_oracle.py`, `gated_replace.py`, `masking_test.py`).

**Provenance.** AMD EPYC-Milan (Zen 3), 1 socket × 4 cores × 2 threads = 8 vCPU;
`--release`; `RAYON_NUM_THREADS=1`, Locus `threads=1` (single-thread, apples-to-apples);
`locus_v1_tag36h11_1920x1080` (50 tags); pose mode **Accurate**.

## Result 1 — GWLF (apriltag-style edge-line refit) is a rotation↔translation trade

Locus already implements apriltag's `refine_edges` algorithm as **GWLF**
(`gwlf.rs`: gradient-weighted per-edge projective line fit → intersect → calibrated
2×2 covariance → weighted PnP). It is not enabled in any shipped profile. Routing
`high_accuracy` large markers through it:

| Variant | recall | rot p99 (°) | rot p95 (°) | trans p99 (mm) | trans mean (mm) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| V0 EdLines + None (shipped) | 100% | 0.600 | 0.385 | **18.6** | **1.95** |
| V-GWLF EdLines + Gwlf | 100% | **0.398** | 0.297 | 63.0 | 8.89 |

GWLF nearly reaches the rotation target but **regresses translation 3.4×**
(49/50 tags worse). Corner-error decomposition against GT explains why: GWLF corners
are uniformly ~3× worse in *absolute* position (mean RMSE 0.63 px vs EdLines 0.21 px —
the ~0.6 px edge-line floor on Blender PSF) but far more *consistent* (p99 0.82 vs
1.30 px). Consistency pulls in the rotation tail; the raised absolute error inflates
translation broadly. **Edge-line intersection swaps Locus's error profile for
apriltag's own** — which is exactly why OpenCV-apriltag also has ~55 mm translation.
Not a Pareto win; **not shipped**.

## Result 2 — the tail is 2–3 tags, not distributed noise

Sorting the shipped rotation errors and dropping the worst-k tags:

| remove worst-k | rot p99 (°) | rot p95 (°) |
| ---: | ---: | ---: |
| 0 | 0.600 | 0.385 |
| **2** | **0.384** | 0.339 |
| 3 | 0.359 | 0.310 |

Removing the worst **two** tags already lands at the apriltag target. The p95 (0.385°)
is *already* competitive. The entire gap is a handful of EdLines **Phase-1
arc-partition gross failures** (e.g. tag 563 / scene_0008: one corner 3.83 px off),
not the refinement quality of the other 47 tags.

## Result 3 — reweighting can't fix it (EdgeLineGated, built and reverted)

An opt-in `CornerRefinementMode::EdgeLineGated` was prototyped: keep EdLines corner
positions, and snap a corner to the GWLF edge-line intersection **only** when they
disagree by more than a gate τ (isotropic pose, no covariance). The idea was to
correct EdLines' outlier corners while preserving its accurate majority.

| Variant | outlier_drop | rot p99 (°) | trans p99 (mm) |
| :--- | ---: | ---: | ---: |
| baseline (shipped) | on (25) | 0.600 | 18.6 |
| baseline | off | 0.771 | 18.9 |
| gated τ=1.5 | on/off | 0.700 | 17.8 |

Gating is a *real* outlier handler (0.70 beats no-handling 0.77) but **loses to the
shipped `outlier_drop`** (0.60): *dropping* a catastrophic corner beats *correcting*
it to the 0.6 px edge-line floor. Worse, a corrected corner no longer trips the
post-pose d² gate, so gating **neutralises** `outlier_drop` (on = off) — it fights the
existing, better handler. Reverted; no dead config knob shipped.

## Conclusion & the open lever

Shipped `high_accuracy` (0.600° / 18.6 mm) sits on the **single-frame single-detector
Pareto frontier**: reaching apriltag's rotation requires the common-mode corner
profile that structurally costs translation, and every reweighting lever loses to the
existing Huber + `outlier_drop`. The residual gap is 2–3 EdLines arc-partition
failures, and *dropping* their corners (what shipped does) caps at ~0.66° on those
tags — you must **repair** the corner to near-truth to reach the 0.38° oracle ceiling.

**Untried, Pareto-plausible next lever — surgical edge repair:** detect the gross
EdLines failure with the pose-independent GWLF-disagreement signal (it flags tag 563
at 2.75 px), high τ so only the 2–3 real failures fire; repair by substituting the
GWLF edge **line** for the mis-partitioned EdLines edge and re-running the
chord-coupled Phase-5 GN (keeping the shared-corner regulariser the Phase-5 negative
proved essential); do it *instead of* `outlier_drop` for those tags. Touches only the
failing tags, so translation is preserved by construction. Oracle ceiling rot
p99 → ~0.36–0.38°. Not yet built.

## Reproduce

```bash
PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache RAYON_NUM_THREADS=1 \
  uv run --group bench tools/bench/refine_variants_eval.py
# other resolutions: RENDER_TAG_SOTA_CONFIG=locus_v1_tag36h11_3840x2160 ...
```
