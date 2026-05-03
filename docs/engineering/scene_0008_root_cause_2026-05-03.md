# scene_0008 root-cause deep-dive (2026-05-03)

`scene_0008_cam_0000` is the single outlier in V1's pose-covariance audit
on `locus_v1_tag36h11_1920x1080`: detected `d²_pose = 12 405` (the next-worst
scene is 6 344) with `min_irls = 0.219` (the only scene where Huber fired).
Track A flagged it as `corner_geometry_outlier`. This memo captures the
first-principles investigation of *why* it fails and what the practical fix is.

## §1 What the data says

The defect is a **single-corner localisation failure on canonical corner 1**
(image top-left of the tag, since the tag is rotated ≈ 90° in image space):

| Corner | GT (px) | Detected (px) | Δ (px) | ‖r‖ | IRLS |
|---|---|---|---|---:|---:|
| 0 (BL) | (880.73, 602.61) | (879.40, 602.87) | (−1.33, +0.26) | 1.35 | 1.000 |
| **1 (BR, image TL)** | **(903.44, 456.41)** | **(907.25, 456.84)** | **(+3.81, +0.42)** | **3.83** | **0.219** |
| 2 (TR) | (1049.71, 455.25) | (1049.89, 454.58) | (+0.18, −0.67) | 0.69 | 1.000 |
| 3 (TL) | (1031.40, 607.67) | (1031.52, 607.92) | (+0.12, +0.25) | 0.28 | 1.000 |

Corner 1's residual is **almost entirely along the top edge of the tag**
(angle from +x of the residual: 6° vs angle of top edge: −0.5°). It is
*not* perpendicular slip — it is corner-along-edge slip by 3.8 px.

## §2 The image edges are clean

Linear regression of the **actual image edge points** (sub-pixel 50% transition,
143 rows from y=458 to y=600) against the GT line gives:

| | Image fit | GT line | Δ |
|---|---|---|---|
| slope | −0.15566 | −0.15527 | +0.00039 |
| intercept | 974.21 | 974.26 | −0.05 |
| Per-row deviation RMS | 0.057 px | — | — |

**The actual image LEFT edge is at the GT line within 0.06 px RMS over
143 measured rows.** Same for the TOP edge. Their geometric intersection
is at (903.16, 456.27) — 0.24 px from GT.

The detector reports (907.25, 456.84), **4 px from the actual image-edge
intersection in a direction that has no support in the pixel data.** This
is unambiguously a detector-internal failure, not an image artefact.

## §3 Why it's not a refinement problem

`high_accuracy` runs `EdLines + None` for high-PPB candidates (this scene's
tag is ~150 px → PPB ≈ 25). EdLines internally does Phases 1–5 (boundary
segmentation → IRLS line fit → sub-pixel re-fit → IRLS re-fit → joint
Gauss-Newton), then emits sub-pixel corners. Post-EdLines `refine_corner`
is skipped. So the wrong corner originates **inside** EdLines.

We swept the available compatible refinement modes to verify it's not
"polishing" that's missing:

| `high_refinement` | scene_0008 corner 1 ‖r‖ | corpus mean ‖r‖ | corpus KL |
|---|---:|---:|---:|
| `None` (baseline) | **3.83 px** | 0.19 px | 13.9 |
| `Gwlf` | 2.87 px | 0.51 px (**+170 %**) | 20.7 |
| `Edge` | 2.87 px (identical to Gwlf) | 0.51 px | 20.7 |
| `Erf` | n/a — blocked by validator | — | — |

`Edge` and `Gwlf` produce **byte-identical** outputs because the dispatch
in `crates/locus-core/src/quad.rs:412` flags `use_erf = (mode == Erf)`,
so anything-not-Erf goes through the same `fit_edge_line` path.

`Erf` is blocked at both the Pydantic validator (`_config.py:382`) and
the Rust config layer — `EdLines + Erf` is explicitly rejected.

**No compatible refinement mode reduces scene_0008's residual below ~3 px,
and the only one that helps it at all (1 px) damages the corpus by 2.7×.**
The post-EdLines refinement architecture cannot fix this scene because:

- `fit_edge_line` only adjusts the line's perpendicular *offset*; it locks
  the line's *direction* to the vector between input corners.
- If the input corner is wrong by 4 px (as it is here), the line direction
  inherits the slope error.
- Refinement falls back to the raw input via a 2 px sanity check
  (`quad.rs:1320`) — large legitimate corrections are filtered out as
  noise.

## §4 So what's wrong inside EdLines?

Without instrumenting Phase 1–5 directly we can only narrow the suspects.
The geometry of scene_0008 makes three failure modes plausible:

1. **Phase 1 boundary segmentation.** The tag is ~90° rotated in image,
   so its corners are very near the T/R/B/L extrema — exactly the
   degenerate configuration the imbalance gate at `edlines.rs:1088`
   explicitly warns about. The gate fires on `min-arc < 16 % AND
   max-arc > 40 %`; on a balanced tag like this it would not fire, but
   the partition can still be subtly off, placing the L↔T arc boundary
   1–2 contour pixels away from the true corner.
2. **Phase 4 sub-pixel re-fit on the near-horizontal top edge.** The top
   edge has angle −0.5° from horizontal, so the 50% transition row spans
   over 30 px laterally before the perpendicular gradient peaks deviate
   meaningfully. Sub-pixel localisation in this regime depends sensitively
   on the gradient sample step.
3. **Phase 5 joint Gauss-Newton.** Same slope-locking dynamic as
   `refine_corner` — if Phase 3/4 fed it biased points along the L→T arc,
   the joint GN cannot recover slope information from the points alone.

A definitive answer needs Phase-by-phase telemetry exposure (the PyO3
hook `_bench_refine_pose_lm_weighted_with_telemetry` for the LM stage
has a clear precedent). That is multi-day work and should only be
prioritised if the runtime gate (§5) proves insufficient.

## §5 The practical fix is at runtime

Track A already detects scene_0008 at `min_irls = 0.219` and
`max_corner_d² = 37.7` (vs χ²(1) at α=10⁻⁴ = 15.137). A small follow-up
track can:

1. **Drop the failing corner from the LM weighted pose solve** (remove
   one row pair from `JᵀWJ`; the remaining 3 corners are still over-determined
   for a 6-DoF planar PnP).
2. **Inflate `Σ_pose` only when the gate fires**, so downstream Kalman /
   factor-graph consumers see honest covariance for this *one* scene.
3. **Reject the detection** entirely if the application can tolerate a
   missed tag.

Estimated effort ~4 h. Doesn't fix EdLines, but resolves scene_0008's
contribution to the residual tail (counterfactual p99 rotation drop:
0.171° per Track A).

## §6 What's NOT worth pursuing

- Switching to `Edge` or `Gwlf` refinement: net regresses the corpus
  by 2.7× while only partially fixing scene_0008 (§3 sweep).
- Bypassing the `Erf + EdLines` validator block: the 2-px `refine_corner`
  sanity check at `quad.rs:1320` would still gate ERF's corrections, and
  ERF has its own ~0.6 px corner-RMSE floor on Blender PSF
  (`project_phase_c5_render_tag_hub_negative.md`).
- Increasing the 2-px sanity check threshold: would let *bad* corrections
  through on the 49 typical scenes (currently at 0.14 px noise floor),
  worsening the corpus.
- Deep EdLines Phase 1–5 instrumentation **on synthetic data alone**:
  the failure mode is real but its prevalence on real cameras is
  unknown. Investing multi-day work to fix one scene out of fifty
  on Blender renders is poor ROI vs. shipping the runtime gate first.

## §7 Reproducing

```bash
uv run maturin develop --release \
    --manifest-path crates/locus-py/Cargo.toml --features bench-internals

# Audit + per-corner residual capture (extends V1's harness).
RAYON_NUM_THREADS=8 PYTHONPATH=. \
uv run --group bench tools/bench/pose_cov_audit.py \
    --hub-config locus_v1_tag36h11_1920x1080 \
    --output-dir diagnostics/pathb_gt_corners

# Image-side analysis (sub-pixel edge fit vs GT pose projection):
PYTHONPATH=. uv run --group bench python <<'PY'
import cv2, numpy as np, json
img = cv2.imread('tests/data/hub_cache/locus_v1_tag36h11_1920x1080/images/scene_0008_cam_0000.png',
                 cv2.IMREAD_GRAYSCALE)
samples = json.loads(open('diagnostics/pathb_gt_corners/samples.json').read())
s = next(x for x in samples if x['scene_id'] == 'scene_0008_cam_0000')
gt = np.array(s['gt_corners_px']); det = gt + np.array(s['corner_residuals_px'])
print('GT  corners:', gt.tolist())
print('Det corners:', det.tolist())
PY

# Refinement-mode sweep to confirm post-EdLines refinements can't help.
# Edit crates/locus-core/profiles/high_accuracy.json: high_refinement to
# {"None", "Gwlf", "Edge"}; "Erf" is rejected by the validator.
```
