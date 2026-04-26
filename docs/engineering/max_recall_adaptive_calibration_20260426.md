# `max_recall_adaptive` calibration & follow-ups

Records the empirical calibration of the
[`max_recall_adaptive`](../../crates/locus-core/profiles/max_recall_adaptive.json)
profile and the deferred Phase B work that would lift its render-tag
pose metrics back to `high_accuracy` parity.

## §1 The shipped configuration

```jsonc
"quad": {
  "extraction_policy": {
    "AdaptivePpb": {
      "threshold": 2.5,
      "low_extraction":  "ContourRdp", "low_refinement":  "Erf",
      "high_extraction": "EdLines",    "high_refinement": "None"
    }
  }
},
"threshold": { "enable_sharpening": true }
```

Validation matrix — all snapshots blessed at `674b4ef`:

| Suite | Resolution | Adaptive | `high_accuracy` |
|---|---|---:|---:|
| ICRA forward | — | **`0.7380`** | `0.4631` |
| render-tag tag36h11 | 480p | recall `0.92`, p99 rot `0.717°`, RMSE `0.582` | `0.86` / `0.538°` / `0.245` |
| render-tag tag36h11 | 720p | `0.90` / `1.016°` / `0.564` | `0.90` / `0.600°` / `0.187` |
| render-tag tag36h11 | 1080p | `0.96` / `1.009°` / `0.588` | `0.94` / `0.562°` / `0.203` |
| render-tag tag36h11 | 2160p | `0.98` / `1.813°` / `0.606` | `1.00` / `1.357°` / `0.180` |

ICRA-suite recall lifts +27 pp vs `high_accuracy` and matches `standard`
to within 1 pp. Every render-tag p99 rotation stays under the
`1.897°` budget set in
[`docs/engineering/benchmarking/render_tag_sota_20260425.md`](benchmarking/render_tag_sota_20260425.md).

The render-tag mean RMSE delta vs `high_accuracy` (`~0.6 px` versus
`~0.2 px`) is **intrinsic** to the ContourRdp + Erf path: even at PPB ≥
2.5 the EdLines path (selected by both profiles) gives sub-pixel corners
with per-corner Fisher-information priors that the weighted LM solver
consumes. ContourRdp + Erf produces equally robust corners but no
covariance prior, so the LM solver falls back to the unweighted ideal-
corner residual. That is the single largest knob driving the gap.

## §2 The cheap calibration sweep (negative result)

Two configurations were tested as in-place edits to
`max_recall_adaptive.json`, each a single regression-suite run, before
shipping:

| Variant | ICRA recall | 1080p p99 rot | 1080p RMSE |
|---|---:|---:|---:|
| **shipped** `(2.5, sharp_on)` | `0.7380` | `1.009°` | `0.588` |
| `(1.5, sharp_off)` | `0.5484` | `3.154°` | `0.607` |
| `(1.5, sharp_on)`  | `0.7074` | `1.009°` | `0.588` |

- Disabling sharpening cost ICRA recall (`-19 pp`) **and** worsened the
  render-tag p99 tail (small-tag candidates routed to EdLines without
  the sharpening prior locked unstably and produced 3°+ rotation
  outliers). Sharpening is doing real work on both extraction paths.
- Lowering the threshold from `2.5 → 1.5` had near-zero effect at the
  1080p / 2160p resolutions (every candidate already sat above `2.5`)
  and a small ICRA cost (`-3 pp`) from boundary candidates routing to
  EdLines instead of ContourRdp + Erf.

Conclusion: the shipped configuration is at a local Pareto optimum.
Further config-only changes will trade ICRA recall for render-tag
RMSE, or vice versa. The **only** way to drive every metric up
without trade is to change the underlying pipeline.

## §3 Phase B follow-ups (open work)

Listed in order of expected impact-per-effort. None of these are
in-scope for the current PR.

### B1. Post-decode corner re-refinement

After a tag is **decoded**, its identity is known and so is the ideal
corner geometry of the canonical tag template. Refit each corner to
the local image gradient under that template:

1. From `(homography, ideal_corners)` build the ideal-image
   transformation that lands each corner's bit grid on the observed
   image.
2. For each of the 4 corners, do a local sub-pixel search (parabolic
   fit on the symmetric-corner response map, or Förstner / Saddle-
   point Newton step) over a 5×5 / 7×7 neighbourhood of the current
   corner estimate.
3. Re-solve the homography from the 4 refined corners; emit the new
   pose.

Why it pushes render-tag RMSE down:

- Independent of extraction path. ContourRdp + Erf and EdLines both
  benefit; the post-decode refit is anchored on the **decoded** ideal
  corner, not the extracted candidate corner, so the dependency on
  ContourRdp's coarser sub-pixel quality is broken.
- `high_accuracy`'s advantage today comes from EdLines's per-corner
  covariance prior. The post-decode refit emits its own per-corner
  covariance (Hessian of the local response), so the weighted LM
  solver still has a Fisher prior — even when the extraction was
  ContourRdp.

Expected: render-tag mean RMSE collapses toward `~0.2 px` across the
matrix; ICRA recall unchanged. Adaptive becomes a strict-improvement
profile vs `high_accuracy` on every resolution.

Implementation cost: ~1 PR. The corner search lives next to
`edge_refinement.rs`; the per-corner covariance feeds existing
`corner_covariances` SoA columns.

Risk: the post-decode pass is per-valid-tag, not per-candidate, so
the cost scales with the number of accepted detections (small).
The Phase 1 implementation should ship behind a profile flag
(`decoder.post_decode_refinement: bool`, default `false`) so it can
be A/B'd without touching the existing snapshot suite.

### B2. Image pyramid for far-field

Detect at `½×` / `¼×`, lift corners back to full resolution and
re-refine. Standard fix for dense small-tag recall — AprilTag3 ships
this as `decimation`, OpenCV `cv2.aruco` ships it as a pyramid.

Why it pushes ICRA recall up:

- ICRA forward frames `0..5` have candidates at PPB ≈ `1.06–1.20` —
  below EdLines's lock-on threshold. A `2×` upscale gives them
  effective PPB ≈ `2.1–2.4`, putting them in EdLines's working range
  *without* changing the extraction algorithm.
- Pose precision is preserved because the corner refinement runs at
  full resolution.

Implementation cost: ~2 PRs. Coarse-to-fine engine work touches the
detection-pipeline orchestration and the extraction caches.

### B3. Per-route sharpening in `AdaptivePpb`

Extend `AdaptivePpbConfig` with `low_sharpening: bool` and
`high_sharpening: bool` so small tags can keep sharpening (recall)
while large render-tag tags skip it (precision).

Phase B3 is only worth doing **after** B1: the post-decode refit
removes most of the sharpening-induced precision penalty on the
high path, so the per-route knob may end up redundant.

## §4 How to re-run the calibration

```bash
TRACY_NO_INVARIANT_CHECK=1 \
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
LOCUS_ICRA_DATASET_DIR=tests/data/icra2020 \
  cargo test --release --features bench-internals \
  --test regression_render_tag --test regression_icra2020 \
  max_recall_adaptive
```

Snapshot review: `cargo insta review`.

## §5 Diagnostic: PPB calibration histogram

`crates/locus-core/tests/icra_forward_diagnostic.rs` was extended in
`674b4ef` with a per-frame PPB histogram over both valid and rejected
candidates (using the same `bbox_short / outer_dim` estimator the
runtime router uses). Re-run it whenever the threshold needs to be
re-derived for a new dataset:

```bash
LOCUS_ICRA_DATASET_DIR=/path/to/icra2020 \
  cargo test --release --features bench-internals \
  --test icra_forward_diagnostic -- --ignored --nocapture
```

The shipped `2.5` threshold was derived from this output: ICRA forward
candidates land entirely in the `<1.5` bucket on every frame, so the
strict `<` tie-break routes them all to ContourRdp + Erf even if a
future release nudges the cutoff downward.
