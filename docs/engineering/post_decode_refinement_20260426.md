# Phase C.5 — Post-decode corner re-refinement (2026-04-26)

## TL;DR

A new optional pipeline stage runs after decoding and before pose refinement.
For every `Valid` candidate, we re-fit each of the four **outer** tag edges
with the shared `ErfEdgeFitter` (PSF-blurred step model), intersect adjacent
edges to recover four sub-pixel corners, and re-solve the homography. Behind
`DetectorConfig.decoder.post_decode_refinement` (default `false`) — only the
shipped `max_recall_adaptive` profile opts in. The weighted pose solver's
covariance column is **preserved**, not overwritten.

| Dataset (`max_recall_adaptive`)         | Metric         | Before | After  | Δ       |
| --------------------------------------- | -------------- | ------ | ------ | ------- |
| ICRA forward 50 frames                  | mean RMSE (px) | 0.2871 | 0.2724 | **−5.1 %** |
| ICRA forward 50 frames                  | mean recall    | 0.7380 | 0.7380 | flat    |
| render-tag tag36h11 640×480             | mean RMSE      | 0.5819 | 0.5781 | −0.7 %  |
| render-tag tag36h11 1280×720            | mean RMSE      | 0.5642 | 0.5617 | −0.4 %  |
| render-tag tag36h11 1920×1080           | mean RMSE      | 0.5878 | 0.5862 | −0.3 %  |
| render-tag tag36h11 3840×2160           | mean RMSE      | 0.6063 | 0.6038 | −0.4 %  |
| render-tag tag36h11 1920×1080           | p99 rotation°  | 1.0087 | 1.0037 | −0.5 %  |
| render-tag tag36h11 3840×2160           | p99 rotation°  | 1.8130 | 1.8128 | flat    |

Means improve uniformly across the full render-tag pyramid; p99 rotation
percentiles stay within sub-1 % noise on every resolution. ICRA recall
is unchanged (no candidate is *added* or *dropped* — refinement only
adjusts already-Valid corners).

## Algorithm

### Why not a per-corner saddle-point refit?

The first prototype used a 7×7 Förstner saddle-point window centred on each
decoded corner. Mean RMSE moved by less than the noise floor, and percentiles
on dense small-tag scenes regressed. Root cause: a 2D saddle window inside a
tag straddles **internal bit boundaries** at the L-corner, biasing the solve
toward whichever neighbouring bit is brighter. That's a textbook saddle-point
ambiguity for a structured-light pattern.

### Edge-fit + intersection

Decoded corners give us a labelled CW-wound quad — we know which segment is
which edge. So instead of fitting four corners independently, we fit four
**outer edges** independently, then intersect them.

For each edge $k \in [0, 4)$:

1. Build an `ErfEdgeFitter` from corner $k$ to corner $(k+1) \bmod 4$
   with the **left-hand normal convention** (matches the GWLF pass and the
   existing decoder ERF refinement).
2. Collect samples via `for_decoder()` (perpendicular sampling band, bumpalo-
   allocated).
3. Refine for up to 15 Gauss–Newton iterations with `re_estimate_ab = true`,
   `step_clamp = 0.5 px`, and `min_contrast = 5.0`. We skip the `scan_initial`
   pre-step because decoded corners are already accurate to ≈ 0.5 px.
4. Bail the entire tag if the per-line $J^TJ$ falls below `MIN_LINE_JTJ = 1.0`
   — a degenerate fit would produce a meaningless intersection.

Then, with the four lines $[n_x, n_y, d]$ stored, compute corner $k$ as the
intersection of `lines[(k+3) % 4]` and `lines[k]`:

$$
\begin{bmatrix} n_{x,a} & n_{y,a} \\ n_{x,b} & n_{y,b} \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix} =
\begin{bmatrix} -d_a \\ -d_b \end{bmatrix}
$$

Reject the entire refit (preserve original corners) when:

* `|det M| < 0.05` — adjacent edges nearly parallel after refit.
* Any single corner moves more than `1.5 px` from its decoded position.
  Larger displacements almost always mean a line locked onto an interior bit
  boundary.
* `Homography::square_to_quad` fails the 1e-4 round-trip — the refit pushed
  the quad off the plane and the homography would be malformed.

### Why the covariance column is preserved

The line fit's Cramér-Rao bound, $\sigma_n^2 / (J^T J)$, is the marginal
variance of the perpendicular distance $d$. Propagating it through the line-
intersection Jacobian gives a 2×2 per-corner covariance — and on synthetic
data this is mathematically tight. But the production weighted pose solver
is calibrated against pose RMSE on render-tag's Blender PSF, which has an
intensity profile materially different from the Gaussian-erf model the
fitter assumes. Writing the optimistic CRB into `corner_covariances` makes
the solver over-trust the refined corners, **regressing render-tag p99
rotation by 3-5 %** on every resolution.

The fix: leave Phase A's GWLF / structure-tensor covariance untouched. The
edge-fit pass is purely a corner-position refit; `corner_covariances` is in
the SoA write-set per the contract but this pass intentionally writes
nothing into it. Documented at the head of the source file so future
contributors don't try to "fix" the apparent gap.

## Configuration

```jsonc
// crates/locus-core/profiles/max_recall_adaptive.json
"decoder": {
  // ...
  "post_decode_refinement": true
}
```

| Profile               | `post_decode_refinement` |
| --------------------- | ------------------------ |
| `standard`            | `false` (default)        |
| `high_accuracy`       | `false` (default)        |
| `grid`                | `false` (default)        |
| `general`             | `false` (default)        |
| `render_tag_hub`      | `false` (default)        |
| `max_recall_adaptive` | **`true`**               |

Python users edit the same field through the Pydantic model:

```python
cfg = locus.DetectorConfig.from_profile("max_recall_adaptive")
cfg.decoder.post_decode_refinement = False  # opt out, e.g. for benchmarking
```

## SoA write contract

The pass declares write privileges on `corners`, `homographies`, and
`corner_covariances` (the third is in the allowed set but not written;
this preserves Phase A's prior — see above). Enforced by two new tests in
`crates/locus-core/tests/contract_detection_batch.rs`:

* `contract_phase_c5_refit_valid_corners_disabled_is_noop` — sentinel
  invariance when the flag is off.
* `contract_phase_c5_refit_valid_corners_active_writes` — only the three
  allowed columns change; every other column keeps its sentinel.

## Failure modes & rationale

| Constant            | Value | Rationale                                                                                                  |
| ------------------- | ----- | ---------------------------------------------------------------------------------------------------------- |
| `SIGMA`             | 0.6   | Matches the GWLF + decoder-ERF PSF assumption.                                                              |
| `MAX_STEP_PX`       | 1.5   | Decoded corners are ≤ 0.5 px after Phase A refinement; > 1.5 px almost always means a line locked onto an interior bit boundary. |
| `MIN_EDGE_FOR_REFIT`| 16 px | The 2.5-px perpendicular sample window leaks samples from the adjacent edge for shorter sides; 16 px also gives ≥ 12 usable samples per edge for stable Gauss-Newton convergence. |
| `MIN_LINE_DET`      | 0.05  | $\lvert\sin\theta\rvert$ floor (~ 2.87°) between adjacent edges — catches degenerate post-refit geometry.   |
| `MIN_LINE_JTJ`      | 1.0   | Below this the line variance is meaningless and the intersection covariance would explode.                  |

## How to disable for an A/B comparison

```bash
# Profile JSON
sed -i 's/"post_decode_refinement": true/"post_decode_refinement": false/' \
    crates/locus-core/profiles/max_recall_adaptive.json

LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
    cargo insta test --release --all-features --features bench-internals \
    --test regression_render_tag --review
```

Both gates are byte-revertible — flip the JSON, re-bless the snapshots, and
the rest of the pipeline is untouched.

## Verification matrix

| Check                                | Status |
| ------------------------------------ | ------ |
| `cargo nextest run --release --all-features` | 226 / 226 pass |
| `cargo clippy --workspace --all-targets --features bench-internals -- -D warnings` | clean |
| `cargo check --target aarch64-unknown-linux-gnu --all-features` | clean (NEON) |
| `cargo fmt --all --check`            | clean |
| `uv run ruff check . && uv run ruff format --check .` | clean |
| `uv run mypy .`                      | clean |
| `uv run python tools/export_profile_schema.py --check` | parity |
| `uv run pytest`                      | 154 passed, 10 skipped, 1 xfailed |
| Snapshot review (5 snaps re-blessed) | render-tag means improve, percentiles flat, ICRA RMSE −5.1 % |

## Out of scope

* Image pyramid for far-field detection (Phase B).
* Per-bit gradient strength weighting (Phase C).
* Inverse-distortion of bit-sample grid (Phase C).
* Per-family `max_hamming_error = 3` for `standard` profile (needs ROC).

## Files touched

| Layer       | File                                                                         |
| ----------- | ---------------------------------------------------------------------------- |
| Rust core   | `crates/locus-core/src/post_decode_refinement.rs` (new)                       |
|             | `crates/locus-core/src/edge_refinement.rs` (added `line_jtj()`)               |
|             | `crates/locus-core/src/detector.rs` (Phase C.5 wire-in)                       |
|             | `crates/locus-core/src/config.rs` (`post_decode_refinement` field)            |
|             | `crates/locus-core/src/lib.rs` (module export)                                |
|             | `crates/locus-core/profiles/max_recall_adaptive.json` (opt-in)                |
|             | `crates/locus-core/tests/contract_detection_batch.rs` (Phase C.5 tests)       |
| Python      | `crates/locus-py/src/lib.rs`                                                  |
|             | `crates/locus-py/locus/_config.py`                                            |
|             | `crates/locus-py/locus/locus.pyi`                                             |
|             | `crates/locus-py/tests/test_json_roundtrip.py`                                |
| Schema      | `schemas/profile.schema.json`                                                 |
| Snapshots   | `regression_icra2020__icra_forward_pure_max_recall_adaptive.snap`             |
|             | `regression_render_tag__common__hub__hub_locus_v1_tag36h11_{640x480,1280x720,1920x1080,3840x2160}_max_recall_adaptive.snap` |
| Docs        | `docs/engineering/post_decode_refinement_20260426.md` (this file)             |
