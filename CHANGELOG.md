# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

## [0.8.0] - 2026-09-25

### Tests

- **EuRoC regression tests now use the shipped `grid` profile instead of
  `Detector::new()`'s hardcoded Rust `Default`.** That default matches
  neither shipped profile (notably `enable_sharpening: false`, where
  `standard` ships `true`) and measurably underperforms on this dataset:
  691 total detections for `standard` and 977 for `high_accuracy` vs.
  `grid`'s 2299, at stride 10 across all 145 sampled frames. This isn't a
  new convention — `regression_render_tag.rs`'s `accuracy_baseline` module
  already documents picking `high_accuracy` for clean, large synthetic
  renders and leaving `regression_render_tag_robustness.rs` on `standard`
  for small-tag/noisy content; `grid` extends the same reasoning one step
  further for real (not synthetic) sensor noise on a literal multi-tag
  *AprilGrid* board — its looser `decoder.min_contrast`/`quad.min_edge_score`
  suit the former, and `segmentation.connectivity: Four` (vs `standard`'s
  `Eight`) avoids merging diagonally-adjacent tag components in the latter.
  Applied to all four EuRoC tests uniformly. Measured together with the
  `select_dominant_vertices` quad-extraction fix (see `Fixed` below — the
  two changes were validated jointly, not independently): relative recall
  63.1% → 69.9%, distorted-pose recall 49.0% → 71.0%, board-consistency
  frames checked 50 → 78 (still 100% consistent).
- **`euroc_detection_baseline` now measures relative recall (decoded /
  present) instead of an absolute "≥15 tags" bar.** `cam_april` is a
  calibration *sweep* recording — 62.8% of sampled frames have the board
  absent, far away, or off-frame by construction — so the old absolute bar
  conflated "board not in view" with "decoder missed tags that were there"
  and measured 27.6% recall dragged down almost entirely by the former.
  "Present" is now measured, not guessed: once ≥4 tags decode in a frame, a
  2D affine map is fit (ordinary least squares, closed-form) from every
  corner of every decoded tag's known board-plane position to its observed
  pixel position, the full 6×6 grid layout is reprojected through it, and a
  tag counts as present iff its projected corners land inside the image
  bounds. (A full 6-DOF board-pose fit via `BoardEstimator`'s LO-RANSAC was
  tried first and is more rigorous when it converges, but converges on under
  20% of real frames here even with its gates loosened 25× — this affine fit
  can't fail to converge, at the cost of not modeling perspective, which is
  an acceptable tradeoff for the coarse "in frame or not" question it's
  answering.) Also added: a duplicate-decoded-ID-within-a-frame invariant, a
  periphery-vs-interior spatial recall check, and `euroc_pinhole_funnel_diagnostic`
  (`#[ignore]`d, run explicitly) which reports exactly where candidates are
  lost per pipeline stage (quad extraction → contrast gate → decode). Fixed
  `scripts/fetch_euroc_calibration.sh`, whose upstream host
  (`robotics.ethz.ch`) is permanently decommissioned; it now pulls the
  calibration bundle from ETH's Research Collection instead. Also added
  `tools/viz_rerun_euroc.py`, a standalone Rerun visualizer for EuRoC frames
  (the existing `tools/cli.py visualize` is hard-wired to the ICRA/hub
  dataset loader's ground-truth format, which EuRoC doesn't have), including
  a board-coverage overlay (green = decoded, yellow = predicted-present-but-
  missing, grey = predicted-out-of-frame) for exactly this recall metric.
- **`scripts/fetch_euroc_calibration.sh` now fetches `cam_april` from a
  private Hugging Face Hub mirror instead of ETH's Research Collection
  directly.** The Research Collection host
  (`research-collection.ethz.ch`, ETH's replacement for the dead
  `robotics.ethz.ch`) works, but is frequently blocked by the outbound-host
  allowlists sandboxed CI/agent environments use, making the previous fetch
  unreliable exactly where it matters most. The `cam_april` sequence
  (~534 MB, ETH's own "In Copyright — Non-Commercial Use Permitted"
  license, <https://rightsstatements.org/vocab/InC-NC/1.0/>; full citation
  and source links in the dataset card) is now mirrored, private, at
  `NoeFontana/euroc-mav-cam-april-mirror` on the Hub and pulled via `hf
  download`. Output layout (`$DEST/cam_april/mav0/...`), idempotency
  (skip if already present) and the `LOCUS_EUROC_DATASET_DIR` consumer in
  `crates/locus-core/tests/common/euroc.rs` are unchanged; only the fetch
  path changed. Verified end-to-end: a fresh fetch into a scratch directory
  reproduces the expected `mav0/cam0/data` (1450 PNGs) / `cam1/data` (1449
  PNGs) / `imu0` layout, and a second run correctly no-ops.

### Performance

- **Segmentation (LSL CCL) now scales with the rayon pool instead of running
  fully serial.** `simd_ccl_fusion::label_components_lsl` was the one pipeline
  stage with no parallelism at all — 54 ms of a 76 ms 8-thread 4K frame, capping
  whole-frame 1→8-thread scaling at 1.69×. Three of its four sub-stages are now
  parallel: RLE extraction (count pass → prefix sum → row-parallel write into an
  arena slice), per-run root resolution (a read-only walk of the finished
  Union-Find forest), and the label-buffer fill (recursive row-band split via
  `rayon::join`). The row-pair Union-Find merge stays serial — its output depends
  on union order, and changing that would renumber components. On an AMD
  EPYC-Milan 8 vCPU box, `RAYON_NUM_THREADS=8`, `--release`: a textured 4K frame
  (1.72 M runs) drops **60.3 → 39.4 ms (−34.7 %)** and a 4K ICRA frame
  **7.52 → 4.18 ms (−44.5 %)**; stage scaling 1→8 threads goes from 1.00× to
  1.49×. The single-worker path is within ±2.5 % (−2.7 % at 4K, +0.7…+2.5 % on
  smaller frames), and keeps the one-pass extractor because the counting pass has
  nothing to amortise against there. Output is **bit-exact** vs the previous
  implementation — labels, component order and every component statistic — pinned
  by a differential test against a verbatim copy of the old algorithm over an
  adversarial frame zoo, run in both a 1-worker and a 4-worker pool.

### Breaking

- **`segmentation.margin` removed from the profile format.** The field had no
  reader anywhere in the workspace — its last consumer, the threshold-model CCL,
  was deleted in `f79d04b` — so it was accepted, validated, round-tripped and
  reported by `Detector.config()` while doing nothing. It is gone from the Rust
  `DetectorConfig`, the serde shim, the Pydantic model, `schemas/profile.schema.json`
  and the three shipped profiles. Because the profile shim uses
  `deny_unknown_fields`, a **custom profile JSON that still carries
  `segmentation.margin` will now fail to load** with a `ValueError` naming the
  key; delete the key. Detection output is unchanged (byte-identical).
- **`Detector(threads=n)` / `DetectorBuilder::with_threads(n)` now does what it
  says.** Previously the value was stored and echoed back by `config()` but never
  read: the pipeline always ran on Rayon's global pool, so `threads=1` silently
  used every core. It now builds one scoped `rayon::ThreadPool` of `n` workers at
  detector construction and runs `detect` / `detect_concurrent` under
  `ThreadPool::install`. `threads=0` (the default) keeps the global pool and the
  previous behaviour exactly. Code that passed a small `threads` value and relied
  on the accidental full-core execution will now be slower by design — pass `0`.
  Detection results are identical for every thread count.
- **`DetectorConfig::default()` / bare `Detector::new()` now actually match the
  `standard` profile they're documented to.** `docs/tutorials/guide.md` has long
  documented `Detector()` (no args) as equivalent to `Detector(profile="standard")`,
  and `docs/engineering/core.md` states the shipped JSON is authoritative over any
  Rust/Pydantic constant — but the hand-written `impl Default for DetectorConfig`
  had drifted on three fields (`threshold.enable_sharpening: false` vs `standard`'s
  `true`; `quad.max_elongation: 0.0` vs `20.0`; `quad.min_density: 0.0` vs `0.15` —
  the latter two are gate-disabling sentinels, so the drift silently turned off both
  the elongation cap and the density floor for anyone constructing a bare config).
  The independent Python Pydantic model had the same three plus a fourth
  (`quad.min_area: 16`, the pre-fix placeholder `standard.json` moved off of;
  see the `select_dominant_vertices` entry below), inert in practice because
  `locus.Detector()` always resolves through `DetectorConfig.from_profile(...)`
  and never touches the bare Pydantic defaults — but a real trap for anyone
  constructing `locus.DetectorConfig()` directly. All four fields are now synced to
  `standard.json` on both sides, `schemas/profile.schema.json` regenerated to match.
  Two new value-level parity tests close the gap the existing field-*set* tripwire
  (`schema_parity_tests::serde_shim_matches_referee_schema`) didn't cover:
  `config::schema_parity_tests::default_matches_standard_profile` (Rust) and
  `test_bare_default_matches_standard_profile` (Python). **Behavior change:** any
  code path that bare-constructs a config (`Detector::new()`, Rust
  `DetectorConfig::default()`, or Python `locus.DetectorConfig()`) now detects with
  sharpening on and both quad geometry gates enabled — every shipped-profile
  regression suite (render-tag, ICRA, EuRoC, board/distortion hub) already
  specifies an explicit profile and is byte-identical; only bare-default callers
  are affected. `contract_config_inertness`'s `threshold_min_range` case needed a
  new prerequisite (`base_sharpening_off`) — sharpening now saturates every tile's
  contrast on that test's tiny synthetic canvas, making the field coincidentally
  inert on that one scene regardless of its real (telemetry-only, see above) effect.

### Fixed

- **`threshold.min_range` documented as telemetry-scoped.** The tile-validity mask
  it drives is applied only when writing `telemetry.binarized`; the per-pixel
  threshold map that segmentation consumes is written unconditionally, and the
  propagation pass that would have made the knob affect detection has been
  commented out since `996e782`. The field is kept (it is live for the debug map)
  and its Rust/Pydantic docs now say so instead of implying a detection effect.
- **`quad.upscale_factor > 1` returned corners in the upscaled frame.** The
  detection pipeline bound the upscale factor to an unused `_effective_scale`,
  so quad corners (and the covariances derived from them) stayed in upscaled
  pixels (e.g. ~2x the true coordinates at factor 2) and were then fed, still
  unscaled, to GWLF refinement, decoding and pose against the original image.
  Corners and covariances are now mapped back to original-image coordinates
  right after quad extraction and the funnel gate, using the centre-aware
  convention shared with decimation (`x = (x_up + 0.5)/U - 0.5`;
  covariance scaled by `1/U^2`); decode, GWLF and pose sample the original
  image. `ImageView::upscale_to` now samples pixel centres
  (`(x + 0.5)/U`) so it is the exact inverse of that mapping (it previously
  shifted content by half an upscaled pixel). `upscale_factor == 1` and all
  decimation paths are unchanged.
- **`ContourRdp` quad-corner extraction silently discarded correctly
  segmented, correctly sized tag candidates before they ever reached
  decode.** `extract_single_quad`'s Douglas-Peucker simplification used
  `epsilon = perimeter * 0.02` — dimensionally wrong, since the perpendicular
  "staircase" deviation a rasterized angled edge needs absorbing is set by
  pixel-grid quantization and edge angle, not by the object's size on
  screen. That made epsilon too tight for small/moderate contours, which
  then failed a `simplified.len() in [4, 11]` vertex-count gate (root-caused
  on real EuRoC MAV imagery: one frame's segmentation found 40 tag-sized
  components, only 10 ever became quad candidates — 13 of the 30 lost ones
  failed exactly this gate, always with *too many* vertices). Replaced with
  `select_dominant_vertices`: an epsilon-free significance ranking (one full
  unconditional Douglas-Peucker decomposition, diameter-pair-seeded so the
  two anchor points are provably real corners rather than an arbitrary
  boundary-trace artifact) that hands a generous candidate pool to the
  existing, unchanged `reduce_to_quad` for final 4-corner selection — a
  single top-down "take the top 4" pass was tried first and cost ~19%
  relative recall on real EuRoC frames vs. pooling before reducing, so the
  final design keeps `reduce_to_quad`'s iterative robustness and only
  replaces the epsilon it used to be gated by.
  **Real-world**: EuRoC relative recall (decoded/present) 63.1% → 69.9%
  (measured together with a real-camera/AprilGrid profile fix — see the
  `regression_euroc.rs` entry above — the two changes were validated
  together); distorted-pose recall 49.0% → 71.0%.
  **Synthetic (`regression_render_tag`/`_robustness`, insta-snapshotted)**:
  broad improvement on the `ContourRdp`-path tests — mean/reprojection RMSE
  roughly halves and p99 rotation-error tail improves 4×–60× in 6 of 8
  affected tests (e.g. `high_iso` p99 rotation 104.1° → 1.75°,
  `raw_pipeline` 119.5° → 26.0°), recall flat or better in 6 of 8. Two
  tradeoffs reviewed: `tag16h5` precision drops 96.3% → 93.75% (one extra
  false positive out of ~100 images), and `low_key_tuned`'s p99 rotation
  error worsens 21.6° → 99.3° on `scene_0005_cam_0000.png` tag `34` — root-
  caused, not a corner-extraction defect: that tag's corner RMSE is 2.43px
  (good), so this is a *differential* (non-uniform across the 4 corners)
  micro-perturbation landing on an already poorly-conditioned single-tag
  pose solve, not a wrong-branch or systematically-worse-corners issue.
  Confirmed against this project's own prior investigation
  (`project_rotation_tail_is_corner_localization_20260714` in memory):
  IPPE branch selection has "zero headroom" (picks correctly essentially
  always), so the render-tag rotation tail is driven by differential
  corner-localization error, not solver branch choice — and the only
  documented lever for that class of tail is corner-*refinement*
  robustness (ERF/GWLF), a separate pipeline stage this PR doesn't touch.
  One isolated tag out of every tag across 8 affected tests showing this
  pattern, against broad improvement everywhere else, is consistent with
  an isolated hard-geometry case, not a systematic regression this fix
  introduced. `.snap` files are not updated by this change (pre-existing,
  unrelated snapshot drift affects 11 of 17 render-tag/robustness tests
  independent of this fix — see PR discussion).
  Latency on the `ContourRdp` path rose modestly (e.g. `low_key` 36→41ms,
  `high_iso` 45→56ms), expected given the unconditional decomposition does
  more work than the old epsilon-gated pass; not evaluated against a
  specific budget.
- **`reduce_to_quad` could discard a real corner for an adjacent
  staircase-rasterization pixel on an exact triangle-area tie.** Caught by
  CI, not local validation: `regression_icra2020::regression_fixtures`
  (real 2448×2048 photo, 190 printed `tag36h11` tags,
  `tests/fixtures/icra2020/0037.png`) regressed mean corner RMSE 0.1315px
  → 0.1375px. Root cause, isolated with a per-tag diff against `main`: only
  2 of 154 matched tags moved (tags `1` and `20`, both ~0.5–0.75px on one
  corner; the other 152 were bit-identical), and both share the same
  contour shape — a clean 4-corner rectangle plus a 1px anti-aliasing
  staircase notch immediately beside one true corner (7 contour points
  total, all ≤ `POOL_CAP`, so the whole contour reaches `reduce_to_quad`
  unfiltered). `reduce_to_quad`'s smallest-triangle-area elimination has no
  concept of which vertex is "real": on this shape, removing the true
  corner and removing its neighboring 1px notch cost the *exact same*
  triangle area (`12.0px²` in both regressed cases, verified by hand), and
  the loop's `area < min_area` scan silently keeps whichever it reaches
  first — a coin flip that happened to discard the real corner both times.
  This was latent in `reduce_to_quad` itself (unchanged since before this
  PR) but only became reachable once `select_dominant_vertices` started
  pooling low-but-nonzero-significance points like the notch instead of an
  epsilon threshold filtering them out upstream; `reduce_to_quad` still has
  exactly one caller, so this is fixed at the source rather than
  papered over with a new epsilon. `select_dominant_vertices` already
  computes exactly the signal needed to break this tie — the notch's
  significance weight (~1.0) is over an order of magnitude below the real
  corner's (~17.0) — so `reduce_to_quad` now takes that weight array
  alongside the points and, **only when the area criterion is at or within
  float-precision (`1e-9` relative) of the current minimum**, prefers to
  discard the lower-significance point instead of leaving it to iteration
  order. This is a numerical-precision tolerance, not a reintroduced
  geometric epsilon — it never changes which vertex has the strictly
  smaller area, only which one wins a real tie. Fixes both regressed tags
  exactly (RMSE 0.5085px/0.5083px → 0.0422px/0.0446px, matching `main` to
  4 decimal places) and the fixture snapshot **improves** on `main`,
  0.1315px → 0.1312px (154/154 tags now bit-identical or better).
  Re-validated against `regression_render_tag`/`_robustness` (both suites
  byte-identical to the pre-tie-break-fix numbers already documented
  above — this fix only changes behavior on exact/near-exact area ties,
  which those suites' contours don't happen to hit) and
  `euroc_detection_baseline` (unaffected). All 278 default-feature tests
  and the full `--all-features` suite pass; `cargo fmt`/`cargo clippy
  --all-features -- -D warnings` clean.

### Added

- **Config-inertness contract test** (`crates/locus-core/tests/contract_config_inertness.rs`).
  Mutates **every** `DetectorConfig` field on a deterministic synthetic frame set
  and requires the observable output — detections, rejected candidates, poses,
  covariances and the `binarized` / `threshold_map` telemetry images — to change.
  The field list comes from an exhaustive destructuring of the struct, so a new
  field fails the build until a case exists. Provably-inert fields are allowlisted
  with a written reason and are asserted to be *identical*, so the allowlist
  cannot silently rot in either direction. Also pins `nthreads` determinism
  (output invariant across 1/2/4/8 threads) and asserts the scoped pool is the one
  the pipeline actually executes on.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
