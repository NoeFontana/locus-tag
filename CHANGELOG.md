# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Tests

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
