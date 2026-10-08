# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Fixed

- **The EuRoC corner cells across the benchmarking docs are re-baselined for the inexact
  undistortion inverse.** Recomputing the fourteen archived EuRoC runs both ways shows the
  artefact is a near-fixed **absolute** shift of -0.005 to +0.011 px, close to constant across
  detectors, so its *relative* size falls as a detector's own error grows: ~4 % at 0.27 px,
  +0.75 % at 0.48 px, indistinguishable from zero above 2 px. That is the whole mechanism by
  which it flattered the best corners, and why no single rescaling factor could undo it.
  `euroc_sota_20261004.md` is **re-derived exactly** — its detections were archived, and
  recomputing them with the old inverse reproduces the published cell to the thousandth, which
  identifies the runs (`locus_main`, `locus_final`, `opencv_apriltag`): the cell moves
  `0.316 / 0.634 | 0.283 / 0.575 | 0.516 / 0.996` to
  **`0.325 / 0.650 | 0.295 / 0.594 | 0.520 / 1.008`**, with recall and false positives moving
  by at most 0.01 pp. The runs behind `sota_scoreboard_20261002/03/04.md` and
  `liu4k_euroc_sota_20261001.md` were not archived and the common-tag set each cell used
  depended on the directory composition at the time, so those pages are annotated with the
  measured bound rather than rewritten — a figure rescaled by a factor measured on other runs
  is not a measurement. **No win/loss verdict on any scoreboard changes**, and only the EuRoC
  `LOO` rows are affected at all: `_undist` is reachable only from `score_euroc`, so the
  liu4k / ICRA / hub `Corner RMSE` cells never undistorted.

- **The EuRoC scorer's undistortion inverse was approximate, and it flattered the better
  detector.** `_undist` was `cv2.undistortPoints`, which runs a fixed, small iteration count
  with no convergence test; on this lens it stopped short by a *radius-growing* amount --
  0.046 px median at r in [300, 380) and 0.134 px at worst, where a converged fixed point
  reaches 3e-13. Because the shortfall is a smooth function of position a homography absorbs
  part of it, so the reported corner median came out **low**: -3.98 % for Locus `standard`
  against -0.82 % for OpenCV APRILTAG and -0.02 % for OpenCV CONTOUR+SUBPIX. It therefore did
  **not** cancel in the head-to-head comparison the scorer exists to make -- Locus's relative
  corner margin was overstated by about 3 pp. The inverse is now a fixed point that *asserts*
  convergence rather than returning an approximation, and a test pins the achieved round-trip
  error (it fails at 0.131 px against the old implementation). Recall, false positives and the
  reference-frame count are unmoved (`locus_b_std` 92.485 -> 92.491 %, FP 0 -> 0, 1029 frames
  both ways) and no SOTA ordering changes, so this re-baselines the corner columns only.

### Changed

- **The dataset-backed snapshots are blessed, and the suites are green for the first time since
  2026-07-19.** `cargo nextest run --profile datasets --release --features bench-internals`
  reports 50 passed / 0 pending. The twelve snapshots that moved with the 179/400 tile cut were
  held back deliberately until the cut had been re-examined; it has been, the constant is
  unchanged, and they now record the measured result rather than a pending question. The wins:
  low-key recall **0.80 -> 0.98** with corner RMSE 0.1776 -> 0.0650 and rotation p99
  **1.4606 -> 0.1811 deg**, low-key tuned recall 0.20 -> 0.44, raw-pipeline recall 0.96 -> 1.00,
  raw-pipeline tuned recall 0.62 -> 0.74 with rotation p99 **3.4167 -> 0.4746 deg**. The costs,
  recorded rather than buried: `tag16h5` loses one marker of 100 in both variants (recall
  1.00 -> 0.99), ICRA forward checkerboard-grid recall 0.6886 -> 0.6736, the EdLines-variant
  rotation p99 rises 0.5794 -> 0.8356 deg, and raw-pipeline mean corner RMSE rises
  0.1015 -> 0.1839 px as four percent more markers enter the average.
- **A profile optimisation pass found no change worth shipping, which is itself the result.**
  Recorded in `profiles/README.md` so it is not re-derived:
  - `segmentation.connectivity` and `threshold.enable_sharpening` **interact** and the two
    shipped profiles sit at the two self-consistent corners of a 2x2. Sharpening overshoots at a
    border, widening the dark region into single-pixel *diagonal* bridges between neighbouring
    markers; 8-connectivity fuses across them, 4-connectivity cannot see them. EuRoC
    `high_accuracy` recall: `Eight`+off **53.4 %**, `Four`+off 87.6 %, `Eight`+on 25.9 %,
    `Four`+on **91.6 %**. Changing either knob alone moves the detector to a worse corner than
    it started in.
  - Moving `high_accuracy` to `Four` + sharpening would buy 38 pp of EuRoC recall and cost the
    metric the profile exists for: **AprilGrid board p99 rotation 0.0186 -> 0.1749 deg, 9.4x
    worse**, deterministic to 16 digits over two runs, plus one 4K render-tag marker. Rejected.
    Its low EuRoC recall is a use-case mismatch — those are small, motion-blurred tags and
    `standard` reaches 92.0 % on the same frames — not mis-tuning.
  - Measured and **inert**: `decoder.min_contrast` 20 -> 15 -> 10 (identical recall),
    `quad.min_edge_score` 4.0 -> 2.0 (identical), `quad.subpixel_refinement_sigma`
    0.6 -> 1.4 (4th decimal only), and `high_accuracy`'s `min_area` 400 -> 36 plus the
    filled-blob gates switched off (identical, because connectivity is the binding constraint:
    markers are merged, not rejected for being small). The sigma result retires a plausible
    hypothesis: the difference between synthetic corner error (0.0370 px) and the real EuRoC
    figure (0.2845 px) is not the Erf model's assumed PSF width. It is also not a 7.7x
    detector gap — see the `### Fixed` entry above, which shows the two were never the same
    quantity.

- **The EuRoC corner number is now documented for what it measures**, in
  `docs/engineering/benchmarking/euroc_error_budget.md`. It is a leave-one-tag-out
  *self-consistency residual* against a nominal board, with no ground truth in it, and it had
  been compared against the render-tag synthetic RMSE as though the two were one quantity. They
  are not, and the 7.7x "gap" between them was an artefact of that comparison. Measured:
  the metric reports **1.365x** the underlying iid corner noise (and its p90 reports **2.6x**),
  and of Locus's 0.436 px RMS residual, 24 % of the variance is static per image cell and 13 %
  is a *deterministic per-`(tag, corner)`* bias -- which turns out to be the **estimator**, not
  the board, because it ranges over 13x across detectors (Locus 0.333 mm, OpenCV APRILTAG
  0.747 mm, OpenCV CONTOUR 4.226 mm) where a physical board would give them all the same
  number. Locus is best of the four detectors on every component of the budget. Falsified and
  recorded so they are not re-attempted: board-model error (7 %; the pitch/tag ratio fits to
  *exactly* the Kalibr nominal 1.3000 and a free 288-parameter board removes 2.2 % on held-out
  frames), motion blur (flat over a 40x velocity range), board non-planarity (non-monotonic in
  tilt), and apparent tag size (error *grows* as L^+0.68, the opposite sign to the
  variance-limited L^-0.5, and extrapolates to 0.46 px where render-tag measures 0.037 px).

- **`high_accuracy` now shares `standard`'s geometry.** It ran `EdLines` whole-edge corners
  under an `AdaptivePpb` router; it now uses `ContourRdp` contours, `corner_subpix` and `Erf`
  refinement on a `Static` route, like `standard` and `grid`. What makes it the accuracy
  profile is its pose layer (model-edge refinement, the χ² consistency gate, single-corner
  outlier drop) and its stricter detection gates (`min_area: 400`, fill/elongation/density,
  8-connectivity, no sharpening) — not a separate corner estimator.
  - It was measured worse on every axis, **including the only real imagery in the repo**.
    EuRoC `cam_april`, 1450 frames, one scoring pass, GT-free LOO protocol, `RAYON_NUM_THREADS=1`:
    recall **38.43 -> 53.75 %**, precision 99.734 -> **100.000 %**, false positives **32 -> 0**,
    LOO corner median **0.4138 -> 0.2825 px**, LOO p90 0.8471 -> 0.5856 px. Synthetic render-tag,
    four resolutions: corner RMSE **-69 to -71 %**, translation p99 **-24 to -39 %**, rotation
    p99 and p50 **unchanged** (±0.0001 deg) — the tail the profile exists for is untouched.
    Board pose: AprilGrid p99 rotation **0.1098 -> 0.0186 deg**, p99 translation -94 %; ChArUco
    p99 rotation **0.1757 -> 0.0477 deg**. ICRA forward on this profile: recall
    **17.04 -> 24.30 %**, corner RMSE -66 %.
  - The cost is latency: EuRoC **3.47 -> 5.28 ms**, because `min_area: 400` plus EdLines was
    discarding candidates it should have kept. Still well inside `standard`'s 11.77 ms.
  - **Emitted uncertainty changes meaning.** Board `mean_board_translation_std_m` rises ~14x
    while the actual error falls ~86 %, so the covariance moves from over-confident to markedly
    conservative. That is the safer direction for a consumer that gates on it, but it is a
    change, not a wash.
  - This also removes the last cut-sensitive corner path in the tree. Corner error under
    EdLines moved **+34 %** between tile cuts 0.5000 and 0.4475; under `standard` geometry it is
    bit-identical at both. The 179/400 cut had looked like a geometry regression on render-tag,
    and it was this profile, not the cut: the four `accuracy_baseline` snapshots were the only
    place the cost appeared, and they run `high_accuracy`. A sweep of seven cuts found **no
    knee** — render-tag cost is superlinear in the offset from 0.5 while recall gain is roughly
    linear, with marginal efficiency falling 3.03 -> 1.18 EuRoC pp per 0.01 px — so no single
    constant was defensible, and the constant was not the thing to change.
  - **`EdLines`, `AdaptivePpb` and `edlines_imbalance_gate` are retained deliberately.** No
    shipped profile uses them; they are **not** dead code to be removed. The evidence above is
    synthetic plus one real sequence, whole-edge corners may yet prove better on real imagery
    this repo does not have, and they stay reachable through the config surface and exercised by
    the `quad_extraction_variants` cases so they keep snapshot coverage. `profiles/README.md`
    records the historical settings and the reason for keeping them.
  - `high_accuracy_profile_routes_low_ppb_to_contour_rdp` becomes
    `high_accuracy_profile_shares_standard_geometry` and asserts the four geometry fields
    **against `standard`** rather than against literals, so the two cannot drift apart
    silently — that drift is what made the cut look like a geometry regression.
- **The dark/bright cut no longer sits on the midpoint of the tile extremes.**
  `ThresholdMode::TileMidExtreme` thresholded at `(min + max) / 2` over a tile's 3x3
  neighbourhood. That is the unbiased cut for an edge the optics resolved, and it is the wrong
  cut for *topology*: segmentation reads this map to decide which pixels are one marker, and a
  bright separation narrower than the point-spread function never reaches the bright level, so
  two markers either side of it come out as one connected component. Nothing downstream can
  take them apart — a component is traced once and reduced to one quadrilateral — so both
  markers are lost. The cut is now at **179/400 (0.4475) of the range**, which shrinks every
  dark region by a fraction of the blur width and holds those separations open.
  - The cut is applied with **rounding**, not truncation. Truncating `9 * range / 20` biases the
    cut downward by up to one grey level, and the bias is proportionally largest where the
    range is smallest — the low-contrast tiles where erosion is most likely to cost a marker
    its border, i.e. exactly backwards from what the cut is for. At a range of 2 it reached the
    whole fraction: the threshold equalled the neighbourhood minimum, and under a
    `pixel < threshold` rule that tile could never hold foreground at all, where the old
    midpoint still admitted its darkest pixel. The constant is integer throughout, so the map
    stays bit-identical across targets and SIMD widths, but it is **not** exact for most
    ranges.
  - **The fraction reads 0.4475 and not 0.45 because the frontier sweep that chose it was
    itself truncated.** Every path of the foreground test is a strict `pixel < threshold`
    (scalar, AVX2, NEON), so the faithful integer cut for a fraction is `ceil(fraction *
    range)`; averaged over the 8-bit ranges, truncation lands 0.95 grey levels under that and
    rounding 0.45 under it. Rounding a nominal 0.45 therefore moves the realized cut *up* by
    half a grey level, off the point that was measured — and that half level is the entire
    regression it caused. Half a level back down is `0.5/range` of fraction, about 0.003 at the
    ranges this stage sees, which is why the denominator moved 20 -> 400: no coarser grid can
    write the number down, and re-denominating is exact, so the swept grid stays readable on
    the new one (`(180 * range + 200) / 400` equals `(9 * range + 10) / 20` for every 8-bit
    range). No constant reproduces truncation exactly — truncation's realized fraction depends
    on the range, which is the defect being removed — but 0.4475 is the constant that lands on
    the same integer cut for most of the data the frontier was measured on: **81 % of EuRoC
    tile neighbourhoods and 76 % of Liu4K's, against 47 % and 45 % for 0.45**. It was derived
    before it was run. Measured, it recovers EuRoC **exactly** — 92.05 %, against the 92.05 the
    truncated cut scored and 91.65 at 0.45 — and **two thirds of Liu4K**: 44.56 %, against 44.80
    and 44.06. That shortfall is the range dependence itself rather than a mis-derivation.
    Liu4K's neighbourhoods are lower-contrast (median range 86 against EuRoC's 107) and so want
    a larger correction than any one constant can give them while also giving EuRoC its own.
  - On **EuRoC `cam_april`**, the only real-camera dataset in the suite, the board's tags are
    separated from the connector squares by a strip that closes under motion blur and
    obliquity; whole frames were resolving to a single black lattice. Recall
    **86.02 -> 92.05 %** with precision rising 99.996 -> **100.000 %** (1450 frames,
    `standard`, 1 thread). Leave-one-tag-out corner error barely moves (0.2833 -> 0.2845 px
    median) and is identical to four decimals at every cut in the sweep below, which is the
    expected result: the cut decides topology, corners come from the grey-scale refinement
    either way.
  - Elsewhere: **Liu4K 37.18 -> 44.56 %** recall (precision 100 %), **render-tag low-key
    66 -> 96 %**, **render-tag raw-pipeline 96 -> 100 %**, AprilGrid 99.089 -> 99.108 %, ICRA
    forward 72.026 -> 72.039 %, ICRA random 99.995 -> 99.997 %. Every other benchmark holds
    at its previous recall, and the shipped cut is at or above the midpoint baseline on all
    thirteen. (`hub-charuco` scores zero recall for every arm including the baseline — the
    ChArUco board is not evaluable by this harness's matcher, not a regression.) Two extra
    false positives appear on the 100-frame tag16h5 set (precision 100 -> 98 %); they are
    borderline candidates that flicker with the constant, not
    a systematic loss. **That precision side is not guarded by any assertion** — the render-tag
    robustness suite is snapshot-only and this set lives in the bench harness — while the
    recall side now has a hard floor, so the two directions are not symmetrically protected.
  - The re-expression was swept rather than assumed, four fractions over the whole board in
    one scoring pass (recall %, `standard`, 1 thread):

    | cut | EuRoC | Liu4K | 640 | tag16h5 | low-key | AprilGrid | ICRA fwd |
    | :-- | --: | --: | --: | --: | --: | --: | --: |
    | midpoint (`main`) | 86.02 | 37.18 | 100 | 99.00 | 66.00 | 99.089 | 72.026 |
    | 0.4500 = rounded 9/20 | 91.65 | 44.06 | 100 | 99.00 | 96.00 | 99.127 | 72.065 |
    | **0.4475 (shipped)** | **92.05** | **44.56** | **100** | **99.00** | **96.00** | **99.108** | **72.039** |
    | 0.4450 | 92.38 | 45.01 | 100 | 99.00 | 96.00 | 99.127 | 72.013 |
    | 0.4425 | 92.40 | 45.23 | 100 | 99.00 | 96.00 | 99.127 | 72.013 |

    0.4475 is kept because it is both the derived value and the last point at which nothing
    regresses against the midpoint baseline. Lower cuts are **not** free and are not taken:
    they keep buying EuRoC, but ICRA forward turns over at 0.4450, and `recall_by_side` puts
    every bit of that movement in the `[0,20)` bin — the `[20,45)` and `[45,100)` bins sit at
    100.0000 % for every arm, baseline included. Small markers eroded out of their one-module
    borders, in other words: 57.615 % of the sub-20-px bin at the midpoint, 57.635 % at the
    shipped cut, 57.595 % at 0.4450 — the same trade the frontier above describes. The
    remaining gap to the measured 99.6 % / 74.6 % frontier is not another constant; it needs
    the seeded split.
  - Latency, serialised A-B-B-A on an idle host (AMD EPYC-Milan, 4 cores / 8 threads,
    Linux 6.8.0, rustc 1.92.0, `--release`, `RAYON_NUM_THREADS=1`, two `detect()` calls per
    image, best of two interleaved arms), midpoint -> shipped cut, within-arm spreads in
    parentheses: render-tag 4K 78.00 -> **68.99 ms** (-11.5 %, 0.82 / 0.14), 1080p
    20.23 -> **18.66 ms** (-7.7 %, 0.07 / 0.23). EuRoC is flat at 7.38 -> 7.41 ms (+0.4 %,
    0.01 / 0.03): it now finds 6.0 pp more markers, and decoding them is work the old cut was
    not doing. The extra foreground the re-expressed fraction admits costs nothing measurable
    against 0.4500, which scored 69.55 ms and 18.50 ms on the same bench — both inside these
    within-arm spreads.
  - How far below the midpoint is a stated assumption about the thinnest *dark* stroke that
    must survive, and it was measured rather than derived: the full scoreboard is flat from the
    midpoint down to 9/20 and then starts costing small and low-contrast markers their
    one-module borders (render-tag 640 falls off at 8/20, tag16h5 at 8.4/20). Those are the
    sweep's *nominal truncated* fractions, which is the distinction the bullet above turns on. The
    constant is
    documented at `threshold::CUT_NUM`, and a `const` assertion rejects a numerator above the
    denominator or wide enough to overflow the `u16` the tile loop multiplies in.
  - Two tests bracket the constant from both sides, which the first version of this change did
    not do. `the_cut_holds_open_gaps_the_midpoint_closed` sweeps contrast and point-spread
    width over a pair of dark squares split by a one-pixel gap and requires the shipped cut to
    separate strictly more of the sweep than the midpoint rule does — reverting the cut to the
    midpoint makes the two counts equal and fails. `the_cut_is_the_lowest_that_keeps_a_one_pixel_dark_stroke`
    requires a one-pixel stroke to survive at the shipped cut and to be **lost** one twentieth
    lower, so lowering the constant fails too; it steps by `CUT_DEN / 20` rather than by 1, because
    a bracket written as `CUT_NUM - 1` stops bracketing anything the moment the cut is
    re-denominated onto a finer grid — which is exactly what just happened to it.
    `the_cut_is_rounded_and_leans_neither_way` pins the discretizer itself: over the 8-bit
    ranges the cut must stay within half a grey level of the exact fraction and show no
    systematic lean, which truncation (0.5 low) and `ceil` (0.5 high) both fail. That half
    level has now been lost twice without anything failing, so it is a test and not a comment.
    `counterfactual_map_reproduces_the_shipped_cut` pins the test-side cut
    arithmetic to the production map so neither can drift.
  - `regression_euroc`'s relative-recall floor moves 0.50 -> 0.75. It is a per-frame mean over
    whichever frames the sampling stride selects, so it was calibrated across strides rather
    than at the default alone: 0.806 / 0.807 / 0.802 / 0.802 / 0.823 / 0.802 at strides
    1 / 3 / 7 / 10 / 23, minimum 0.802.
  - Board snapshots moved: the shift is deterministic (three consecutive runs agree bit for
    bit, so it is not the known ~1e-13 flake) and comes from marginal tags entering and leaving
    the board solve. It is small and mixed in both directions — the largest single movements
    are AprilGrid p95 translation -33.8 % and p99 translation +15.3 %, `high_accuracy`
    AprilGrid p95 rotation +13.2 %, and ChArUco `high_accuracy` p95 rotation -14.0 %; tag
    coverage rises slightly on every AprilGrid arm. Relative to `main` the `high_accuracy`
    AprilGrid arm also loses two of 150 frames to `frames_no_estimate` while its mean rotation
    error improves 39 % and its p99 78 %. ICRA's fixture corner RMSE improves
    0.1312 -> 0.1048 px. **Every render-tag snapshot is byte-identical.**
  - The EuRoC figures come from a single scoring pass. `tools/bench/sota/score.py` builds the
    presence reference from every self-consistent run in the directory, which includes the
    Locus runs under test, so absolute EuRoC recall moves at the ~0.01 pp level with the run
    set; deltas within one pass are unaffected. The gap is now documented at the pool
    construction with the fix and why it needs its own re-baselining pass.

- **The tile thresholder stops doing a frame of telemetry work on every production frame, and
  its 3x3 reduction vectorises.** `apply_threshold_with_map` drove its row loop from the
  binarized image — which `detect()` sizes to zero unless debug telemetry is on — so a
  production frame allocated a full-size scratch buffer purely to have something to iterate,
  ran the whole compare-and-store pass that fills it, expanded a per-tile validity mask only
  that pass reads, and computed the 3x3 tile reduction a **second** time, serially, in an
  Amdahl section beside the parallel one. The loop is now driven by the threshold map that
  segmentation actually reads, and the binarize work happens only when a caller asks for the
  image. Disjointness between the two output buffers now comes from `par_chunks_mut` on both
  rather than from a pointer-aliasing argument, which removes the **only `unsafe` block in the
  module** (and its `#![allow(unsafe_code)]`), along with two per-worker scratch rows.
  - The reduction itself is now separable. Each tile is packed as `(min, !max)`; `!max` is
    `255 - max`, which turns the maximum into a *minimum*, so one byte-wise `pminub` lane
    carries both fields, and the minimum over a clamped 3x3 window factors exactly into a
    horizontal and a vertical 3-tap over contiguous bytes. Reducing `TileStats` where it lies
    cannot vectorise: it interleaves the two fields, so every lane would want a strided gather
    and the maximum would want the opposite instruction. `TileStats` stays as it is — the
    packing is internal, into the arena.
  - Stage cost on the 2448x2048 ICRA frame, `--release`, one thread, median of 100 samples:
    **2.510 -> 0.289 ms (-88.5 %)**. Attributable as 2.510 -> 1.102 for dropping the telemetry
    work, -> 1.038 for hoisting the per-tile bounds checks out of the reduction, -> 0.289 for
    the separable form. A diagnostic that replaced the reduction with a single-tile read put it
    at 65 % of the stage, which is what justified the rewrite over the map write.
  - **This does not reach frame latency.** Interleaved A-B-B-A at 1080p against the same cut
    family measures 18.69 -> 18.67 ms (-0.1 %, spreads 0.18 and 0.07): the ~0.9 ms the stage
    saves at that resolution is absorbed downstream, consistent in both A-B-B-A runs with the
    extra foreground that rounding admits costing a comparable amount in segmentation and quad
    extraction. The stage win is real and the frame win is not, and both are reported.
  - `bench_threshold_real_icra_apply_map` is new and benchmarks the production shape (empty
    binary output). Nothing did, which is why a full frame of discarded work survived.
  - **Fixed:** a frame narrower or shorter than `threshold_tile_size` produced a zero-sized
    tile grid and reached `par_chunks_mut(0)`, which panics. Both the statistics pass and the
    threshold pass now return early, leaving the map at "never foreground".

- **The distortion path is one pipeline again, and ~20 % faster.** `decode_batch_soa_with_camera_inner`
  was a second implementation of the decode loop; that duplication was the *mechanism* by which
  every corner-estimator change since #426 reached the pinhole route only. It is gone. The loop
  now holds the decode policy once — scale retries, decode-first ordering with refined-quad
  verification, border-ring budgets, the near-miss recovery search — and routes image reads
  through a `Warp`: the identity for a rectified camera, which monomorphizes to the ROI-cached
  SIMD kernels unchanged, and the lens for a distorted one, whose working plane is the ideal
  plane where a marker's edges are straight. `decoder.rs` is net shorter despite gaining the
  trait, two implementations and a per-candidate sampler.
  - The one genuinely route-dependent operation is corner refinement, because refinement fits
    edges and an edge is straight only in the frame the warp defines. `Warp::refine_seed` and
    `Warp::refine_erf` supply it, so **decode-first ordering now applies on the straight-space
    route too**: refinement runs on the tags that decode (~32 per frame on the Brown-Conrady
    hub) instead of on every surviving candidate (~250).
  - **BREAKING:** `decode_batch_soa_with_camera` takes the frame's `RadialInverseTable`.
    `RoiCache::disabled()` is new.

- **The radial inverse is tabulated once per frame instead of solved per point.** Straight-space
  extraction unprojected 83,623 contour points per frame at 122.7 ns each — 29 % of that
  stage's CPU, almost all of it iteration and division (up to eight radial Newton steps, a 2-D
  polish, a verification evaluation; around nine `f64` divisions). Because the radial forward
  map is odd, `scale(s) = r_u/r_d` is smooth in `s = r_d²`, so `RadialInverseTable` indexes by
  `s` and a lookup needs **no square root and no division**. Tangential terms are then removed
  by a fixed point that reuses the same table, also division-free. Accuracy is *verified at
  build time* at the knot midpoints against a 1e-7 normalized budget (achieved 2.1e-7 px
  Brown-Conrady, 2.0e-5 px Kannala-Brandt), and the table is also structurally safer than the
  solve it replaces: knots are filled outward from the origin along one monotone branch, so a
  lookup cannot reach the mirrored far branch that an unguarded polish converges onto.
  Outside the verified domain lookups fall back to the iterative solve.

### Fixed

- **Every quad edge was being line-fitted twice.** `refine_corner_with_camera` fitted both of a
  corner's edges, so each of the four edges was fitted once as corner `i`'s trailing edge and
  again as corner `i+1`'s leading edge — with *identical arguments*, making the dedupe exact
  rather than an approximation. This was the single most expensive part of distorted extraction,
  48.8 % of its CPU time.

- **The +0.5 px lattice expansion was wrongly gated off for distorted cameras**, on the grounds
  that straight-space RDP corners are "projectively exact intersections, not integer-midpoint
  artifacts". That mistakes where the vertices were *chosen* for what the contour *is*: the
  boundary trace follows the dark side of the outline, and unprojecting a stepped contour is a
  smooth map, which locally is affine and therefore carries the half-pixel bias straight
  through. Under decode-first it stops being cosmetic, since those corners are the quad the bit
  grid is sampled through.

- **The contour was rectified before being simplified.** `chain_approximation` rejects a point
  when its two adjacent segments are exactly collinear — meaningful only on the pixel lattice,
  where adjacent traced points are an integer apart. On the rectified contour the same test
  compared a curvature residual against an absolute epsilon and kept nearly every point. It now
  simplifies first and unprojects the survivors, which is both more correct and much cheaper.
  The reordering also lets this route use the pinhole extractor's cheap compactness pre-gate,
  with area and perimeter both in pixel space so the gate means the same thing at every field
  angle.

- **`high_accuracy` built a full-frame label image on every distorted frame for a consumer that
  can never run.** `may_use_edlines()` did not know about distortion, and EdLines is
  geometrically incompatible with a declared lens.


- **`regression_distortion_hub` measured the pinhole detector on fisheye frames.** The harness's
  `build_intrinsics` falls through to `CameraIntrinsics::new` (no distortion) whenever the
  declared model cannot be represented, which is exactly what happens without
  `--features non_rectified`, since the `DistortionCoeffs` variants are themselves feature-gated.
  The test target did not require that feature, so the committed baselines were blessed from
  distortion-blind runs and the entire distortion path — straight-space quad extraction, the
  distortion-aware pose LMs, the `AdaptivePpb` fallback — had **no regression coverage at all**.
  Running the suite without the feature reproduces the committed snapshots to every digit, and
  passes. Every re-blessing to date (#412, #432, and the 0.9.0 line) recorded pinhole numbers.
  - The target now declares `required-features = ["non_rectified"]`, so it disappears instead of
    blessing a meaningless baseline, and the suite asserts its own premise — checked per frame,
    against `gt.intrinsics.or(options.intrinsics)`, which is what the detector is actually
    handed. Asserting on the dataset-level fallback alone would pass while every frame ran
    pinhole.
  - Both baselines are re-blessed from true distortion-path runs. Reading the old numbers as
    evidence about distorted detection is what made the "reprojection RMSE is worse than corner
    RMSE under distortion" anomaly look like a geometry defect; it was a pinhole pose fitted to
    distorted observations. The reproj/corner ratio on the real path is ~0.96, not 2.4
    (Brown-Conrady) or 8.6 (Kannala-Brandt).
- **The distortion-aware decode route ran no corner refinement at all.**
  `decode_batch_soa_with_camera_inner` is a second implementation of the decode loop, so its tail
  never received the gradient-orthogonality corner pass (#426, #430, #431, #432), the
  marker-inset photometric calibration (#434), or the frame-clipping check (#436). Toggling
  `decoder.corner_subpix` under declared distortion was a no-op. Every pass involved is a local
  photometric or geometric test on the raw image — none needs a camera model, and a lens cannot
  make a corner stop being a corner.
  - Both loops now share one `finalize_decoded_candidate` (sub-pixel pass, marker-inset
    calibration, frame-clipping check, rotation reorder of the `corner_refined` bits, homography
    recompute). The duplication *was* the defect's mechanism, so removing it is the fix: a new
    corner pass added there now reaches both routes.
  - The distorted route also sizes its sub-pixel window from the decoder that actually matched
    (`cells` travels with the accepted match), rather than a conservative minimum over every
    registered family.
  - **Run:** `cargo test --release --features bench-internals,non_rectified --test
    regression_distortion_hub -- --test-threads=1`, profile `standard`, **pose mode Accurate**
    (the harness supplies intrinsics + `tag_size`), 50 frames per hub, `rayon` default pool,
    `RAYON_NUM_THREADS` unset. Host: AMD EPYC-Milan, x86_64, 4 cores / 8 threads (`lscpu`, same
    session). Accuracy figures are deterministic and host-independent.
  - **Comparison base is the distortion path with the shared tail inert vs active** — *not* the
    previously committed baseline, which was a pinhole run and is not comparable:

    | | corner RMSE | reproj RMSE | trans p50 | rot p50 |
    | :-- | --: | --: | --: | --: |
    | Brown-Conrady, before | 0.957 px | 0.938 px | 16.8 mm | 0.305° |
    | Brown-Conrady, after | **0.234 px** | **0.225 px** | **0.4 mm** | **0.059°** |
    | Kannala-Brandt, before | 1.257 px | 1.249 px | 6.0 mm | 0.260° |
    | Kannala-Brandt, after | **0.306 px** | **0.290 px** | **0.3 mm** | **0.053°** |

  - Recall and precision are unchanged to four decimals; the pinhole path is untouched by
    construction, and its snapshots are byte-identical. Latency cost of the tail on this route,
    one run per arm: Brown-Conrady 16.19 → 16.59 ms per frame, Kannala-Brandt 14.48 → 15.34 ms —
    **2-6 %**, with a single sample per arm, so treat the spread as indicative.
  - **Still open, and pinned rather than hidden** (an `insta` baseline fails on any movement, so
    a further regression in either is caught): the distortion route remains behind the pinhole
    route on Brown-Conrady recall (89.4 % vs 98.4 %) and corners (0.234 vs 0.142 px) on the same
    frames, while winning decisively on pose (trans p50 0.4 vs 37.1 mm, rot p50 0.059° vs
    1.72°); and Brown-Conrady keeps a ~1 % catastrophic tail (rot p99 98.6°, trans p99 0.40 m)
    with the planar two-fold-ambiguity signature, which Kannala-Brandt's wider field of view does
    not show (rot p99 1.27°). The decode-first work of #433/#435 also landed on the generic loop
    only, which is the next instalment of this unification.
- **Brown-Conrady `undistort` had not converged.** The documented "Newton refinement" was a
  fixed-point iteration (`xu ← (xd − dx(xu))/radial(xu)`), only linearly convergent, capped at
  5 steps. At the shipped hub dataset's own coefficients (`k1 = -0.28`, `k2 = 0.08`) it left
  **0.42 px** of round-trip error at the image corner — above `quad.rs`'s undistort gate, which
  silently discarded every candidate whose contour touched ~1.8 % of the frame. Replaced by a 1-D
  Newton solve of the monotone radial problem followed by a 2-D Newton polish against the
  analytic Jacobian: worst-case **1.4e-9 px** over the frame (the polish's deliberate residual
  floor, five orders under the gate it feeds), in 1-2 polish steps.
  - **Out-of-domain radii are rejected, not silently inverted onto the wrong branch.** Past the
    model's radial turning point `r_d` has no preimage on the invertible branch, but the full map
    still has the mirrored one (`g(−r) = −g(r)`), which re-distorts to ~1e-14 and would therefore
    *pass* the caller's residual gate. A first cut of this fix did exactly that (`k1 = −0.1`,
    `r_d = 1.335` returned `r_u = −3.69`, residual 4e-14, accepted — where the superseded fixed
    point left residual 0.5 and was rejected). The radial solve now restores the last on-branch
    iterate and reports non-convergence, and the polish runs only on a converged solve.
  - The polish is skipped outright when `p1 == p2 == 0`, where the radial solve is already exact
    to 2e-16 — the common calibration case. The forward map and its Jacobian are evaluated
    together so the radial monomials are computed once per step, and the Jacobian is symmetric
    (`∂xd/∂yn == ∂yd/∂xn`, now asserted), so the 2x2 solve uses `det = a·d − b²`.
  - The model unit tests previously probed only `r ≤ 0.5` with hand-picked coefficients. They now
    also pin both inverters and both Jacobians at the shipped datasets' real operating point —
    `r_d = 0.80` for Brown-Conrady, `θ = 85.1°` for Kannala-Brandt — using central differences
    with a radius-scaled step, and the tuning constants are named rather than inline (the step
    tolerance is now *relative*: an absolute `1e-15` sits below 1 ulp for `r_u > 4.5`, so the
    loop always burned its full iteration budget on wide-angle coordinates).
- **Unprojection could fail silently, and three consumers trusted it anyway.**
  `CameraModel::undistort` cannot fail by signature, so outside a model's invertible domain it
  returns a point that is not a preimage of its argument. Only `quad.rs` re-distorted and
  checked; the IPPE seed of the single-tag pose path (`pose.rs`), the ideal homography the whole
  distorted decode hangs off (`decoder.rs`), and the board DLT/IPPE seed (`board.rs`) all used
  `undistort_pixel` and could not tell. New `CameraModel::undistort_checked` and
  `CameraIntrinsics::undistort_pixel_checked` perform the round trip — rejecting non-finite
  residuals explicitly, since a NaN fails every comparison and a bare `>` would *admit* them —
  and the three consumers now handle failure: no pose, a `FailedDecode` candidate, and a skipped
  correspondence respectively, rather than geometry built on a point the lens model cannot
  explain. The residual budget has one definition (`camera::MAX_UNDISTORT_RESIDUAL`) instead of
  living in `quad.rs`.
  - Identity, and so always `Some`, for `DistortionCoeffs::None`: the pinhole path is unchanged.
  - `camera::MAX_UNDISTORT_RESIDUAL` is public: it is the bound the default
    `undistort_checked` applies, so it is observable contract for anyone implementing
    `CameraModel` rather than a private tuning knob.
- **No `regression_*` suite ran in CI.** `.config/nextest.toml`'s `default-filter` excludes
  `binary(~regression)`, the `ci` profile inherits it, and `cargo insta test` defaults to
  `--test-runner auto`, which picks the installed nextest and inherits the same filter. Most of
  those suites need a dataset, but `regression_straight_space` renders its own fisheye and
  polynomial scenes, so it is a free gate — and it is the only CI coverage of the
  distortion-aware extraction and decode routes. It now runs as its own step, and it gained a
  test asserting the sub-pixel pass is live on the distorted route (mutation-checked: disabling
  the pass fails it).

#### Review findings addressed

A principal-level review of the above turned up fifteen findings; fourteen are fixed here, all
accuracy-neutral (every suite, including both distortion hubs, is bit-identical to the state
before this batch).

The one that mattered: **`RadialInverseTable` was silently truncating its domain on the fisheye
hub.** `RADIAL_SOLVE_REL_TOL` is a relative *step* tolerance, and a Newton step is
`(g - r_d)/g'` — so where `g'` is small the step's own round-off floor exceeds the tolerance and
the solve can never report convergence however correct its answer is. Kannala-Brandt has
`g' = dθ_d/dθ · 1/(1+r²)`, which decays as `r = tan θ` grows, so the knot fill truncated at
**90.4 % of the frame radius: 2.45 % of a 1920x1080 lattice — the frame corners, where fisheye
markers sit — permanently falling back to the per-point solve, precisely where that solve is
most expensive.** Nothing signalled it: `is_enabled()` stayed true, `verified_error()` reported
1.7e-11 over the *shrunk* domain, and the unit test asserted only "≥95 % tabulated". The fix is
a residual test alongside the step test; coverage is now 100 % with zero fallbacks and a
verified error of 8.2e-9. This is the **third** instance of the same bug class after the pose
LM's step gate (#341) and the Brown-Conrady inverter's sub-ulp absolute tolerance (#443) —
prefer a residual or relative-cost test over a step test.

Also fixed: `branch_reach` bisected on `g' > 0` while its probe also broke on `!g.is_finite()`,
so an overflowing model returned a non-finite reach that `f64::min` then silently discarded,
voiding the clamp in exactly the case it exists for. `RoiCache::disabled()` was a zero-width
`Arena`, whose `get` clamps every coordinate to 0 and indexes an empty slice — so a type
documented as "every lookup falls outside it" would panic on *every* lookup; it is now its own
variant that reads as zero. `fit_edge_line_curved` indexed a fixed 16-element buffer with
`2·(decimation+1)+1`, which panics for any `decimation >= 7` that `validate()` happily accepts.
The two minimum-edge-length gates still compared a *chart* length against a pixel-calibrated
4.0, loosening by ~6.5x at the Kannala-Brandt periphery — the same correction as the area gate,
with a square root. The label-image gate keyed on coefficient *values* while the extractor
dispatch keys on the declared *variant*, so an all-zero `BrownConrady` still allocated an 8.3 MB
label image for a consumer that can never run. `RadialInverseTable::scale` held a slice rather
than a fixed-size array reference, so the hot loop's two index operations each carried a
compare-and-panic branch LLVM could not elide — in the function whose whole purpose is to be
branch-free. `LensWarp::refine_seed` could never return `None`, making the shared loop's
`continue` arm unreachable on that route. And the merged decode loop had dropped one of seven
disjoint-slice `debug_assert`s — `corner_refined`, the most recently added column.

**`upscale_factor > 1` with a declared lens model is now rejected** rather than silently
mis-projected: straight-space extraction runs on the upscaled grid but is handed the
un-upscaled intrinsics, so every normalized radius it computes is inflated by `upscale_factor`.
Decimation has no such problem because `ScaledIntrinsics` rescales to match. New
`ConfigError::UpscaleUnsupportedWithDistortion`, mirroring the existing distortion +
static-EdLines rejection.

One finding was **measured and rejected**: adding the pinhole extractor's +/-1 px seed band to
the curved edge gate. The reasoning was sound — an unrefined decode-first seed sits up to a
pixel off the edge — but it cost **0.63 pp of precision on Kannala-Brandt and 0.12 pp on
Brown-Conrady for 0.20 pp and 0.11 pp of recall**, and pairing it with the post-refinement
re-gate recovered only 0.06 pp of that. Unlike the pinhole chord, this one is already sampled
along the *curved* edge, so it does not miss the gradient ridge the same way and the band is
close to pure loosening. The plumbing is kept so the trade can be re-measured on a wider lens.

One finding is **not addressed**: the shared loop returns on the first match that clears its
budget rather than keeping the lowest-Hamming match across all three scale retries, and the
review correctly notes that a cross-scale best could be kept for *both* routes rather than
diverging. Measured in isolation the early return is worth mean Hamming 0.0182 → 0.0844. It is
left for its own change because deferring acceptance past verification on all three scales is a
behavioural change to the **pinhole** route, which needs its own measured PR rather than a
late amendment to this one.

### Changed

- **Segmentation was 65 % of the frame; it is now 45 % of a much shorter frame.** It is
  camera-independent, so this lands on every dataset. Phase timers first, because the obvious
  suspect was wrong: the threshold + RLE pixel scan is **7.6 %** of segmentation, and the other
  92 % is run/component bookkeeping. The counts say why — a 1920x1080 frame averages **237k runs
  resolving to 63k components, of which 940 survive `min_area`** (326k components at worst).
  - **Describe the survivors, not everything.** `ComponentStats` is 56 bytes, so the stats
    scatter was a `num_components`-sized array (3.5 MB per frame, 18 MB at worst) written at one
    random offset per run — a cache miss per run, for data discarded 98.5 % of the time. Only
    `pixel_count` decides survival, so the pass over every run now accumulates just that, into a
    `u32` per component (253 KB, L2-resident); the full description runs over the survivors alone
    (940 x 56 B, L1-resident). The counting sort that groups runs by component asked every run
    the same question, so it is now the same pass.
  - **Spatial moments are opt-in.** `accumulate_run` computed five integer moments per run —
    fifteen multiplies and three divides — that nothing reads on the default path:
    `compute_moment_shape` is gated on `quad_max_elongation` / `quad_min_density`, both `0.0` in
    every shipped profile, and EdLines reads `m10`/`m01` only when it can run at all.
  - **Union-find attaches by minimum index, not by rank.** A set's root is then always its
    smallest member, so the forest is a function of the partition alone rather than of the order
    unions were applied in. That buys determinism (component numbering follows scan order) and,
    more importantly, parallelism.
  - **Labelling is striped.** A worker given a row range only unions runs from those rows, and
    because roots are minima it can only ever write parent entries inside its own contiguous id
    range — so `S` workers take `S` disjoint sub-slices with no synchronisation and no `unsafe`.
    The row pairs that straddle a stripe edge are merged afterwards on the whole array, and the
    result does not depend on the interleaving. 32 rows per stripe gives ~34 stripes at 1080p,
    several per worker, so rayon can balance rows of very different run density.

#### Measured

50 frames per hub, `standard`, pose mode Accurate, `--release --features
bench-internals,non_rectified`, `--test-threads=1`, rayon default pool, `RAYON_NUM_THREADS`
unset, AMD EPYC-Milan x86_64 4c/8t via `lscpu` in the same session. Latency is a serialised
A-B-B-A comparison of `Detector::detect` on an otherwise idle host; within-arm spread was
0.07–0.15 ms.

| | before | after |
| :-- | --: | --: |
| Brown-Conrady recall | 89.40 % | **91.39 %** |
| Brown-Conrady precision | 99.671 % | **99.825 %** |
| Brown-Conrady corner RMSE | 0.2335 px | **0.2163 px** |
| Brown-Conrady reprojection RMSE | 0.2246 px | **0.2038 px** |
| Brown-Conrady translation p99 | 0.3989 m | **0.3470 m** |
| Brown-Conrady latency | 17.96 ms | **14.27 ms** |
| Kannala-Brandt precision | 99.145 % | **99.231 %** |
| Kannala-Brandt corner RMSE | 0.3056 px | **0.2847 px** |
| Kannala-Brandt reprojection RMSE | 0.2899 px | **0.2720 px** |
| Kannala-Brandt translation p99 | 0.0439 m | **0.0347 m** |
| Kannala-Brandt rotation p99 | 1.2669 deg | **1.0241 deg** |
| Kannala-Brandt latency | 16.50 ms | **13.55 ms** |

Every pinhole regression suite is **bit-identical** throughout: `regression_render_tag`,
`regression_render_tag_robustness`, `regression_board_hub`, `regression_icra2020`,
`regression_euroc`, `regression_pose_consistency_roc`, `regression_straight_space`.

Not improved, and recorded rather than hidden: mean Hamming rose (Brown-Conrady 0.0196 →
0.1053, Kannala-Brandt 0.0069 → 0.0557) and Brown-Conrady's rotation p90/p99 rose (0.2687 →
0.2977 deg, 98.57 → 103.01 deg). About half of the Hamming rise is the shared acceptance policy
— returning on the first match that clears its budget rather than searching all three scale
retries for the lowest-Hamming one, worth 0.0182 → 0.0844 on its own at unchanged recall and
corner accuracy — and keeping a cross-scale best for one route only would reintroduce exactly
the divergence this work removed. The rest, and the rotation percentiles, are consistent with
composition: corner RMSE *improved* 7 % while recall rose 2 pp, so the newly recovered tags are
not degrading the corner population, and rotation is this pipeline's ill-conditioned degree of
freedom at the high incidence angles those tags sit at. Proving that needs per-tag matching
between the two arms, which the harness does not expose; adding it is the follow-up, not an
assertion.

Segmentation, which the lens never touches, was then the dominant stage at 8.59 ms of
Brown-Conrady's 13.25 ms of instrumented spans (64.9 %). After the work above it is **4.53 ms**,
and the frame is:

| | `main` | now |
| :-- | --: | --: |
| Brown-Conrady hub | 17.96 ms | **9.73 ms** |
| Kannala-Brandt hub | 16.50 ms | **8.51 ms** |

Segmentation phase by phase, Brown-Conrady hub, measured with temporary timers inside
`label_components_lsl_opts`:

| phase | before | after |
| :-- | --: | --: |
| union | 3.219 ms | striped across workers |
| stats | 1.779 ms | survivors only |
| run grouping | 1.451 ms | fused with stats |
| root resolution | 1.420 ms | shorter walks |
| threshold + RLE scan | 0.647 ms | unchanged |
| **total** | **8.52 ms** | **4.53 ms** |

Every suite is bit-identical except `regression_board_hub`, whose six snapshots are re-blessed
for a last-digit shift: minimum-index attachment renumbers some components, which changes the
order the board LM accumulates its residuals in. Verified deterministic — three consecutive runs
agree to the 17th digit — and confined to *mean* and *std* aggregates plus one p99 that moves
from 0.4904113 deg to 0.4904126 deg; every other percentile, the tag coverage and the frame
counts are unchanged. `regression_render_tag`, the byte-stable gate, is untouched.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.9.0](docs/changelogs/v0.9.0.md) - 2026-10-04
- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
