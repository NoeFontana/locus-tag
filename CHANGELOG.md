# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Fixed

- **Split pieces get their corners back.** The seeded split's opening cuts the neck where a
  Kalibr tag touches a connector square, which chamfers that corner; the contour's vertex then
  sits at one end of the chamfer, up to ~3 px along a side, and the bit grid sampled through it
  misses. Against Kalibr's own detections on the TUM-VI calibration frames, these tags, not
  blurred ones, were the decode failures: decoded from Kalibr's corners, 93-98 % of the missed
  tags read correctly, from the detector's quad 0.3 %. A piece's vertex now moves to the
  intersection of its two sides' lines when that lies outward by more than 1 px (and at most a
  quarter side, on sides of at least 14 px). Measured:
  - TUM-VI `calib-cam1`..`cam4` (cam0), share of Kalibr's tags found: 97.10 → 98.62 %,
    95.52 → 97.48 %, 97.51 → 98.60 %, 96.37 → 97.91 %; decode failures down 58-62 %; no
    wrong ids.
  - Rendered Kalibr AprilGrids (`grid`): 70° field angle 95.5 → 96.5 %, ink 0.4 mm
    97.0 → 97.6 %, noiseless 96.9 → 97.6 %, the rest equal or up; no wrong ids.
  - Dataset suites byte-identical (only `grid` splits).

  Measured and not shipped: ISI equalisation of the cell samples (a per-tag PSF scale fitted on
  the known border, then a box-constrained deconvolution of the payload). On renders it is large
  (blur 0.8 px with truth corners 86 → 100 %, 12 px tags 72 → 96 %), but on the TUM-VI frames it
  recovers 4 % of the missed tags (about +0.1 pp), because their misses are geometric. Applying
  the side-line intersection to every quad (not only split pieces) moved ChArUco board coverage
  and percentiles for no real-data gain.

- **`quad.split_fused` (`grid`) no longer breaks rim tags into fragments.** The seeded split
  opened fused components with a 3 × 3 erosion, which also opens borders 2 px wide. At the
  fisheye rim, radial compression thins a Kalibr tag's 2-cell border to about that, so the
  erosion broke the tag's ring into cores that were no marker; 14 % of the tags at a rendered
  70° field angle were lost this way, and 5-12 % elsewhere. The erosion is now 2 × 2, the
  smallest that opens anything: it keeps strokes 2 px wide and still cuts the diagonal pinch
  where ink spread joins two squares at an X-junction (cores are labelled 4-connected). The
  core floor follows, `4·(2·cells/border − 2)` px. Measured:
  - Rendered Kalibr AprilGrid recall: baseline 89.4 → 97.2 %, 70° field angle
    77.9 → 95.3 %, 40° field angle 89.0 → 97.5 %, ink 0.4 mm 82.9 → 95.6 %, 32 px tags
    94.4 → 99.7 %, 50° incidence 45 → 67 %; no wrong ids. The recovered tags' corners match
    the others' (median 0.017-0.022 px, p90 0.037-0.050 px).
  - Real TUM-VI calibration frames, share of Kalibr's corners also found: 86-93 % → 95-97 %;
    between 0.6 and 0.8 of the half-diagonal 61-65 % → 88-94 % on `calib-cam1`/`cam2`.
    Corners more than half a corner spacing from Kalibr's rise from 0.13 % to 0.20 % of the
    matched ones, mostly where locus-tag now finds rim corners Kalibr does not.
  - EuRoC `cam_april` (`grid`): recall 99.41 → 99.52 %, false positives (1) and
    leave-one-out error unchanged.
  - Latency, 1 thread: EuRoC +0.3 % (noise), rendered baseline +4 %, rendered 70° field
    angle +12 % (23 % more tags refined).
  - Dataset suites: no `grid` snapshot moves.
- **A refused X-junction fit whose window was off-centre is refitted.** #485 refitted a fit
  that landed past the move limit. Any fit refused only by the move limit or the residual gate,
  after landing more than 0.5 px from its start, is now refitted once over a window centred where it landed, and only the refit is judged: a
  window a pixel off-centre holds the junction but its far side is mostly one wedge, so the
  model fails the residual gate. A split piece's seed sits 1 px into the neck that way.
  Rendered 70° field angle: recovered tags with a corner off by more than 0.5 px 5 → 2 of 122.
  AprilGrid board (`standard`): translation mean −7.5 %, p95 0.58 → 0.38 mm, rotation p99
  0.076° → 0.064°; `forstner` within 2 %; EuRoC (`standard`) unchanged.

- **AprilGrid corners at the frame edge and on the smallest tags are fitted, not left at
  their seeds.** Against rendered truth (Kalibr's AprilGrid), tags whose mean corner error
  exceeds 0.1 px appear only under hard geometry. In every case one corner had kept its quad
  seed, 0.4-3.5 px off, because the X-junction fit refused it:
  - Within a few pixels of the frame (the fisheye rim) the fit window left the image. The
    window is now clipped to the image. The fit stays consistent; only its robustness to a
    wrong edge profile needed the symmetric window.
  - On 12 px tags the window holds 21-30 pixels, under the 28-sample floor. The floor is now
    3 per parameter (21).
  - At 50° incidence, seeds of foreshortened tags start 1.5-2 px off, past what one window
    can vouch for. A fit that lands past the move limit is now refitted once over a window
    centred where it landed, and kept only if that refit stays within its own limit. Tags
    0.1 px inward 0.95 % → 0.47 %; every other case, EuRoC and the dataset suites unchanged.

  Measured:
  - Rendered 70° field angle: such tags 0.75 % inward / 0.94 % outward → 0.19 % / 0.19 %;
    error over the bound 3.24 → 3.09, p90 0.058 → 0.045 px.
  - Rendered 12 px tags: error over the bound 6.8 → 4.6, p90 0.196 → 0.098 px; tags
    0.1 px inward 1.5 % → 0.8 %.
  - Baseline cases unchanged.
  - AprilGrid board (`standard`): rotation mean −12 %, p99 0.080° → 0.076°, translation mean
    −5.1 %. Translation p95 rises 0.42 → 0.58 mm from two frames, by cancellation: in one,
    the small-tag corners off by more than 0.3 px fall from 91 to 26 while the board's error
    rises 0.42 → 3.28 mm, because the remaining biased corners no longer cancel.
  - `high_accuracy` rotation mean −4.9 %, translation p95 −11 %; `forstner` rotation p99 −14 %.
  - EuRoC `cam_april` (`standard`) unchanged: leave-one-out median 0.2263 → 0.2262 px, recall
    and false positives the same.

  Measured and dropped: a symmetric clip (a pixel kept only if its mirror is in frame) is
  worse at the frame edge (p90 0.059 → 0.068 px), and a move limit of 0.75-1.0 of the reach
  matches the refit's gain without requiring the final window to hold the junction. The
  remaining 70° incidence outliers are 13-16 px tags whose seeds are 3 px off on a ~1 px
  border.

- **The board's per-estimator inset needs four markers of its own class.** `BoardEstimator`
  required four inlier markers before estimating an estimator class's photometric inset, but
  counted markers with no corner in that class. Since #483 moved X-junction-fit corners into
  their own class, a class could be held by one marker's corners and still pass. It now counts
  only contributing markers, as documented. AprilGrid board (`standard`): translation mean
  0.109 → 0.104 mm, rotation p95 0.0233° → 0.0209°; `high_accuracy` rotation mean
  0.00126° → 0.00130°; ChArUco boards within 0.13 % on their means. Measured and dropped:
  a marker-clustered standard error for the inset's significance test, on the grounds that a
  marker's corners share errors. It also drops ChArUco insets that are real: ChArUco
  `high_accuracy` translation mean +28 %, rotation p99 +52 %.

  Two related corrections in the same estimator: after the board drops its worst marker it
  no longer falls back to the inset fitted with that marker when the re-estimate fails
  (unreachable on the hub boards, results identical), and the inset's significance test uses
  the unbiased variance over Kish's effective sample size, which at four corners the biased
  form understated by ~13 % (ChArUco boards move by 0.01 %).
- **The X-junction fit now reaches AprilGrid tags under the cell floor, and the board gives its
  corners their own inset class.** On the hub AprilGrid, 24 px tags (one-cell border, 3 px)
  had corners sitting at their quad seeds, 0.8 px inward on average. Seen from there, the
  X-junction test's outward probe fell short of the connector square, and the fit, though it
  converged within 0.025 px of the truth, had to move more than its 1 px limit. The test now
  reaches 0.5 px past the half-border, and the move limit is half the window's shorter reach
  when that is larger. The board pose puts fitted corners (`corner_refined` bits 4-7, new) in
  a third estimator class with no inset, instead of pooling them with the gradient-orthogonality
  corners. Measured on `aprilgrid_v1_golden`:
  - `standard`: translation mean 0.154 → 0.109 mm, p95 0.76 → 0.42 mm, p99 4.38 → 3.24 mm;
    rotation mean 0.0060° → 0.0048°, p95 0.046° → 0.023°, p99 0.084° → 0.080°. Median
    corner error on tags under 29 px 0.18 → 0.057 px.
  - `high_accuracy`: translation p95 0.041 → 0.033 mm; rotation p99 0.0148° → 0.0075°.
  - `forstner`: within 2 % except translation p99 +7 % and rotation p95 −9 %.
  - EuRoC `cam_april` (`standard`): leave-one-out median 0.229 → 0.226 px, p90 0.488 → 0.483
    px, recall and false positives unchanged. Rendered Kalibr AprilGrid unchanged.
  - Measured and dropped:
    - fitting X-junctions one corner at a time, now that seeds are reachable: per-corner error
      on small tags improves (p90 0.63 → 0.50 px), but `standard` board p95 translation rises
      0.42 → 0.60 mm, with or without the separate class;
    - an X-test probe a full border wide: it reaches the data bits on large tags;
    - a probe margin of 0.25 px: not enough;
    - a move limit of 0.35 of the reach: worse.

- **AprilGrid corners: the junction pass now reaches small tags, and the marker inset no longer
  misreads Kalibr's 2-bit border.** Measured against rendered truth (Kalibr's printed
  2-bit-border AprilGrid, `grid` profile) and on EuRoC `cam_april`:
  - Under the L-corner cell floor (~3.3 px), `decoder.corner_subpix` tries the 2 px window at a
    corner whose seed looks like an X-junction and keeps the solution only if the window is
    point-symmetric (asymmetry ≤ 0.1, read on the estimator's own patch). 20 px Kalibr tags go
    from 0.30 to 0.11 px median error, unbiased within 0.005 px.
  - `marker_inset` declines a marker whose ring inside the border reads uniformly dark, read a
    quarter cell inside the border's edge: no codeword of a supported family has a uniform outer
    ring, so that is a wider border than the layout. A Kalibr border read as a 1-bit layout
    passed the other guards about one time in eight and pushed the corners 0.3–0.4 px outward
    (+0.087 px tag-scale bias on 32 px tags; now within 0.005 px).
  - EuRoC leave-one-out corner median 0.278 → 0.243 px (`grid`) and 0.279 → 0.243 px
    (`standard`), recall unchanged. AprilGrid board (`standard`): rotation mean 0.0104° →
    0.0077°, p99 0.167° → 0.101°; translation mean 0.21 → 0.17 mm, p95 0.76 → 1.02 mm, p99
    5.7 → 4.2 mm. `high_accuracy`: translation mean 0.024 → 0.026 mm, p99 0.19 → 0.29 mm,
    rotation unchanged. Every other snapshot, ChArUco included, is unchanged.
- **`cargo deny` and `cargo fmt` gates green again.** `askama` (build-only) drops its default
  `config`/`urlencode` features, removing the unmaintained `basic-toml` (RUSTSEC-2026-0374); the
  generated `dictionaries.rs` is byte-identical. The `paste` advisory ignore is gone, since
  `nalgebra` 0.35 no longer pulls it into the production graph.

### Added

- **Recall autopsy** (`examples/recall_autopsy.rs`, `bench-internals`). For each ground-truth tag,
  the first stage that lost it: border not foreground, component too small, fused with another
  tag or other structure, no quad, decode failed, wrong id. Truth from a hub dataset or a JSONL
  file. On a render of Kalibr's printed AprilGrid it splits every lost tag between decoding
  (13–95 % per case, mostly 3–9 bits off) and ink-spread fusion (2–67 %); see
  `docs/engineering/benchmarking/recall_autopsy_20261104.md`.
- **`decoder.marker_border_bits`** (default 1; OpenCV's `markerBorderBits`, Python
  `DecoderConfig.marker_border_bits`, 1–3): the printed black-border width. The detected quad
  is the border's outline, so a marker printed with a wider border had its bits sampled
  off-centre. Kalibr's AprilGrid prints tag36h11 with 2 border bits, and the 10-cell tag decoded
  only at the 0.9 scale retry, about 0.3 bit from its cells' edges. Declared, the payload is
  sampled through the outline homography composed with `diag(s, s, 1)`, `s = (d + 2)/(d + 2b)`
  (exact under perspective). The scale retries apply to that lattice, the border evidence
  covers all `(d + 2b)² − d²` border cells, and the corner pass and marker inset size their
  windows and layout from the border.
  - On a physically based render of Kalibr's printed AprilGrid (TUM-VI photometry, `grid`,
    `marker_border_bits = 2`), against the rendered truth:
    - recall 0.536 → 0.780 at the baseline (Kalibr's extractor 0.788);
    - 0.685 → 0.912 without ink, 0.161 → 0.522 at 0.8 px blur, 0.006 → 0.300 on 12 px tags;
    - error over the Cramér–Rao bound 15.5 → 13.7 (baseline) and 13.0 → 9.2 (0.8 px blur);
    - tag-scale bias within the ±0.01 px target in every case with blur ≤ 0.8 px and ink
      ≤ 0.2 mm, except 70° incidence (−0.020 ± 0.027 px over ~170 corners).
  - EuRoC `cam_april` (`grid`): recall 88.0 → 88.3 %, one extra false positive (a correct id
    5.2 px off the reference outline). Leave-one-out corner median 0.2423 → 0.2443 px.
    1-thread latency +12 % (5.72 → 6.42 ms, serial A-B-B-A).
  - No shipped profile sets it, and with the default every dataset snapshot is byte-identical.
- **`quad.split_fused`** (default `false`, on in `grid`; Python `QuadConfig.split_fused`): split
  components fused at their corners. Ink spread and blur close the diagonal contact where two
  dark squares meet corner to corner, and on Kalibr's AprilGrid every tag meets its neighbours'
  connector squares that way. One component then holds a tag and its connectors, or several
  tags, and its outline is no single marker's: it either decodes nothing or traces no quad.
  After the decode pass, each such component able to hold two markers is eroded by a 3 × 3
  square, which opens the one-to-two-pixel necks. Every core of at least a marker's size is
  dilated back (a morphological opening, labelled per core), and the pieces are traced and
  decoded as candidates appended after the first pass, so they never displace one. The whole
  pass runs on runs, so its cost follows the component's run count, not its bounding box.
  - On the physically based render of Kalibr's AprilGrid (`grid`, `marker_border_bits = 2`),
    against the rendered truth:
    - recall 0.780 → 0.894 at the baseline (Kalibr's extractor 0.788);
    - 0.361 → 0.829 at 0.4 mm ink (Kalibr 0.736), 0.438 → 0.944 on 32 px tags (0.904);
    - 0.389 → 0.725 at 70° field angle (0.705), 0.853 → 0.991 at 0° incidence (0.992).
    Locus now matches or exceeds Kalibr's recall in every case. Bias stays within the
    ±0.01 px target (0.4 mm ink moves inside it, +0.0106 → +0.0075 px). Error over the
    Cramér–Rao bound is unchanged or slightly lower.
  - Real TUM-VI calibration frames (`grid`, `marker_border_bits = 2`), against Kalibr's corners
    on the same frames, over eight recordings and cameras:
    - share of Kalibr's corners also found: 0.29–0.52 → 0.86–0.93;
    - at 0.6–0.8 of the half-diagonal: 0.09–0.29 → 0.61–0.85.
    The localization-outlier rate is unchanged at about 0.14 %. The tag-scale offset against
    Kalibr is unchanged except on `calib-cam4` (+0.002 → +0.014 px).
  - EuRoC `cam_april` (`grid`): recall 88.0 → 99.4 %, leave-one-out corner median 0.2424 →
    0.2461 px, since the recovered tags are the hard ones. One false positive (0 → 1): tag 3 in
    the frame where `marker_border_bits = 2` already had its one, 5.1 px off the reference.
    With `marker_border_bits = 2`, recall is 88.3 → 99.7 %, with false positives 1 → 2.
  - Cost: about 30 % of detect-to-decode time on EuRoC frames (in-process timing; the
    serial A-B-B-A wall clock gave +24–28 % under external load). On textured frames it is
    33–36 % and finds nothing, which is why `standard` and `high_accuracy` leave it off. Every
    dataset snapshot is byte-identical.
  - Measured and not shipped:
    - running the split on every component: textured frames fill the candidate cap with
      pieces and push real tags out (ICRA grid recall 0.67 → 0.26);
    - geodesic regrowth over the bounding box: up to 2.6 pt more recall on the renders, but
      up to 424 ms on a frame-sized texture component, where the opening on runs stays
      bounded by the run count;
    - a 5 × 5 erosion: it opens small markers' own borders;
    - requiring two cores: a tag fused only to its connectors has one;
    - skipping a lone piece that keeps ≥ 90 % of its component: −5 % cost, but it loses up
      to 0.9 pt of recall.

### Changed

- **Decoding accepts a codeword past the Hamming budget when the marker's evidence supports
  it.** Hard bits discard how sure each one is. On small or blurred markers blur flips the bits
  next to a bright neighbour, but those sit near their thresholds, while a non-marker
  contradicts a codeword with confident bits or a bright border. A match within half the
  family's minimum distance (5 bits for 36h11, where the nearest codeword is unique) now stands
  when the confidence of the disagreeing bits plus the bright border cells is at most 2 % of
  the marker's total; decode-first matches are verified on the refined quad by the same rule.
  It applies only where random bits rarely reach that radius at all (it lies within the
  family's recovery window: 36h11 and ArUcoMip36h12, not tag16h5), so the evidence is a
  second, independent guard, and never to an in-budget match whose border ring failed.
  The 2 % comes from rendered Kalibr AprilGrids against 102 595 candidates from tag16h5 and
  ChArUco renders: negatives never fell below 3.3 %, true tags' median is 0.5-1.4 %. An
  explicit `decoder.max_hamming_error` still caps it. Measured:
  - Rendered Kalibr AprilGrid recall (`grid`): blur 0.8 px 63.8 → 75.2 %, 12 px tags
    31.9 → 45.6 %, 50° incidence 67.4 → 71.2 %, ink 0.4 mm 95.6 → 97.0 %, baseline
    97.2 → 97.8 %; no wrong ids.
  - ICRA `forward` recall +1.0 pp (`grid`), +0.7 pp (`standard`), no extra detections;
    AprilGrid board tag coverage 0.9917 → 0.9924. The tag16h5, ChArUco and render-tag
    suites are byte-identical.
  - EuRoC `cam_april`: false positives unchanged (`grid` 1, `standard` 0).
  - Latency, 1 thread: EuRoC −0.5 %, rendered blur 0.8 −7 % (an accepted candidate skips the
    remaining scales and recovery).

- **AprilGrid X-junctions are located by a model fit (`decoder.corner_subpix`).** A marker whose
  four corners are X-junctions (connector squares touching them) is re-placed by a weighted
  Levenberg–Marquardt fit of `A + B·erf(n₁·(p − c)/s)·erf(n₂·(p − c)/s)` to the raw pixels of an
  elliptical window spanning the black border along each edge (`xjunction.rs`). Gradient
  orthogonality read the corner from a resampled patch, whose error depends on where the corner
  falls within its pixel; the fit reads pixel values as they are, and its point symmetry keeps
  it unbiased under any point-symmetric blur and tone curve. All four corners or none, and the
  marker inset leaves fitted corners alone. Measured:
  - Rendered Kalibr AprilGrid (TUM-VI lens and photometry, 2-bit border, `grid`): median error
    over the Cramér–Rao bound 13.7 → 3.1 at baseline; 2.6-3.4 across ink, 32 px tags, 0.8-1.2
    px blur, 50-70° incidence and 40-70° field angle; 4.4 at 0° incidence, 4.5 on 12 px tags
    and 7.4 at 0.3 px blur (from 7.5-18.8 before). Kalibr's own extractor reads 13-14. Recall
    unchanged; bias within ±0.01 px except at 70° incidence, where 32 corners give a −0.05 ±
    0.02 px tag-scale bias (−0.02 ± 0.03 before).
  - EuRoC `cam_april` leave-one-out median 0.242 → 0.228 px (`grid`) and 0.243 → 0.229 px
    (`standard`), p90 0.53 → 0.49 px; recall and false positives unchanged.
  - AprilGrid board (`high_accuracy`): translation mean 0.026 → 0.010 mm, p95 0.104 → 0.041
    mm, p99 0.29 → 0.10 mm; rotation p99 0.0186° → 0.0148°. `standard`: translation mean 0.166
    → 0.154 mm, p95 1.02 → 0.76 mm, p99 4.23 → 4.38 mm; rotation p99 0.101° → 0.084°. Per-corner
    error on 29-60 px tags 0.035 → 0.010 px, on larger tags 0.034 → 0.007 px. `forstner`
    moves within 4 %. Tags under 29 px
    whose seeds sit ~0.8 px off stay at their seeds: replacing corners one at a time instead
    left such tags half fitted and raised `standard`'s p95 translation 28 %.
  - Cost: an AVX2+FMA kernel (runtime-detected; portable fallback) keeps dense small-tag
    AprilGrid frames at +12 % detect time and EuRoC within noise (+1.8 %), single thread.
  - Measured and dropped: a curvature term per line for lens bow (median error +60 %),
    box-integrating the model over each pixel (no change), robust residual weights (worse),
    four free levels (no change).

- **`helicoid` 0.0.4.** Its `SO3::from_matrix` returns Shepperd's quotients whenever they are
  within `16u` of the unit sphere, so exact rotations stay unit and the chosen pose branch's
  quaternion matches `tf_tree_math`'s accuracy (helicoid `0086`). The seven board snapshots'
  float metrics move by at most 4.2e-9 relative; every detection and count is unchanged.
- **`helicoid` 0.0.8.** Updated dependencies `helicoid` and `helicoid-linalg` to 0.0.8 (incorporating 0.0.6, 0.0.7, and 0.0.8).
  - Adopts `SE3::inv_act` and `SO3::inv_act` for inverse point action $R^\top(p - t)$ without catastrophic cancellation; exposed as [`Pose::inverse_transform_point`] and [`Pose::transform_point`].
  - Adopts Frobenius-nearest polar rotation projection onto SO(3) via `helicoid_linalg::nearest_rotation`; exposed as [`Pose::from_matrix_nearest`] and [`pose::nearest_rotation`].
  - Integrates `helicoid`'s single-rounding exact SO(3) logarithm and portable `Real::atan2` kernel.
- **`helicoid` 0.0.5.** An accuracy and speed release with no API break: the SO(3)
  coefficients and `Log`, `Quat::to_matrix`'s diagonal, and the SE(3) `Exp` and Jacobians
  change in their last bits. The seven board snapshots' float metrics move by at most 1.9e-9
  relative (67 values); every detection and count is unchanged.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.10.0](docs/changelogs/v0.10.0.md) - 2026-10-13
- [0.9.0](docs/changelogs/v0.9.0.md) - 2026-10-04
- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
