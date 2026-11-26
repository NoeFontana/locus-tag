# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Changed

- **AprilGrid corners are polished over each pixel's aperture, along sides bent as the lens
  bends them.** The X-junction fit compared its erf·erf model with each pixel at the pixel's
  centre, and drew straight lines. Both cost more than the noise does. Once a marker's four
  fits are accepted, each now takes one Gauss–Newton step of a model that
  - averages each line's profile over the pixel exactly: along a line's normal the pixel
    projects to the convolution of two boxes, and `erf` convolved with both has a closed form.
    The covariance of the two profiles over the pixel is kept to first order;
  - bends each line by its side's curvature. A lens images a side as an arc, and the two fits
    at its ends report the arc's tangents, so their turn over the distance between the two
    windows is the curvature: no pixel is read for it. Only half the difference between opposite sides is used, which
    drops what the sides share (ink spread reads as every side bowing alike).

  The gates still judge the centre-sampled fit, so which markers are fitted is unchanged, and
  that fit is bit-identical to before. All four corners of a marker are polished or none: a
  step that does not factor, moves a corner more than 0.5 px or off the image, or lands where
  the marker model no longer accepts it, leaves the marker its four fits. AVX2+FMA kernel with
  a portable reference. Measured:
  - **Exact renders through a 195° fisheye** (each pixel the box ⊗ Gaussian-PSF integral of
    the board, sensor noise; 24 frames of 36 tags per case), as the median corner error over
    the Cramér–Rao bound. Recall is identical in all 13 cases.

    | Case | before | aperture only | aperture and curvature |
    |---|---|---|---|
    | 20 px tags, image centre | 1.80 | 1.47 | 1.20 |
    | 0.2 px PSF | 4.81 | 1.73 | 1.51 |
    | 12 px tags | 1.71 | 1.16 | 1.16 |
    | 32 px tags | 3.51 | 3.23 | 1.74 |
    | 50° incidence | 1.93 | 1.64 | 1.31 |
    | 40° / 70° off axis | 2.79 / 4.12 | 2.56 / 4.02 | 1.52 / 1.86 |
    | without noise | 1.53 | 1.07 | 0.71 |
    | geometric mean, 13 cases | 2.30 | 1.79 | 1.32 |

  - **Hub AprilGrid against its corner truth** (`standard`, 5 128 tags): median corner error
    0.0085 → 0.0079 px, 95th percentile 0.063 → 0.046 px, and lower in every tag-size bin
    (under 20 px: median 0.087 → 0.045 px). These renders are pinhole, so the gain is the
    aperture's; the curvature term finds nothing and moves the median by 0.0001 px.
  - **Real images** (no corner truth; `grid` with a 2-bit border): EuRoC `cam_april` cam0
    (1 450 frames, 32 392 tags) and TUM-VI `calib-cam1` cam0 / cam1 (195° fisheye, 436
    frames each, 15 137 / 15 188 tags). The decoded tags are the same in every frame. The
    polish moves a corner by 0.008–0.011 px in the median (95th percentile 0.019 px).
    - *Consistency*: each tag's corners predicted by a homography fitted to its 8 nearest
      tags in the frame through the published lens model, median residual, with a 95 %
      bootstrap interval over frames. TUM-VI 0.0192 → 0.0185 px (−3.5 %, −4.2 to −2.9) and
      0.0189 → 0.0186 px (−2.0 %, −2.7 to −1.4), −11 % and −9 % within 20° of the axis;
      EuRoC 0.0821 → 0.0812 px (−1.1 %, −1.4 to −0.9). All of it is the aperture's. Under
      the repository's EuRoC scorer (`tools.bench.sota.score euroc`) nothing moves: recall
      99.87 %, one false positive, leave-one-out median 0.258 → 0.259 px, a residual set by
      the board and the lens model.
    - *Curvature*: a lens's bend shifts neighbouring tags alike, so no consistency measure
      sees it; only rendered truth does (above). What real images show is that the bend the
      fits measure is the lens's. The correction the curvature term applies, against κR²/6
      with κ from the published lens model, correlates at r = 0.996 with slope 0.87 on
      TUM-VI (0.008 px RMS, 0.010 px beyond 55° off axis) and at r = 0.85 on EuRoC's milder
      lens. Measuring it adds 0.4 % to the consistency residual.
  - **Snapshots.** The five AprilGrid suites are re-blessed; every other dataset suite is
    byte-identical, and every count (recall, precision, coverage, frames) is unchanged.

    | Suite | Metric | before | now |
    |---|---|---|---|
    | distortion Kannala–Brandt | corner RMSE (px) | 0.0781 | 0.0691 |
    | | p90 rotation / translation | 0.0430 / 0.0009 | 0.0306 / 0.0003 |
    | distortion Brown–Conrady | corner RMSE (px) | 0.1540 | 0.1521 |
    | | p90 rotation / translation | 0.1127 / 0.0014 | 0.0996 / 0.0011 |
    | board `high_accuracy` | mean / p95 / p99 translation (µm) | 9.75 / 29.2 / 98.4 | 7.97 / 25.1 / 71.8 |
    | | mean / p95 / p99 rotation (mdeg) | 1.23 / 4.03 / 7.49 | 1.03 / 3.27 / 4.56 |
    | board `standard` | mean / p50 / p90 translation (µm) | 88.1 / 4.82 / 67.0 | 94.1 / 4.61 / 46.2 |
    | | p95 / p99 translation (µm) | 486 / 2529 | 714 / 1481 |
    | | mean / p95 / p99 rotation (mdeg) | 3.97 / 20.9 / 63.7 | 3.75 / 26.7 / 61.5 |
    | board Förstner | mean / p95 / p99 translation (mm) | 1.702 / 8.65 / 17.9 | 1.674 / 8.90 / 17.2 |

    The `standard` board's p95 rises 47 % while its p99 falls 41 %, its p90 31 % and its
    median 4 %. The suite cannot resolve a change in that percentile: frame by frame 78 of
    149 improve and 71 worsen (geometric mean of the ratios 0.93), and a bootstrap over the
    frames puts the p95's ratio between 0.50 and 7.9 (95 %). Its tail is the frames whose
    tags are all under 29 px, with corners 0.25–0.42 px off (RMS) either way: their board
    error moves tenfold in both directions (116 → 1481 µm, 2529 → 912 µm) while their corner
    RMS falls 1–3 %, the cancellation 0.11.0's notes record for the same suite. Against
    corner truth the same detections improve in every size bin (above). The Förstner
    variant's errors are set by its saddle refiner.
  - **Cost.** Whole detection, sum of each frame's fastest pass, on an AMD EPYC-Milan host
    (8 vCPUs, `lscpu`), `--release`, one thread (`RAYON_NUM_THREADS=1`, pinned with `taskset`),
    best of interleaved rounds. Detections are identical in number.

    | | before | now | |
    |---|---|---|---|
    | hub AprilGrid 1080p, 150 frames, `standard` | 3 898 ms | 3 969 ms | +1.8 % |
    | same, `high_accuracy` | 2 236 ms | 2 280 ms | +2.0 % |
    | same, `grid` | 4 390 ms | 4 434 ms | +1.0 % |
    | 512 × 512 fisheye renders, 24 frames of 36 tags of 20 px, `grid` | 126.1 ms | 130.7 ms | +3.6 % |
    | same, 32 px tags | 168.0 ms | 174.7 ms | +4.0 % |
    | EuRoC `cam_april` cam0, 752 × 480, 1 450 frames, `grid` | 11 300 ms | 11 621 ms | +2.8 % |
    | TUM-VI `calib-cam1` cam0, 512 × 512, 436 frames, `grid` | 3 113 ms | 3 201 ms | +2.8 % |

    This is net of the faster window below, which the fit shares. On a 56 px window a
    polish takes 1.77 µs: the aperture kernel 1.21 µs, against 0.47 µs for the
    centre-sampled one, and the window 0.38 µs.
  - **Measured and kept out.** A 3 × 3 Gauss–Legendre quadrature of the model over the pixel,
    iterated to convergence: 1.87 geometric mean with straight lines (2.11 at the 0.2 px PSF,
    where its points are farther apart than the blur) for 19–26 % more detection time on the
    fisheye renders. The curvature from the whole turn of each side, without the difference
    between opposite sides: 3.76 over the bound at 0.4 mm of ink spread, against 1.47. A
    looser convergence of the centre-sampled fit before the polish: under 1 % faster.

### Performance

- **X-junction windows: one per marker, scanned four pixels at a time.** Each fit cleared
  its own 17 KB window; the fits and polishes of a marker now refill one (gathering a 57 px
  window 965 → 707 ns). The scan that fills it was scalar with three branches per pixel and
  is now an AVX2 kernel that packs the kept pixels without a branch, with the portable scan
  as its reference: 712 → 381 ns on a 56 px window. A fit goes from 5.03 to 4.60 µs with
  the shared window and 8 % further with the scan; the windows, and so the fits, are
  bit-identical (same host and build as above).
- **Pose LM stops when its model predicts no reduction.** `nielsen_lm` rejected every step whose
  predicted reduction was ≤ 1e-12 and kept going to `max_iters`; since rejections only shrink
  that prediction, the pose could no longer move. It now stops there, returning the same pose
  and `JᵀJ` bit for bit. Normal-equation builds per LM call: 7.52 → 5.04 (board suites),
  5.59 → 4.15 (render-tag); calls reaching `max_iters`: 15.8 % → 0.3 %.
- **Board RANSAC seed by an 8×8 solve.** The sample homography fixes `h33 = 1` (it is the depth
  of the sample centroid, so it is positive for any visible board) and solves 8×8 normal
  equations instead of a 9×9 eigenproblem. Board metrics move by at most 1.06e-7 relative; every
  count is unchanged, and the 7 board snapshots are re-baselined.
- Together, instructions per call (callgrind, `--release`, run-to-run spread about ±3 %):
  board `estimate()` −15.9 %, tag pose weighted −29.6 % / unweighted −49.8 %,
  `refine_soa` standard −28.1 %, high_accuracy −0.2 %.

### Fixed

- **A non-finite homography no longer yields a NaN pose.** `solve_ippe_square` returned `Some`
  NaN-filled poses for a NaN/∞ homography, since NaN fails every comparison guard as `false`; it
  now returns `None`, as for every other degenerate input. Finite inputs are untouched: every
  dataset suite is byte-identical.

### Dependencies

- **`helicoid` 0.0.11** (from 0.0.8, incorporating 0.0.9 and 0.0.10). No output changes:
  0.0.9 adds `eig_sym`, 0.0.10 adds `Jac::add`/`Jac::scale` (breaking only for external `Jac`
  impls, of which Locus has none) and the `helicoid-spline` crate, and 0.0.11 adds the `rq3`/`lq3`
  triangular factorizations. Every dataset regression suite and snapshot is unchanged.
- **Python tooling**: pydantic 2.14.2 (fixes GHSA-qpmf-fgh3-ff25 and GHSA-3r8g-cqwr-jr29; Locus
  configs use no aliases, so the stricter JSON alias handling does not apply), maturin 1.16.0,
  pytest 9.2.0, ruff 0.17.3, huggingface-hub 2.4.0 and refreshed type stubs.
- **`pyo3-stub-gen` 0.23.2**; the generated stubs are unchanged.
- **CI**: `astral-sh/setup-uv` 10.3.3 everywhere. The `setup-toolchain` composite action was still
  on v8.2.0 because dependabot only scanned `.github/workflows`; it now also scans
  `.github/actions/*`, and the composite's `enable-uv-cache` defaults to setup-uv's `auto`.
- **Hub downloads use `HF_XET_HIGH_PERFORMANCE`.** `cargo xtask data` and the SOTA `fetch` set it
  to `1` for their Python child unless it is already set. It replaces `HF_HUB_ENABLE_HF_TRANSFER`,
  which `huggingface_hub` 2.x no longer uses and warns about.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.11.0](docs/changelogs/v0.11.0.md) - 2026-11-09
- [0.10.0](docs/changelogs/v0.10.0.md) - 2026-10-13
- [0.9.0](docs/changelogs/v0.9.0.md) - 2026-10-04
- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
