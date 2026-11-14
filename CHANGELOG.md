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
    at its ends report the arc's tangents, so their turn over the chord is the curvature: no
    pixel is read for it. Only half the difference between opposite sides is used, which
    drops what the sides share (ink spread reads as every side bowing alike).

  The gates still judge the centre-sampled fit, so which markers are fitted is unchanged, and
  that fit is bit-identical to before. AVX2+FMA kernel with a portable reference. Measured:
  - **Exact renders through a 195° fisheye** (each pixel the box ⊗ Gaussian-PSF integral of
    the board, sensor noise; 24 frames of 36 tags per case), as the median corner error over
    the Cramér–Rao bound. Recall is identical in all 13 cases.

    | Case | before | aperture only | aperture and curvature |
    |---|---|---|---|
    | 20 px tags, image centre | 1.80 | 1.47 | 1.20 |
    | 0.2 px PSF | 4.81 | 1.73 | 1.51 |
    | 12 px tags | 1.71 | 1.16 | 1.17 |
    | 32 px tags | 3.51 | 3.23 | 1.73 |
    | 50° incidence | 1.93 | 1.64 | 1.32 |
    | 40° / 70° off axis | 2.79 / 4.12 | 2.56 / 4.02 | 1.52 / 1.86 |
    | without noise | 1.53 | 1.07 | 0.71 |
    | geometric mean, 13 cases | 2.30 | 1.79 | 1.32 |

  - **Hub AprilGrid against its corner truth** (`standard`, 5 128 tags): median corner error
    0.0085 → 0.0079 px, 95th percentile 0.063 → 0.046 px, and lower in every tag-size bin
    (under 20 px: median 0.087 → 0.046 px). These renders are pinhole, so the gain is the
    aperture's; the curvature term finds nothing and moves the median by 0.0001 px.
  - **Snapshots.** The five AprilGrid suites are re-blessed; every other dataset suite is
    byte-identical, and every count (recall, precision, coverage, frames) is unchanged.

    | Suite | Metric | before | now |
    |---|---|---|---|
    | distortion Kannala–Brandt | corner RMSE (px) | 0.0781 | 0.0690 |
    | | p90 rotation / translation | 0.0430 / 0.0009 | 0.0306 / 0.0003 |
    | distortion Brown–Conrady | corner RMSE (px) | 0.1540 | 0.1521 |
    | | p90 rotation / translation | 0.1127 / 0.0014 | 0.0993 / 0.0011 |
    | board `high_accuracy` | mean / p95 / p99 translation (µm) | 9.75 / 29.2 / 98.4 | 7.96 / 25.1 / 73.7 |
    | | mean / p95 / p99 rotation (mdeg) | 1.23 / 4.03 / 7.49 | 1.03 / 3.27 / 4.61 |
    | board `standard` | mean / p50 translation (µm) | 88.1 / 4.82 | 93.9 / 4.59 |
    | | p95 / p99 translation (µm) | 486 / 2529 | 701 / 1439 |
    | | mean / p95 / p99 rotation (mdeg) | 3.97 / 20.9 / 63.7 | 3.74 / 26.7 / 61.2 |
    | board Förstner | mean / p95 / p99 translation (mm) | 1.702 / 8.65 / 17.9 | 1.675 / 8.90 / 17.2 |

    The `standard` board's p95 rises 44 % while its p99 falls 43 % and its median 5 %. The
    tail is heavy (median 4.6 µm, p95 0.7 mm: the eight worst frames of 150), and its
    percentiles move with any change to those frames' corners, as 0.11.0's notes record for
    the same suite. Against corner truth the same detections improve in every size bin
    (above). The Förstner variant's errors are set by its saddle refiner.
  - **Cost.** Whole detection, sum of each frame's fastest pass, on an AMD EPYC-Milan host
    (8 vCPUs, `lscpu`), `--release`, one thread (`RAYON_NUM_THREADS=1`, pinned with `taskset`),
    best of interleaved rounds. Detections are identical in number.

    | | before | now | |
    |---|---|---|---|
    | hub AprilGrid 1080p, 150 frames, `standard` | 3 924 ms | 4 018 ms | +2.4 % |
    | same, `high_accuracy` | 2 238 ms | 2 325 ms | +3.9 % |
    | same, `grid` | 4 418 ms | 4 498 ms | +1.8 % |
    | 512 × 512 fisheye renders, 24 frames of 36 tags of 20 px, `grid` | 127.3 ms | 135.2 ms | +6.3 % |
    | same, 32 px tags | 169.7 ms | 180.8 ms | +6.5 % |

    One fit takes 4.60 µs and its polish 2.15 µs on a 57 px window (the aperture kernel
    1.30 µs, against 0.50 µs for the centre-sampled one).
  - **Measured and kept out.** A 3 × 3 Gauss–Legendre quadrature of the model over the pixel,
    iterated to convergence: 1.87 geometric mean with straight lines (2.11 at the 0.2 px PSF,
    where its points are farther apart than the blur) for 19–26 % more detection time on the
    fisheye renders. The curvature from the whole turn of each side, without the difference
    between opposite sides: 3.76 over the bound at 0.4 mm of ink spread, against 1.47. A
    looser convergence of the centre-sampled fit before the polish: under 1 % faster.

### Performance

- **X-junction fits reuse one pixel window per marker.** Each fit cleared its own 17 KB
  window; the fits and polishes of a marker now refill one. Gathering a 57 px window goes from
  965 to 707 ns and a whole fit from 5.03 to 4.60 µs (same host and build as above), with
  bit-identical fits.
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
