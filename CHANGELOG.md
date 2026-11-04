# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Fixed

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

### Changed

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
