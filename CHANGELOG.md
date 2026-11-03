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
  - `marker_inset` declines a marker whose ring inside the border reads uniformly dark: no
    codeword of a supported family has a uniform outer ring, so that is a wider border than the
    layout. A Kalibr border read as a 1-bit layout passed the other guards about one time in
    eight and pushed the corners 0.3–0.4 px outward (+0.087 px tag-scale bias on 32 px tags;
    now within 0.005 px).
  - EuRoC leave-one-out corner median 0.278 → 0.246 px (`grid`) and 0.279 → 0.246 px
    (`standard`), recall unchanged. AprilGrid board (`standard`): rotation mean 0.0104° →
    0.0077°, p99 0.167° → 0.101°; translation mean 0.21 → 0.17 mm, p95 0.76 → 1.02 mm, p99
    5.7 → 4.2 mm. `high_accuracy`: translation mean 0.024 → 0.026 mm, p99 0.19 → 0.29 mm,
    rotation unchanged. Every other snapshot, ChArUco included, is unchanged.

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
