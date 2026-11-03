# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Fixed

- **AprilGrid corners: the junction pass now reaches small tags, and no longer inherits a
  misread layout's inset.** Measured against rendered truth (Kalibr's printed 2-bit-border
  AprilGrid, `grid` profile) and on EuRoC `cam_april`:
  - `decoder.corner_subpix` refines an X-junction below the L-corner cell floor (~3.3 px) when
    its window is point-symmetric. 20 px Kalibr tags go from 0.30 to 0.11 px median error, and
    16–20 px tags on the hub's one-cell-border grid from 0.43–0.53 to 0.06–0.09 px.
  - The marker-inset calibration is discarded when it moves the marker's refined X-junctions
    more than 0.15 px RMS. A Kalibr 2-bit border read as a 1-bit layout passed its guards about
    one time in eight and pushed the corners 0.3–0.4 px outward (+0.087 px tag-scale bias over
    the refined corners of 32 px tags; now +0.000).
  - EuRoC leave-one-out corner median 0.278 → 0.244 px (`grid`) and 0.279 → 0.244 px
    (`standard`), recall unchanged. AprilGrid board (`standard`): rotation mean 0.0104° →
    0.0085°, p99 0.167° → 0.097°; translation mean 0.21 → 0.18 mm, p95 0.76 → 0.86 mm.
    `high_accuracy` translation p99 0.19 → 0.42 mm, from two frames whose corners improve
    (bias −0.09 → −0.04 px and −0.09 → −0.02 px). Every other snapshot is unchanged.
- **`cargo deny` and `cargo fmt` gates green again.** `askama` (build-only) drops its default
  `config`/`urlencode` features, removing the unmaintained `basic-toml` (RUSTSEC-2026-0374); the
  generated `dictionaries.rs` is byte-identical. The `paste` advisory ignore is gone, since
  `nalgebra` 0.35 no longer pulls it into the production graph.

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
