# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Changed

- **`helicoid` 0.0.4.** Its `SO3::from_matrix` returns Shepperd's quotients whenever they are
  within `16u` of the unit sphere, so exact rotations stay unit and the chosen pose branch's
  quaternion matches `tf_tree_math`'s accuracy (helicoid `0086`). The seven board snapshots'
  float metrics move by at most 4.2e-9 relative; every detection and count is unchanged.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.10.0](docs/changelogs/v0.10.0.md) - 2026-10-13
- [0.9.0](docs/changelogs/v0.9.0.md) - 2026-10-04
- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
