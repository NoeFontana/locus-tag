# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Performance

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
