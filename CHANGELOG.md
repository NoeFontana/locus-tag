# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

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
