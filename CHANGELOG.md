# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### BREAKING CHANGES

Detector configuration has been unified around JSON profiles. Four
overlapping config paths have been collapsed into one; migration is
documented in [docs/migration/config-refactor.md](docs/migration/config-refactor.md).
The public version number is not bumped in this release — the bump will
be batched with other breaking changes in a later cleanup PR.

- **Removed** `DetectorPreset` enum; pass `profile="standard" | "grid" | "high_accuracy"` as a string.
- **Removed** flat detector kwargs on `Detector.__init__` (`threshold_tile_size`, `quad_min_edge_score`, `refinement_mode`, etc.). Use `locus.DetectorConfig.from_profile(...)` and edit the nested model.
- **Removed** `Detector.standard_config()`; use `locus.DetectorConfig.from_profile("standard")`.
- **Removed** the `**kwargs` escape hatch on `Detector.__init__`. Unknown kwargs now raise `TypeError`.
- **Removed** duck-typed integer conversion for typed PyO3 enums at the FFI boundary. Pass the real enum instance, or a string variant name when editing a JSON-shaped dict.
- **Removed** Grid + 8-connectivity `UserWarning` branch. The grid profile pins 4-connectivity by construction.
- **Removed** `locus._profile` module. `DetectorConfig` lives in `locus._config` and is re-exported from the package root.
- **Removed** Rust `DetectorConfig::standard_default() / grid_default() / high_accuracy_default()`. Use `DetectorConfig::from_profile("standard" | "grid" | "high_accuracy")` or `DetectorConfig::from_profile_json(&str)`.
- **Removed** Rust `ConfigPreset` test enum. The regression harness now takes `.with_profile(&str)` / `.with_profile_json(&str)`.

### Added

- Shipped JSON profiles (`crates/locus-core/profiles/*.json`) — single source of truth for Rust and Python. Embedded into Rust via `include_str!` and re-exposed to the Python wheel through the `_shipped_profile_json` FFI hook so both readers see byte-identical input.
- `schemas/profile.schema.json` — JSON Schema exported from the Pydantic model; CI diffs it against the runtime-generated schema to catch drift.
- `locus._config.DetectorConfig` — nested Pydantic model with cross-group invariant checks.
- `DetectorConfig::from_profile_json(&str)` on the Rust side for user-supplied profiles.
- `docs/engineering/profiles.md` — rationale for the refactor.
- `docs/migration/config-refactor.md` — before/after table for every removal.
- **Pose-consistency gate** (`pose.pose_consistency_fpr`, default `0.0` = disabled): final reprojection-consistency check inside `estimate_tag_pose_with_config` using Mahalanobis distance with χ²-derived critical values. Aggregate gate at χ²(2; fpr); per-corner gate at χ²(1; fpr). Rejected poses surface as `Detection.pose = None`. Only `high_accuracy` and `render_tag_hub` opt in (both at `1.0e-3`); `standard`, `grid`, and `general` profiles are byte-identical. Mahalanobis-aware IPPE branch selection swaps to the alternate branch when its aggregate `d²` is at least 2× better AND statistically defensible — gated on `fpr > 0` so disabled profiles preserve the legacy ideal-corner Euclidean comparison.
- `bench-internals` SoA telemetry columns: `pose_consistency_d2`, `pose_consistency_d2_max_corner`, `ippe_branch_d2_ratio`. NaN sentinel when the gate did not run.
- `crates/locus-core/tests/regression_pose_consistency_roc.rs` — synthetic-isotropic ROC sweep with a hard `realized FPR ∈ [1e-4, 1e-2]` acceptance gate at the production `fpr = 1e-3`. Snapshot lives at `regression_pose_consistency_roc__pose_consistency_roc__synthetic_isotropic.snap`.
- `docs/engineering/track2_precision_threshold.md` — calibration rationale, ROC harness usage, and the empirical-fallback procedure if the χ²(2) tail breaks on a future codebase change.

### Fixed

- **Quad-extraction truncation order**: reverted the pre-filter `pixel_count` truncation introduced in `5a2f438` that capped `component_stats` at `MAX_CANDIDATES` *before* per-component geometric filtering. On dense small-tag scenes (ICRA forward) the cap discarded tag-sized candidates in favour of large background blobs that the geometric gates would have rejected anyway. Now: filter first via `extract_single_quad`, then partition + truncate by `pixel_count` only if survivors still exceed the SoA ceiling. ICRA forward `standard` recall recovers from `0.6149` to `0.7236` (parent-commit baseline). Distortion suite improves as a side-effect — Brown–Conrady recall `0.870 → 0.935` (+6.5 pp). Render-tag SOTA results unchanged. See [docs/engineering/benchmarking/quad_truncation_fix_20260426.md](docs/engineering/benchmarking/quad_truncation_fix_20260426.md).
- **Packaging**: relocated the shipped JSON profiles from `crates/locus-py/locus/profiles/` to `crates/locus-core/profiles/` so `cargo package -p locus-core` no longer fails with "couldn't read `src/../../locus-py/locus/profiles/standard.json`: No such file or directory". Python now queries the embedded bytes through a new `_shipped_profile_json` FFI hook instead of `importlib.resources`, which means the wheel ships zero profile data files of its own — deleting one is impossible by construction.

### Internal

- Pre-release code cleanup: deleted orphan `crates/locus-core/src/homography.rs` (never declared in `lib.rs`; production `Homography` lives in `decoder.rs`), dropped unused `img_height` parameter from `extract_boundary_segments`, removed dead `sample_grid_generic` / `sample_grid_soa` helpers, removed unused `DecodingStrategy::distance` trait method, gated test/bench-only helpers behind `cfg(any(test, feature = "bench-internals"))` (or `cfg(feature = "bench-internals")` for items with no in-crate test consumers), tightened `unsafe` SAFETY comments across `filter.rs`, `threshold.rs`, `quad.rs`, and `locus-py` to cite concrete invariants (rayon disjointness, multiversion runtime checks, pyo3-numpy contracts), removed redundant inner `cfg(feature = "bench-internals")` blocks from tests already gated by `required-features` in `Cargo.toml`, and replaced a stream-of-consciousness comment in `threshold.rs` with a one-line invariant statement.
- Pre-release docs hygiene: untracked the leaked `.claude/settings.local.json` and added it to `.gitignore`; stripped an absolute developer path from `docs/engineering/erf-simd-attempt-postmortem.md`; renamed the legacy `*_sota` regression test functions and their snapshot files to align with the public preset taxonomy (`*_highaccuracy` for the EdLines + GN + None metrology preset, `*_standard` for the default `ContourRdp + Hard` preset, `*_grid` for the 4-connectivity profile), deleted four stale duplicate hub snapshots at `tests/snapshots/regression_render_tag__hub_*_sota.snap` (the live copies live under `tests/common/snapshots/...`), reconciled `sota_metrology_20260321.md` with the v0.3.1 shipped JSON profiles (Standard and Grid both ship `decode_mode: Hard` since PR #172 — the historical Soft-decode figures are now flagged as such), updated `release_performance_20260322.md`'s Hub table to use the current preset name (`HighAccuracy` rather than `SOTA`), and removed a dangling `regression_hub_tag36h11_*_sota_pure_tags` TODO sentence that had outlived its context.
