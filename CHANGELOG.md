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

- Shipped JSON profiles (`crates/locus-py/locus/profiles/*.json`) — single source of truth for Rust and Python.
- `schemas/profile.schema.json` — JSON Schema exported from the Pydantic model; CI diffs it against the runtime-generated schema to catch drift.
- `locus._config.DetectorConfig` — nested Pydantic model with cross-group invariant checks.
- `DetectorConfig::from_profile_json(&str)` on the Rust side for user-supplied profiles.
- `docs/engineering/profiles.md` — rationale for the refactor.
- `docs/migration/config-refactor.md` — before/after table for every removal.
