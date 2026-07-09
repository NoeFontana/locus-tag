# Config refactor — migration notes

This document covers the breaking changes in the current `Unreleased`
section of the project `CHANGELOG.md` (at the repository root). Each
removal lists the before/after spelling so consumers can mechanically
port their call sites.

## `DetectorPreset` enum → `profile=` string

Before:

```python
import locus
detector = locus.Detector(preset=locus.DetectorPreset.Standard)
```

After:

```python
detector = locus.Detector(config=locus.DetectorConfig.from_profile("standard"))
```

Valid profile names: `"standard"`, `"grid"`, `"high_accuracy"`.
Custom JSON profiles round-trip through
`DetectorConfig.from_profile_json(text)`.

## Flat detector kwargs → nested `DetectorConfig`

Before:

```python
detector = locus.Detector(
    threshold_tile_size=24,
    quad_min_edge_score=20.0,
    refinement_mode=locus.CornerRefinementMode.Erf,
)
```

After:

```python
cfg = locus.DetectorConfig.from_profile("standard").model_copy(deep=True)
cfg.threshold.tile_size = 24
cfg.quad.min_edge_score = 20.0
cfg.decoder.refinement_mode = locus.CornerRefinementMode.Erf
detector = locus.Detector(config=cfg)
```

`Detector.__init__` no longer takes the `**kwargs` escape hatch; unknown
keywords raise `TypeError`. The pre-typed PyO3 enums no longer accept
integer values implicitly — pass the enum instance, or the variant name
as a string when editing a JSON-shaped dict.

## `Detector.standard_config()` removed

Use `locus.DetectorConfig.from_profile("standard")`.

## `locus._profile` module removed

`DetectorConfig` lives in `locus._config` and is re-exported from the
package root. Imports of `locus._profile` should switch to
`locus.DetectorConfig` directly.

## Rust `DetectorConfig::*_default()` removed

Before:

```rust
let cfg = locus_core::DetectorConfig::high_accuracy_default();
```

After:

```rust
let cfg = locus_core::DetectorConfig::from_profile("high_accuracy")?;
// or:
let cfg = locus_core::DetectorConfig::from_profile_json(include_str!("..."))?;
```

`ConfigPreset` (the test-only enum) is also gone — regression harnesses
now take `.with_profile(&str)` / `.with_profile_json(&str)`.

## `DecodeMode::Soft` removed

The `DecodeMode` enum (Rust `locus_core::config::DecodeMode` and Python
`locus.DecodeMode`) and the `decode_mode` field on `DetectorConfig`,
`DetectorConfigBuilder`, the shipped JSON profiles, and the public
Python API are gone. Hard-decision Hamming is the only path.

Migration: drop any `decode_mode=` argument; if you relied on
`DecodeMode::Soft` for ICRA-style noisy decodes, accept that the
detector now uses the same hard-decision path that every shipped
profile already used (Soft was opt-in only on a deleted synthetic
fixture).

The historical Soft-vs-Hard analysis is preserved in
`docs/engineering/benchmarking/lessons.md` as a record-of-decisions.

## `PoseEstimationMode` removed

The `Fast` / `Accurate` toggle is gone. The pose estimator now selects
its cost surface automatically from per-corner covariance availability:
image view (or external GWLF covariances) routes to weighted
Mahalanobis LM with a 6×6 covariance output; no image and no
covariances routes to unweighted Huber LM with no covariance.

Migration:

```diff
- detections = detector.detect(img, pose_estimation_mode=locus.PoseEstimationMode.Accurate)
+ detections = detector.detect(img)
```

`tools/cli.py`'s `--pose-mode` flag is removed in the same change.

## `CornerRefinementMode::Edge` removed

`Edge` (the gradient-peak `fit_edge_line`-only refiner) is gone. No
shipped profile referenced it; the variant existed only as a public
surface with no production consumer. The PyO3 enum is now
`{None=0, Erf=1, Gwlf=2}` (dense integer values; the Rust enum order
matches).

Migration:

```diff
- refinement_mode=locus.CornerRefinementMode.Edge
+ refinement_mode=locus.CornerRefinementMode.Erf
```

If you depended on Edge's specific failure-mode characteristics, note
that `Erf` already falls back to the same gradient-peak fit
(`fit_edge_line`) internally on sample shortfall or low contrast —
choosing `Erf` is a strict superset of the old `Edge` path.

If your config used `Edge` simply to skip the PSF Gauss-Newton cost on
high-volume scenes, `None` is the right choice with `EdLines`
extraction (whose GN corners are already sub-pixel — see
`docs/engineering/benchmarking/lessons.md §4.1`).
