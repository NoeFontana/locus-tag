# Shipped detector profiles

Three JSON files — `standard.json`, `grid.json`, `high_accuracy.json` — are
the **single source of truth** for Locus detector configuration. They are
embedded into the Rust crate (via `include_str!`) and re-exposed to the
Python wheel through the `_shipped_profile_json` FFI hook, so `locus-core`
and `locus._profile.DetectorConfig` always read identical bytes.

If the Rust defaults and these JSONs ever disagree, **the JSON wins**.

The JSON Schema that validates them lives at workspace root in
[`schemas/profile.schema.json`](../../../schemas/profile.schema.json),
regenerated from the Pydantic model via
`tools/export_profile_schema.py`.

## Conventions

- **`snake_case` everywhere.** Every field in a profile file must match a
  declared field on the Pydantic model; unknown fields are rejected by
  `extra="forbid"` and by Rust's `#[serde(deny_unknown_fields)]`.
- **No comments in JSON.** The JSON specification has none, and per-value
  rationale is documented here, not inline. Use meaningful `name` and
  (for custom profiles) `extends` values instead.
- **Enums are named, not numbered.** `"refinement_mode": "Erf"`, not
  `"refinement_mode": 2`. The Pydantic loader also accepts integer
  discriminants for backward compatibility in tests, but shipped
  profiles always use names.
- **`decimation` and thread count are not profile fields.** They are
  per-call orchestration concerns handed to the `Detector` constructor,
  not detection logic.

## Loading a profile

```python
from locus._profile import DetectorConfig

cfg = DetectorConfig.from_profile("standard")            # shipped
cfg = DetectorConfig.from_profile_json(path.read_text()) # user-supplied
```

From Rust:

```rust
let cfg = DetectorConfig::from_profile("standard");
let cfg = DetectorConfig::from_profile_json(&text)?;
```

## Per-profile rationale

### `standard`

General-purpose recall-optimised configuration. The default choice for
most detection workloads.

Values that differ from the Rust `DetectorConfig::default()`:

| Field | Default | `standard` | Reason |
| --- | --- | --- | --- |
| `threshold.enable_sharpening` | `false` | `true` | Laplacian pre-sharpening recovers small-tag edges on low-texture scenes. |
| `quad.max_elongation` | `0.0` (off) | `20.0` | Moments gate rejects thin-line false positives before contour tracing. |
| `quad.min_density` | `0.0` (off) | `0.15` | Moments gate rejects sparse noise blobs. |

### `grid`

Touching-tag / checkerboard-grid configuration for scenes where adjacent
tags share borders (ICRA 2020 `forward/checkerboard_corners_images` and
similar).

Non-negotiable invariants that make this profile work:

| Field | `standard` | `grid` | Reason |
| --- | --- | --- | --- |
| `segmentation.connectivity` | `Eight` | `Four` | 8-connectivity merges adjacent tag regions into one component at shared borders; 4-connectivity separates them. |
| `decoder.min_contrast` | `20.0` | `10.0` | Packed tags are low-contrast; the default 20.0 causes the Otsu gate to reject valid tags at shared borders. |
| `quad.min_edge_score` | `4.0` | `2.0` | Touching borders produce weaker edge scores; this relaxed threshold prevents false negatives on interior edges. |
| `threshold.enable_sharpening` | `true` | `false` | Laplacian sharpening creates halos at shared borders, biasing the threshold and producing merged components. |

### `high_accuracy`

State-of-the-art metrology configuration. Bridges the deterministic
EdLines 2D solver to the probabilistic weighted-LM 3D solver for maximum
pose accuracy on well-isolated tags. Trades recall on packed/small tags
for precision.

| Field | Default | `high_accuracy` | Reason |
| --- | --- | --- | --- |
| `quad.extraction_mode` | `ContourRdp` | `EdLines` | EdLines produces per-corner 2×2 covariances used by the weighted pose solver. |
| `decoder.refinement_mode` | `Erf` | `None` | EdLines already yields sub-pixel corners; Erf on top degrades precision. Also: EdLines + Erf is incompatible (validator enforces). |
| `threshold.enable_sharpening` | `false` | `false` | Raw PSF is passed to the solver unfiltered. |
| `quad.max_elongation`, `quad.min_density` | `0.0` | `20.0`, `0.15` | Same moments gate as `standard`. |

The pose-tuning knobs (`pose.huber_delta_px`, `pose.tikhonov_alpha_max`,
`pose.sigma_n_sq`, `pose.structure_tensor_radius`) are held at their
defaults in this profile; sweep them against your sensor profile for
production metrology.

## Authoring a custom profile

A custom profile is any JSON document that validates against the schema.
Load via `from_profile_json`:

```python
custom = """
{
  "name": "low_light",
  "threshold": { "tile_size": 8, "enable_sharpening": true },
  "decoder":   { "min_contrast": 12.0, "max_hamming_error": 3 }
}
"""
cfg = DetectorConfig.from_profile_json(custom)
```

Unspecified fields fall back to the Pydantic defaults (which mirror the
Rust `DetectorConfig::default()`). The `extends` field is declared in
the schema but resolution is not yet implemented — inline parent values
for now.
