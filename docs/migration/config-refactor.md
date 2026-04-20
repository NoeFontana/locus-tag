# Migration: Flat kwargs → JSON profiles

This release collapses four parallel config paths (Rust `*_default()`,
Rust `ConfigPreset`, Python `DetectorPreset` + flat kwargs, scaffolded
Pydantic) into one: detector settings are loaded from a named profile
(`"standard"`, `"grid"`, or `"high_accuracy"`) or from a `DetectorConfig`
Pydantic model. Every other surface was deleted — there is no deprecation
window.

This document is the complete "if you had X, do Y" map.

## Python

### 1. The `DetectorPreset` enum

**Before**

```python
d = locus.Detector(preset=locus.DetectorPreset.Standard)
d = locus.Detector(preset=locus.DetectorPreset.Grid)
d = locus.Detector(preset=locus.DetectorPreset.HighAccuracy)
```

**After**

```python
d = locus.Detector(profile="standard")
d = locus.Detector(profile="grid")
d = locus.Detector(profile="high_accuracy")
```

Pass a string that matches the `name` field in the shipped JSON profile.

### 2. Flat detection kwargs (`threshold_tile_size`, `quad_min_edge_score`, etc.)

**Before**

```python
d = locus.Detector(
    threshold_tile_size=16,
    quad_min_edge_score=2.0,
    decode_mode=locus.DecodeMode.Soft,
    refinement_mode=locus.CornerRefinementMode.Edge,
)
```

**After**

```python
base = locus.DetectorConfig.from_profile("standard").model_dump()
base["threshold"]["tile_size"] = 16
base["quad"]["min_edge_score"] = 2.0
base["decoder"]["decode_mode"] = "Soft"
base["decoder"]["refinement_mode"] = "Edge"

d = locus.Detector(config=locus.DetectorConfig.model_validate(base))
```

Field groupings:

| Old flat kwarg | New location |
| --- | --- |
| `threshold_tile_size` | `threshold.tile_size` |
| `threshold_min_range` | `threshold.min_range` |
| `threshold_min_radius` / `threshold_max_radius` | `threshold.min_radius` / `threshold.max_radius` |
| `enable_sharpening`, `enable_adaptive_window` | `threshold.enable_sharpening` / `enable_adaptive_window` |
| `adaptive_threshold_constant` | `threshold.constant` |
| `adaptive_threshold_gradient_threshold` | `threshold.gradient_threshold` |
| `quad_min_area`, `quad_max_aspect_ratio` | `quad.min_area`, `quad.max_aspect_ratio` |
| `quad_min_fill_ratio`, `quad_max_fill_ratio` | `quad.min_fill_ratio`, `quad.max_fill_ratio` |
| `quad_min_edge_length`, `quad_min_edge_score` | `quad.min_edge_length`, `quad.min_edge_score` |
| `subpixel_refinement_sigma`, `upscale_factor` | `quad.subpixel_refinement_sigma`, `quad.upscale_factor` |
| `quad_max_elongation`, `quad_min_density` | `quad.max_elongation`, `quad.min_density` |
| `quad_extraction_mode` | `quad.extraction_mode` |
| `decoder_min_contrast` | `decoder.min_contrast` |
| `refinement_mode` | `decoder.refinement_mode` |
| `decode_mode` | `decoder.decode_mode` |
| `max_hamming_error` | `decoder.max_hamming_error` |
| `gwlf_transversal_alpha` | `decoder.gwlf_transversal_alpha` |
| `segmentation_connectivity`, `segmentation_margin` | `segmentation.connectivity`, `segmentation.margin` |

Per-call orchestration kwargs (`decimation`, `threads`, `families`) stay at
the `Detector.__init__` level — they describe the *call*, not the
detection logic.

### 3. `Detector.standard_config()`

**Before**

```python
cfg = locus.Detector.standard_config()
```

**After**

```python
cfg = locus.DetectorConfig.from_profile("standard")
```

### 4. Integer values for enum fields

**Before** (duck-typed)

```python
d = locus.Detector(refinement_mode=0)
```

**After** — pass a real enum instance (or a string when editing a dict):

```python
d = locus.Detector(config=locus.DetectorConfig.from_profile("standard"))
# or, inside a config dict:
base["decoder"]["refinement_mode"] = "None"  # string matching the Rust variant
```

The FFI entrypoint now refuses bare integers for typed-enum fields. This is
enforced by the `test_ffi_enum_strictness.py` tripwire.

### 5. Grid + 8-connectivity warning

**Before** — the kwarg combination emitted `UserWarning("Grid preset
relies on 4-connectivity")`.

**After** — inexpressible. The `grid` profile JSON sets
`segmentation.connectivity = "Four"`. If you want 8-connectivity, write
your own profile and own the trade-off:

```python
base = locus.DetectorConfig.from_profile("grid").model_dump()
base["segmentation"]["connectivity"] = "Eight"
d = locus.Detector(config=locus.DetectorConfig.model_validate(base))
```

### 6. `**kwargs` escape hatch on `Detector.__init__`

Gone. The constructor accepts only `profile`, `config`, `decimation`,
`threads`, and `families`. Anything else raises `TypeError`.

### 7. `locus._profile` module

Deleted. Everything moved to `locus._config`; `DetectorConfig` is also
re-exported from the package root:

```python
from locus import DetectorConfig              # preferred
from locus._config import DetectorConfig      # equivalent
```

## Rust

### 8. `DetectorConfig::standard_default()` / `grid_default()` / `high_accuracy_default()`

**Before**

```rust
let cfg = DetectorConfig::standard_default();
let cfg = DetectorConfig::grid_default();
let cfg = DetectorConfig::high_accuracy_default();
```

**After**

```rust
let cfg = DetectorConfig::from_profile("standard");
let cfg = DetectorConfig::from_profile("grid");
let cfg = DetectorConfig::from_profile("high_accuracy");
```

`from_profile` panics on an unknown name — the set is closed and mis-typing
is a programming error. For user-supplied profiles use
`DetectorConfig::from_profile_json(&str) -> Result<Self, ConfigError>`.

### 9. `ConfigPreset` test enum

Deleted. The regression harness now takes `.with_profile(&str)` or
`.with_profile_json(&str)`. The ICRA grid tuning that used to live inside
`ConfigPreset::Grid` is now a fixture at
`crates/locus-core/tests/fixtures/icra_grid.json`.

## Finding stragglers

If you have in-tree code that still references the old surface, this
sweep catches it:

```bash
rg -tpy -trs -tmd -tpyi \
  'standard_default|grid_default|high_accuracy_default|ConfigPreset|DetectorPreset|standard_config' \
  --glob '!CHANGELOG.md' \
  --glob '!docs/migration/**'
```

A match outside the migration directory or changelog is a missed call
site — migrate it using the tables above.
