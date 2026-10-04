# Shipped detector profiles

Three JSON files — `standard.json`, `grid.json`, and `high_accuracy.json` —
are the **single source of truth** for Locus detector configuration. They
are embedded into the Rust crate (via `include_str!`) and re-exposed to the
Python wheel through the `_shipped_profile_json` FFI hook, so `locus-core`
and `locus.DetectorConfig` always read identical bytes.

`standard` is the default: `DetectorConfig::default()` in Rust and a bare
`locus.DetectorConfig()` in Python equal it field for field
(`config::schema_parity_tests::default_matches_standard_profile` and
`test_bare_default_matches_standard_profile` enforce this).

`standard` and `grid` carry `quad.extraction_policy: "Static"`. Only
`high_accuracy.json` opts into the per-candidate `AdaptivePpb` router.

If the Rust defaults and these JSONs ever disagree, **the JSON wins**.

The JSON Schema that validates them lives at workspace root in
[`schemas/profile.schema.json`](../../../schemas/profile.schema.json),
regenerated from the Pydantic model via
`tools/export_profile_schema.py`.

## Conventions

- **`snake_case` everywhere.** Every field in a profile file must match a
  declared field on the Pydantic model; unknown fields are rejected by
  `extra="forbid"` and by Rust's `#[serde(deny_unknown_fields)]`.
- **Every key is optional.** A key omitted from a profile takes its
  `standard` value, identically in Rust and Python.
- **No comments in JSON.** The JSON specification has none, and per-value
  rationale is documented here, not inline.
- **Enums are named, not numbered.** `"refinement_mode": "Erf"`, not
  `"refinement_mode": 1`. The Pydantic loader also accepts integer
  discriminants, but shipped profiles always use names.
- **`decimation` and thread count are not profile fields.** They are
  per-call orchestration concerns handed to the `Detector` constructor,
  not detection logic.

## The `threshold` group

`threshold.mode` chooses how the per-pixel foreground threshold that
segmentation reads is built. All three shipped profiles carry
`"TileMidExtreme"`:

| `mode` | Rule | Reads |
| --- | --- | --- |
| `TileMidExtreme` | midpoint of min/max over the 3×3 tile neighbourhood | `tile_size` |
| `LocalMean` | mean of a `(2·local_mean_radius + 1)²` window, minus `clamp(round(noise_k · σ̂ₙ), 2, 20)` grey levels | `local_mean_radius`, `noise_k` |

`LocalMean` is opt-in and **changes detector output on every frame**; it
exists for scenes where the tile rule's dependence on local *extremes*
fails — textured or dark backgrounds that fuse with a marker, and uniform
regions that speckle with foreground. Its foreground is a hollow ring around
dark markers, so the filled-blob quad gates (`quad.min_fill_ratio`,
`min_density`, `max_elongation`) must stay off with it. Its one offset knob,
`noise_k` (must be > 0), scales σ̂ₙ, the frame's estimated sensor noise: a flat
pixel turns foreground with probability ≈ Φ(−k), whatever the camera. The
defaults (`local_mean_radius` 7, `noise_k` 4.0) are the measured values, so
switching the mode is enough:

```json
"threshold": { "enable_sharpening": false, "mode": "LocalMean" }
```

Run `LocalMean` **unsharpened**. The noise σ is estimated on the raw frame and
carried through the pre-filters' white-noise gain; the Laplacian sharpen alone
multiplies it by √29 ≈ 5.4, so with sharpening on the offset `k · σ` exceeds the
20-level ceiling on essentially every real sensor and degenerates to a
constant 20.

`local_mean_radius` and `noise_k` are inert under `TileMidExtreme`; the
shipped profiles carry their defaults.

## Loading a profile

```python
from locus import DetectorConfig

cfg = DetectorConfig.from_profile("standard")  # shipped
cfg = DetectorConfig.from_profile_json(path.read_text())  # user-supplied
```

From Rust:

```rust
let cfg = DetectorConfig::from_profile("standard");
let cfg = DetectorConfig::from_profile_json(&text)?;
```

## Per-profile rationale

### `standard`

General-purpose configuration and the default (`Detector()` /
`Detector::new()`). Candidates are judged by marker evidence rather than
blob shape:

| Field | `standard` | Reason |
| --- | --- | --- |
| `segmentation.connectivity` | `Four` | Where two dark squares touch at a corner (AprilGrid connectors, a tag touching dark structure diagonally) 8-connectivity fuses them through the diagonal pixels. |
| `quad.min_fill_ratio`, `quad.max_elongation`, `quad.min_density` | `0.0` (off) | The filled-blob gates reject markers merged with neighbouring structure or with hollow threshold rings although they decode. |
| `quad.extraction_mode` / `decoder.refinement_mode` | `ContourRdp` / `Erf` | Decode-first: candidates are decoded from their contour corners and only those that decode, or nearly do, are ERF-refined and re-verified. |
| `decoder.max_border_error_rate` | omitted (per family) | Each family's dark border ring may carry its codeword's own error density (one ring cell of 28 for tag36h11, none for tag16h5). |
| `decoder.corner_subpix` | `true` | Sub-pixel junction refinement of every decoded marker, calibrated against its own bit edges (undistorted cameras; markers with cells under ~3.3 px keep their seed corners). |
| `threshold.enable_sharpening` | `true` | Laplacian pre-sharpening recovers small-tag edges. |
| `quad.min_area`, `quad.min_edge_score`, `decoder.min_contrast` | `36`, `4.0`, `20.0` | `36` px² is one pixel per bit on the smallest supported family (6×6 cells). |

### `grid`

`standard` with the filled-blob gates kept and the contrast floors relaxed,
for touching-tag / checkerboard-grid scenes where adjacent tags share
borders (ICRA 2020 `forward/checkerboard_corners_images` and similar):

| Field | `standard` | `grid` | Reason |
| --- | --- | --- | --- |
| `quad.min_fill_ratio` | `0.0` | `0.10` | Filled-blob gates reject the sparse and thin components a dense grid produces. |
| `quad.max_elongation` | `0.0` | `20.0` | |
| `quad.min_density` | `0.0` | `0.15` | |
| `decoder.min_contrast` | `20.0` | `10.0` | Packed tags are low-contrast; 20.0 rejects valid tags at shared borders. |
| `quad.min_edge_score` | `4.0` | `2.0` | Touching borders produce weaker edge contrast; the relaxed floor prevents false negatives on interior edges. |
| `threshold.enable_sharpening` | `true` | `false` | Laplacian sharpening creates halos at shared borders, biasing the threshold and merging components. |

Everything else (4-connectivity, ContourRdp + Erf decode-first, per-family
ring check, `corner_subpix`) is as in `standard`.

### `high_accuracy`

Pose-precision configuration for large, well-resolved markers. EdLines
whole-edge corners feed the weighted-LM pose solver; `AdaptivePpb` routing
sends small candidates (below ~2.5 pixels per bit) to `ContourRdp + Erf`
instead of failing inside EdLines.

| Field | `standard` | `high_accuracy` | Reason |
| --- | --- | --- | --- |
| `quad.extraction_policy` | `Static` | `AdaptivePpb(2.5, ContourRdp+Erf, EdLines+None)` | Per-candidate routing: small tags use ContourRdp/Erf, large tags EdLines/None (sub-pixel Gauss-Newton corners). |
| `quad.extraction_mode` | `ContourRdp` | `EdLines` | Inert under `AdaptivePpb` (the routes decide); kept consistent with the high route. |
| `quad.edlines_imbalance_gate` | `"Disabled"` | `"Enabled"` | AXIS→DIAG rescue: when one boundary arc is > 40 % and another < 16 % (two corners collapsing onto the same extremal on near-axis-aligned tags), re-partition along the diagonals. Distorted cameras never reach EdLines (`AdaptivePpb` falls back to ContourRdp there). |
| `decoder.refinement_mode` | `Erf` | `None` | Required by `AdaptivePpb`: the routes carry their own refinement. |
| `quad.min_area` | `36` | `400` | Suppresses small textured-quad false positives; the profile targets large markers. |
| `quad.min_fill_ratio`, `quad.max_elongation`, `quad.min_density` | off | `0.10`, `20.0`, `0.15` | Filled-blob gates on. |
| `decoder.max_hamming_error` | per family | `1` | Tightened against false positives at Hamming distance 2. |
| `decoder.max_border_error_rate` | per family | `1.0` | Border-ring check off. |
| `decoder.corner_subpix` | `true` | `false` | EdLines' whole-edge corners are not re-refined. |
| `segmentation.connectivity` | `Four` | `Eight` | Joins borders thinner than a pixel on the diagonal. |
| `threshold.enable_sharpening` | `true` | `false` | The raw PSF is passed to the corner solvers unfiltered. |
| `pose.pose_consistency_fpr` | `0.0` (off) | `0.001` | χ² gate active to catch IPPE branch ambiguity; a decisive branch winner (alternate ≥ 5× primary d²) bypasses it so single-scene noise outliers are not nulled. |
| `pose.pose_consistency_gate_sigma_px` | `1.0` | `0.5` | Tighter gate σ for clean high-PPB corner residuals. |
| `pose.outlier_drop_d2_threshold` | `0.0` (off) | `25.0` | Drop a single catastrophic corner (≈ 5σ²) and re-solve on the other three when that improves their fit. |
| `pose.pose_edge_refinement_enabled` | `false` | `true` | Refine the pose against the decoded marker's internal bit edges (Accurate mode). |

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

Unspecified fields take their `standard` values (the Rust
`DetectorConfig::default()`), on both sides of the FFI.
