# Shipped detector profiles

Four JSON files — `standard.json`, `grid.json`, `high_accuracy.json`,
and `max_recall_adaptive.json` — are the **single source of truth**
for Locus detector configuration. They are
embedded into the Rust crate (via `include_str!`) and re-exposed to
the Python wheel through the `_shipped_profile_json` FFI hook, so
`locus-core` and `locus._profile.DetectorConfig` always read identical
bytes.

All shipped profiles except `max_recall_adaptive` carry
`quad.extraction_policy: "Static"`. Only `max_recall_adaptive.json` opts
into the per-candidate `AdaptivePpb` router — it is the only shipped
profile where that machinery runs.

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

State-of-the-art pose-precision configuration. Bridges the
deterministic EdLines 2D solver to the probabilistic weighted-LM 3D
solver via per-corner GN covariance propagation, with the EdLines
axis-aligned imbalance gate enabled to recover near-axis-aligned tags
and `AdaptivePpb` routing so tags below ~2.5 PPB fall back to
`ContourRdp + Erf` instead of failing inside EdLines. Beats AprilTag-C
on every translation percentile on `locus_v1_tag36h11_1920x1080`
(rot p99 = 0.77° vs 65°, trans p99 = 19 mm vs 54 mm).

| Field | Default | `high_accuracy` | Reason |
| --- | --- | --- | --- |
| `quad.extraction_policy` | `Static` | `AdaptivePpb(2.5, ContourRdp+Erf, EdLines+None)` | Per-candidate routing: small tags use ContourRdp/Erf (where EdLines collapses), large tags use EdLines/None (metrology-grade sub-pixel corners). |
| `quad.edlines_imbalance_gate` | `"Disabled"` | `"Enabled"` | AXIS→DIAG rescue: when one boundary arc is > 40 % and another < 16 % (the signature of two corners collapsing onto the same TRBL extremal on near-axis-aligned tags) divert to NW/NE/SE/SW partition. Safe to leave on because the upstream guard at `detector.rs:194-198` rejects EdLines under any non-trivial distortion — so the gate is unreachable on distorted inputs where it would have collapsed legitimate aprilgrid sub-tags. |
| `quad.min_area` | `36` | `800` | Tag16h5's dense codebook produces small textured-quad false positives (TP_min ≈ 1329 px² vs FP_max ≈ 1158 px² per PR #214). 800 sits cleanly between, killing the FP class without losing any tag36h11 TP. |
| `decoder.refinement_mode` | `Erf` | `None` | EdLines already yields sub-pixel corners; Erf on top degrades precision (validator-enforced incompatibility on the EdLines route). |
| `threshold.enable_sharpening` | `false` | `false` | Raw PSF is passed to the solver unfiltered. |
| `pose.pose_consistency_fpr` | `0.0` (off) | `0.001` | χ² gate active to catch IPPE branch ambiguity, with branch-ratio escape (`min_decisive_ratio = 5.0`) so single-scene noise outliers aren't lossily nulled. |
| `pose.pose_consistency_gate_sigma_px` | `1.0` | `0.5` | Tighter gate σ for clean-render PPB > 2.5 corner residuals. |

On the `locus_v1_tag36h11_*` Hub render-tag subsets this delivers
100 % recall across 480p / 720p / 1080p (2160p stays at 94 % due to
upstream segmentation fragmentation, unrelated to pose). See
`docs/engineering/benchmarking/render_tag_sota_20260425.md` for the
full A/B evaluation.

### `max_recall_adaptive`

Opt-in profile that enables the `AdaptivePpb` per-candidate router. Each
candidate quad is classified by a pixels-per-bit estimate (bbox short
side / min outer tag dimension across registered families) and routed to
one of two extraction + refinement pairings:

| Route | Pixels per bit | Extraction | Refinement | Rationale |
| --- | --- | --- | --- | --- |
| Low | `< threshold` | `ContourRdp` | `Erf` | Robust on small / blurry tags where EdLines cannot lock. |
| High | `≥ threshold` | `EdLines` | `None` | Metrology-grade sub-pixel corners for well-resolved tags. |

The shipped `threshold` value (2.5) is a Phase 1 sentinel and will be
tightened to the empirically derived cutoff produced by the render-tag
`locus_ppb_sweep_v1` dataset during Phase 0. Document your application
expectations before flipping to this profile — behaviour of existing
frames will change relative to `standard`.

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
