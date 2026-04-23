# How to Use the `max_recall_adaptive` Profile

`max_recall_adaptive` is an opt-in shipped profile that enables the
**adaptive PPB router**: each candidate quad is classified by a
pixels-per-bit estimate (bbox short side / minimum outer tag dimension
across registered families) and routed to one of two extraction +
refinement pairings.

| Route | Pixels per bit | Extraction | Refinement | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| Low  | `< threshold` | `ContourRdp` | `Erf`  | Robust on small / blurry tags where EdLines cannot lock. |
| High | `≥ threshold` | `EdLines`    | `None` | Metrology-grade sub-pixel corners for well-resolved tags. |

The router runs per-candidate, so a single frame can mix close-up and
distant tags and serve each optimally. The default profile (`standard`)
can only apply one mode to every candidate in every frame.

## When to use it

- **Mixed-distance scenes** — robot arm approaching a workstation, AMR
  navigating a warehouse with tags at different stand-off distances,
  drone capturing ground + wall markers in one frame.
- **Recall-leaning workloads** — you want to accept a candidate whenever
  *either* mode would detect it, at the cost of a small per-frame
  routing overhead.

## When **not** to use it

- **Single-distance workflows.** If every tag in every frame is at
  roughly the same stand-off, pick `standard` (close / blurry) or
  `high_accuracy` (well-resolved, metrology-grade) directly — the
  router adds nothing.
- **Metrology-only workflows.** `high_accuracy` + `EdLines` + weighted
  LM is the designed path for maximum pose precision. The adaptive
  router's low route intentionally sacrifices precision for recall on
  low-PPB quads.
- **All-small-tag workflows.** If nothing in your scene reaches the PPB
  threshold, the adaptive profile behaves identically to `standard` but
  pays the routing cost. Use `standard` directly. If you need more
  recall on small tags, consider layering ROI rescue (below).

## Loading the profile

Python:

```python
from locus._config import DetectorConfig

cfg = DetectorConfig.from_profile("max_recall_adaptive")
```

Rust:

```rust
let cfg = DetectorConfig::from_profile("max_recall_adaptive");
```

The loaded `DetectorConfig` carries
`quad.extraction_policy = AdaptivePpb { threshold, low_extraction,
high_extraction, low_refinement, high_refinement }`. Shipped values are
documented in `crates/locus-core/profiles/max_recall_adaptive.json`.

## Tuning knobs

Most users should load the profile as-is. If you need to tune, clone the
JSON and adjust these fields under `quad.extraction_policy.AdaptivePpb`:

| Field | Range | What it does |
| :--- | :--- | :--- |
| `threshold` | `(1.0, 5.0)` | PPB cutoff. Higher → more candidates go to the low (ContourRdp) route. |
| `low_extraction` / `high_extraction` | `ContourRdp` / `EdLines` | Extraction mode per route. |
| `low_refinement` / `high_refinement` | `None` / `Erf` / `Gwlf` | Corner refinement per route. |

Rules the validator enforces:

- `low_extraction != high_extraction` — a degenerate policy uses `Static`.
- Neither route may pair `EdLines` with `Erf` (solvers conflict) or
  `Soft` decode (no bit-probabilities from EdLines).
- `threshold` must be finite in `(1.0, 5.0)`.
- Distortion (non-rectified) is incompatible with any `EdLines` route;
  the runtime check fires on detector construction.

## Combining with ROI super-resolution rescue

`max_recall_adaptive` ships with `rescue.enabled = false`. Enabling
rescue on top is an orthogonal knob — layer it via a custom profile
when you need to recover candidates that pass the funnel but miss the
Hamming budget on the first decode pass:

```python
from locus._config import DetectorConfig, RescueConfig, RescueInterpolation

cfg = DetectorConfig.from_profile("max_recall_adaptive")
cfg.rescue = RescueConfig(
    enabled=True,
    upscale_factor=2,
    max_roi_side_px=128,
    max_rescues_per_frame=16,
    rescue_max_hamming=1,
    require_first_pass_agreement=True,
    interpolation=RescueInterpolation.Lanczos3,
)
```

Field-level assignment skips Pydantic cross-group validation, but Rust's
`DetectorConfig::validate()` remains the final gate — an invalid
combination (e.g. `rescue_max_hamming >= decoder.max_hamming_error`) still
fails fast at detector construction.

See `docs/explanation/pipeline.md` for where rescue sits in the pipeline
(after the first decode pass, before partition + pose refinement).

## Validation before switching

Detections from `max_recall_adaptive` are a **strict superset** of
`standard` on the evaluation corpus (checked by
`crates/locus-core/tests/diff_adaptive.rs`): every tag `standard`
detects is also detected by `max_recall_adaptive`, within 0.5 px corner
deviation, with the same ID. Extras from the adaptive profile are
allowed.

Before flipping a production deployment, run both profiles in parallel
on a representative stream and diff the detection sets — the extras are
where the recall win lives *and* where any regression will show up.

## Caveats

- **Sentinel threshold.** The shipped `threshold` value (2.5) is a
  Phase 1 placeholder. The empirical cutoff is derived from the
  `locus_ppb_sweep_v1` dataset (owned by the render-tag repo) and will
  replace the sentinel in a subsequent release.
- **Family mix changes routing.** The PPB estimate divides by the
  minimum outer tag dimension across registered families (AprilTag
  36h11 = 8, ArUco 4×4 = 6). Registering a smaller-grid family shifts
  every candidate toward the high route. Prefer single-family
  detectors when precision matters.
- **Behavioural delta vs `standard`.** Flipping profiles changes the
  frame-by-frame output. Downstream code that memoises per-frame
  detection counts, or snapshots detection results for replay, needs
  re-validation.
