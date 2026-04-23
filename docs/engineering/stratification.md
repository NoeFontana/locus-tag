# Stratification Schema

> **Purpose.** Define how regression metrics are sliced along physically
> meaningful axes so we catch regressions that averaging hides.
>
> **Status.** Axes are frozen for v1. **Bucket boundaries are TBD** — to be
> filled during the perception-engineer review (see §6).

## 1. Motivation

The current baseline (`docs/engineering/benchmarking/baseline.json`) reports
one recall number per dataset per library. A 1% drop in that number could be
anything: a 4% drop in the 4K stratum concealed by gains at 720p, a systematic
loss of oblique-view tags masked by improved frontal detection, or a silent
break on distant small tags.

Concretely, the four `locus_v1_tag36h11_{640x480, 1280x720,
1920x1080, 3840x2160}` hub-configs already live under
`tests/data/hub_cache/`. Today they are reported as a single number each;
tomorrow every metric splits further by PPM, angle of incidence, distance,
and motion — because those are the axes that correlate with the *physical
regime* the detector runs in, not the *rendering configuration*.

Stratification gives us:

1. **Early-warning signal.** A stratum-scoped tolerance gate fires on a 5%
   regression in `dist=far` even when the global number holds.
2. **Debuggability.** When a regression fires, the `stratum_id` narrows the
   search from "everything" to a handful of images.
3. **Cross-dataset comparability.** A stratum means the same thing in every
   hub-config and in the ICRA 2020 corpus.

## 2. The five axes

Every ground-truth record is assigned a bucket on each axis at load time.
Source fields come from `tests/data/hub_cache/*/rich_truth.json`; see
`tools/bench/utils.py` for the existing loader wired to the
`TagGroundTruth` / `DatasetMetadata` dataclasses.

| Axis | Source field(s) | Unit | Bucket count | Notes |
| --- | --- | --- | --- | --- |
| **Resolution** | `resolution: [W, H]` | pixels | 4 (TBD) | Bucket on `H` (height). Expected slugs: `sd`, `hd`, `fhd`, `uhd`. |
| **PPM** (pixels per metre of tag) | `ppm` (preferred) else derived from `corners` + `tag_size_mm` | px/m | 3 (TBD) | `ppm == 0.0` in current `rich_truth.json` files — see §3. Slugs: `lo`, `mid`, `hi`. |
| **Angle of incidence (AOI)** | `angle_of_incidence` | degrees | 3 (TBD) | 0° = frontal. Slugs: `frontal`, `oblique`, `grazing`. |
| **Distance** | `distance` | metres | 3 (TBD) | Slugs: `near`, `mid`, `far`. |
| **Motion** | `velocity` or `shutter_time_ms` × `‖velocity‖` | m/s (or px blur) | 2 (TBD) | Most current records have `velocity: null` ⇒ `static`. Slugs: `static`, `motion`. |

### Axis selection rationale

- **Resolution, PPM, AOI, distance** are the canonical four of fiducial-detector
  evaluation literature. They are present, per-tag, in the existing ground
  truth.
- **Motion** is nascent. Today most records are `velocity: null` and
  `rolling_shutter_ms: 0.0`. Rather than drop it and rebuild the schema later,
  v1 reserves the slot with a degenerate `static` bucket; synthetic datasets
  or future captures populate the `motion` bucket.

## 3. Known data-quality gaps

These are called out so reviewers don't discover them mid-triage:

- **`ppm` is 0.0** in the current hub-cache `rich_truth.json` files. The axis
  is still load-bearing — the loader must derive PPM from:
  `ppm ≈ max_edge_px(corners) / (tag_size_mm / 1000.0)`. Boundaries for the
  `ppm` axis should be chosen on the *derived* value, not the raw field.
- **`velocity: null`** for all current hub-cache records. Treated as
  `static`. When a dataset with motion lands, the loader reads `velocity`
  directly (or computes effective blur as
  `shutter_time_ms * ‖velocity‖ * ppm`).
- **ICRA 2020** has no per-tag metadata; every record there collapses to
  `res=<from image shape>|ppm=unk|aoi=unk|dist=unk|mot=unk`. The schema must
  accept `unk` for any axis without crashing the loader.

## 4. Canonical `stratum_id` format

`stratum_id` is a single string, deterministic across runs, diffable in
`baseline_v2.json` entries.

### Grammar

```
stratum_id := axis_pair ("|" axis_pair){4}
axis_pair  := key "=" bucket
key        := "res" | "ppm" | "aoi" | "dist" | "mot"
bucket     := slug | "unk"
slug       := [a-z0-9]+          # short, lowercase, alphanumeric
```

### Rules

1. **Key order is fixed**: `res`, `ppm`, `aoi`, `dist`, `mot`. Always five
   pairs, always in that order, always `|`-separated.
2. **Slugs are short and lowercase**. No numeric ranges inside the id —
   ranges live in the boundary table (see §5). This keeps ids stable across
   re-bucketing.
3. **Unknown bucket is `unk`**. Any axis that cannot be derived from a
   record (missing source field, NaN, out-of-range) uses `unk`. Reporters
   treat `unk` as its own legitimate stratum — it is visible in diffs and has
   its own tolerances.
4. **Escape rules.** Slugs must not contain `=` or `|`. The validator in
   `tools/bench/schema.py:Tolerances` (A0.2) rejects ids that fail the
   grammar.

### Examples

```
res=fhd|ppm=hi|aoi=frontal|dist=near|mot=static
res=uhd|ppm=lo|aoi=grazing|dist=far|mot=static
res=hd|ppm=unk|aoi=unk|dist=unk|mot=unk       # ICRA 2020 record
```

## 5. Bucket boundary table (TBD)

The concrete cut points for each axis are chosen during the engineer review
(§6). The table below is a skeleton; reviewers fill the numeric columns.

| Axis | Slug | Range | Rationale |
| --- | --- | --- | --- |
| res | sd | `H ≤ ?` | |
| res | hd | `? < H ≤ ?` | |
| res | fhd | `? < H ≤ ?` | |
| res | uhd | `? < H` | |
| ppm | lo | `ppm ≤ ?` | |
| ppm | mid | `? < ppm ≤ ?` | |
| ppm | hi | `? < ppm` | |
| aoi | frontal | `angle ≤ ?` | |
| aoi | oblique | `? < angle ≤ ?` | |
| aoi | grazing | `? < angle` | |
| dist | near | `d ≤ ?` | |
| dist | mid | `? < d ≤ ?` | |
| dist | far | `? < d` | |
| mot | static | `‖velocity‖ ≤ ?` | |
| mot | motion | `? < ‖velocity‖` | |

## 6. Extension protocol

Adding, removing, or renaming an axis is a **schema minor bump** (v1.x → v1.(x+1))
and requires:

1. Update this doc: add the axis row to §2, update the `stratum_id` grammar
   in §4, add boundary entries in §5.
2. Update `tools/bench/schema.py` validators.
3. Bump `BaselineV2.schema_version` (see `tools/bench/schema.py`).
4. Write a migration for existing baselines (old ids need the new axis
   populated with `unk`).

Re-bucketing an **existing** axis (changing cut points only) is a v1 *patch*
bump and does **not** change the `stratum_id` grammar — but it does invalidate
existing baselines because the same raw record may map to a different bucket.
Re-run the baseline after re-bucketing.

## 7. Review checklist

Two perception engineers sign off on:

- [ ] Axis selection is sufficient and non-redundant.
- [ ] Bucket counts per axis (total strata = ∏ bucket counts — keep this
      manageable; target <60 strata with `unk` excluded).
- [ ] Concrete cut points in §5, chosen with a histogram over
      `tests/data/hub_cache/*/rich_truth.json` fields.
- [ ] `stratum_id` grammar is unambiguous.
- [ ] Data-quality gaps in §3 are acceptable for v1 or have mitigations
      planned.
- [ ] Extension protocol §6 is understood by both reviewers.

## 8. Sources

- Field schema: `tests/data/hub_cache/*/rich_truth.json` (sampled).
- Loader types: `tools/bench/utils.py` — `TagGroundTruth`,
  `DatasetMetadata`, `HubDatasetResult`.
- Related benchmarking contract: `docs/engineering/benchmarking.md`.
