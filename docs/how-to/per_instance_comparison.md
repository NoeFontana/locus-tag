# Compare detectors per image and find Locus's levers

The per-instance comparison (`tools/bench/compare/`) takes the tuned frontier
configs from a `bench tune` run and compares locus-tag against OpenCV ArUco and
pupil_apriltags **image-by-image, tag-by-tag**. It answers *which library wins on
which strata* and *which specific images Locus should improve on*, and produces a
structured report, embeddable SVG figures, and a scrubbable rerun deep-dive.

It reuses the Tier-1 record substrate, the tuning wrappers, and stratification.
The analysis layer is [polars](https://pola.rs); figures are matplotlib → SVG.

## Prerequisites

- A `bench tune` output directory with `pareto/<library>.json` for locus,
  opencv_aruco, and apriltag (see `how-to/tune_and_compare.md`).
- Synced render-tag hub datasets under `tests/data/hub_cache/`.
- The `bench` dependency group: prefix commands with `uv run --group bench`.

## One-shot

```bash
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
uv run --group bench python tools/cli.py bench compare-instances \
  --pareto-dir out/tune/pareto \
  --metric repro --top-n 25 \
  --markdown-out docs/engineering/benchmarking/comparative_2026-07-05.md \
  --out out/compare
```

This runs **four series** — `locus:tuned`, `locus:shipped` (default profile
`high_accuracy`), `opencv_aruco:tuned`, `apriltag:tuned` — across the four
render-tag resolutions, then writes:

- `out/compare/index.html` — the report bundle with **two sections** (tuned-Locus
  and shipped-Locus, each vs the tuned competitors): per-stratum win-rate
  heatmaps, per-instance error ECDFs/violins, Locus-vs-best-competitor paired
  scatters, delta histograms, and the **worst-Locus lever table** (the exact
  images/tags where Locus trails and by how much).
- `out/compare/*.parquet` — `instances_wide`, `winrate_by_stratum`,
  `worst_locus_{tuned,shipped}` for further analysis.
- `out/compare/recordings/compare_deepdive.rrd` — a rerun recording: scrub the
  `frame_idx` timeline through the worst-Locus cases, with GT and each library's
  corners as **separately toggleable overlays** and per-library corner/pose error
  as time-series. Open with `rerun out/compare/recordings/compare_deepdive.rrd`.
- The markdown report for the docs (tables only).

`--metric` picks the lever ranking metric (`repro` = corner RMSE, `trans`, `rot`).

## Iterating

Split the two phases (same pattern as `bench tune` ↔ `bench compare-report`):

```bash
# 1. generate the combined parquet once (the slow part)
bench compare-generate --pareto-dir out/tune/pareto --out out/compare
# 2. re-render the report / try metrics without re-detecting
bench compare-report-instances --records out/compare/instance_records.parquet \
  --out out/compare --metric trans --pareto-dir out/tune/pareto
```

## Notes on measurement fidelity

- **Corner error is order-preserving.** Each library's corners are normalised to
  the ground-truth convention by a *fixed* per-library adapter (e.g. apriltag's
  corner order is a fixed `[1,0,3,2]` relabel). We do **not** use an
  orientation-independent metric — a genuine wrong-orientation detection must
  still surface as a large error, since orientation drives pose sign and decode.
- **Best-vs-best.** The comparison uses each library's tuned config, so a Locus
  loss is a real gap, not an un-tuned-competitor artifact. Section B additionally
  shows what the shipped Locus profile delivers.
