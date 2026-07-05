# Tune detectors and compare against other libraries

The tuning harness (`tools/bench/tune/`) runs many detector configurations
across CPU cores, selects an optimal frontier under hard constraints, and
compares tuned Locus against tuned competitors (OpenCV ArUco, pupil_apriltags)
to expose *which levers move which metric*.

It is a research tool layered on the existing bench substrate: it reuses the
detector wrappers, the Tier-1 `ObservationRecord` records, and the 5-axis
stratification. It does **not** touch the Rust core or the insta regression
snapshots.

## Prerequisites

- A synced hub dataset (see `.agent/skills/testing/SKILL.md`), e.g.
  `tests/data/hub_cache/locus_v1_tag36h11_1920x1080/`.
- The `bench` dependency group: prefix commands with `uv run --group bench`.

## 1. Sweep a search space (accuracy, parallel)

A *search space* declares tunable parameters per library as JSON in
`tools/bench/tune/spaces/`. `bench sweep` fans the space across cores and writes
tidy result tables. Accuracy only — latency is **not** measured here (parallel
timing is contention-poisoned).

```bash
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
uv run --group bench python tools/cli.py bench sweep \
  --library locus \
  --hub-config locus_v1_tag36h11_1920x1080 \
  --strategy random --n 64 --seed 0 \
  --out out/sweep
```

Writes `out/sweep/tune_results.parquet` (long-form `(library, param_hash,
dataset, stratum_id, metric) → value`) and `tune_configs.parquet` (the
`param_hash → param_values` sidecar).

Strategies: `grid`, `random` (both dependency-free), or `bayes` (needs the
optional extra: `pip install -e '.[tune]'`).

## 2. Tune to a frontier (adds selection + serial latency)

`bench tune` runs the sweep, then selects the Pareto frontier over
*(maximize recall, minimize p99 pose error)* subject to a precision floor (and
optional latency budget), verifies latency **serially** with production
threading, and guards against regressing the render-tag tail/mean versus shipped
Locus profiles. Repeat `--library` to tune competitors in the same run.

```bash
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
uv run --group bench python tools/cli.py bench tune \
  --library locus --library opencv_aruco --library apriltag \
  --hub-config locus_v1_tag36h11_1920x1080 \
  --strategy random --n 64 \
  --precision-floor 0.99 --tail-metric trans_p99 \
  --out out/tune
```

Writes `out/tune/pareto/<library>.json` — the feasible frontier you pick a
deployment config from. A `✓promotable` config improves without regressing the
render-tag tail/mean; a `⚠tail-regress` one does not.

## 3. Comparative lever report

```bash
uv run --group bench python tools/cli.py bench compare-report \
  --results out/tune/tune_results.parquet \
  --pareto-dir out/tune/pareto \
  --out out/report
```

Produces `out/report/index.html` with:

- a **lever-sensitivity heatmap** — for each library, which knob most moves each
  metric (e.g. `decoder.refinement_mode` drives the pose tail, not recall);
- **per-stratum deltas** — where tuned Locus trails / leads the best tuned
  competitor, so you know which implementation to improve and where.

## Notes

- **Fair comparison** needs the competitor spaces (`spaces/opencv_default.json`,
  `spaces/apriltag_default.json`) reviewed for breadth — narrow competitor
  spaces make the deltas misleading.
- **Priority**: render-tag hub datasets outrank ICRA; the tuner never promotes a
  config that trades render-tag tail/mean for recall.
