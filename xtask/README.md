# xtask — repository automation

`cargo xtask <command>` (alias in `.cargo/config.toml`). Dependency-free: it only
orchestrates `git`, `cmake`, a C++ compiler and `uv`.

## `cargo xtask data` — pinned dataset provisioning

Every external dataset (Hub synthetic suites, ICRA 2020, EuRoC, Liu4K) is declared once in
[`datasets.toml`](datasets.toml): source, **pin** (Hugging Face repository commit or URL
checksum), size, licence, citation, destination under `tests/data/`, and readiness markers.

```bash
cargo xtask data list                                # presence, kind, destination, licence
cargo xtask data fetch euroc liu4k                   # idempotent; --force to refetch
cargo xtask data fetch hub --subsets all             # or a comma-separated config list
cargo xtask data verify                              # missing or stale-pin copies → exit 1
```

The implementation is `tools/bench/dataset_registry.py`; the Python bench CLI
(`bench prepare`, `bench real`), `prepare_liu4k` / `DatasetLoader.prepare_icra` and
`cargo xtask sota fetch` all go through it. Rust integration tests and Python loaders read
the same `tests/data/` paths, so they need no configuration; `LOCUS_HUB_DATASET_DIR`,
`LOCUS_ICRA_DATASET_DIR` and `LOCUS_EUROC_DATASET_DIR` relocate a dataset for both (`--dest`
overrides everything).

Readiness markers appear only on success (archives are extracted into a staging directory
and renamed into place; a Hub subset's `annotations.jsonl` is renamed in last), so an
interrupted fetch is retried rather than mistaken for a complete one. Each successful fetch
records its pin in a `.locus-dataset.json` stamp (per subset for the Hub); `verify` exits 1
for missing or stale-pin copies and warns for copies that predate stamping (`--force`
refetches and stamps them).

Bumping a pin is a reviewable one-line change to `datasets.toml`. The EuRoC mirror is
private: authenticate with `hf auth login` or `HF_TOKEN`.

## `cargo xtask sota` — comparative benchmarking against pinned references

Reproducible, long-lived comparison of Locus against the published state of the
art on every dataset the references can decode, so competitiveness is a re-runnable
measurement rather than a number in a dated report.

```bash
uv sync --group bench                                   # once
uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml
cargo xtask sota setup                                  # once (~10 min): OpenCV + aruco_nano + runner
cargo xtask sota list                                   # benchmark names
cargo xtask sota all liu4k                              # fetch, run, score, report
cargo xtask sota run euroc --jobs 4 --stride 2          # quick accuracy-only pass
cargo xtask sota run liu4k --detectors 'locus:standard,locus:mine=standard+my.json,aruco_nano'
cargo xtask sota scoreboard                             # win table over every scored benchmark
```

`locus:<name>=<profile>+<file.json>` merges the JSON into the profile. A top-level
`"detector"` object in it holds per-call `Detector` options instead of profile keys, e.g.
`{"detector": {"decimation": 2}}`.

Outputs land in `target/sota/runs/<benchmark>/`: one `<detector>.jsonl` per
detector (ids + corners + per-image latency), `meta.json` (verified hardware,
git revision, threads, protocol), `score.json`, and `report.md`.
`scoreboard` writes `target/sota/scoreboard.md`.

### Win table

Every report ends with a **win table**: the champion Locus run (`locus_standard`;
`scoreboard --champion <label>` for another) against the *best* reference operating point
(`aruco_nano`, `opencv`, `opencv_subpix`, `opencv_apriltag`) on each metric, with the margin.
A tie is a win; latency is judged only for runs with valid timing. AprilTag 3 is scored and
reported but is not part of the criterion. "Industrial SOTA" here means every judged cell
green on every benchmark.

### References (pinned, never patched)

| Reference | Pin | Configuration |
| :-- | :-- | :-- |
| OpenCV `aruco` | `4.10.0`, minimal module build | `errorCorrectionRate = 0` (as aruco_nano's `testperf.cpp`), `markerBorderBits` from the dataset; three published corner refiners: `opencv` (`CORNER_REFINE_NONE`), `opencv-subpix` (`SUBPIX`), `opencv-apriltag` (`APRILTAG`) |
| aruco_nano | `961b18b` | library defaults, dataset dictionary |
| AprilTag 3 | `pupil-apriltags` from the `bench` group | `quad_decimate = 1`, `refine_edges = True` |

References are compared **as published**. When one cannot decode a dataset
through its public parameters (aruco_nano and AprilTag 3 on Kalibr's 2-bit-border
AprilGrid), it is listed as unsupported in the report, not patched.

### Benchmarks

Declared as `[sota.<name>]` tables in [`datasets.toml`](datasets.toml) (dictionary, border
width, scorer, ground truth and its pixel convention, unsupported references) and resolved by
`tools/bench/sota/spec.py`; adding one is a manifest edit.

| Benchmark | Content | Ground truth | Scorer |
| :-- | :-- | :-- | :-- |
| `liu4k` | 924 4K photos, `ARUCO_MIP_36h12` (Zenodo 10.5281/zenodo.18667018, CC BY 4.0) | ids + corners | aruco_nano `testperf.cpp` rule: same id, centre ≤ 10 px, first-match TP/FP/FN; recall by marker side; corner error |
| `euroc` | 1450 frames of the EuRoC `cam_april` sequence, 6×6 Kalibr AprilGrid (tag36h11, **2-bit border**), strong radtan distortion | none per corner | GT-free: pooled-detector homography after undistortion defines presence/precision; leave-one-tag-out corner error on a common tag set |
| `icra-{forward,circle,random}` | ICRA 2020 AprilTag localization dataset, `pure_tags` images, tag36h11 | corners (`tags.csv`) | `gt-csv` |
| `hub-{640,720p,1080p,4k,high-iso,low-key,raw-pipeline,tag16h5}` | Locus render-tag suites (Blender), one tag per frame | corners (`rich_truth.json`) | `gt-hub` |
| `hub-aprilgrid`, `hub-charuco` | Board renders (tag36h11 1-bit AprilGrid; ArUco 6x6_250 ChArUco), scored per marker | corners | `gt-hub` |

`gt-*` scorers: a detection is a TP when it pairs with a same-id GT tag whose centre is
within 20 px (`tools/bench/matching.py`); detections of tags the GT marks partly visible or
not evaluable are neither TP nor FP. Corner error is **order-preserving**: one dihedral
relabelling of the GT corners is fixed per (benchmark, detector), never chosen per instance,
and per-tag error is the RMSE of the four corners. `corner_common_*` restricts to tags every
detector with ≥ 20 % recall matched, so detectors are compared on the same tags.

### Protocol

- Image decode is outside every timer; one untimed warm-up call; best of
  `--reps` `detect()` calls per image.
- One thread per detector by default (`RAYON_NUM_THREADS` for Locus,
  `cv::setNumThreads` / `nthreads` for the references); `OMP_NUM_THREADS` and
  `OPENBLAS_NUM_THREADS` pinned to 1.
- `--jobs > 1` runs detectors concurrently: fine for accuracy, and `meta.json`
  marks the latency column invalid. Publish latency only from `--jobs 1` runs on
  an idle machine (`docs/engineering/constraints.md` §6).
- Corners carry a `convention` tag (`locus` = pixel centre at +0.5, `opencv` =
  pixel centre at integer); scorers convert before comparing.
