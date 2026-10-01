# xtask — repository automation

`cargo xtask <command>` (alias in `.cargo/config.toml`). Dependency-free: it only
orchestrates `git`, `cmake`, a C++ compiler and `uv`.

## `cargo xtask sota` — comparative benchmarking against pinned references

Reproducible, long-lived comparison of Locus against the published state of the
art on real-image datasets, so competitiveness is a re-runnable measurement
rather than a number in a dated report.

```bash
uv sync --group bench                                   # once
uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml
cargo xtask sota setup                                  # once (~10 min): OpenCV + aruco_nano + runner
cargo xtask sota all liu4k                              # fetch, run, score, report
cargo xtask sota run euroc --jobs 4 --stride 2          # quick accuracy-only pass
cargo xtask sota run liu4k --detectors 'locus:standard,locus:mine=standard+my.json,aruco_nano'
```

Outputs land in `target/sota/runs/<dataset>/`: one `<detector>.jsonl` per
detector (ids + corners + per-image latency), `meta.json` (verified hardware,
git revision, threads, protocol), `score.json`, and `report.md`.

### References (pinned, never patched)

| Reference | Pin | Configuration |
| :-- | :-- | :-- |
| OpenCV `aruco` | `4.10.0`, minimal module build | `errorCorrectionRate = 0` (as aruco_nano's `testperf.cpp`), `CORNER_REFINE_NONE` (`opencv-subpix` for `CORNER_REFINE_SUBPIX`), `markerBorderBits` from the dataset |
| aruco_nano | `961b18b` | library defaults, dataset dictionary |
| AprilTag 3 | `pupil-apriltags` from the `bench` group | `quad_decimate = 1`, `refine_edges = True` |

References are compared **as published**. When one cannot decode a dataset
through its public parameters (aruco_nano and AprilTag 3 on Kalibr's 2-bit-border
AprilGrid), it is listed as unsupported in the report, not patched.

### Datasets

| Dataset | Content | Ground truth | Scorer |
| :-- | :-- | :-- | :-- |
| `liu4k` | 924 4K photos, `ARUCO_MIP_36h12` (Zenodo 10.5281/zenodo.18667018, CC BY 4.0) | ids + corners | aruco_nano `testperf.cpp` rule: same id, centre ≤ 10 px, first-match TP/FP/FN; recall by marker side |
| `euroc` | 1450 frames of the EuRoC `cam_april` sequence, 6×6 Kalibr AprilGrid (tag36h11, **2-bit border**), strong radtan distortion | none per corner | GT-free: pooled-detector homography after undistortion defines presence/precision; leave-one-tag-out corner error on a common tag set |

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
