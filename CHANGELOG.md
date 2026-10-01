# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Added

- **`ArUcoMip36h12` tag family** (`cv2.aruco.DICT_ARUCO_MIP_36h12`: 250 codes, 6x6 bits,
  minimum Hamming distance 12). New dictionary JSON generated with
  `examples/dictionary_generation/extract_opencv.py`, `TagFamily` variant in `locus-core` and
  `locus-py` (discriminant 5), decoder, regenerated `locus.pyi`, dictionary parity snapshot.
- **Liu4K real-photo benchmark** (`bench real --dataset liu4k`, `tools/bench/liu4k.py`). Runtime
  download from Zenodo (10.5281/zenodo.18667018, CC BY 4.0) into gitignored `tests/data/liu4k/`
  with md5 verification; reports id-agnostic quad recall and, with `--family ArUcoMip36h12`,
  id-aware decode recall/precision. Corners/ids ground truth only, so no pose metrics. No dataset
  files are committed or packaged. Attribution in `docs/engineering/benchmarking.md`.
  The scorer mirrors aruco_nano's `testperf.cpp` (same id, centre distance `<= 10 px`, first-match
  TP/FP/FN); the 10 px radius is a Liu4K-specific constant and the repo-wide match threshold is
  unchanged. Adds `bench real --sharpening/--no-sharpening` and the
  [Liu4K report](docs/engineering/benchmarking/liu4k_20260919.md) (config sweep, comparison against
  OpenCV 4.10 and aruco_nano, and open detector findings).

- **`cargo xtask sota` comparative benchmarking** (`xtask/`, `tools/bench/sota/`). Builds pinned,
  unpatched references (OpenCV 4.10.0 minimal build, aruco_nano `961b18b`) plus a C++ runner,
  fetches Liu4K / EuRoC, runs Locus, aruco_nano, OpenCV and AprilTag 3 under one timing protocol,
  and scores them (aruco_nano rule on Liu4K; ground-truth-free board-consistency protocol on
  EuRoC). Every run records verified hardware/thread metadata (`xtask/README.md`).

- **`cargo xtask data`: pinned dataset provisioning** (`xtask/datasets.toml`,
  `tools/bench/dataset_registry.py`). One manifest declares every external dataset (Hub suites, ICRA
  2020 scenarios, EuRoC, Liu4K) with its Hugging Face revision or URL checksum, licence, citation
  and destination; `list` / `fetch` / `verify` replace four ad-hoc downloaders. Hugging Face sources
  are now pinned to a repository commit (previously the default branch); readiness markers appear
  only on success (staged extraction, Hub `annotations.jsonl` renamed in last); per-dataset /
  per-subset stamps let `verify` flag missing or stale-pin copies; `LOCUS_*_DATASET_DIR` overrides
  are honoured. `bench prepare`, `prepare_liu4k`,
  `DatasetLoader.prepare_icra`, `xtask sota fetch` and `sync_hub.py`'s CLI delegate to it.
  `sync_subset_to_local` gains a `revision` argument and now raises on image-write or auxiliary
  download failures (a file absent from the repo stays benign) instead of only logging them.

### Fixed

- **Resampling coordinate maps (`decimation > 1`, `upscale_factor > 1`).** Corners found on a
  decimated or upscaled grid were mapped back with the integer-pixel-centre formula
  `(v + 0.5)·d − 0.5` inside Locus's +0.5 convention: seeds landed 1 px off at `decimation = 2`
  and refined corners 0.25 px off at `upscale_factor = 2`. One pair of maps,
  `image::decimated_to_full` / `image::upscaled_to_full` (pure scalings), now serves quad
  extraction, the camera-aware path, `ScaledIntrinsics` and the upscale corner/covariance mapping,
  with resampler-consistency tests. `ImageView::decimate_to` now area-averages each `d × d`
  block (it point-sampled one pixel, aliasing fine texture), and `upscale_to` rounds instead of
  truncating (which darkened every pixel by 0.5 grey). Shipped profiles use neither, so their
  output is unchanged.
- **`sample_gradient_bilinear` border fallback** sampled 0.5 px off the requested point within
  ~1.5 px of the image border (only AprilGrid board snapshots move, by ~1e-6 relative).
- **AVX2 branch of the ERF gradient projection** sampled `px` instead of the pixel centre
  `px + 0.5` (compiled only with `target_feature = "avx2"`; regression test added).
- **Detector-level GWLF** no longer refines candidates the contrast funnel already rejected.
- `detection-batch-contract.md` documents the real phase execution order and the GWLF phase.

### Removed

- `scripts/fetch_euroc_calibration.sh` (and its `LOCUS_EUROC_HF_REPO` override): use
  `cargo xtask data fetch euroc`, which no longer needs the `hf` CLI or `unzip`.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
