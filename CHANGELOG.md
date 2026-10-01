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
  [Liu4K report](docs/engineering/benchmarking/liu4k_euroc_sota_20261001.md) (config sweep, comparison against
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

### Removed

- `scripts/fetch_euroc_calibration.sh` (and its `LOCUS_EUROC_HF_REPO` override): use
  `cargo xtask data fetch euroc`, which no longer needs the `hf` CLI or `unzip`.

### Documentation

- **Real-image competitiveness root causes (Liu4K, EuRoC).** Dated `lessons/` subsections record why
  Locus trailed aruco_nano / OpenCV on real photos and how a Locus configuration closes the recall
  gap: the segmentation threshold model, filled-blob quad pre-gates, a grey-level (not noise-scaled)
  offset, Kalibr's 2-bit tag border, refine-before-decode latency, and photometric (sRGB gamma)
  corner bias, including an EdLines outward bias that the sRGB render-tag benchmark masks.
  [Real-image competitiveness snapshot](docs/engineering/benchmarking/liu4k_euroc_sota_20261001.md)
  supersedes the 2026-09-19 Liu4K report; controlled reproducer
  `tools/bench/photometric_corner_bias.py`; `coordinates.md` gains an OpenCV/Kalibr pixel-centre
  interop note.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
