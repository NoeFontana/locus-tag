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

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
