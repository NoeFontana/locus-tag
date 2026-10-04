# locus-tag

[![CI](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml)
[![Docs](https://github.com/NoeFontana/locus-tag/actions/workflows/docs.yml/badge.svg)](https://noefontana.github.io/locus-tag/latest/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Locus** detects AprilTag and ArUco markers, as well as AprilGrid and ChArUco boards. It's implemented in Rust with zero-copy Python bindings. It targets a balance of low latency, high pose accuracy and high recall.

> [!WARNING]
> **Experimental: pre-1.0, not recommended for production yet.**
> - The API may break until 1.0.0. The road to 1.0 is a smaller API surface and broader validation on real-camera data (EuRoC and Liu4K are benchmarked today). Contribution of permissively licensed real datasets greatly appreciated (open an issue to chat about it)!
> - Distortion-model support is experimental and slated for a redesign.
> - The shipped tag families are intentionally minimal.

## Some Features

- **Zero-copy ingestion**: images cross the Rust↔Python boundary through the NumPy Buffer Protocol; no copies.
- **Releases the GIL** during detection, so it parallelizes cleanly across Python threads.
- **6-DOF pose**: an IPPE-Square seed refined by a weighted Levenberg–Marquardt solver, and internal-edge based optional refinement.
- **Allocation light**: per-frame arena allocation, no `malloc` inside `detect()`.

## Supported markers & requirements

- **Tag families:** AprilTag (`16h5`, `36h11`) and ArUco (`4x4_50`, `4x4_100`, `6x6_250`, `mip_36h12`). More can be registered: [Add a dictionary](https://noefontana.github.io/locus-tag/latest/how-to/add_dictionary/).
- **Boards:** AprilGrid and ChArUco layouts over those families.
- **Python:** 3.10+ (abi3 wheels).
- **Platforms:** prebuilt wheels for Linux (x86_64 / aarch64, glibc + musl), macOS (Intel + Apple Silicon), and Windows (x64).

## Installation

```bash
pip install locus-tag
```

The PyPI wheel targets rectified (pinhole) imagery. For unrectified cameras (Brown–Conrady, Kannala–Brandt fisheye), see [Install with distortion support](https://noefontana.github.io/locus-tag/latest/how-to/install-with-distortion/).

## Quick start

### Detect markers

```python
import cv2
import locus

img = cv2.imread("tags.jpg", cv2.IMREAD_GRAYSCALE)
detector = locus.Detector(families=[locus.TagFamily.AprilTag36h11])

batch = detector.detect(img)  # parallel NumPy arrays
print(batch.ids)  # (N,)
print(batch.corners.shape)  # (N, 4, 2)
```

### Estimate 6-DOF pose

```python
from locus import Detector, CameraIntrinsics

intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

batch = detector.detect(
    img,
    intrinsics=intrinsics,
    tag_size=0.10,  # physical side length, meters
)

if batch.poses is not None:
    print(batch.poses[0])  # [tx, ty, tz, qx, qy, qz, qw]
```

Configuration is nested and Pydantic-validated: start from a shipped profile, edit the group you care about, and hand it back to the detector. The [detection guide](https://noefontana.github.io/locus-tag/latest/tutorials/guide/) walks through the `DetectorConfig` API.

## Performance

Profiles are selected by name and embedded in the wheel:

<!-- --8<-- [start:performance-profiles] -->
| `profile` | Best for | Notes |
| :--- | :--- | :--- |
| `"standard"` | General detection, real cameras | Default. Decode-first with a dark-border-ring check, 4-connectivity, fused and photometrically calibrated corners. |
| `"grid"` | Calibration boards, low-contrast prints | `standard`'s pipeline with blob-shape gates kept, sharpening off and lower contrast gates. |
| `"high_accuracy"` | Metrology, AV pose | EdLines quads and model-edge pose refinement for single-tag rotation tails. Needs camera intrinsics + `tag_size`. |
<!-- --8<-- [end:performance-profiles] -->

`standard` is benchmarked against pinned OpenCV `aruco` 4.10 (three corner refiners) and aruco_nano on real and rendered data with `cargo xtask sota` (2026-10-04, 1 thread, AMD EPYC-Milan):

| Benchmark | Metric | Locus `standard` | Best reference |
| :--- | :--- | :---: | :---: |
| EuRoC `cam_april` (real camera, Kalibr AprilGrid) | Recall | **86.0 %** | 57.1 % |
| | Precision | **99.996 %** | 99.993 % |
| | Leave-one-out corner error, median / p90 | **0.283 / 0.575 px** | 0.516 / 0.996 px |
| render-tag 1080p (Blender) | Corner error, debiased mean | **0.062 px** | 0.230 px |
| Liu4K (real 4K photos, ArUco MIP 36h12) | Recall | 37.2 % | **66.3 %** |

Liu4K recall is an open gap (threshold model; see the [recall lessons](https://noefontana.github.io/locus-tag/latest/engineering/lessons/recall-quad-icra/)). Sources: the [EuRoC report](https://noefontana.github.io/locus-tag/latest/engineering/benchmarking/euroc_sota_20261004/) and the [SOTA scoreboard](https://noefontana.github.io/locus-tag/latest/engineering/benchmarking/sota_scoreboard_20261004/) (2026-10-04).

For single-tag pose, `high_accuracy` had the lowest rotation and translation tails on the 1080p render-tag suite in the 2026-07 snapshot (rotation p99 0.249° vs 0.376° for OpenCV's `apriltag` refiner). Details, methodology and hardware are in the [performance docs](https://noefontana.github.io/locus-tag/latest/explanation/performance/).

## Visual debugging

Built-in integration with the **[Rerun SDK](https://rerun.io)** emits intermediate pipeline stages:

```python
batch = detector.detect(img, debug_telemetry=True)
if batch.telemetry:
    print(batch.telemetry.subpixel_jitter)
```

## Documentation

- [Detection guide](https://noefontana.github.io/locus-tag/latest/tutorials/guide/) for end-to-end usage and the config API
- [Python API reference](https://noefontana.github.io/locus-tag/latest/reference/api/)
- [Performance & benchmarks](https://noefontana.github.io/locus-tag/latest/explanation/performance/)
- [Architecture & memory model](https://noefontana.github.io/locus-tag/latest/explanation/architecture/)
- [Coordinate conventions](https://noefontana.github.io/locus-tag/latest/explanation/coordinates/)

## Contributing

See the [engineering workflow](https://noefontana.github.io/locus-tag/latest/engineering/workflow/) for the dev setup and PR gates.

## License

Dual-licensed under [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
