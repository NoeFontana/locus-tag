# locus-tag

[![CI](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml)
[![Docs](https://github.com/NoeFontana/locus-tag/actions/workflows/docs.yml/badge.svg)](https://noefontana.github.io/locus-tag/latest/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Locus** detects AprilTag and ArUco markers, as well as AprilGrid and ChArUco boards. It's implemented in Rust with zero-copy Python bindings. It targets a balance of low latency, high pose accuracy and high recall.

> [!WARNING]
> **Experimental: pre-1.0, not recommended for production yet.**
> - The API may break until 1.0.0. The road to 1.0 is a smaller API surface and validation on real-camera (not just synthetic) data. Contribution of permissively licensed real datasets greatly appreciated (open an issue to chat about it)!
> - Distortion-model support is experimental and slated for a redesign.
> - The shipped tag families are intentionally minimal.

## Some Features

- **Zero-copy ingestion**: images cross the Rust↔Python boundary through the NumPy Buffer Protocol; no copies.
- **Releases the GIL** during detection, so it parallelizes cleanly across Python threads.
- **6-DOF pose**: an IPPE-Square seed refined by a weighted Levenberg–Marquardt solver, and internal-edge based optional refinement.
- **Allocation light**: per-frame arena allocation, no `malloc` inside `detect()`.

## Supported markers & requirements

- **Tag families:** AprilTag (`16h5`, `36h11`) and ArUco (`4x4_50`, `4x4_100`, `6x6_250`). More can be registered: [Add a dictionary](https://noefontana.github.io/locus-tag/latest/how-to/add_dictionary/).
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

batch = detector.detect(img)          # parallel NumPy arrays
print(batch.ids)                      # (N,)
print(batch.corners.shape)            # (N, 4, 2)
```

### Estimate 6-DOF pose

```python
from locus import Detector, CameraIntrinsics

intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

batch = detector.detect(
    img,
    intrinsics=intrinsics,
    tag_size=0.10,                    # physical side length, meters
)

if batch.poses is not None:
    print(batch.poses[0])             # [tx, ty, tz, qx, qy, qz, qw]
```

Configuration is nested and Pydantic-validated: start from a shipped profile, edit the group you care about, and hand it back to the detector. The [detection guide](https://noefontana.github.io/locus-tag/latest/tutorials/guide/) walks through the `DetectorConfig` API.

## Performance

Profiles are selected by name and embedded in the wheel:

<!-- --8<-- [start:performance-profiles] -->
| `profile` | Best for | Notes |
| :--- | :--- | :--- |
| `"standard"` | General detection | Balanced recall and precision; highest recall on small/distant tags. |
| `"grid"` | ChArUco / AprilGrid boards | 4-connectivity recovers touching tags that `"standard"` merges. |
| `"high_accuracy"` | Metrology, AV pose | Best pose accuracy and rotation-tail control. Needs camera intrinsics + `tag_size`. |
<!-- --8<-- [end:performance-profiles] -->

On our high-fidelity `render-tag` suite (1080p, single-thread), `high_accuracy` leads both the translation **and** rotation tails while running ~13× faster than OpenCV's best-accuracy configuration:

| Detector | Rot p99 | Trans p99 | Latency |
| :--- | :---: | :---: | :---: |
| **Locus (`high_accuracy`)** | **0.249°** | **20.1 mm** | **15.2 ms** |
| OpenCV (`cv2.aruco`, apriltag) | 0.376° | 55.3 mm | 195.8 ms |

Full results — both benchmark suites (render-tag + ICRA 2020), every percentile, methodology, and hardware — are in the [performance docs](https://noefontana.github.io/locus-tag/latest/explanation/performance/).

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
