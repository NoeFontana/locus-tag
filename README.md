# locus-tag

[![CI](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml)
[![Docs](https://github.com/NoeFontana/locus-tag/actions/workflows/docs.yml/badge.svg)](https://noefontana.github.io/locus-tag/latest/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Locus** detects AprilTag and ArUco markers as well as AprilGrid and ChArUco boards. The library is implemented in Rust and provides zero-copy Python bindings.

> [!WARNING]
> **Experimental Status**: API is subject to breaking changes until 1.0.0 ships. The main workstreams towards 1.0.0 are reducing the API surface and validating against non-synthetic data. Until then, this library isn't recommended for production systems. The support for distortion models is experimental and requires a ground-up redesign. Addition of default tag families may happen on request, the current defaults were chosen to be minimal and support most commonly used tags.

## Technical Capabilities

- **Zero-Copy Ingestion**: Accesses NumPy arrays via the Python Buffer Protocol.
- **Parallel Execution**: Releases the Python GIL during detection to allow multi-threaded use.
- **Vectorized Results**: Returns a `DetectionBatch` with parallel arrays for IDs, corners, and poses.
- **Memory**: Uses `bumpalo` arena allocation for zero heap allocations in the detection loop.
- **Solvers**: 6-DOF recovery using IPPE-Square or weighted Levenberg-Marquardt with corner uncertainty.

<!-- --8<-- [start:performance-profiles] -->
## Performance Profiles

Locus optimises for **high recall**, **low corner RMSE**, and **low
latency**. Profiles are selected by name; the three shipped profiles
are authored as JSON files and embedded in the wheel.

| `profile` | Primary characteristic |
| :--- | :--- |
| `"standard"` | Production default; balanced recall + precision. |
| `"grid"` | 4-connectivity for touching tags — ChArUco / AprilGrid boards. |
| `"high_accuracy"` | EdLines + axis-imbalance gate + adaptive PPB; prioritises pose precision and tail-rotation control. |
<!-- --8<-- [end:performance-profiles] -->

<!-- --8<-- [start:icra-comparison] -->
### ICRA 2020 Forward (community benchmark)

[ICRA 2020 Forward](https://github.com/aprilrobotics/apriltag-comparison)
is the closest thing the AprilTag community has to a neutral
benchmark. The 50-frame subset we report on is **synthetic** (not
real-camera), but it's public, peer-reviewed, and the basis for prior
detector comparisons — we report on it for continuity with the
literature.

| Detector | Recall | Corner RMSE |
| :--- | :---: | :---: |
| **Locus (`standard`)** | **96.2 %** | **0.315 px** |
| AprilTag 3 (UMich) | 62.3 % | 0.22 px |
| OpenCV (`cv2.aruco`) | 52.6 % | 0.98 px |

The OpenCV row is its recall-best OpenCV 5.0 config (tuned `subpix`); the
tag-aware `apriltag` refinement more than halves corner RMSE (0.39 px) but
rejects ICRA's marginal small tags, dropping recall to ~30 %.
<!-- --8<-- [end:icra-comparison] -->

<!-- --8<-- [start:render-tag-comparison] -->
### render-tag (high-fidelity Blender + PSF)

`render-tag` is our in-house render suite — Blender with calibrated
PSF, exposure, sensor noise, and lens distortion models. The
detection scenes carry pixel-accurate ground truth for both corners
and 6-DOF pose, which lets us report translation / rotation
percentiles in addition to recall. **Numbers below are the 2026-07-13
single-threaded SOTA snapshot on the 1080p 50-scene subset (OpenCV 5.0.0,
re-tuned)** (see
[`docs/engineering/benchmarking/render_tag_sota_20260713.md`](https://github.com/NoeFontana/locus-tag/blob/main/docs/engineering/benchmarking/render_tag_sota_20260713.md)
for methodology, the 2160p table, and OpenCV's two operating points).

| Detector | Recall | Trans p50 | Trans p99 | Rot p50 | Rot p99 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Locus (`high_accuracy`)** | **100 %** | **0.4 mm** | **18.6 mm** | **0.057 °** | 0.600 ° | **13.8 ms** |
| Locus (`standard`) | 100 % | 3.5 mm | 50.3 mm | 0.288 ° | 27.248 ° | 32.7 ms |
| OpenCV (`cv2.aruco`, subpix) | 100 % | 3.5 mm | 66.6 mm | 0.127 ° | 0.569 ° | 101.1 ms |
| OpenCV (`cv2.aruco`, apriltag) | 100 % | 3.0 mm | 55.3 mm | 0.067 ° | **0.376 °** | 195.8 ms |
| AprilTag-C (pupil) | 100 % | 2.9 mm | 54.4 mm | 0.061 ° | 65.365 ° | 78.5 ms |

Latencies are single-thread. Locus `high_accuracy` wins the translation tail and
is 14× faster than OpenCV's best-accuracy `apriltag` config; that config in turn
has the best rotation tail (0.376°). OpenCV ships two operating points — fast
`subpix` and accurate-but-~2×-slower `apriltag`. AprilTag-C's median rotation is
best in class (0.06°) but its p99 explodes to 65° on symmetric-tag IRLS
branch-ambiguity failures.
<!-- --8<-- [end:render-tag-comparison] -->

## Installation

```bash
pip install locus-tag
```

The PyPI wheel is compiled for rectified (pinhole) imagery. For unrectified cameras (Brown-Conrady polynomial, Kannala-Brandt equidistant fisheye), see [Install with distortion support](https://noefontana.github.io/locus-tag/latest/how-to/install-with-distortion/).

## Quick Start

### Basic Detection

```python
import cv2
import locus

img = cv2.imread("tags.jpg", cv2.IMREAD_GRAYSCALE)
detector = locus.Detector(families=[locus.TagFamily.AprilTag36h11])

# batch contains parallel NumPy arrays
batch = detector.detect(img)
print(f"IDs: {batch.ids}")
print(f"Corners: {batch.corners.shape}") # (N, 4, 2)
```

### 6-DOF Pose Estimation

```python
from locus import Detector, CameraIntrinsics

# fx, fy, cx, cy
intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

# Returns [tx, ty, tz, qx, qy, qz, qw] for each tag
batch = detector.detect(
    img,
    intrinsics=intrinsics,
    tag_size=0.10,  # physical side length in meters
)

if batch.poses is not None:
    # First tag translation
    print(batch.poses[0, :3])
```

### Configuration Overrides

Settings are nested and validated by Pydantic. Start from a shipped profile,
edit the group you care about, and hand it back to the detector:

```python
base = locus.DetectorConfig.from_profile("high_accuracy").model_dump()
base["quad"]["upscale_factor"] = 2
base["decoder"]["max_hamming_error"] = 1

detector = locus.Detector(config=locus.DetectorConfig.model_validate(base))
```

## Visual Debugging

Built-in integration with the **[Rerun SDK](https://rerun.io)**:

```python
batch = detector.detect(img, debug_telemetry=True)
if batch.telemetry:
    print(batch.telemetry.subpixel_jitter)
```

## Documentation

- **[Architecture & Memory Model](https://noefontana.github.io/locus-tag/latest/explanation/architecture/)**
- **[Coordinate Systems](https://noefontana.github.io/locus-tag/latest/explanation/coordinates/)**
- **[Benchmarking Methodology](https://noefontana.github.io/locus-tag/latest/engineering/benchmarking/)**

## License

Dual-licensed under [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
