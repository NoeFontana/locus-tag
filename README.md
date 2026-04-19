# locus-tag

[![CI](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Locus** detects AprilTag and ArUco markers using a Rust core with zero-copy Python bindings. It implements sub-pixel corner refinement, releases the GIL during detection, and returns results as vectorized NumPy arrays.

> [!WARNING]
> **Experimental Status**: API is subject to breaking changes. Not recommended for production safety-critical systems.

## Technical Capabilities

- **Zero-Copy Ingestion**: Accesses NumPy arrays via the Python Buffer Protocol.
- **Parallel Execution**: Releases the Python GIL during detection to allow multi-threaded use.
- **Vectorized Results**: Returns a `DetectionBatch` with parallel arrays for IDs, corners, and poses.
- **Memory**: Uses `bumpalo` arena allocation for zero heap allocations in the detection loop.
- **Solvers**: 6-DOF recovery using IPPE-Square or weighted Levenberg-Marquardt with corner uncertainty.

## Performance Presets

Presets are selected via the `preset` argument in the `Detector` constructor:

| `DetectorPreset` | ICRA 2020 Recall | Corner RMSE | Primary Characteristics |
| :--- | :---: | :---: | :--- |
| `Standard` | **96.2%** | 0.315 px | Production default; balanced recall/precision. |
| `Grid` | **91.4%** | 0.458 px | 4-connectivity for touching tags (checkerboards). |
| `HighAccuracy` | 46.3%* | **0.16 px** | EdLines + GN optimizer; prioritized for metrology. |

*\*HighAccuracy is optimized for high-resolution near-field images (Hugging Face Hub datasets).*

### Comparison (ICRA 2020 Forward - 50 images)

| Detector | Recall | RMSE |
| :--- | :---: | :---: |
| **Locus (`Standard`)** | **96.2%** | 0.315 px |
| AprilTag 3 (UMich) | 62.3% | 0.22 px |
| OpenCV | 33.2% | 0.92 px |

## Installation

```bash
pip install locus-tag
```

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
from locus import Detector, CameraIntrinsics, PoseEstimationMode

# fx, fy, cx, cy
intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

# Returns [tx, ty, tz, qx, qy, qz, qw] for each tag
batch = detector.detect(
    img, 
    intrinsics=intrinsics, 
    tag_size=0.10, # physical side length in meters
    pose_estimation_mode=PoseEstimationMode.Accurate
)

if batch.poses is not None:
    # First tag translation
    print(batch.poses[0, :3])
```

### Configuration Overrides

```python
detector = locus.Detector(
    preset=locus.DetectorPreset.HighAccuracy,
    decode_mode=locus.DecodeMode.Soft,
    upscale_factor=2
)
```

## Visual Debugging

Built-in integration with the **[Rerun SDK](https://rerun.io)**:

```python
batch = detector.detect(img, debug_telemetry=True)
if batch.telemetry:
    print(batch.telemetry.subpixel_jitter)
```

## Documentation

- **[Architecture & Memory Model](https://noefontana.github.io/locus-tag/explanation/architecture/)**
- **[Coordinate Systems](https://noefontana.github.io/locus-tag/explanation/coordinates/)**
- **[Benchmarking Methodology](https://noefontana.github.io/locus-tag/engineering/benchmarking/)**

## License

Dual-licensed under [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
