# locus-tag

[![CI](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Locus** is a high-performance fiducial marker detector (AprilTag & ArUco) written in Rust with zero-copy Python bindings. Designed for robotics and autonomous systems, it aims to balance low latency, high recall, and sub-pixel precision.

> [!WARNING]
> **Experimental Status**: Locus is currently an experimental project. The API is subject to breaking changes. While performance exceeds alternatives on ICRA2020, it is **not recommended for production systems**. Photo-realistic benchmarks are being developed under [render-tag](https://github.com/NoeFontana/render-tag).

## Key Features

- **High-Performance Core**: Written in Rust (2024 Edition) with a focus on Data-Oriented Design.
- **Encapsulated Facade**: Simple, ergonomic `Detector` API that manages complex memory lifetimes (arenas, SoA batches) internally.
- **Runtime SIMD Dispatch**: Automatically utilizes AVX2, AVX-512, or NEON based on host CPU capabilities.
- **Vectorized Python API**: Returns detection results as a single `DetectionBatch` object containing parallel NumPy arrays for maximum throughput.
- **GIL-Free Execution**: Releases the Python Global Interpreter Lock (GIL) during detection to enable true multi-threaded applications.
- **Memory Efficient**: Uses `bumpalo` arena allocation to achieve zero heap allocations in the detection hot-path.
- **Advanced Pose Estimation**: High-precision 6-DOF recovery using IPPE-Square or weighted Levenberg-Marquardt with corner uncertainty modeling.
- **Visual Debugging**: Native integration with the **[Rerun SDK](https://rerun.io)** for real-time pipeline inspection.

## Performance (ICRA 2020 Dataset)

Evaluated on the standard ICRA 2020 benchmark (50 images). Latency measured on a modern desktop CPU.

| Detector | Recall | RMSE | Latency (1080p avg) |
| :--- | :---: | :---: | :---: |
| **Locus (Soft)** | **93.16%** | 0.26 px | 79.2 ms |
| **Locus (Hard)** | 74.35% | 0.24 px | **59.5 ms** |
| AprilTag 3 | 62.34% | **0.22 px** | 105.9 ms |
| OpenCV | 33.16% | 0.92 px | 108.2 ms |

*Note: Locus utilizes a Structure of Arrays (SoA) layout to achieve ~3.8x speedup over previous versions in dense tag environments.*

## Quick Start

### Installation

```bash
pip install locus-tag
```

For development, build from source using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/NoeFontana/locus-tag
cd locus-tag
uv run maturin develop -r
```

### Basic Usage

The simplest way to detect tags using default settings:

```python
import cv2
import locus

# Load image in grayscale
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Create detector and detect tags (defaults to AprilTag 36h11)
detector = locus.Detector()
batch = detector.detect(img)

# batch is a vectorized DetectionBatch object
for i in range(len(batch)):
    print(f"ID: {batch.ids[i]}, Center: {batch.centers[i]}")
```

### Advanced Configuration

Use semantic keyword arguments for fine-grained control and performance tuning:

```python
from locus import Detector, TagFamily, DecodeMode

# Configure for maximum recall on small, blurry tags
detector = Detector(
    decode_mode=DecodeMode.Soft,
    upscale_factor=2,
    families=[TagFamily.AprilTag36h11, TagFamily.ArUco4x4_50]
)

batch = detector.detect(img)
```

### 3D Pose Estimation

Recover the 6-DOF transformation between the camera and the tag:

```python
from locus import CameraIntrinsics, PoseEstimationMode

# Camera parameters (fx, fy, cx, cy)
intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

# Pass intrinsics and physical tag size (meters)
batch = detector.detect(
    img, 
    intrinsics=intrinsics, 
    tag_size=0.10,
    pose_estimation_mode=PoseEstimationMode.Accurate
)

if batch.poses is not None:
    # batch.poses is (N, 7) array: [tx, ty, tz, qx, qy, qz, qw]
    print(f"First tag translation: {batch.poses[0, :3]}")
    print(f"First tag quaternion: {batch.poses[0, 3:]}")
```

## Visual Debugging with Rerun

Locus provides a powerful visualization tool to inspect every stage of the pipeline (thresholding, segmentation, quad candidates, bit grids).

```bash
# Run the visualizer on a dataset using the dev/bench dependency groups
uv run --group dev --group bench tools/cli.py visualize --scenario forward --limit 5
```

## Development & Benchmarking

Locus includes a rigorous suite to ensure detection quality and latency targets.

```bash
# Prepare local datasets
uv run --group dev --group bench tools/cli.py bench prepare

# Run full evaluation suite and compare with competitors
uv run --group dev --group bench tools/cli.py bench real --compare
```

Detailed documentation for profiling, architecture, and coordinate systems is available in the **[Docs Site](https://noefontana.github.io/locus-tag/)**.

## License

Dual-licensed under [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
