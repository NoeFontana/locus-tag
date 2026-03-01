# locus-tag

[![CI](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFontana/locus-tag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Locus** is a high-performance fiducial marker detector (AprilTag & ArUco) written in Rust with zero-copy Python bindings. Designed for robotics and autonomous systems, it aims to balance low latency, high recall, and sub-pixel precision.

> [!WARNING]
> **Experimental Status**: Locus is currently an experimental project. The API is subject to breaking changes. While performance exceeds alternatives on ICRA2020, it is **not recommended for production systems**. Photo-realistic benchmarks are being developed under [render-tag](https://github.com/NoeFontana/render-tag).

## Key Features

- **High-Performance Core**: Written in Rust (2024 Edition) with a focus on Data-Oriented Design.
- **Runtime SIMD Dispatch**: Automatically utilizes AVX2, AVX-512, or NEON based on host CPU capabilities.
- **Zero-Copy Python API**: Direct ingestion of NumPy arrays via `pyo3` and `numpy` bindings.
- **Memory Efficient**: Uses `bumpalo` arena allocation to achieve zero heap allocations in the detection hot-path.
- **Soft-Decoding**: Optional Log-Likelihood Ratio (LLR) decoding for maximum recall on blurry or noisy tags (+11.5% boost).
- **Advanced Pose Estimation**: High-precision 6-DOF recovery using IPPE-Square or weighted Levenberg-Marquardt with corner uncertainty modeling.
- **Visual Debugging**: Native integration with the **[Rerun SDK](https://rerun.io)** for real-time pipeline inspection.

## Performance (ICRA 2020 Dataset)

Evaluated on the standard ICRA 2020 benchmark (50 challenging images). Latency measured on a modern desktop CPU.

| Detector | Recall | RMSE | Latency (avg) |
| :--- | :---: | :---: | :---: |
| **Locus (Soft)** | **95.42%** | 0.26 px | 129.1 ms |
| **Locus (Hard)** | **83.90%** | 0.25 px | **97.9 ms** |
| AprilTag 3 | 62.34% | **0.22 px** | 121.0 ms |
| OpenCV | 33.16% | 0.92 px | 113.0 ms |

*Note: Locus' higher recall (detecting more challenging tags) correlates with its aggregate RMSE. On identical detections, Locus' precision is within **+0.0024 px** of AprilTag.*

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

# Detect tags (defaults to AprilTag 36h11)
tags = locus.detect_tags(img)

for t in tags:
    print(f"ID: {t.id}, Center: {t.center}, Hamming: {t.hamming}")
```

### Advanced Configuration

Use the `Detector` class for fine-grained control and performance tuning:

```python
from locus import Detector, DetectorConfig, TagFamily

# Configure for maximum recall on small, blurry tags
config = DetectorConfig(
    decode_mode="Soft",
    upscale_factor=2,
    enable_sharpening=True
)
detector = Detector(config)

# Set specific families and detect
detector.set_families([TagFamily.AprilTag36h11, TagFamily.ArUco4x4_50])
tags = detector.detect(img)
```

### 3D Pose Estimation

Recover the 6-DOF transformation between the camera and the tag:

```python
from locus import CameraIntrinsics

# Camera parameters (fx, fy, cx, cy)
intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

# Pass intrinsics and physical tag size (meters)
tags = detector.detect(img, intrinsics=intrinsics, tag_size=0.16)

for t in tags:
    if t.pose:
        print(f"Tag {t.id} Position: {t.pose.translation}")
```

## Visual Debugging with Rerun

Locus provides a powerful visualization tool to inspect every stage of the pipeline (thresholding, segmentation, quad candidates, bit grids).

```bash
# Install rerun-sdk
pip install rerun-sdk

# Run the visualizer on a dataset
uv run python scripts/debug/visualize.py --scenario forward --limit 5
```

## Development & Benchmarking

Locus includes a rigorous suite to ensure detection quality and latency targets.

```bash
# Prepare local datasets
uv run python scripts/locus_bench.py prepare

# Run full evaluation suite and compare with competitors
uv run python scripts/locus_bench.py run real --compare
```

Detailed documentation for profiling, architecture, and coordinate systems is available in the **[Docs Site](https://noefontana.github.io/locus-tag/)**.

## License

Dual-licensed under [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
