# Locus (`locus-vision`)

## The Project Goal

**Locus** is a high-performance fiducial marker detector (AprilTag & ArUco) written in Rust with zero-copy Python bindings. It targets a balance of low-latency, high F1 score and precise pose estimation.

> [!WARNING]
> **Experimental Status**: Locus is currently in an active research and development phase. The API is subject to breaking changes, and while performance and recall are competitive with SOTA on the current benchmarks, it is **not recommended for production systems**. An extended photo-realistic benchmark dataset is currently in development under the [render-tag](https://github.com/NoeFontana/render-tag) repository.

Future development will focus on:
- Integrating with [render-tag](https://github.com/NoeFontana/render-tag) to provide a photo-realistic benchmark dataset.
- Benchmarking Locus against well-tuned SOTA detectors.
- Integrating custom trained neural networks optimized for edge inference.
- Adding support for more tag families and profiles.

## Performance (Full ICRA 2020 Dataset, 50 images)

| Detector | Recall | RMSE | Latency |
| :--- | :--- | :--- | :--- |
| **Locus (Soft)** | **95.42%** | 0.31 px | 110.7 ms |
| **Locus (Hard)** | **83.90%** | 0.25 px | **91.5 ms** |
| AprilTag | 62.34% | **0.22 px** | 101.5 ms |
| OpenCV | 33.16% | 0.92 px | 95.5 ms |

Note the higher aggregate RMSE for Locus is mostly correlated with its significantly higher recall (detecting more challenging, blurry tags).
Comparing the RMSE of the **same tags** detected by both Locus and AprilTag shows that Locus' precision is only slightly worse than AprilTag (Delta: **+0.0024 px**).

## Quick Start

### Install
```bash
uv run maturin develop -r
```

### Basic Usage
The simplest way to detect tags using default settings:

```python
import cv2
import locus

# Load image in grayscale
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Detect all tag families (default AprilTag 36h11)
tags = locus.detect_tags(img)

for t in tags:
    print(f"ID: {t.id}, Center: {t.center}, Corners: {t.corners}")
```

### Advanced Usage

For performance-critical applications, use the `Detector` class to maintain state and customize timing.

#### Performance Tuning & Decimation
If you are processing high-resolution images but your tags are large, use `decimation` to significantly speed up detection.

```python
detector = locus.Detector(
    threshold_tile_size=16, # Larger tiles are faster
    enable_bilateral=False  # Disable for 20-30% speedup if noise is low
)

# Use decimation=2 to process at half resolution (4x speed boost)
# Note: Use decimation=1 if some tags are very small (< 15-20 pixels)
tags, stats = detector.detect_with_stats(img, decimation=2)

print(f"Detected {len(tags)} tags in {stats.total_ms:.2f}ms")
```

#### Soft-Decision Decoding (Maximum Recall)
For challenging conditions (tiny, blurry, or noisy tags), Locus supports **Soft-Decision Decoding**. This uses Log-Likelihood Ratios (LLRs) instead of hard bit-binarization, providing a massive **+11.5% recall boost**.

```python
# Enable Soft-Decision mode for difficult tags
config = locus.DetectorConfig(
    decode_mode=locus.DecodeMode.Soft,
    decoder_min_contrast=10.0 # Recommended for soft mode
)
detector = locus.Detector(config=config)

# Soft mode is ~20% slower than Hard mode but detects significantly more tags
tags = detector.detect(img)
```

| Mode | Use Case | Latency |
| :--- | :--- | :--- |
| `Hard` (Default) | **High Resolution / Clean Imagery**. Minimum latency and maximum precision. | ~95ms |
| `Soft` | **Small / Blurry / Noisy Tags**. Use when missed detections are unacceptable. | ~117ms |

#### Checkerboard & Dictionary Support
Locus supports multiple dictionaries and specialized profiles for densely packed tags.

```python
# 1. Specialized Profile for Checkerboards
# Uses 4-way connectivity to prevent merging touching black squares
detector = locus.Detector.checkerboard()

# 2. Selecting Specific Tag Families
# Faster than searching for all families
detector.set_families([
    locus.TagFamily.AprilTag36h11,
    locus.TagFamily.ArUco4x4_50
])

tags = detector.detect(img)
```

| Profile | Connection | Use Case |
| :--- | :--- | :--- |
| `Default` | `Eight` | **Isolated Tags**. Best robustness to broken borders/noise. |
| `Checkerboard`| `Four` | **Dense Patterns**. Required when tag corners touch other black elements. |

#### Precise Configuration
The `DetectorConfig` class allows tuning every stage of the pipeline:

```python
config = locus.DetectorConfig(
    quad_min_area=16,
    subpixel_refinement_sigma=0.8,
    decoder_min_contrast=10.0  # Boost recall on blurry tags
)
detector = locus.Detector(config=config)
```

### Pose Estimation (IPPE-Square)
Locus implements **IPPE-Square** (Infinitesimal Plane-Based Pose Estimation) for SOTA pose accuracy. This solves the "perspective flip" ambiguity that plagues generic PnP solvers and refines the result using **Levenberg-Marquardt**.

```python
# To get pose, simply pass intrinsics and tag size
# You can pass a tuple/list or use the helper class
intrinsics = locus.CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

# Standard detection with pose estimation
tags = detector.detect(
    img,
    intrinsics=intrinsics,
    tag_size=0.16, # meters
    pose_estimation_mode=locus.PoseEstimationMode.Accurate # Optional: Use high-precision mode
)

for t in tags:
    if t.pose:
        print(f"Translation: {t.pose.translation}")
        print(f"Rotation: {t.pose.rotation}")
```

### Running Benchmarks & Regression Tests

Locus includes a rigorous suite to ensure detection quality and latency targets are met. All benchmarking and data preparation tools are unified in the `locus_bench.py` CLI.

#### 1. Dataset Preparation
```bash
uv run python scripts/locus_bench.py prepare
```

#### 2. Python Benchmarking CLI
```bash
# Compare recall, RMSE, and latency against other libs
uv run python scripts/locus_bench.py run real --compare

# Synthetic stress testing
uv run python scripts/locus_bench.py run synthetic --targets 1,10,50,100
```

#### 3. Rust Regression Suite (Core Engine)
The source of truth for core engine performance:
```bash
export LOCUS_DATASET_DIR=/path/to/icra2020
cargo test --release --test regression_icra2020 -- --test-threads=1
```

#### 4. Debug Visualization (with Rerun)
```bash
uv run python scripts/debug/visualize.py --scenario forward --limit 10
```

For detailed documentation, see the [Benchmarking Guide](docs/benchmarking.md).

1. **Set Environment Variable**: Point to the dataset root.
   ```bash
   export LOCUS_DATASET_DIR=/path/to/icra2020
   ```
2. **Run Tests**:
   ```bash
   # Quick check (fixtures + forward only, approx 10s)
   cargo test --release --test regression_icra2020

   # Comprehensive check (all datasets, approx 2 mins)
   cargo test --release --test regression_icra2020 -- --ignored

   # Accurate latency measurement (sequential)
   cargo test --release --test regression_icra2020 -- --test-threads=1
   ```
   *Note: `--release` is mandatory for performance benchmarking.*

## License

This project is licensed under either of:

 * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 * [MIT license](http://opensource.org/licenses/MIT)

at your option.

Any contribution intentionally submitted
for inclusion in `locus-tag` by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
