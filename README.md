
# Locus

A high-performance AprilTag and ArUco detector in Rust with Python bindings.
This project is an experiment in LLM-assisted library development, targeting 1-10ms latencies for modern computer vision tasks.

## Performance (Full ICRA 2020 Dataset, 50 images)

| Detector | Recall | RMSE | Latency |
| :--- | :--- | :--- | :--- |
| **Locus** | **72.52%** | 0.30 px | **70.0 ms** |
| AprilTag (SOTA) | 62.34% | **0.22 px** | 93.9 ms |
| OpenCV | 33.16% | 0.92 px | 100.7 ms |

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

## Development & Testing

Locus includes a rigorous regression suite to ensure detection quality on standard datasets (ICRA 2020).

### Running Regression Tests

The regression suite validates that `Locus` strictly matches or exceeds the ground truth for thousands of images. It is skipped by default if the dataset is missing.

The suite runs 8 parallel tests covering different scenarios:
- **Standard (8-way)**: `forward`, `rotation`, `random`, `circle`
- **Checkerboard (4-way)**: `checkerboard_forward`, `checkerboard_rotation`, `checkerboard_random`, `checkerboard_circle`

1. **Download the Dataset**: Ensure you have the ICRA 2020 dataset (images and `tags.csv`).
2. **Set Environment Variable**: Point to the dataset root.
   ```bash
   export LOCUS_DATASET_DIR=/path/to/icra2020
   ```
3. **Run the Test**:
   ```bash
   cargo test --test regression_icra2020 --release
   ```
   *Note: `--release` is recommended for performance on large datasets.*

**Strict Mode**: The test will fail (panic) if any image yields fewer detections than the ground truth.
