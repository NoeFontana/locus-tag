
# Locus

A high-performance AprilTag and ArUco detector in Rust with Python bindings.
This project is an experiment in LLM-assisted library development, targeting 1-10ms latencies for modern computer vision tasks.

## Performance (1080p, 1-thread)

| Detector | 1 Tag | 200 Tags |
| :--- | :--- | :--- |
| **Locus** | **5.4 ms** | **10.9 ms** |
| AprilTag (SOTA) | 14.2 ms | 38.6 ms |
| OpenCV | 20.7 ms | 65.4 ms |

## Quick Start

### Install
```bash
uv run maturin develop -r
```

### Usage
```python
import cv2, locus
img = cv2.imread("image.jpg", 0)
tags = locus.detect_tags(img)

for t in tags:
    print(f"ID: {t.id}, Corners: {t.corners}")
```

### Checkerboard Support

Locus supports detecting tags embedded in checkerboard patterns (where corners touch black squares).
This requires switching to **4-way connectivity**, as 8-way connectivity (default) merges touching corners.

```python
import locus

# Configure for checkerboard tags
detector = locus.Detector(
    segmentation_connectivity=locus.SegmentationConnectivity.Four
)

tags = detector.detect(img)
```

| Mode | Use Case |
| :--- | :--- |
| `Eight` (Default) | **Isolated Tags**. Best performance and robustness to broken borders. |
| `Four` | **Checkerboards**. Required when tag corners touch other black elements. |

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
