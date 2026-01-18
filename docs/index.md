# Locus

**Locus** is a high-performance AprilTag and ArucoTag detector written in Rust with Python bindings. It targets 1-10ms latency and was created as an experiment on LLM-assisted library development.

## Features
- **High Performance**: Written in Rust with SIMD optimizations.
- **Zero-Copy**: Efficient Python integration via NumPy.
- **Easy to Use**: Pythonic API with type hints.
- **Documented Architecture**: See the [System Architecture](architecture.md) for design details.

## Installation

```bash
pip install locus-tag
```

## Basic Usage

```python
import locus
import cv2

# Load image
image = cv2.imread("tag.png", cv2.IMREAD_GRAYSCALE)

# Create detector with custom configuration
detector = locus.Detector(
    threshold_tile_size=32,  # Window size for adaptive thresholding
    quad_min_area=100,       # Minimum pixel area for a candidate quad
    quad_min_edge_score=20.0 # Strictness of edge detection
)

# Optional: Set families to decode (default is none)
detector.set_families([locus.TagFamily.AprilTag36h11, locus.TagFamily.ArUco4x4_50])

# Detect tags
detections = detector.detect(image)

for detection in detections:
    print(f"Found tag {detection.id} at {detection.center}")
```
