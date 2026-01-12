# Locus

High-performance AprilTag and ArUco detector in Rust with zero-copy Python bindings.

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
