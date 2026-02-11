# Locus (`locus-vision`)

## The Project Goal

**Locus** is a high-performance fiducial marker detector (AprilTag & ArUco) written in Rust with zero-copy Python bindings. It targets a balance of low latency, high recall, and precise pose estimation.

> [!WARNING]
> **Experimental Status**: Locus is an experiment on AI-assisted coding and the code is still actively iterated on. The API is subject to breaking changes. While performance is high on current benchmarks, it is **not recommended for production systems** yet. Photo-realistic benchmarks are in development under [render-tag](https://github.com/NoeFontana/render-tag).

## Performance (ICRA 2020 Dataset)

| Detector | Recall | RMSE | Latency (avg) |
| :--- | :--- | :--- | :--- |
| **Locus (Soft)** | **95.42%** | 0.26 px | 129.1 ms |
| **Locus (Hard)** | **83.90%** | 0.25 px | **97.9 ms** |
| AprilTag 3 | 62.34% | **0.22 px** | 121.0 ms |
| OpenCV | 33.16% | 0.92 px | 113.0 ms |

*Note: Locus' higher recall (detecting more challenging tags) correlates with its aggregate RMSE. On identical detections, Locus' precision is within **+0.0024 px** of AprilTag.*

## Quick Start

### Install
```bash
pip install locus-tag
# Or from source
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
    print(f"ID: {t.id}, Center: {t.center}")
```

### Advanced Features
For performance tuning, pose estimation, and specialized profiles, see the **[User Guide](guide.md)**.

- **Decimation**: Speed up processing on high-res images.
- **Soft-Decoding**: Maximum recall for blurry or small tags (+11.5% boost).
- **Pose Estimation**: Advanced IPPE-Square and Probabilistic refinement.
- **Profiles**: Optimized settings for checkerboards and dense patterns.

## Development & Benchmarking

Locus includes a rigorous suite to ensure detection quality and latency targets.

```bash
# Prepare datasets
uv run python scripts/locus_bench.py prepare

# Run evaluation suite
uv run python scripts/locus_bench.py run real --compare
```

Detailed instructions for profiling and regression testing are available in the **[Benchmarking Guide](benchmarking.md)**.

## License

Dual-licensed under [Apache 2.0](https://github.com/NoeFontana/locus-tag/blob/main/LICENSE-APACHE) or [MIT](https://github.com/NoeFontana/locus-tag/blob/main/LICENSE-MIT).
