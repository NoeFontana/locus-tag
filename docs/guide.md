# User Guide

This guide covers advanced configuration and features of the **Locus** detector.

## Performance Tuning & Decimation

For performance-critical applications, using `decimation` can significantly speed up the detection pipeline by processing a downsampled version of the image.

```python
import locus

detector = locus.Detector(
    threshold_tile_size=16, # Larger tiles are faster (8-16 is typical)
    enable_bilateral=False  # Disable for 20-30% speedup if image noise is low
)

# Use decimation=2 to process at half resolution (4x speed boost)
# Note: Use decimation=1 if tags are small (< 20 pixels)
tags, stats = detector.detect_with_stats(img, decimation=2)

print(f"Detected {len(tags)} tags in {stats.total_ms:.2f}ms")
```

| Parameter | Impact | Recommendation |
| :--- | :--- | :--- |
| `decimation` | Linear speedup in preprocessing. | Use `2` for 1080p+ images if tags are >40px. |
| `enable_bilateral` | High. Improves noise rejection. | Disable for high-SNR sensors. |
| `threshold_tile_size` | Low. Affects local adaptive threshold. | Use `8` or `16`. |

## Soft-Decision Decoding (Maximum Recall)

For challenging conditions where tags are tiny, blurry, or noisy, Locus supports **Soft-Decision Decoding**. This mode uses Log-Likelihood Ratios (LLRs) instead of hard bit-binarization, typically providing a **+10-15% recall boost** on difficult datasets.

```python
# Enable Soft-Decision mode
config = locus.DetectorConfig(
    decode_mode=locus.DecodeMode.Soft,
    decoder_min_contrast=10.0 # Lower threshold to capture faint tags
)
detector = locus.Detector(config=config)

tags = detector.detect(img)
```

| Mode | Use Case | Latency |
| :--- | :--- | :--- |
| `Hard` (Default) | High Resolution / Clean Imagery. | Minimum latency. |
| `Soft` | Small / Blurry / Noisy Tags. | ~20% overhead. |

## Specialized Profiles

Locus includes pre-configured profiles for specific use cases.

### Checkerboard Detection
Used for calibration patterns or densely packed tags where black squares touch. This profile uses 4-way connectivity to prevent component merging.

```python
detector = locus.Detector.checkerboard()
```

### Targeted Families
Searching for fewer families reduces the decoding search space and improves latency.

```python
detector.set_families([
    locus.TagFamily.AprilTag36h11,
    locus.TagFamily.ArUco4x4_50
])
```

## Precise Configuration

The `DetectorConfig` class allows fine-tuning every stage of the pipeline:

```python
config = locus.DetectorConfig(
    quad_min_area=16,           # Filter small components early
    subpixel_refinement_sigma=0.8, # Gaussian kernel for corner refinement
    decoder_min_contrast=10.0      # Sensitivity to bit transitions
)
detector = locus.Detector(config=config)
```

## Pose Estimation

Locus implements SOTA pose solvers to recover the 6-DOF transformation between the camera and the tag.

### IPPE-Square
For standard detection, we use **IPPE-Square** (Infinitesimal Plane-Based Pose Estimation). It provides an analytical solution that resolves the Necker reversal (perspective flip) ambiguity.

```python
intrinsics = locus.CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

# Pass intrinsics and tag size (meters) to enable pose estimation
tags = detector.detect(
    img,
    intrinsics=intrinsics,
    tag_size=0.16
)

for t in tags:
    if t.pose:
        print(f"Translation: {t.pose.translation}") # [x, y, z]
        print(f"Rotation: {t.pose.rotation}")       # 3x3 Matrix
```

### High-Precision (Probabilistic) Mode
When `PoseEstimationMode.Accurate` is selected, Locus computes the **Structure Tensor** for each corner to estimate position uncertainty, then performs an **Anisotropic Weighted Levenberg-Marquardt** refinement.

```python
tags = detector.detect(
    img,
    intrinsics=intrinsics,
    tag_size=0.16,
    pose_estimation_mode=locus.PoseEstimationMode.Accurate
)

for t in tags:
    if t.pose_covariance:
        # Full 6x6 covariance matrix [R|t]
        print(f"Pose Uncertainty: {t.pose_covariance}")
```
