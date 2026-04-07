# API Reference

This page provides detailed information about the Locus Python API.

## Core Interface

The primary entry point for single-threaded use is the `Detector` class.

::: locus.Detector
    options:
        show_root_toc_entry: false
        members:
            - __init__
            - detect

## Concurrent Detection

`Detector` supports concurrent multi-frame processing via `detect_concurrent`. Set `max_concurrent_frames` at construction time to control the internal pool size.

See the [Concurrent Detection how-to](../how-to/concurrent_detection.md) for full usage examples.

### `DetectorBuilder`

The fluent builder constructs `Detector` instances.

```python
import locus

detector = (
    locus.DetectorBuilder()
    .with_family(locus.TagFamily.AprilTag36h11)
    .with_threads(4)
    .with_max_concurrent_frames(8)
    .build()
)
```

| Method | Description |
| :--- | :--- |
| `with_family(family)` | Add a tag family to detect. |
| `with_decimation(n)` | Spatial decimation factor (default 1). |
| `with_threads(n)` | Rayon intra-frame thread count (0 = all cores). |
| `with_corner_refinement(mode)` | `CornerRefinementMode` for subpixel accuracy. |
| `with_decode_mode(mode)` | `DecodeMode.Hard` or `DecodeMode.Soft`. |
| `with_max_concurrent_frames(n)` | Pool size for `detect_concurrent` (default 1 = sequential). |
| `build()` | Build the `Detector`. |

**`detect_concurrent(frames, *, intrinsics=None, tag_size=None, pose_estimation_mode=PoseEstimationMode.Fast) -> list[DetectionResult]`**

Detect tags in multiple frames concurrently using Rayon. Releases the GIL for the entire parallel section. Pool contexts are managed internally. Rejected-corner data and telemetry are not available via this method.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `frames` | `list[np.ndarray]` | List of (H, W) uint8 grayscale frames. |
| `intrinsics` | `CameraIntrinsics \| None` | Camera intrinsics for 3D pose estimation. |
| `tag_size` | `float \| None` | Physical tag side length in metres. |
| `pose_estimation_mode` | `PoseEstimationMode` | `Fast` or `Accurate`. |

## Configuration

Locus uses Pydantic for robust configuration validation.

::: locus.DetectorConfig
    options:
        heading_level: 3

::: locus.DetectOptions
    options:
        heading_level: 3

## Data Models

These classes represent the output and internal state of the detection pipeline.

::: locus.DetectionBatch
    options:
        heading_level: 3

::: locus.Pose
    options:
        heading_level: 3

## Geometry

::: locus.CameraIntrinsics
    options:
        heading_level: 3

## Board Pose Estimation

Locus supports board-level 6-DOF pose estimation for AprilGrid and ChAruco boards.
All board types enforce dictionary bounds at construction time — a `ValueError` is raised
if the board requires more marker IDs than the target `TagFamily` provides.

For the underlying algorithm, see [Board-Level Pose Estimation](../explanation/algorithms.md#7-board-level-pose-estimation).

### `AprilGrid`

```python
locus.AprilGrid(
    rows: int,
    cols: int,
    spacing: float,        # gap between markers (metres)
    marker_length: float,  # physical marker side length (metres)
    family: TagFamily,
)
```

Immutable topology for an AprilTag grid board. `spacing = square_length - marker_length`.
Raises `ValueError` if `rows * cols > family.max_id_count()`.

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `rows` | `int` | Number of tag rows. |
| `cols` | `int` | Number of tag columns. |

### `CharucoBoard`

```python
locus.CharucoBoard(
    rows: int,
    cols: int,
    square_length: float,  # physical checkerboard square side (metres)
    marker_length: float,  # physical ArUco marker side (metres)
    family: TagFamily,
)
```

Immutable topology for a ChAruco board. Markers occupy squares where `(row + col)` is even.
Saddle points are at the outer corners of each square (`(rows-1)*(cols-1)` total).
Raises `ValueError` if the required marker count exceeds `family.max_id_count()`.

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `rows` | `int` | Number of square rows. |
| `cols` | `int` | Number of square columns. |

### `BoardEstimator`

Unified stateful estimator for both AprilGrid and ChAruco boards. All scratch buffers
are pre-allocated at construction; `estimate()` performs **zero heap allocations**.

**Construction:**

```python
# AprilGrid board
estimator = locus.BoardEstimator(board: AprilGrid)

# ChAruco board
estimator = locus.BoardEstimator.from_charuco(board: CharucoBoard)
```

**Estimation:**

```python
result: BoardEstimateResult = estimator.estimate(
    detector: locus.Detector,
    img: np.ndarray,           # (H, W) uint8 grayscale
    intrinsics: CameraIntrinsics,
)
```

`BoardEstimateResult` attributes:

| Attribute | Shape / Type | Description |
| :--- | :--- | :--- |
| `ids` | `(N,) int32` | Decoded tag IDs visible in this frame. |
| `corners` | `(N, 4, 2) float32` | Refined tag corner image coordinates. |
| `board_pose` | `(7,) float64` or `None` | `[tx, ty, tz, qx, qy, qz, qw]` in camera frame — `None` if insufficient observations. Pose origin is the board's top-left marker corner. |
| `board_cov` | `(6, 6) float64` or `None` | Pose covariance in $\mathfrak{se}(3)$ tangent space `[t, ω]`. |

## Enumerations

::: locus.TagFamily
    options:
        heading_level: 3

::: locus.DecodeMode
    options:
        heading_level: 3

::: locus.PoseEstimationMode
    options:
        heading_level: 3

::: locus.CornerRefinementMode
    options:
        heading_level: 3

::: locus.SegmentationConnectivity
    options:
        heading_level: 3
