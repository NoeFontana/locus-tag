# API Reference

This page provides detailed information about the Locus Python API.

## Core Interface

The primary entry point for using Locus is the `Detector` class.

::: locus.Detector
    options:
        show_root_toc_entry: false
        members:
            - __init__
            - detect

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

### `CharucoRefiner`

```python
locus.CharucoRefiner(board: CharucoBoard)
```

Stateful estimator that extracts ChAruco saddle points from a decoded `DetectionBatch`
and estimates the board pose. All scratch buffers are pre-allocated at construction;
`estimate()` performs **zero heap allocations**.

```python
result: dict = refiner.estimate(
    img: np.ndarray,           # (H, W) uint8 grayscale
    batch_view,                # internal DetectionBatchView from CharucoRefiner
    intrinsics: CameraIntrinsics,
) -> dict
```

Returned dictionary:

| Key | Shape / Type | Description |
| :--- | :--- | :--- |
| `ids` | `(N,) int32` | Decoded ArUco tag IDs. |
| `corners` | `(N, 4, 2) float32` | Refined tag corner coordinates. |
| `saddle_ids` | `(S,) int32` | Accepted saddle-point indices into `CharucoBoard.saddle_points`. |
| `saddle_pts` | `(S, 2) float32` | Refined saddle image coordinates. |
| `saddle_obj` | `(S, 3) float64` | Board-frame 3D coordinates of accepted saddles. |
| `board_pose` | `(7,) float64` or `None` | `[tx, ty, tz, qx, qy, qz, qw]` — `None` if fewer than 4 saddles accepted. |
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
