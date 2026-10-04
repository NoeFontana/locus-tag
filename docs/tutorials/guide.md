# User Guide

This guide covers advanced configuration and features of the **Locus** detector.

## Profiles: the one knob you should reach for first

Detector settings are loaded from JSON **profiles**. Three are shipped in the
wheel: `standard`, `grid`, and `high_accuracy`.

```python
import locus

detector = locus.Detector(profile="standard")  # default; dense multi-tag
tags = detector.detect(img)
```

Per-call orchestration (`decimation`, `threads`, `families`) stays outside
the profile because it describes *how* the detector is invoked, not *what*
it detects:

```python
detector = locus.Detector(
    profile="standard",
    decimation=2,  # 4x preprocessing speedup
    families=[locus.TagFamily.AprilTag36h11],
)
```

## Tweaking a profile

Load a shipped profile, edit the relevant nested group, then pass it back:

```python
base = locus.DetectorConfig.from_profile("standard").model_dump()
base["threshold"]["tile_size"] = 16  # larger tiles run faster
base["decoder"]["min_contrast"] = 10.0

custom = locus.DetectorConfig.model_validate(base)
detector = locus.Detector(config=custom)
```

`model_validate` runs the full invariant suite — per-field ranges, fill-ratio
ordering, and the cross-group compatibility checks (e.g. `EdLines` refuses
`Erf` refinement) — so any inconsistency surfaces as a
`pydantic.ValidationError` before the Rust detector is constructed. The Rust
side re-validates the config when the detector is built.

## Loading a custom profile from JSON

For reproducibility, teams typically keep their tuned profile under version
control as a JSON file and load it via `from_profile_json`:

```python
with open("my_profile.json") as f:
    detector = locus.Detector(config=locus.DetectorConfig.from_profile_json(f.read()))
```

The shipped `standard.json` is a good starting template; copy it, edit the
nested groups, and load the copy. The JSON Schema at
`schemas/profile.schema.json` powers editor autocomplete.

## Specialized Profiles

### What `standard` does

`standard` is the default and the profile benchmarked against OpenCV and
aruco_nano on real and rendered data. In order, it:

- sharpens the image, thresholds it against local tile extremes
  (`TileMidExtreme`), and labels dark regions with **4-connectivity**, so
  squares that touch only at a corner (AprilGrid connectors, checkerboard
  cells) stay separate;
- leaves the blob-shape quad gates off and judges candidates by marker
  evidence instead: candidates are **decoded first** from their contour
  corners, and only those that decode (or nearly do) are refined and
  re-decoded to confirm the same id; every match must also show the marker's
  **dark border ring** within a per-family error budget;
- re-estimates each decoded marker's corners (`decoder.corner_subpix`):
  gradient-orthogonality junction corners fused with whole-edge line corners,
  repair of grossly misplaced corners, and a per-marker calibration of the
  photometric edge shift measured from the marker's own bit edges;
- rejects markers cut by the image frame.

The corner fusion and calibration run on undistorted images (no intrinsics, or
intrinsics without a distortion model). See the
[pipeline](../explanation/pipeline.md) for details.

### Checkerboard Detection
For calibration boards and densely packed tags. `grid` shares `standard`'s
4-connectivity, decode-first ordering, ring check and corner stage, and
differs in three ways:

- it keeps the blob-shape quad gates (`min_fill_ratio` 0.10, `min_density`
  0.15, `max_elongation` 20);
- sharpening is off;
- the contrast gates are lower (`decoder.min_contrast` 10 vs 20,
  `quad.min_edge_score` 2 vs 4), for low-contrast board prints.

```python
detector = locus.Detector(profile="grid")
```

### High-Accuracy Metrology
For isolated-tag pose extraction at high resolution: `EdLines` quads for
well-resolved tags (ContourRdp + ERF below 2.5 pixels per bit), no decoder
corner stage, no ring check, 8-connectivity. When called with camera
intrinsics and a `tag_size`, it also runs **model-edge pose refinement** —
aligning the decoded tag's internal bit-grid edges to the image to tighten
rotation accuracy:

```python
detector = locus.Detector(profile="high_accuracy")
```

### Targeted Families
Searching for fewer families reduces the decoding search space and improves
latency:

```python
detector.set_families(
    [
        locus.TagFamily.AprilTag36h11,
        locus.TagFamily.ArUco4x4_50,
    ]
)
```

## Precise Configuration

Every detection knob lives in one of five nested groups (`threshold`,
`quad`, `decoder`, `pose`, `segmentation`). Tweak any of them by round-
tripping a profile through a dict:

```python
base = locus.DetectorConfig.from_profile("standard").model_dump()
base["quad"]["min_area"] = 16  # filter small components
base["quad"]["subpixel_refinement_sigma"] = 0.8  # corner-refinement kernel
base["decoder"]["min_contrast"] = 10.0  # bit-transition sensitivity

detector = locus.Detector(config=locus.DetectorConfig.model_validate(base))
```

## Pose Estimation

Locus implements modern pose solvers to recover the 6-DOF transformation between the camera and the tag.

### IPPE-Square
For standard detection, we use **IPPE-Square** (Infinitesimal Plane-Based Pose Estimation). It provides an analytical solution that resolves the Necker reversal (perspective flip) ambiguity.

```python
intrinsics = locus.CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

# Pass intrinsics and tag size (meters) to enable pose estimation
tags = detector.detect(img, intrinsics=intrinsics, tag_size=0.16)

for t in tags:
    if t.pose:
        print(f"Translation: {t.pose.translation}")  # [x, y, z]
        print(f"Rotation: {t.pose.rotation}")  # 3x3 Matrix
```

### Per-tag Covariance
When the call supplies an `img` view (and tag-size + intrinsics), Locus
computes the **Structure Tensor** at each corner to estimate position
uncertainty and runs **Anisotropic Weighted Levenberg-Marquardt** refinement.
The returned pose carries a 6×6 covariance matrix. Pose-only refits that
skip the image fall back to unweighted Huber LM and report no covariance.

```python
tags = detector.detect(
    img,
    intrinsics=intrinsics,
    tag_size=0.16,
)

for t in tags:
    if t.pose_covariance:
        # Full 6x6 covariance matrix [R|t]
        print(f"Pose Uncertainty: {t.pose_covariance}")
```
