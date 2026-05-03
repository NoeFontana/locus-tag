# User Guide

This guide covers advanced configuration and features of the **Locus** detector.

## Profiles: the one knob you should reach for first

Detector settings are loaded from JSON **profiles**. Three are shipped in the
wheel: `standard`, `grid`, and `high_accuracy`.

```python
import locus

detector = locus.Detector(profile="standard")      # default; dense multi-tag
tags = detector.detect(img)
```

Per-call orchestration (`decimation`, `threads`, `families`) stays outside
the profile because it describes *how* the detector is invoked, not *what*
it detects:

```python
detector = locus.Detector(
    profile="standard",
    decimation=2,                                 # 4x preprocessing speedup
    families=[locus.TagFamily.AprilTag36h11],
)
```

## Tweaking a profile

Load a shipped profile, edit the relevant nested group, then pass it back:

```python
base = locus.DetectorConfig.from_profile("standard").model_dump()
base["threshold"]["tile_size"] = 16               # larger tiles run faster
base["decoder"]["min_contrast"] = 10.0

custom = locus.DetectorConfig.model_validate(base)
detector = locus.Detector(config=custom)
```

`model_validate` runs the full invariant suite — radius ordering, fill-ratio
ordering, and the cross-group compatibility checks (e.g. `EdLines` refuses
`Erf` refinement) — so any inconsistency surfaces as a
`pydantic.ValidationError` before the Rust detector is constructed.

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

### Checkerboard Detection
Used for calibration patterns or densely packed tags where black squares
touch. The `grid` profile uses 4-way connectivity to prevent component
merging:

```python
detector = locus.Detector(profile="grid")
```

### High-Accuracy Metrology
For isolated-tag pose extraction at high resolution (`EdLines` + geometric
corners + no sub-pixel refinement):

```python
detector = locus.Detector(profile="high_accuracy")
```

### Targeted Families
Searching for fewer families reduces the decoding search space and improves
latency:

```python
detector.set_families([
    locus.TagFamily.AprilTag36h11,
    locus.TagFamily.ArUco4x4_50,
])
```

## Precise Configuration

Every detection knob lives in one of five nested groups (`threshold`,
`quad`, `decoder`, `pose`, `segmentation`). Tweak any of them by round-
tripping a profile through a dict:

```python
base = locus.DetectorConfig.from_profile("standard").model_dump()
base["quad"]["min_area"] = 16                    # filter small components
base["quad"]["subpixel_refinement_sigma"] = 0.8  # corner-refinement kernel
base["decoder"]["min_contrast"] = 10.0           # bit-transition sensitivity

detector = locus.Detector(config=locus.DetectorConfig.model_validate(base))
```

## Pose Estimation

Locus implements modern pose solvers to recover the 6-DOF transformation between the camera and the tag.

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
