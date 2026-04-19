# Install with distortion support

The PyPI wheel (`pip install locus-tag`) is compiled **without** the `non_rectified` Cargo feature. It supports the ideal pinhole camera model only. To detect markers in imagery from unrectified cameras — i.e. cameras with significant lens distortion (Brown-Conrady polynomial, Kannala-Brandt equidistant fisheye) — you must build from source with the `non_rectified` feature enabled.

## Prerequisites

- A working **Rust toolchain** (stable). Install via [rustup](https://rustup.rs/).
- A C compiler (for any transitive C dependencies of `numpy` / `pyo3`).
- Python ≥ 3.10 with `pip` or `uv`.

## Install from source with the feature enabled

Pass `--features locus-py/non_rectified` to the maturin build backend via `MATURIN_PEP517_ARGS`:

```bash
MATURIN_PEP517_ARGS="--features locus-py/non_rectified" \
    pip install --no-binary=locus-tag --force-reinstall locus-tag
```

The equivalent `uv` invocation:

```bash
MATURIN_PEP517_ARGS="--features locus-py/non_rectified" \
    uv pip install --no-binary=locus-tag --reinstall locus-tag
```

`--no-binary=locus-tag` tells pip to ignore the prebuilt wheel on PyPI and install from the source distribution instead. `MATURIN_PEP517_ARGS` is then forwarded to `cargo build`.

## Verify

```python
import locus

assert locus.HAS_NON_RECTIFIED
assert hasattr(locus.DistortionModel, "BrownConrady")
assert hasattr(locus.DistortionModel, "KannalaBrandt")
```

## Use

```python
from locus import CameraIntrinsics, DistortionModel

# Brown-Conrady polynomial distortion (OpenCV convention)
intrinsics = CameraIntrinsics(
    fx=800.0, fy=800.0, cx=640.0, cy=360.0,
    distortion_model=DistortionModel.BrownConrady,
    dist_coeffs=[-0.3, 0.1, 0.001, -0.002, 0.0],  # [k1, k2, p1, p2, k3]
)

# Kannala-Brandt equidistant fisheye
fisheye = CameraIntrinsics(
    fx=380.0, fy=380.0, cx=320.0, cy=240.0,
    distortion_model=DistortionModel.KannalaBrandt,
    dist_coeffs=[0.1, -0.01, 0.001, 0.0],  # [k1, k2, k3, k4]
)
```

## Error: `LocusFeatureError`

Using a distortion model on the lean (PyPI) wheel raises `LocusFeatureError` with the install recipe above:

```python
>>> import locus
>>> locus.DistortionModel.BrownConrady
Traceback (most recent call last):
  ...
locus.LocusFeatureError: DistortionModel.BrownConrady is unavailable.

Distortion models require the `non_rectified` Cargo feature, which is not
compiled into this wheel. Reinstall from source:
    MATURIN_PEP517_ARGS="--features locus-py/non_rectified" \
        pip install --no-binary=locus-tag --force-reinstall locus-tag
...
```

Catch it programmatically with `locus.LocusFeatureError`:

```python
import locus

try:
    model = locus.DistortionModel.BrownConrady
except locus.LocusFeatureError:
    model = locus.DistortionModel.Pinhole  # fall back to rectified input
```

## Why isn't this a separate PyPI wheel?

Two wheels (`locus-tag`, `locus-tag-unrectified`) would both ship a `locus.abi3.so` native extension and mechanically overwrite each other on `pip install`. The sdist-with-feature path keeps the PyPI surface clean while letting downstream robotics and AV users opt in to distortion at build time.
