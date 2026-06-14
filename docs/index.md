# Locus

A production-grade, memory-safe fiducial-marker detector for robotics,
autonomous vehicles, and perception engineering. Locus detects
**AprilTag**, **ArUco**, **AprilGrid**, and **ChArUco** markers and
boards — implemented in Rust with zero-copy Python bindings via
PyO3 (abi3).

!!! warning "Experimental status"
    API is subject to breaking changes until 1.0.0 ships. The main
    workstreams toward 1.0.0 are reducing the API surface and
    validating against non-synthetic data. Until then, this library
    isn't recommended for production systems. Distortion-model
    support is experimental and requires a ground-up redesign.

## At a glance

- **Zero-copy ingestion** — NumPy arrays accessed via the Python
  Buffer Protocol; the FFI boundary releases the GIL during
  `detect()`.
- **Arena-allocated hot path** — `bumpalo`-backed per-frame
  allocator; zero system-allocator calls in the detection loop.
- **SoA results** — `DetectionBatch` exposes parallel NumPy arrays
  for IDs, corners, and poses; cache-conscious and vectorizable.
- **OpenCV parity** — tag layout, bit ordering, and canonical
  orientation follow `cv2.aruco` conventions for ecosystem
  interoperability.
- **6-DOF pose recovery** — IPPE-Square or weighted
  Levenberg-Marquardt with per-corner uncertainty.
- **Cross-platform wheels** — Linux (manylinux + musllinux × x86_64
  + aarch64), macOS (x86_64 + aarch64), Windows x64.

## Install

```bash
pip install locus-tag
```

The PyPI wheel is built for **rectified (pinhole)** imagery. For
Brown-Conrady or Kannala-Brandt distortion models, build from
source with the `non_rectified` feature — see
[Install with distortion support](how-to/install-with-distortion.md).

## Quick start

```python
import cv2
import locus

img = cv2.imread("tags.jpg", cv2.IMREAD_GRAYSCALE)
detector = locus.Detector(families=[locus.TagFamily.AprilTag36h11])

batch = detector.detect(img)
print(batch.ids)            # (N,)
print(batch.corners.shape)  # (N, 4, 2)
```

Pass `intrinsics` and `tag_size` to recover full 6-DOF poses.
The [Detection guide](tutorials/guide.md) walks through the full
configure → detect → solve flow.

## Where to next

### Tutorials
Hands-on, end-to-end walkthroughs for first-time users.

- [Detection guide](tutorials/guide.md) — install, configure,
  detect, and recover 6-DOF poses.

### How-To guides
Targeted recipes for specific tasks.

- [Add a custom dictionary](how-to/add_dictionary.md)
- [Debug with Rerun](how-to/debug_with_rerun.md)
- [Concurrent detection](how-to/concurrent_detection.md)
- [Install with distortion support](how-to/install-with-distortion.md)

### Explanation
Architecture, algorithms, and conventions — the *why* behind the
code.

- [System architecture](explanation/architecture.md)
- [Detection pipeline](explanation/pipeline.md)
- [Memory model (SoA / arena / FFI)](explanation/memory_model.md)
- [Algorithms](explanation/algorithms.md)
- [Coordinate conventions](explanation/coordinates.md)
- [Performance](explanation/performance.md) — headline recall / RMSE numbers and detector comparison.

### Reference
Generated API documentation.

- [Python API](reference/api.md)

### Migration
- [Config refactor (0.5.0+)](migration/config-refactor.md)
