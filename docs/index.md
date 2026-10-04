# Locus

A production-grade, memory-safe fiducial-marker detector for robotics,
autonomous vehicles, and perception engineering. Locus detects
**AprilTag**, **ArUco**, **AprilGrid**, and **ChArUco** markers and
boards — implemented in Rust with zero-copy Python bindings via
PyO3 (abi3).

!!! warning "Experimental status"
    API is subject to breaking changes until 1.0.0 ships. The main
    workstreams toward 1.0.0 are reducing the API surface and
    broadening validation on real-camera data (EuRoC and Liu4K are
    benchmarked today). Until then, this library
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
- **6-DOF pose recovery** — IPPE-Square seed refined by weighted
  Levenberg-Marquardt with per-corner uncertainty.
- **Validated on real and rendered data** — against pinned OpenCV
  4.10 and aruco_nano (`cargo xtask sota`, 2026-10-04): on the EuRoC
  `cam_april` real-camera AprilGrid sequence, `standard` decodes
  86.0 % of markers (best reference 57.1 %) at 99.996 % precision,
  with a leave-one-out corner error of 0.283 px (reference 0.516 px);
  on the render-tag suites its debiased corner error is ≈ 0.06 px
  (references ≈ 0.22 px). Liu4K recall (37 % vs aruco_nano's 66 %) is
  an open gap. See [Performance](explanation/performance.md).
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
print(batch.ids)  # (N,)
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
- [Tune detectors and compare against other libraries](how-to/tune_and_compare.md)
- [Compare detectors per image](how-to/per_instance_comparison.md)

### Explanation
Architecture, algorithms, and conventions — the *why* behind the
code.

- [System architecture](explanation/architecture.md)
- [Detection pipeline](explanation/pipeline.md)
- [Memory model (SoA / arena / FFI)](explanation/memory_model.md)
- [Algorithms](explanation/algorithms.md)
- [Coordinate conventions](explanation/coordinates.md)
- [Performance](explanation/performance.md) — current results against OpenCV and aruco_nano on real (EuRoC, Liu4K) and rendered data.

### Reference
Generated API documentation.

- [Python API](reference/api.md)

### Migration
- [Config refactor (0.5.0+)](migration/config-refactor.md)
