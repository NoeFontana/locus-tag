# How to Run Concurrent Detection

`Detector` supports two modes controlled by `max_concurrent_frames` at construction:

| `max_concurrent_frames` | Behaviour |
| :--- | :--- |
| `1` (default) | Sequential. `detect_concurrent` processes frames one at a time. |
| `> 1` | Parallel. `detect_concurrent` processes up to N frames simultaneously via Rayon. |

`threads` (intra-frame parallelism) and `max_concurrent_frames` (inter-frame parallelism) are independent — set both to match your workload.

## Building a concurrent detector

```python
import locus

detector = (
    locus.DetectorBuilder()
    .with_family(locus.TagFamily.AprilTag36h11)
    .with_threads(4)               # Rayon threads per frame
    .with_max_concurrent_frames(8) # up to 8 frames in parallel
    .build()
)
```

## Single-frame detection (default)

```python
import numpy as np

frame = np.zeros((480, 640), dtype=np.uint8)
result = detector.detect(frame)
print(f"{len(result.ids)} tags")
```

Single-frame `detect` supports debug telemetry; `detect_concurrent` does not.

## Batch detection

```python
frames: list[np.ndarray] = [...]   # list of (H, W) uint8 arrays

results = detector.detect_concurrent(
    frames,
    intrinsics=locus.CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0),
    tag_size=0.166,
)

for i, r in enumerate(results):
    print(f"frame {i}: {len(r.ids)} tags")
```

The GIL is released for the entire Rayon section. Results are returned in the same order as `frames`.

**Limitations of `detect_concurrent`:**
- `rejected_corners` and `rejected_error_rates` are always empty.
- Debug telemetry is not available. Use `detect(debug_telemetry=True)` for diagnostic work.

## Thread-pool pattern (one detector per thread)

For workloads where each thread processes a continuous stream (e.g. one camera per thread), create one `Detector` per thread and call `detect` on each frame:

```python
import threading
import locus

def make_detector() -> locus.Detector:
    return locus.DetectorBuilder().with_family(locus.TagFamily.AprilTag36h11).build()

_local = threading.local()

def detect_one(frame):
    if not hasattr(_local, "detector"):
        _local.detector = make_detector()
    return _local.detector.detect(frame)
```

This avoids the overhead of `detect_concurrent`'s pool management for per-frame streaming workloads.

## Choosing `max_concurrent_frames`

A good starting point is the number of CPU cores, or the expected batch size if it is smaller:

```python
import os
detector = (
    locus.DetectorBuilder()
    .with_max_concurrent_frames(os.cpu_count())
    .build()
)
```

If more frames arrive simultaneously than the pool size, Locus allocates temporary overflow contexts (~200 KB each) rather than blocking. This is acceptable for burst traffic but suboptimal under sustained over-subscription.
