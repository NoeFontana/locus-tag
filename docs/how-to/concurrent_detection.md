# How to Run Concurrent Detection

`Detector` batches frames through `detect_concurrent`; `max_concurrent_frames` sizes the
reusable-context pool it draws from, and `threads` bounds the workers it runs on.

!!! note "API Constraint"
    Currently, `max_concurrent_frames` is only exposed through the `DetectorBuilder` API. Using the standard `locus.Detector()` constructor will default to a single-frame pool.

| `max_concurrent_frames` | Behaviour |
| :--- | :--- |
| `1` (default) | One pooled `FrameContext`. `detect_concurrent` still fans frames out via Rayon, but every frame beyond the first in flight allocates a temporary overflow context (~200 KB). |
| `> 1` | N pooled contexts, so up to N frames run in parallel without allocating. |

## How `threads` and `max_concurrent_frames` relate

They are **not** two independent parallelism dials:

* `max_concurrent_frames` sizes the pool of reusable `FrameContext`s. It
  allocates no threads at all — it decides how many frames can be in flight
  before Locus falls back to temporary overflow contexts.
* `threads` sizes the **worker pool**, and it is the only knob that creates
  threads. With `threads = n > 0` the detector builds one scoped Rayon pool of
  `n` workers at construction and runs `detect` **and the whole
  `detect_concurrent` fan-out** inside it (`LocusEngine::run_scoped`). The
  frame-level and intra-frame parallelism therefore *share* those `n` workers
  instead of multiplying: the detector never uses more than `n` threads,
  whatever `max_concurrent_frames` is set to.
* With `threads = 0` (the default) both levels run on Rayon's global pool,
  sized by `RAYON_NUM_THREADS` or the core count.

So size `threads` to the CPU budget you want the detector to occupy, and
`max_concurrent_frames` to your batch size (to avoid overflow allocations).
See [Thread control](../reference/api.md#thread-control) for the full contract.

## Building a concurrent detector

```python
import locus

# DetectorBuilder is required to set max_concurrent_frames
detector = (
    locus.DetectorBuilder()
    .with_family(locus.TagFamily.AprilTag36h11)
    .with_threads(4)  # 4 Rayon workers for this detector, total
    .with_max_concurrent_frames(8)  # 8 pooled frame contexts (no extra threads)
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
frames: list[np.ndarray] = [...]  # list of (H, W) uint8 arrays

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

detector = locus.DetectorBuilder().with_max_concurrent_frames(os.cpu_count()).build()
```

If more frames arrive simultaneously than the pool size, Locus allocates temporary overflow contexts (~200 KB each) rather than blocking. This is acceptable for burst traffic but suboptimal under sustained over-subscription.
