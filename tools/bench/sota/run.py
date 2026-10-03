"""Run Locus or AprilTag 3 over an image list and write the shared SOTA JSONL schema.

Usage::

    python -m tools.bench.sota.run locus <family> <profile> <overrides-json> <list> <out> [reps]
    python -m tools.bench.sota.run apriltag3 <family> <list> <out> <threads>

``overrides-json`` merges into the profile; its optional ``"detector"`` object holds per-call
``Detector`` options instead (e.g. ``{"detector": {"decimation": 2}}``).

Image decode is outside every timer; ``ms`` is the best of ``reps`` ``detect()``
calls after one untimed warm-up call. Threads are controlled by the caller
(``RAYON_NUM_THREADS`` for Locus, ``nthreads`` for AprilTag 3).
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, cast

import cv2
import numpy as np

from tools.bench.utils import _APRILTAG_CORNER_TO_GT as APRILTAG_TO_OPENCV_ORDER  # noqa: PLC2701


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into ``base`` (in place) and return ``base``."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(img)


def _paths(list_file: str) -> list[str]:
    with open(list_file) as f:
        return [line.strip() for line in f if line.strip()]


def run_locus(
    family: str, profile: str, overrides: str, list_file: str, out: str, reps: int = 2
) -> None:
    import locus  # noqa: PLC0415 - deferred so `apriltag3` runs without the wheel

    base = locus.DetectorConfig.from_profile(cast(locus.ProfileName, profile))
    over = json.loads(overrides)
    # `detector` holds per-call options (e.g. `decimation`), not profile keys.
    options = over.pop("detector", {})
    cfg = merge(base.model_dump(mode="json"), over)
    det = locus.Detector(
        config=locus.DetectorConfig.model_validate(cfg),
        families=[getattr(locus.TagFamily, family)],
        **options,
    )
    warmed = False
    with open(out, "w") as f:
        for p in _paths(list_file):
            img = _read(p)
            if not warmed:
                det.detect(img)
                warmed = True
            best, batch = float("inf"), None
            for _ in range(max(1, reps)):
                t0 = time.perf_counter()
                batch = det.detect(img)
                best = min(best, (time.perf_counter() - t0) * 1e3)
            assert batch is not None
            corners = np.asarray(batch.corners, dtype=np.float64).reshape(-1, 4, 2)
            rec = {
                "image": p,
                "ms": best,
                "ids": [int(i) for i in batch.ids],
                "corners": corners.round(4).tolist(),
                "convention": "locus",
            }
            f.write(json.dumps(rec) + "\n")


def run_apriltag3(family: str, list_file: str, out: str, threads: int) -> None:
    from pupil_apriltags import Detector  # noqa: PLC0415

    det = Detector(families=family, nthreads=threads, quad_decimate=1.0, refine_edges=True)
    warmed = False
    with open(out, "w") as f:
        for p in _paths(list_file):
            img = _read(p)
            if not warmed:
                det.detect(img)
                warmed = True
            t0 = time.perf_counter()
            # pupil_apriltags' stub types detect() as a single Detection; it returns a list.
            res = cast(list[Any], det.detect(img))
            ms = (time.perf_counter() - t0) * 1e3
            rec = {
                "image": p,
                "ms": ms,
                "ids": [int(r.tag_id) for r in res],
                "corners": [
                    np.asarray(r.corners)[APRILTAG_TO_OPENCV_ORDER].round(4).tolist() for r in res
                ],
                "convention": "opencv",
            }
            f.write(json.dumps(rec) + "\n")


def main(argv: list[str]) -> None:
    kind, *rest = argv
    if kind == "locus":
        family, profile, overrides, list_file, out, *reps = rest
        run_locus(family, profile, overrides, list_file, out, int(reps[0]) if reps else 2)
    elif kind == "apriltag3":
        family, list_file, out, threads = rest
        run_apriltag3(family, list_file, out, int(threads))
    else:
        raise SystemExit(f"unknown runner {kind!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
