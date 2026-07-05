"""Metric primitives shared between the bench loop and the plot modules.

Kept tiny and import-light so it can be pulled in from anywhere without
dragging in pandas/matplotlib. The scoring conventions here are the
single source of truth — change here and every consumer updates in lockstep.
"""

from __future__ import annotations


def compute_recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN). Returns 0.0 when no GT was present."""
    total = tp + fn
    return tp / total if total > 0 else 0.0


def compute_precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP). Returns 0.0 when no detections were emitted."""
    total = tp + fp
    return tp / total if total > 0 else 0.0


def percentiles(values: object, qs: list[float]) -> list[float]:
    """Percentiles of ``values`` at the ``qs`` positions (0–100 scale).

    Returns one ``float`` per entry in ``qs``; an empty input yields ``0.0`` for
    each. The single reduction used by both the headline pose aggregator
    (``aggregate_pose_stats``) and the tuner's per-cell summariser so tail
    numbers are computed identically everywhere. ``numpy`` is imported lazily to
    keep the recall/precision primitives above dependency-free.
    """
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [0.0] * len(qs)
    return [float(v) for v in np.percentile(arr, qs)]
