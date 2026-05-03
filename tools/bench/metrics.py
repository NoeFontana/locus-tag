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
