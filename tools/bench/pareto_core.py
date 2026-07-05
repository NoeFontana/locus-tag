"""Pareto-frontier primitive shared by the plot layer and the tuner.

One ``pareto_mask`` covers both call sites:

- ``plots/pareto.py`` wants "strictly better on *both* axes" domination (its
  operating-frontier semantics, which keep more points) plus a precision
  feasibility filter.
- ``tune/pareto_select.py`` wants standard Pareto domination (≥ on all
  objectives, > on at least one) over ``(recall, -trans_p99)``.

Both are the same routine with a ``strict`` flag and an optional ``feasible``
mask. NumPy-only; no bench-specific imports.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def pareto_mask(
    objectives: np.ndarray | Sequence[Sequence[float]],
    senses: Iterable[int],
    *,
    strict: bool = False,
    feasible: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows in ``objectives``.

    Args:
        objectives: ``(n, k)`` array — one row per candidate, one column per
            objective.
        senses: length-``k`` iterable; ``+1`` maximizes a column, ``-1``
            minimizes it. Applied so every column becomes "higher is better".
        strict: when ``False`` (default), row ``i`` is dominated if some feasible
            ``j`` is ``>=`` on every objective and ``>`` on at least one
            (standard Pareto). When ``True``, ``j`` must be *strictly* greater on
            *every* objective to dominate — a weaker relation that keeps ties and
            axis-equal points on the frontier.
        feasible: optional length-``n`` bool mask; infeasible rows are neither on
            the frontier nor allowed to dominate others.

    Returns:
        Length-``n`` bool array; ``True`` marks a frontier (non-dominated,
        feasible) row.
    """
    obj = np.asarray(objectives, dtype=np.float64)
    if obj.ndim != 2:
        raise ValueError(f"objectives must be 2-D (n, k); got shape {obj.shape}")
    n = obj.shape[0]
    sense_arr = np.asarray(list(senses), dtype=np.float64)
    if sense_arr.shape[0] != obj.shape[1]:
        raise ValueError(f"senses length {sense_arr.shape[0]} != objective count {obj.shape[1]}")
    # Orient every column so larger is better; NaNs stay NaN and never satisfy
    # the strict/non-strict comparisons below (so NaN rows never dominate and
    # are never dominated — matching the original plot behaviour).
    better = obj * sense_arr

    feas = np.ones(n, dtype=bool) if feasible is None else np.asarray(feasible, dtype=bool)

    optimal = feas.copy()
    for i in range(n):
        if not feas[i]:
            continue
        for j in range(n):
            if i == j or not feas[j]:
                continue
            if strict:
                dominated = bool(np.all(better[j] > better[i]))
            else:
                dominated = bool(np.all(better[j] >= better[i]) and np.any(better[j] > better[i]))
            if dominated:
                optimal[i] = False
                break
    return optimal
