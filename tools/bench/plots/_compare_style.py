"""Shared styling for the per-instance comparison plots.

Stable per-series colors (Locus variants in the blue/cyan family, competitors
distinct) so a series reads the same across every figure and the report.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

_SERIES_COLORS = {
    "locus:tuned": "#1f77b4",  # blue
    "locus:shipped": "#17becf",  # cyan — Locus family
    "opencv_aruco:tuned": "#ff7f0e",  # orange
    "apriltag:tuned": "#d62728",  # red
}


def series_color_map(series: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Return a stable color per series (known series fixed; others from tab10)."""
    cmap = plt.get_cmap("tab10")
    out: dict[str, tuple[float, float, float, float]] = {}
    extra = 0
    for s in series:
        if s in _SERIES_COLORS:
            out[s] = mcolors.to_rgba(_SERIES_COLORS[s])
        else:
            out[s] = cmap(extra % 10)
            extra += 1
    return out
