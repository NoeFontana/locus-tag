"""Histogram of the per-instance Locus − best-competitor delta for a metric.

Negative mass = instances where Locus wins; the positive tail is the lever set.
Consumes a polars ``compare_section`` frame. matplotlib → SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from tools.bench.compare.analysis import METRIC_LABEL


def plot(section_df: pl.DataFrame, out_path: Path | str, *, metric: str = "repro") -> Path:
    """Render the delta histogram (Locus − best competitor) for ``metric``."""
    deltas = section_df["delta"].to_numpy().astype(float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        raise ValueError("delta_hist: no finite deltas (need instances both sides detected)")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(deltas, bins=40, color="#4c72b0", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0)
    frac_worse = float(np.mean(deltas > 0))
    ax.set_xlabel(f"Locus − best competitor  ({METRIC_LABEL[metric]})")
    ax.set_ylabel("instances")
    ax.set_title(
        f"Per-instance delta ({metric}) — {frac_worse:.0%} of instances Locus is worse\n"
        "(left of 0 = Locus wins)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
