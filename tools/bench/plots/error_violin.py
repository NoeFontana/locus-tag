"""Per-instance error violin — distribution shape per series (pooled over strata).

Complements the ECDF: shows where each series' error mass sits. Log-scaled y so
sub-pixel and coarse errors are both legible. Consumes the polars ``long`` frame.
matplotlib ``violinplot`` (no seaborn). → SVG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from tools.bench.compare.analysis import METRIC_COLUMN, METRIC_LABEL
from tools.bench.plots._compare_style import series_color_map


def plot(long_df: pl.DataFrame, out_path: Path | str, *, metric: str = "repro") -> Path:
    """Render one violin per series of the matched-instance ``metric`` errors."""
    col = METRIC_COLUMN[metric]
    matched = long_df.filter(pl.col("matched")).filter(pl.col(col).is_not_null())
    if matched.is_empty():
        raise ValueError(f"error_violin: no matched rows with {col!r}")
    series = sorted(matched["series"].unique().to_list())
    colors = series_color_map(series)

    datasets: list[np.ndarray] = []
    kept: list[str] = []
    for s in series:
        vals = matched.filter(pl.col("series") == s)[col].to_numpy().astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]  # log scale needs > 0
        if vals.size:
            datasets.append(vals)
            kept.append(s)

    fig, ax = plt.subplots(figsize=(1.6 * len(kept) + 2.0, 5.0))
    parts = ax.violinplot(datasets, showmedians=True, showextrema=False)
    bodies = cast("list[Any]", parts["bodies"])
    for body, s in zip(bodies, kept, strict=True):
        body.set_facecolor(colors[s])
        body.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(kept) + 1))
    ax.set_xticklabels(kept, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_title(f"Per-instance {METRIC_LABEL[metric]} distribution")
    ax.grid(True, axis="y", alpha=0.3)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
