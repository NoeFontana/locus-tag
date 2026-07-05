"""Per-instance error ECDF, one line per series, small-multiples by stratum.

ECDF (empirical CDF) reads the *tail* directly — the QA question "how often is
each library within X px/m/deg" — better than a box/violin. Consumes the polars
GT-instance ``long`` frame (matched rows). matplotlib → SVG.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from tools.bench.compare.analysis import METRIC_COLUMN, METRIC_LABEL
from tools.bench.plots._compare_style import series_color_map


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def plot(long_df: pl.DataFrame, out_path: Path | str, *, metric: str = "repro") -> Path:
    """Render per-stratum ECDFs of ``metric`` (one line per series)."""
    col = METRIC_COLUMN[metric]
    matched = long_df.filter(pl.col("matched")).filter(pl.col(col).is_not_null())
    if matched.is_empty():
        raise ValueError(f"error_ecdf: no matched rows with {col!r}")
    strata = sorted(matched["stratum_id"].unique().to_list())
    series = sorted(matched["series"].unique().to_list())
    colors = series_color_map(series)

    ncols = min(3, len(strata))
    nrows = math.ceil(len(strata) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for idx, stratum in enumerate(strata):
        ax = axes[idx // ncols][idx % ncols]
        block = matched.filter(pl.col("stratum_id") == stratum)
        for s in series:
            vals = block.filter(pl.col("series") == s)[col].to_numpy().astype(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                x, y = _ecdf(vals)
                ax.step(x, y, where="post", label=s, color=colors[s], linewidth=1.4)
        ax.set_title(stratum, fontsize=7)
        ax.set_xlabel(METRIC_LABEL[metric], fontsize=8)
        ax.set_ylabel("cumulative fraction", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
    # Blank any unused facets.
    for idx in range(len(strata), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), fontsize=8)
    fig.suptitle(f"Per-instance {METRIC_LABEL[metric]} ECDF by stratum", fontsize=11)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
