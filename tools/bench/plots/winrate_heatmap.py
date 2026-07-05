"""Win-rate heatmap: series (rows) × stratum (cols), cell = win-rate for a metric.

Answers "which library wins where" at a glance. Consumes the polars
``winrate_by_stratum`` frame; matplotlib → SVG (or PNG by extension).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot(winrate_df: pl.DataFrame, out_path: Path | str, *, metric: str = "detection") -> Path:
    """Render the win-rate heatmap for one ``metric`` and save to ``out_path``."""
    sub = winrate_df.filter(pl.col("metric") == metric)
    if sub.is_empty():
        raise ValueError(f"winrate_heatmap: no rows for metric={metric!r}")
    grid = sub.pivot(values="win_rate", index="series", on="stratum_id", aggregate_function="mean")
    series = grid["series"].to_list()
    strata = [c for c in grid.columns if c != "series"]
    data = grid.select(strata).to_numpy().astype(float)

    fig, ax = plt.subplots(figsize=(1.1 * len(strata) + 4.0, 0.6 * len(series) + 2.0))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(strata, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(series, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if 0.25 < v < 0.75 else "white",
                )
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{metric} win-rate")
    ax.set_title(f"Win-rate by stratum — {metric}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
