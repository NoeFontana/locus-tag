"""Paired scatter: Locus vs best-competitor per instance, colored by stratum.

Every point is one GT instance: x = best competitor's error, y = Locus's error.
Points above the y=x line are where Locus is worse (the levers, made visual).
Consumes a polars ``compare_section`` frame. matplotlib → SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from tools.bench.compare.analysis import METRIC_LABEL


def plot(section_df: pl.DataFrame, out_path: Path | str, *, metric: str = "repro") -> Path:
    """Render the Locus-vs-best-competitor paired scatter for ``metric``."""
    pts = section_df.filter(
        pl.col("locus_value").is_not_null() & pl.col("best_competitor_value").is_not_null()
    )
    if pts.is_empty():
        raise ValueError("paired_scatter: no instances where both Locus and a competitor detected")
    strata = sorted(pts["stratum_id"].unique().to_list())
    cmap = plt.get_cmap("tab10")
    color = {s: cmap(i % 10) for i, s in enumerate(strata)}

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for s in strata:
        block = pts.filter(pl.col("stratum_id") == s)
        ax.scatter(
            block["best_competitor_value"].to_numpy(),
            block["locus_value"].to_numpy(),
            s=18,
            alpha=0.7,
            color=color[s],
            label=s,
            edgecolors="none",
        )
    x = pts["best_competitor_value"].to_numpy().astype(float)
    y = pts["locus_value"].to_numpy().astype(float)
    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    hi = float(np.max(finite)) if finite.size else 1.0
    lo = float(np.min(finite[finite > 0])) if np.any(finite > 0) else 1e-4
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x (parity)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"best competitor — {METRIC_LABEL[metric]}")
    ax.set_ylabel(f"Locus — {METRIC_LABEL[metric]}")
    ax.set_title(f"Locus vs best competitor per instance ({metric})\nabove line = Locus worse")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
