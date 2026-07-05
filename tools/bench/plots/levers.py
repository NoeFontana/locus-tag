"""Lever heatmap: normalised knob sensitivity per metric, one facet per library.

Rows are parameters, columns are metrics, each cell is the ``effect_norm`` from
``tune.levers.lever_sensitivity`` (0 = this knob barely moves the metric, 1 = the
most influential knob for that metric). Answers "which lever moves which metric"
at a glance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import AxesImage


def plot(
    sensitivity_df: pd.DataFrame, out_path: Path | str, *, stratum_id: str | None = None
) -> Path:
    """Render one lever heatmap per library and save to ``out_path`` (PNG)."""
    df = sensitivity_df
    if stratum_id is not None:
        df = df[df["stratum_id"] == stratum_id]
    if df.empty:
        raise ValueError("levers.plot: no sensitivity rows to render")

    libraries = sorted(df["library"].unique())
    fig, axes = plt.subplots(
        1, len(libraries), figsize=(1.6 * max(len(libraries), 1) + 5.0, 6), squeeze=False
    )

    im: AxesImage | None = None
    for ax, library in zip(axes[0], libraries, strict=True):
        sub = df[df["library"] == library]
        pivot = sub.pivot_table(
            index="param_name", columns="metric", values="effect_norm", aggfunc="mean"
        ).fillna(0.0)
        data = pivot.to_numpy()
        im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index), fontsize=7)
        ax.set_title(library, fontsize=10)
        # Annotate each cell with the normalised effect.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if data[i, j] < 0.6 else "black",
                    fontsize=6,
                )

    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.7, label="normalised sensitivity")
    title = "Knob sensitivity by metric"
    if stratum_id:
        title += f"  ({stratum_id})"
    fig.suptitle(title, fontsize=12)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_deltas(deltas_df: pd.DataFrame, out_path: Path | str, *, top_n: int = 20) -> Path:
    """Bar chart of the largest tuned-Locus − best-competitor deltas.

    Negative bars (Locus worse) surface first — the strata/metrics to fix.
    """
    if deltas_df.empty:
        raise ValueError("levers.plot_deltas: no deltas to render")
    df = deltas_df.copy()
    df["label"] = df["stratum_id"] + " · " + df["metric"] + " vs " + df["competitor"]
    df = df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(top_n)
    df = df.sort_values("delta")

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.35 * len(df) + 1.0)))
    colors = ["tab:red" if d < 0 else "tab:green" for d in df["delta"]]
    ax.barh(range(len(df)), df["delta"], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(list(df["label"]), fontsize=7)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Locus − best competitor  (negative = Locus worse)")
    ax.set_title("Where tuned Locus trails / leads the best tuned competitor")
    ax.grid(True, axis="x", alpha=0.3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
