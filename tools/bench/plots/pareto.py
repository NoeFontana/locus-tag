"""Pareto plot: recall vs latency_p50, one point per (binary, resolution).

Answers the "should we ship the new profile?" question directly. Points on
the Pareto frontier (no other point both higher-recall and lower-latency)
are highlighted; dominated points are dimmed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pareto_mask(latencies: np.ndarray, recalls: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal points.

    A point is on the frontier if no other point has *both* lower latency and
    higher recall. Ties (equal latency or equal recall) keep both points.
    """
    n = len(latencies)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if latencies[j] < latencies[i] and recalls[j] > recalls[i]:
                optimal[i] = False
                break
    return optimal


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (binary, resolution_h) with TP/FP/FN/recall/latency_p50."""
    g = df.groupby(["binary", "resolution_h"])
    rows: list[dict[str, float | int | str]] = []
    for key, grp in g:
        binary, res = key  # type: ignore[misc]
        tp = int((grp["record_kind"] == "matched").sum())
        fn = int((grp["record_kind"] == "missed_gt").sum())
        fp = int((grp["record_kind"] == "false_positive").sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Per-frame latency: take from any one row per (binary, image) to
        # avoid double-counting the replicated frame_latency_ms column.
        per_frame = grp.drop_duplicates(subset=["image_id"])["frame_latency_ms"]
        latency_p50 = float(np.median(per_frame)) if len(per_frame) else float("nan")
        rows.append(
            {
                "binary": str(binary),
                "resolution_h": int(res),  # type: ignore[call-overload]
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": recall,
                "precision": precision,
                "latency_p50_ms": latency_p50,
                "n_frames": int(len(per_frame)),
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Render the Pareto plot and save to ``out_path`` (PNG).

    Returns the resolved path written.
    """
    agg = _aggregate(df)
    if agg.empty:
        raise ValueError("pareto.plot: no aggregable rows in DataFrame")

    fig, ax = plt.subplots(figsize=(8, 6))
    binaries = sorted(agg["binary"].unique())
    cmap = plt.get_cmap("tab10")
    color_for = {b: cmap(i % 10) for i, b in enumerate(binaries)}

    pareto = _pareto_mask(agg["latency_p50_ms"].to_numpy(), agg["recall"].to_numpy())

    for i, row in agg.iterrows():
        is_pareto = bool(pareto[i])  # type: ignore[call-overload]
        ax.scatter(
            row["latency_p50_ms"],
            row["recall"] * 100.0,
            color=color_for[row["binary"]],
            s=120 if is_pareto else 60,
            alpha=1.0 if is_pareto else 0.35,
            edgecolors="black" if is_pareto else "none",
            linewidths=1.0 if is_pareto else 0,
            zorder=3 if is_pareto else 2,
        )
        ax.annotate(
            f"{row['resolution_h']}p",
            (row["latency_p50_ms"], row["recall"] * 100.0),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="black" if is_pareto else "gray",
        )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color_for[b], markersize=10, label=b
        )
        for b in binaries
    ]
    ax.legend(handles=handles, loc="lower right", title="Binary")
    ax.set_xlabel("Median per-frame latency (ms)")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Pareto: recall vs latency  (filled = Pareto-optimal)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
