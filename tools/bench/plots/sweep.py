"""Sweep plot: a metric vs a continuous physical axis, line per binary.

Catches "binary X loses recall above PPM=15 at 720p but holds at 1080p" or
"binary Y's pose error blows up at grazing angles" — invisible in a single
number, glaringly obvious as a curve.

The continuous axis is binned into ~10 quantile buckets so each bucket has
roughly equal sample count. Per (binary, bin): the metric is computed over
records in that bin (recall = proportion; *_p50 = median).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.bench.metrics import compute_precision, compute_recall
from tools.bench.plots._types import ContinuousAxis, Metric

_AXIS_LABELS = {
    "distance_m": "Distance (m)",
    "aoi_deg": "Angle of incidence (°)",
    "ppm": "Pixels per metre of tag",
}

_METRIC_LABELS = {
    "recall": "Recall (%)",
    "precision": "Precision (%)",
    "trans_err_p50_m": "Translation error p50 (m)",
    "rot_err_p50_deg": "Rotation error p50 (°)",
    "repro_err_p50_px": "Reprojection error p50 (px)",
    "latency_p50_ms": "Frame latency p50 (ms)",
}

# Proportion-style metrics rendered on a 0–100% Y axis.
_PERCENT_METRICS: frozenset[Metric] = frozenset({"recall", "precision"})

# Metrics that aggregate over matched records (need pose/error fields).
_MATCHED_METRICS: dict[Metric, str] = {
    "trans_err_p50_m": "trans_err_m",
    "rot_err_p50_deg": "rot_err_deg",
    "repro_err_p50_px": "repro_err_px",
    "latency_p50_ms": "frame_latency_ms",
}


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges, with degenerate-bin collapse."""
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    return np.unique(edges)


def _binned_metric(
    df: pd.DataFrame, axis: ContinuousAxis, metric: Metric, n_bins: int
) -> pd.DataFrame:
    if metric == "recall":
        sub = df[df["record_kind"].isin(["matched", "missed_gt"])].copy()
    elif metric == "precision":
        # Precision needs the FP partner of TP. false_positive rows without
        # a nearest-GT attribution carry NaN axes and are excluded by the
        # dropna below — those FPs only show up in the global Pareto, not
        # in per-bin curves.
        sub = df[df["record_kind"].isin(["matched", "false_positive"])].copy()
    else:
        sub = df[df["record_kind"] == "matched"].copy()
    sub = sub.dropna(subset=[axis])
    if sub.empty:
        return pd.DataFrame()

    edges = _quantile_bins(sub[axis].to_numpy(), n_bins)
    if len(edges) < 3:
        return pd.DataFrame()
    sub["bin_idx"] = pd.cut(  # pyright: ignore[reportCallIssue]
        sub[axis],
        bins=edges,  # pyright: ignore[reportArgumentType]
        include_lowest=True,
        labels=False,
        duplicates="drop",
    )

    rows = []
    for key, grp in sub.groupby(["binary", "resolution_h", "bin_idx"]):
        binary, res, bin_idx = key
        if pd.isna(bin_idx):
            continue
        bin_idx = int(bin_idx)
        bin_mid = float((edges[bin_idx] + edges[bin_idx + 1]) / 2.0)
        if metric == "recall":
            n = len(grp)
            n_match = int((grp["record_kind"] == "matched").sum())
            value = compute_recall(n_match, n - n_match) * 100.0
        elif metric == "precision":
            n_match = int((grp["record_kind"] == "matched").sum())
            n_fp = int((grp["record_kind"] == "false_positive").sum())
            n = n_match + n_fp
            value = compute_precision(n_match, n_fp) * 100.0
        else:
            field = _MATCHED_METRICS[metric]
            vals = grp[field].dropna()
            if vals.empty:
                continue
            value = float(np.median(vals))
            n = len(vals)
        rows.append(
            {
                "binary": binary,
                "resolution_h": int(res),
                "bin_idx": bin_idx,
                "bin_mid": bin_mid,
                "value": value,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def plot(
    df: pd.DataFrame,
    out_path: Path | str,
    axis: ContinuousAxis = "distance_m",
    metric: Metric = "recall",
    n_bins: int = 10,
) -> Path:
    """Render the sweep plot for ``metric`` vs ``axis``."""
    binned = _binned_metric(df, axis, metric, n_bins=n_bins)
    if binned.empty:
        raise ValueError(f"sweep.plot: no rows after binning {metric} on {axis}")

    resolutions = sorted(binned["resolution_h"].unique())
    binaries = sorted(binned["binary"].unique())
    cmap = plt.get_cmap("tab10")
    color_for = {b: cmap(i % 10) for i, b in enumerate(binaries)}

    n_facets = len(resolutions)
    fig, axes = plt.subplots(1, n_facets, figsize=(5 * n_facets, 5), sharey=True, squeeze=False)

    for ax_idx, res in enumerate(resolutions):
        ax = axes[0][ax_idx]
        for binary in binaries:
            sub = binned[(binned["resolution_h"] == res) & (binned["binary"] == binary)]
            sub = sub.sort_values("bin_mid")  # pyright: ignore[reportCallIssue]
            if sub.empty:
                continue
            ax.plot(
                sub["bin_mid"],
                sub["value"],
                marker="o",
                color=color_for[binary],
                label=binary,
                linewidth=2,
                alpha=0.85,
            )
        ax.set_xlabel(_AXIS_LABELS[axis])
        ax.set_title(f"resolution_h = {res}")
        ax.grid(True, alpha=0.3)
        if metric in _PERCENT_METRICS:
            ax.set_ylim(-5, 105)
        if ax_idx == 0:
            ax.set_ylabel(_METRIC_LABELS[metric])
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"{_METRIC_LABELS[metric]} vs {_AXIS_LABELS[axis]}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
