"""Rejection composition: stacked bars per stratum, segments are RejectionReason.

Locus-only — other libraries don't expose intermediate rejections. Answers
"where do my rejections concentrate, and why?". Two cuts available via the
``group_by`` parameter:

- ``"stratum_id"``: full 5-axis stratum on the x-axis (one bar per populated
  stratum), reasons stacked vertically. Useful when you suspect a specific
  regime drives the rejections.
- ``"resolution_h"``: collapse to resolution. Useful as a top-level scan
  before drilling into strata.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tools.bench.plots._types import GroupBy

# Stable colour mapping so the same reason gets the same shade across plots.
# ``Unknown`` is defensive padding: ``collect.py:_FUNNEL_STATUS_NAMES`` maps
# ``code=0`` (NoneReason) to ``"Unknown"`` for an outcome that should never
# appear among rejected quads. Kept here so a contract drift surfaces as a
# grey segment instead of a missing colour.
_REASON_COLORS = {
    "RejectedDecode": "#fb9a29",  # decoder/Hamming failure
    "RejectedContrast": "#cb181d",  # geometry-only
    "RejectedSampling": "#6a51a3",  # reserved variant — unreachable today
    "Unknown": "#969696",
}


def plot(
    df: pd.DataFrame,
    out_path: Path | str,
    group_by: GroupBy = "stratum_id",
    binary: str | None = None,
) -> Path:
    """Render the rejection-composition plot to ``out_path``.

    If ``binary`` is given, restrict to that wrapper (e.g. ``"Locus"``).
    Otherwise pool all binaries that produced rejected_quad records.
    """
    sub = df[df["record_kind"] == "rejected_quad"].copy()
    if binary is not None:
        sub = sub[sub["binary"] == binary]
    if sub.empty:
        raise ValueError(f"rejection.plot: no rejected_quad rows in DataFrame (binary={binary!r})")

    counts = sub.groupby([group_by, "rejection_reason"]).size().unstack(fill_value=0).sort_index()

    # Drop rows that are entirely zero (defensive) and keep reasons in the
    # order defined by _REASON_COLORS so legends match across plots.
    reasons = [r for r in _REASON_COLORS if r in counts.columns]
    extras = [r for r in counts.columns if r not in _REASON_COLORS]
    reasons = reasons + extras
    counts = counts[reasons]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(counts)), 6))
    counts.plot.bar(
        stacked=True,
        ax=ax,
        color=[_REASON_COLORS.get(r, "#bdbdbd") for r in reasons],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel(group_by)
    ax.set_ylabel("Rejected quads (count)")
    title_suffix = f" — {binary}" if binary else ""
    ax.set_title(f"Rejection composition by {group_by}{title_suffix}")
    ax.legend(title="Reason", loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
