"""Reduce Tier-1 ``ObservationRecord`` rows to metric summaries.

This is the tuner's counterpart to the plot layer's aggregation: the *same*
record substrate (matched / missed_gt / false_positive kinds), the *same*
recall/precision definitions (``metrics.compute_recall`` / ``compute_precision``)
and the *same* ``velocity=None`` stratum derivation as ``plots/_io.py``. Emitting
per-stratum blocks here means the sweep table, the Pareto selection and the
comparative report all read one consistent reduction.

Latency is **opt-in** (``include_latency``): the parallel search phase must not
surface latency percentiles (they are contention-poisoned), so it calls with the
default ``False``; only the serial verification phase passes ``True``.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from tools.bench.metrics import compute_precision, compute_recall, percentiles
from tools.bench.records import ObservationRecord
from tools.bench.strata import stratum_id_series

# The accuracy metric keys every summary carries (latency keys added on demand).
POSE_GROUPS = (
    ("trans_err", "m", "trans_err_m"),
    ("rot_err", "deg", "rot_err_deg"),
    ("repro_err", "px", "repro_err_px"),
)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[~np.isnan(values)]


def _latency_block(df: pd.DataFrame) -> dict[str, float]:
    """Per-frame latency percentiles (dedup by ``image_id`` to avoid replication)."""
    per_frame = _finite(df.drop_duplicates(subset=["image_id"])["frame_latency_ms"].to_numpy())
    if per_frame.size == 0:
        return {
            k: float("nan")
            for k in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms")
        }
    p50, p95, p99 = percentiles(per_frame, [50, 95, 99])
    return {
        "latency_mean_ms": float(np.mean(per_frame)),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
    }


def _metrics_for(df: pd.DataFrame, include_latency: bool) -> dict[str, float]:
    """Compute one metric block from a set of record rows."""
    kind = df["record_kind"]
    tp = int((kind == "matched").sum())
    fn = int((kind == "missed_gt").sum())
    fp = int((kind == "false_positive").sum())
    out: dict[str, float] = {
        "recall": compute_recall(tp, fn),
        "precision": compute_precision(tp, fp),
        "n_gt": float(tp + fn),
        "n_det": float(tp + fp),
        "n_matched": float(tp),
    }
    matched = df[kind == "matched"]
    for prefix, unit, col in POSE_GROUPS:
        arr = _finite(matched[col].to_numpy(dtype=np.float64))
        if arr.size:
            p50, p95, p99 = percentiles(arr, [50, 95, 99])
            out[f"{prefix}_mean_{unit}"] = float(np.mean(arr))
            out[f"{prefix}_p50_{unit}"] = p50
            out[f"{prefix}_p95_{unit}"] = p95
            out[f"{prefix}_p99_{unit}"] = p99
        else:
            for q in ("mean", "p50", "p95", "p99"):
                out[f"{prefix}_{q}_{unit}"] = float("nan")
    out["pose_samples"] = float(_finite(matched["trans_err_m"].to_numpy(dtype=np.float64)).size)
    if include_latency:
        out.update(_latency_block(df))
    return out


def summarize(
    records: list[ObservationRecord], *, include_latency: bool = False
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Reduce a record list to ``(overall, per_stratum)`` metric blocks.

    ``per_stratum`` is keyed by ``stratum_id``. Returns empty dicts for an empty
    record list.
    """
    if not records:
        return {}, {}
    df = pd.DataFrame([asdict(r) for r in records])
    df["stratum_id"] = stratum_id_series(
        df["resolution_h"], df["distance_m"], df["aoi_deg"], df["ppm"], velocity=None
    )
    overall = _metrics_for(df, include_latency)
    per_stratum = {
        str(sid): _metrics_for(group, include_latency) for sid, group in df.groupby("stratum_id")
    }
    return overall, per_stratum
