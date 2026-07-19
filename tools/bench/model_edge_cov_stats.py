"""Calibration stats for the model-edge pose covariance (Phase-A ship gate).

Reads the per-scene ``(δ, Σ, d²)`` dump written by the Rust audit test
``audit_model_edge_covariance`` and reports whether the covariance is honestly
calibrated: a well-specified Σ gives ``d² = δᵀΣ⁻¹δ ~ χ²(6)`` (mean 6). It breaks
the joint 6-DOF statistic into the **translation** (indices 0-2) and **rotation**
(3-5) marginal blocks so the ship decision can be per-block. Gate: **KL < 0.5**
and per-axis ratios ∈ [0.5, 2.0]. Baseline (4-corner weighted LM): mean d²=714.7,
KL=13.93.

Usage:
    uv run --group bench python tools/bench/model_edge_cov_stats.py \\
        diagnostics/model_edge_cov_audit/samples_locus_v1_tag36h11_1920x1080.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

KL_NUM_BINS = 60
KL_MAX_X = 60.0


def _gammainc_lower(s: float, x: float) -> float:
    if x < 0.0 or s <= 0.0:
        raise ValueError(f"_gammainc_lower out of range: s={s}, x={x}")
    if x == 0.0:
        return 0.0
    log_pref = -x + s * math.log(x) - math.lgamma(s)
    if x < s + 1.0:
        term = 1.0 / s
        total = term
        for n in range(1, 200):
            term *= x / (s + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return math.exp(log_pref) * total
    fpmin = 1e-300
    b = x + 1.0 - s
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(log_pref) * h


def _chi2_cdf(x: np.ndarray, k: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(arr)
    s = 0.5 * k
    for idx, v in np.ndenumerate(arr):
        out[idx] = 0.0 if v <= 0.0 else _gammainc_lower(s, 0.5 * float(v))
    return out


def _kl_to_chi2(d2: np.ndarray, k: int) -> float:
    """KL(empirical d² histogram ‖ χ²(k)) over [0, KL_MAX_X]."""
    edges = np.linspace(0.0, KL_MAX_X, KL_NUM_BINS + 1)
    counts, _ = np.histogram(np.clip(d2, 0.0, KL_MAX_X), bins=edges)
    n = max(int(counts.sum()), 1)
    p = counts.astype(np.float64) / float(n)
    q = np.diff(_chi2_cdf(edges, k))
    p_safe = np.clip(p, 1e-12, 1.0)
    q_safe = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _block_stats(delta: np.ndarray, cov: np.ndarray, idx: list[int], label: str) -> dict:
    """d² = δ_blkᵀ (Σ_blk)⁻¹ δ_blk over the marginal sub-block; χ²(len(idx))."""
    k = len(idx)
    sel = np.ix_(idx, idx)
    d2 = np.empty(len(delta))
    for i in range(len(delta)):
        sub = cov[i][sel]
        try:
            inv = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            d2[i] = np.nan
            continue
        dv = delta[i][idx]
        d2[i] = float(dv @ inv @ dv)
    d2 = d2[np.isfinite(d2)]
    ratios = [float(np.mean(delta[:, j] ** 2) / np.mean(cov[:, j, j])) for j in idx]
    return {
        "label": label,
        "dof": k,
        "n": int(d2.size),
        "mean_d2": float(np.mean(d2)),
        "median_d2": float(np.median(d2)),
        "ideal_mean_d2": float(k),
        "kl": _kl_to_chi2(d2, k),
        "per_axis_ratio": ratios,  # δ²/Σ per axis; ideal 1.0
        "p50": float(np.percentile(d2, 50)),
        "p90": float(np.percentile(d2, 90)),
        "p99": float(np.percentile(d2, 99)),
    }


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("diagnostics/model_edge_cov_audit/samples_locus_v1_tag36h11_1920x1080.json")
    )
    samples = json.loads(path.read_text())
    delta = np.array([s["delta"] for s in samples], dtype=np.float64)  # (N,6)
    cov = np.array([np.array(s["cov"]).reshape(6, 6) for s in samples])  # (N,6,6)

    joint = _block_stats(delta, cov, [0, 1, 2, 3, 4, 5], "joint (6-DOF)")
    trans = _block_stats(delta, cov, [0, 1, 2], "translation (3-DOF)")
    rot = _block_stats(delta, cov, [3, 4, 5], "rotation (3-DOF)")

    def gate(b: dict) -> str:
        ok = b["kl"] < 0.5 and all(0.5 <= r <= 2.0 for r in b["per_axis_ratio"])
        return "PASS" if ok else "fail"

    axis_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    print(f"model-edge covariance calibration — {path.name}  (n={len(samples)})")
    print(f"{'block':<20} {'mean d²':>8} {'ideal':>6} {'median':>7} {'KL':>7} {'gate':>5}")
    print("-" * 62)
    for b in (joint, trans, rot):
        print(
            f"{b['label']:<20} {b['mean_d2']:>8.2f} {b['ideal_mean_d2']:>6.1f} "
            f"{b['median_d2']:>7.2f} {b['kl']:>7.3f} {gate(b):>5}"
        )
    print("\nper-axis calibration ratio δ²/Σ (ideal 1.0; <1 = over-confident):")
    for name, r in zip(axis_names, joint["per_axis_ratio"]):
        print(f"  {name}: {r:.3f}")
    print("\nBaseline (4-corner weighted LM, 2026-05-03): mean d²=714.7, KL=13.93")

    report = {"joint": joint, "translation": trans, "rotation": rot, "source": path.name}
    out = path.parent / f"report_{path.stem}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
