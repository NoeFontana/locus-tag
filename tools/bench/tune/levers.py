"""Comparative "lever" analysis over the tidy sweep table.

Two analyses, pandas-only (no scipy):

- :func:`lever_sensitivity` — for each (library, stratum, metric, knob), the
  *main effect* of the knob = variance of the metric's per-knob-value group means
  (a one-way ANOVA main effect read straight off the swept grid). Ranks which
  lever moves which metric. ``effect_norm`` scales effects to ``[0, 1]`` within a
  (library, stratum, metric) so they are comparable across metrics in a heatmap.
- :func:`comparative_deltas` — takes each library's *chosen* frontier config and,
  per stratum, reports where tuned Locus beats / trails the best tuned competitor.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tools.bench.tune.results import OVERALL_STRATUM

# Metrics where a larger value is better (everything else: smaller is better).
_HIGHER_BETTER = {"recall", "precision", "n_matched"}


def expand_configs(configs_df: pd.DataFrame) -> pd.DataFrame:
    """Explode the JSON ``param_values`` sidecar into one column per knob."""
    rows: list[dict[str, object]] = []
    for _, r in configs_df.iterrows():
        values = json.loads(r["param_values"])
        rows.append({"param_hash": r["param_hash"], "library": r["library"], **values})
    return pd.DataFrame(rows)


def _wide_metric_table(results_df: pd.DataFrame, stratum_id: str) -> pd.DataFrame:
    """``param_hash × metric`` table (values averaged across datasets)."""
    scope = "overall" if stratum_id == OVERALL_STRATUM else "stratum"
    df = results_df[(results_df["scope"] == scope) & (results_df["stratum_id"] == stratum_id)]
    piv = df.pivot_table(index="param_hash", columns="metric", values="value", aggfunc="mean")
    return piv.reset_index()


def lever_sensitivity(
    results_df: pd.DataFrame,
    configs_df: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    per_stratum: bool = False,
    min_configs: int = 3,
) -> pd.DataFrame:
    """Rank knobs by their main effect on each metric.

    Returns columns ``library, stratum_id, metric, param_name, effect,
    effect_norm, best_value, n_configs``. ``best_value`` is the knob value that
    optimises the metric (max for recall/precision, min for errors).
    """
    cfg = expand_configs(configs_df)
    param_cols = [c for c in cfg.columns if c not in ("param_hash", "library")]

    if per_stratum:
        strata = sorted(results_df.loc[results_df["scope"] == "stratum", "stratum_id"].unique())
    else:
        strata = [OVERALL_STRATUM]

    out: list[dict[str, object]] = []
    for stratum_id in strata:
        wide = _wide_metric_table(results_df, stratum_id)
        if wide.empty:
            continue
        merged = wide.merge(cfg, on="param_hash", how="left")
        metric_cols = metrics or [c for c in wide.columns if c != "param_hash"]
        for library, g in merged.groupby("library"):
            for metric in metric_cols:
                if metric not in g.columns:
                    continue
                for pcol in param_cols:
                    sub = g[[pcol, metric]].dropna()
                    if sub[pcol].nunique() < 2 or len(sub) < min_configs:
                        continue
                    group_means = sub.groupby(pcol)[metric].mean()
                    effect = float(np.var(group_means.to_numpy()))
                    best_value = (
                        group_means.idxmax() if metric in _HIGHER_BETTER else group_means.idxmin()
                    )
                    out.append(
                        {
                            "library": library,
                            "stratum_id": stratum_id,
                            "metric": metric,
                            "param_name": pcol,
                            "effect": effect,
                            # str so the mixed-kind column (float/str/bool) stays
                            # a single Arrow type when written to parquet.
                            "best_value": str(best_value),
                            "n_configs": int(len(sub)),
                        }
                    )
    df = pd.DataFrame(out)
    if df.empty:
        return df
    # Normalise effects within (library, stratum, metric) for heatmap comparability.
    df["effect_norm"] = df.groupby(["library", "stratum_id", "metric"])["effect"].transform(
        lambda s: s / s.max() if s.max() > 0 else s * 0.0
    )
    return df


def chosen_per_library(pareto_docs: list[dict]) -> dict[str, str]:
    """Pick each library's deployment config: max recall, tie-broken by min tail."""
    chosen: dict[str, str] = {}
    for doc in pareto_docs:
        frontier = doc.get("frontier", [])
        if not frontier:
            continue
        best = max(frontier, key=lambda e: (e["recall"], -e["trans_p99_m"]))
        chosen[doc["library"]] = best["param_hash"]
    return chosen


def comparative_deltas(results_df: pd.DataFrame, chosen: dict[str, str]) -> pd.DataFrame:
    """Per-stratum tuned-Locus − best-tuned-competitor deltas for each metric.

    Returns ``stratum_id, metric, locus_value, best_competitor_value, competitor,
    delta`` sorted so the strata where Locus most underperforms surface first.
    """
    hashes = set(chosen.values())
    df = results_df[(results_df["scope"] == "stratum") & (results_df["param_hash"].isin(hashes))]
    grouped = df.groupby(["library", "stratum_id", "metric"])["value"].mean().reset_index()

    rows: list[dict[str, object]] = []
    for (stratum_id, metric), g in grouped.groupby(["stratum_id", "metric"]):
        by_lib = dict(zip(g["library"], g["value"], strict=True))
        if "locus" not in by_lib:
            continue
        competitors = {k: v for k, v in by_lib.items() if k != "locus" and v == v}
        if not competitors:
            continue
        higher_better = metric in _HIGHER_BETTER
        comp = (max if higher_better else min)(competitors, key=lambda k: competitors[k])
        locus_value = by_lib["locus"]
        # "delta" is oriented so negative = Locus worse (for both metric senses).
        raw = locus_value - competitors[comp]
        delta = raw if higher_better else -raw
        rows.append(
            {
                "stratum_id": stratum_id,
                "metric": metric,
                "locus_value": locus_value,
                "best_competitor_value": competitors[comp],
                "competitor": comp,
                "delta": delta,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("delta").reset_index(drop=True) if not out.empty else out
