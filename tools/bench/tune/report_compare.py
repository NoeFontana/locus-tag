"""Comparative tuning report: lever heatmap + per-stratum win/loss + HTML bundle.

Consumes the tidy ``tune_results.parquet`` (+ its ``tune_configs.parquet``
sidecar) and the per-library ``pareto/<lib>.json`` frontiers, and emits:

- ``lever_sensitivity.parquet`` — which knob moves which metric (per library).
- ``comparative_deltas.parquet`` — where tuned Locus trails / leads the best
  tuned competitor, per stratum.
- ``levers_heatmap.png`` / ``comparative_deltas.png`` and an ``index.html`` bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.bench.plots import levers as levers_plot
from tools.bench.tune.levers import (
    chosen_per_library,
    comparative_deltas,
    lever_sensitivity,
)
from tools.bench.tune.results import read_results


def _load_pareto_docs(pareto_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(pareto_dir.glob("*.json")):
        docs.append(json.loads(path.read_text()))
    return docs


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "<p><em>(no rows)</em></p>"
    return df.head(max_rows).to_html(index=False, float_format=lambda v: f"{v:.5g}", border=0)


def generate(
    *,
    results_path: Path,
    pareto_dir: Path,
    out_dir: Path,
    title: str = "Comparative tuning report",
) -> Path:
    """Build the comparative report; returns the ``index.html`` path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df, provenance = read_results(results_path)
    configs_df = pd.read_parquet(results_path.parent / "tune_configs.parquet")
    pareto_docs = _load_pareto_docs(pareto_dir)

    # Analyses.
    sensitivity = lever_sensitivity(results_df, configs_df)
    chosen = chosen_per_library(pareto_docs)
    deltas = comparative_deltas(results_df, chosen)

    sensitivity.to_parquet(out_dir / "lever_sensitivity.parquet", index=False)
    deltas.to_parquet(out_dir / "comparative_deltas.parquet", index=False)

    # Plots (guarded — a single-library sweep has no competitor deltas).
    plots: list[tuple[str, str]] = []
    if not sensitivity.empty:
        levers_plot.plot(sensitivity, out_dir / "levers_heatmap.png")
        plots.append(("Knob sensitivity (which lever moves which metric)", "levers_heatmap.png"))
    if not deltas.empty:
        levers_plot.plot_deltas(deltas, out_dir / "comparative_deltas.png")
        plots.append(
            ("Tuned Locus vs best tuned competitor (per stratum)", "comparative_deltas.png")
        )

    # Top lever per (library, metric) — a compact "what to tune" table.
    top_levers = (
        sensitivity.sort_values("effect_norm", ascending=False)
        .groupby(["library", "metric"], as_index=False)
        .first()[["library", "metric", "param_name", "best_value", "effect", "n_configs"]]
        if not sensitivity.empty
        else pd.DataFrame()
    )

    index_path = out_dir / "index.html"
    index_path.write_text(
        _render_html(
            title=title,
            provenance=provenance.model_dump_json(indent=2),
            chosen=chosen,
            plots=plots,
            top_levers=top_levers,
            deltas=deltas,
        )
    )
    return index_path


def _render_html(
    *,
    title: str,
    provenance: str,
    chosen: dict[str, str],
    plots: list[tuple[str, str]],
    top_levers: pd.DataFrame,
    deltas: pd.DataFrame,
) -> str:
    imgs = "\n".join(
        f"<h2>{caption}</h2>\n<img src='{fname}' style='max-width:100%'>"
        for caption, fname in plots
    )
    chosen_rows = "".join(f"<li><b>{lib}</b>: {ph}</li>" for lib, ph in sorted(chosen.items()))
    worst = deltas[deltas["delta"] < 0] if not deltas.empty else deltas
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
 table {{ border-collapse: collapse; font-size: 0.85rem; }}
 th, td {{ padding: 4px 8px; border-bottom: 1px solid #ddd; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 code, pre {{ background: #f6f6f6; }}
</style></head><body>
<h1>{title}</h1>
<h2>Chosen (max-recall) frontier config per library</h2>
<ul>{chosen_rows or "<li><em>none</em></li>"}</ul>
{imgs}
<h2>Top lever per (library, metric)</h2>
{_df_to_html_table(top_levers)}
<h2>Strata where tuned Locus most underperforms (delta &lt; 0)</h2>
{_df_to_html_table(worst)}
<h2>Provenance</h2>
<pre>{provenance}</pre>
</body></html>
"""
