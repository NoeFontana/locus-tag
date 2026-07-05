"""Structured per-instance comparison report: HTML bundle + docs markdown.

Two sections — **A** tuned-Locus-vs-tuned-rivals and **B** shipped-Locus-vs-tuned-
rivals — each with per-stratum win-rates, the worst-Locus lever table, a headline
accuracy-by-resolution table, and the statistical figures. polars-native table
rendering (no pandas); figures via the ``tools/bench/plots`` modules → SVG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tools.bench.compare import analysis as A
from tools.bench.plots import (
    delta_hist,
    error_ecdf,
    error_violin,
    paired_scatter,
    winrate_heatmap,
)
from tools.bench.records import read_records

_LOCUS_TUNED = "locus:tuned"
_LOCUS_SHIPPED = "locus:shipped"

_WORST_COLS = [
    "dataset",
    "image_id",
    "tag_id",
    "stratum_id",
    "metric",
    "locus_value",
    "best_competitor",
    "best_competitor_value",
    "delta",
    "failure_kind",
]
_ACC_COLS = ["series", "resolution_h", "recall", "repro_p50", "trans_p50", "rot_p50", "trans_p99"]


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.5g}"
    return "" if v is None else str(v)


def _pl_to_html(df: pl.DataFrame, max_rows: int = 40) -> str:
    if df.is_empty():
        return "<p><em>(no rows)</em></p>"
    df = df.head(max_rows)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>" for row in df.iter_rows()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _pl_to_md(df: pl.DataFrame, max_rows: int = 60) -> str:
    if df.is_empty():
        return "_(no rows)_\n"
    df = df.head(max_rows)
    cols = df.columns
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = "\n".join("| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.iter_rows())
    return f"{head}\n{sep}\n{rows}\n"


def _sections(series: list[str]) -> list[tuple[str, str, list[str]]]:
    """(section id, locus series, competitor series) for each Locus variant present."""
    competitors = sorted(s for s in series if not s.startswith("locus:"))
    out: list[tuple[str, str, list[str]]] = []
    if _LOCUS_TUNED in series and competitors:
        out.append(("tuned", _LOCUS_TUNED, competitors))
    if _LOCUS_SHIPPED in series and competitors:
        out.append(("shipped", _LOCUS_SHIPPED, competitors))
    return out


def generate(
    *,
    records_path: Path,
    out_dir: Path,
    metric: str = "repro",
    top_n: int = 25,
    fmt: str = "svg",
    markdown_out: Path | None = None,
    date: str = "0000-00-00",
    title: str = "Per-instance comparative benchmark",
) -> Path:
    """Build the report bundle; returns the ``index.html`` path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    long = A.load_instances(records_path)
    wide, series = A.build_wide(long)
    _, provenance = read_records(records_path)

    configs_sidecar = records_path.with_suffix(records_path.suffix + ".configs.json")
    configs = json.loads(configs_sidecar.read_text()) if configs_sidecar.exists() else []

    wide.write_parquet(out_dir / "instances_wide.parquet")
    winrate = A.winrate_by_stratum(wide, series)
    winrate.write_parquet(out_dir / "winrate_by_stratum.parquet")
    accuracy = A.accuracy_by_resolution(long, series)

    # Shared figures (all series): (caption, plot fn, data frame, filename stem, metric).
    fig_specs = [
        (
            "Detection win-rate by stratum",
            winrate_heatmap.plot,
            winrate,
            "winrate_detection",
            "detection",
        ),
        (
            f"{metric} win-rate by stratum",
            winrate_heatmap.plot,
            winrate,
            f"winrate_{metric}",
            metric,
        ),
        ("Corner-error ECDF by stratum", error_ecdf.plot, long, "ecdf_repro", "repro"),
        ("Translation-error ECDF by stratum", error_ecdf.plot, long, "ecdf_trans", "trans"),
        (f"{metric} distribution (violin)", error_violin.plot, long, f"violin_{metric}", metric),
    ]
    shared = [
        (caption, plot_fn(data, out_dir / f"{stem}.{fmt}", metric=m).name)
        for caption, plot_fn, data, stem, m in fig_specs
    ]

    # Per-section data, computed ONCE and consumed by both the HTML and markdown
    # renderers (no recompute, no risk of the two paths selecting different columns).
    sections_data: list[_SectionData] = []
    for sid, locus_series, competitors in _sections(series):
        primary = A.compare_section(
            wide, locus_series=locus_series, competitors=competitors, metric=metric
        )
        scatter = paired_scatter.plot(primary, out_dir / f"scatter_{sid}.{fmt}", metric=metric).name
        dhist = delta_hist.plot(primary, out_dir / f"delta_{sid}.{fmt}", metric=metric).name
        # Levers across ALL error metrics — corner AND pose (trans/rot) — so a
        # pose-tail regression can't hide behind a corner-only (repro) ranking.
        worst_by_metric: dict[str, pl.DataFrame] = {}
        for m in A.ERROR_METRICS:
            section_m = (
                primary
                if m == metric
                else A.compare_section(
                    wide, locus_series=locus_series, competitors=competitors, metric=m
                )
            )
            worst = A.worst_locus(section_m, metric=m, top_n=top_n)
            worst_by_metric[m] = worst.select([c for c in _WORST_COLS if c in worst.columns])
        worst_by_metric[metric].write_parquet(out_dir / f"worst_locus_{sid}.parquet")
        sections_data.append(
            _SectionData(sid, locus_series, competitors, scatter, dhist, worst_by_metric)
        )

    index = out_dir / "index.html"
    index.write_text(
        _render_html(
            title=title,
            provenance=provenance.model_dump_json(indent=2),
            configs=configs,
            shared=shared,
            accuracy_html=_pl_to_html(accuracy.select(_ACC_COLS)),
            sections=sections_data,
            primary_metric=metric,
        )
    )

    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(
            _render_markdown(
                title=title,
                date=date,
                configs=configs,
                accuracy=accuracy.select(_ACC_COLS),
                sections=sections_data,
            )
        )
    return index


@dataclass
class _SectionData:
    """One report section, computed once and rendered to both HTML and markdown."""

    sid: str
    locus_series: str
    competitors: list[str]
    scatter: str  # figure filename
    dhist: str  # figure filename
    worst_by_metric: dict[str, pl.DataFrame]  # metric -> worst-Locus table (col-selected)


def _render_html(
    *,
    title: str,
    provenance: str,
    configs: list[dict],
    shared: list[tuple[str, str]],
    accuracy_html: str,
    sections: list[_SectionData],
    primary_metric: str,
) -> str:
    cfg_rows = "".join(
        f"<li><b>{c['library']}:{c['profile']}</b> — {c['param_hash']} "
        f"(space {c['space_name']})</li>"
        for c in configs
    )
    shared_imgs = "\n".join(
        f"<h3>{cap}</h3><img src='{fn}' style='max-width:100%'>" for cap, fn in shared
    )
    section_html = "".join(
        f"<h2>Section {s.sid.upper()} — {s.locus_series} vs {', '.join(s.competitors)}</h2>"
        + "".join(
            f"<h4>Improvement levers by {m} — where Locus underperforms</h4>"
            f"{_pl_to_html(s.worst_by_metric[m])}"
            for m in s.worst_by_metric
        )
        + f"<h4>{primary_metric}: Locus vs best competitor</h4>"
        + f"<img src='{s.scatter}' style='max-width:100%'>"
        + f"<img src='{s.dhist}' style='max-width:100%'>"
        for s in sections
    )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1150px; }}
 table {{ border-collapse: collapse; font-size: 0.8rem; }}
 th, td {{ padding: 3px 7px; border-bottom: 1px solid #ddd; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 pre {{ background: #f6f6f6; padding: 0.5rem; }}
</style></head><body>
<h1>{title}</h1>
<h2>Series (config per library)</h2>
<ul>{cfg_rows or "<li><em>none</em></li>"}</ul>
<h2>Headline: accuracy by resolution</h2>
{accuracy_html}
<h2>Distributions (all series)</h2>
{shared_imgs}
{section_html}
<h2>Provenance</h2>
<pre>{provenance}</pre>
</body></html>
"""


def _render_markdown(
    *,
    title: str,
    date: str,
    configs: list[dict],
    accuracy: pl.DataFrame,
    sections: list[_SectionData],
) -> str:
    parts = [f"# {title}\n", f"_Generated {date}. Tables only; see the HTML bundle for figures._\n"]
    parts.append("## Series\n")
    parts.append(
        "\n".join(
            f"- **{c['library']}:{c['profile']}** — `{c['param_hash']}` (space `{c['space_name']}`)"
            for c in configs
        )
        + "\n"
    )
    parts.append("## Accuracy by resolution\n")
    parts.append(_pl_to_md(accuracy))
    for s in sections:
        parts.append(
            f"\n## Section {s.sid.upper()} — {s.locus_series} vs {', '.join(s.competitors)}\n"
        )
        for m, worst in s.worst_by_metric.items():
            parts.append(f"### Improvement levers by {m} — where Locus underperforms\n")
            parts.append(_pl_to_md(worst))
    return "\n".join(parts)
