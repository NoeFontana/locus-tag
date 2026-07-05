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
        return "<p class='empty'><em>(no rows)</em></p>"
    df = df.head(max_rows)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>" for row in df.iter_rows()
    )
    return (
        "<div class='table-wrap'><table>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _figure(caption: str, fn: str, *, wide: bool = False) -> str:
    """A figure with its caption in a block element (never overlapping the SVG)."""
    cls = " class='wide'" if wide else ""
    return (
        f"<figure{cls}><img src='{fn}' loading='lazy' alt='{caption}'>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


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

    # Large win-rate heatmaps get their own full-width block; the smaller ECDF /
    # violin plots flow into a responsive grid capped so nothing blows out.
    heatmaps = [(cap, fn) for cap, fn in shared if fn.startswith("winrate")]
    smalls = [(cap, fn) for cap, fn in shared if not fn.startswith("winrate")]
    shared_blocks = "".join(_figure(cap, fn, wide=True) for cap, fn in heatmaps)
    if smalls:
        shared_blocks += (
            "<div class='fig-grid'>" + "".join(_figure(cap, fn) for cap, fn in smalls) + "</div>"
        )

    section_blocks = ""
    for s in sections:
        levers = "".join(
            f"<details{' open' if m == primary_metric else ''}>"
            f"<summary>Improvement levers by <b>{m}</b> — where Locus underperforms</summary>"
            f"{_pl_to_html(s.worst_by_metric[m])}</details>"
            for m in s.worst_by_metric
        )
        section_blocks += (
            f"<section class='card' id='section-{s.sid}'>"
            f"<h2>Section {s.sid.upper()} — {s.locus_series} "
            f"<span class='vs'>vs</span> {', '.join(s.competitors)}</h2>"
            "<h3>Improvement levers</h3>"
            f"{levers}"
            f"<h3>{primary_metric}: Locus vs best competitor</h3>"
            "<div class='fig-grid'>"
            f"{_figure(f'{primary_metric}: Locus vs best competitor (paired)', s.scatter)}"
            f"{_figure(f'{primary_metric} delta histogram', s.dhist)}"
            "</div></section>"
        )

    toc_items: list[tuple[str, str]] = [
        ("series", "Series"),
        ("accuracy", "Accuracy by resolution"),
        ("distributions", "Distributions"),
    ]
    toc_items += [(f"section-{s.sid}", f"Section {s.sid.upper()}") for s in sections]
    toc_items.append(("provenance", "Provenance"))
    toc = "".join(f"<li><a href='#{anchor}'>{label}</a></li>" for anchor, label in toc_items)

    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{title}</title>
<style>
 :root {{
   --fg: #1c1e21; --muted: #6b7280; --bg: #ffffff; --card-bg: #fafbfc;
   --border: #e3e6ea; --rule: #cdd2d8; --head-bg: #2f3640; --head-fg: #ffffff;
   --zebra: #f3f5f8; --accent: #2d6cdf;
 }}
 * {{ box-sizing: border-box; }}
 body {{
   font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: var(--fg);
   background: var(--bg); margin: 0 auto; padding: 2rem 2.5rem 4rem;
   max-width: 1200px; line-height: 1.5;
 }}
 h1 {{ font-size: 1.85rem; margin: 0 0 0.25rem; }}
 h2 {{
   font-size: 1.3rem; margin: 0 0 1rem; padding-bottom: 0.35rem;
   border-bottom: 2px solid var(--rule); scroll-margin-top: 1rem;
 }}
 h3 {{ font-size: 1.05rem; margin: 1.6rem 0 0.6rem; color: #33373d; }}
 h2 .vs {{ color: var(--muted); font-weight: 400; font-size: 0.9em; }}
 p, li {{ max-width: 74ch; }}
 a {{ color: var(--accent); }}
 nav.toc {{
   background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
   padding: 0.75rem 1.25rem; margin: 1.25rem 0 2.5rem;
 }}
 nav.toc .lbl {{
   display: block; margin-bottom: 0.4rem; font-size: 0.78rem; color: var(--muted);
   text-transform: uppercase; letter-spacing: 0.06em;
 }}
 nav.toc ul {{
   list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap;
   gap: 0.35rem 1.4rem;
 }}
 nav.toc a {{ text-decoration: none; }}
 nav.toc a:hover {{ text-decoration: underline; }}
 section.card {{
   background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
   padding: 1.4rem 1.75rem; margin: 0 0 2.5rem;
 }}
 .table-wrap {{
   overflow-x: auto; margin: 0.5rem 0 1rem; border: 1px solid var(--border);
   border-radius: 6px;
 }}
 table {{
   border-collapse: collapse; font-size: 0.82rem; width: 100%; background: var(--bg);
 }}
 thead th {{
   position: sticky; top: 0; background: var(--head-bg); color: var(--head-fg);
   text-align: right; padding: 6px 10px; white-space: nowrap; font-weight: 600;
 }}
 thead th:first-child {{ text-align: left; }}
 tbody td {{
   padding: 4px 10px; border-bottom: 1px solid var(--border); text-align: right;
   white-space: nowrap;
 }}
 tbody td:first-child {{ text-align: left; }}
 tbody tr:nth-child(even) {{ background: var(--zebra); }}
 p.empty {{ color: var(--muted); }}
 figure {{
   margin: 0; background: var(--bg); border: 1px solid var(--border);
   border-radius: 8px; padding: 0.75rem;
 }}
 figure img {{ display: block; width: 100%; height: auto; }}
 figure figcaption {{
   margin-top: 0.6rem; font-size: 0.85rem; color: var(--muted); text-align: center;
 }}
 figure.wide {{ margin: 1rem 0; }}
 .fig-grid {{
   display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
   gap: 1.25rem; margin: 1rem 0;
 }}
 .fig-grid figure img {{ max-width: 760px; margin: 0 auto; }}
 details {{
   border: 1px solid var(--border); border-radius: 6px; margin: 0.6rem 0;
   background: var(--bg);
 }}
 summary {{
   cursor: pointer; padding: 0.6rem 0.9rem; font-weight: 500; font-size: 0.95rem;
 }}
 summary:hover {{ background: var(--zebra); }}
 details > .table-wrap {{ margin: 0.25rem 0.75rem 0.75rem; }}
 pre {{
   background: #f6f7f9; border: 1px solid var(--border); border-radius: 6px;
   padding: 0.75rem 1rem; overflow-x: auto; font-size: 0.8rem; line-height: 1.4;
 }}
</style></head><body>
<h1>{title}</h1>
<nav class='toc'>
 <span class='lbl'>On this page</span>
 <ul>{toc}</ul>
</nav>
<section class='card' id='series'>
 <h2>Series (config per library)</h2>
 <ul>{cfg_rows or "<li><em>none</em></li>"}</ul>
</section>
<section class='card' id='accuracy'>
 <h2>Headline: accuracy by resolution</h2>
 {accuracy_html}
</section>
<section class='card' id='distributions'>
 <h2>Distributions (all series)</h2>
 {shared_blocks}
</section>
{section_blocks}
<section class='card' id='provenance'>
 <h2>Provenance</h2>
 <pre>{provenance}</pre>
</section>
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
