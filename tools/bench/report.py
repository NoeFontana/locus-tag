"""Bundle the Tier-1 plots for one or more bench runs.

Reads N parquet files, joins them into a tidy DataFrame, and renders the
plot suite as PNGs into ``out_dir``. An ``index.html`` is emitted alongside
so a reviewer can scroll through the bundle in a browser.

Usage:
    python -m tools.bench.report \\
        runs/v1.5-pre.parquet \\
        --out reports/v1.5-pre/
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import typer

from tools.bench.plots import (
    ContinuousAxis,
    Metric,
    load_records_df,
    pareto,
    rejection,
    sweep,
)


class SweepSpec(NamedTuple):
    metric: Metric
    axis: ContinuousAxis
    title: str


app = typer.Typer(help="Generate the Tier-1 plot bundle")


_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .meta {{ color: #555; margin-bottom: 2rem; }}
  section {{ margin: 2rem 0; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  code {{ background: #f3f3f3; padding: 0.1em 0.4em; border-radius: 3px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  Sources: {sources}<br>
  Total records: {n_records}
</div>
{sections}
</body>
</html>
"""

_SECTION_TEMPLATE = """<section>
  <h2>{title}</h2>
  <p>{caption}</p>
  <img src="{filename}" alt="{title}">
</section>
"""


def generate(
    parquet_paths: list[Path],
    out_dir: Path,
    title: str = "Tier-1 bench report",
) -> dict[str, Path]:
    """Render all plots for the given parquets and emit ``index.html``.

    Returns a dict mapping plot-name → output PNG path.
    """
    df = load_records_df(list(parquet_paths))  # invariant `list[Path]` → `list[Path | str]`
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    sections: list[str] = []

    # Two Pareto views: the unfiltered plot is informational (any low-precision
    # configuration shows up); the operating plot enforces a precision floor so
    # recall-only configurations (e.g., over-permissive Soft decoders running
    # at single-digit precision) cannot sit on the frontier on recall alone.
    written["pareto"] = pareto.plot(df, out_dir / "pareto.png")
    sections.append(
        _SECTION_TEMPLATE.format(
            title="Pareto: recall vs latency (informational)",
            caption=(
                "One point per (binary, resolution). Annotated with global "
                "precision (P=…). Filled markers are on the Pareto frontier — "
                "no other point combines lower latency with higher recall. "
                "<strong>Read with care:</strong> a point on this frontier with "
                "very low precision is not an operating point — see the "
                "<em>operating Pareto</em> below for the precision-gated view."
            ),
            filename="pareto.png",
        )
    )

    written["pareto_operating"] = pareto.plot(
        df, out_dir / "pareto_operating.png", min_precision=0.5
    )
    sections.append(
        _SECTION_TEMPLATE.format(
            title="Pareto: recall vs latency (operating, P ≥ 50%)",
            caption=(
                "Same axes, but configurations with global precision below "
                "50% are excluded from the frontier. This is the plot to use "
                "when picking a deployment configuration."
            ),
            filename="pareto_operating.png",
        )
    )

    # Sweep per (metric, axis). Recall first as the canonical headline; then
    # error and latency metrics that vary even when recall is saturated.
    # Note: ``repro_err_p50_px`` is intentionally omitted. The current
    # AprilTag wrapper at ``tools/bench/utils.py:869+`` returns corners in a
    # canonical orientation that doesn't match the GT corner indexing, so
    # per-corner RMSE picks up tag-diagonal-sized "errors" that are actually
    # just a cyclic shift. Re-enable once the wrapper is canonicalized.
    sweep_specs: list[SweepSpec] = [
        SweepSpec("recall", "distance_m", "Recall vs distance"),
        SweepSpec("recall", "ppm", "Recall vs PPM"),
        SweepSpec("recall", "aoi_deg", "Recall vs angle of incidence"),
        SweepSpec("precision", "distance_m", "Precision vs distance"),
        SweepSpec("precision", "ppm", "Precision vs PPM"),
        SweepSpec("precision", "aoi_deg", "Precision vs angle of incidence"),
        SweepSpec("trans_err_p50_m", "distance_m", "Translation error (p50) vs distance"),
        SweepSpec("trans_err_p50_m", "aoi_deg", "Translation error (p50) vs angle of incidence"),
        SweepSpec("rot_err_p50_deg", "distance_m", "Rotation error (p50) vs distance"),
        SweepSpec("rot_err_p50_deg", "aoi_deg", "Rotation error (p50) vs angle of incidence"),
        SweepSpec("latency_p50_ms", "ppm", "Latency (p50) vs PPM"),
        SweepSpec("latency_p50_ms", "distance_m", "Latency (p50) vs distance"),
    ]
    for spec in sweep_specs:
        slug = f"sweep_{spec.metric}_vs_{spec.axis}"
        try:
            written[slug] = sweep.plot(
                df, out_dir / f"{slug}.png", axis=spec.axis, metric=spec.metric
            )
        except ValueError as e:
            sections.append(f"<section><p><em>{slug} skipped: {e}</em></p></section>")
            continue
        sections.append(
            _SECTION_TEMPLATE.format(
                title=spec.title,
                caption=(
                    "One curve per binary, faceted by resolution. Each marker "
                    "is a quantile bin on the x-axis."
                ),
                filename=f"{slug}.png",
            )
        )

    # Rejection composition — Locus only. Pool across binaries that produced
    # rejected_quad records; collapse to resolution_h for a top-level scan.
    try:
        written["rejection_by_resolution"] = rejection.plot(
            df, out_dir / "rejection_by_resolution.png", group_by="resolution_h"
        )
        sections.append(
            _SECTION_TEMPLATE.format(
                title="Rejection composition by resolution (Locus only)",
                caption=(
                    "Stacked bar segments are the <code>RejectionReason</code> "
                    "labels (RejectedDecode = passed funnel, decoder/Hamming "
                    "rejected; RejectedContrast = geometry-only failure). "
                    "Other libraries don't expose intermediate rejections."
                ),
                filename="rejection_by_resolution.png",
            )
        )
    except ValueError as e:
        sections.append(f"<section><p><em>rejection plot skipped: {e}</em></p></section>")

    index = out_dir / "index.html"
    sources = ", ".join(f"<code>{p.name}</code>" for p in parquet_paths)
    index.write_text(
        _INDEX_HTML_TEMPLATE.format(
            title=title,
            sources=sources,
            n_records=f"{len(df):,}",
            sections="\n".join(sections),
        )
    )
    written["index"] = index
    return written


@app.command()
def cli_report(
    parquet: list[Path] = typer.Argument(..., help="One or more Tier-1 parquet files"),
    out: Path = typer.Option(..., help="Output directory for PNGs + index.html"),
    title: str = typer.Option("Tier-1 bench report", help="HTML page title"),
) -> None:
    """Render plots for the given parquets into ``out``."""
    written = generate(parquet, out, title=title)
    typer.echo(f"\nWrote {len(written)} artifacts to {out}/:")
    for name, path in written.items():
        typer.echo(f"  {name:<28} {path}")


if __name__ == "__main__":
    app()
