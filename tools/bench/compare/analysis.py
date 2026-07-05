"""Per-instance comparative analysis — polars-native.

Turns the combined Tier-1 parquet (one row per detection/GT, keyed by series =
``binary:profile``) into per-*instance* comparisons: for every ground-truth tag
instance ``(dataset, image_id, tag_id)`` it lines up each series' corner/pose
error and detection, computes per-metric winners and per-stratum win-rates, and
ranks the instances where Locus most underperforms the best competitor (the
improvement *levers*).

Two Locus series can coexist (``locus:tuned`` and ``locus:shipped``); a *section*
picks one as the Locus comparator and the tuned competitors as rivals, so the
same wide table drives both report sections.

polars is confined to this module and the plot-data hand-off; matplotlib consumes
numpy extracted from these frames.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tools.bench.strata import stratum_id_series

# GT-instance key and the axis columns carried through for stratification/plots.
KEY = ["dataset", "image_id", "tag_id"]
AXES = ["stratum_id", "resolution_h", "distance_m", "aoi_deg", "ppm"]

# Error metrics compared per instance (all lower-is-better). The wide table uses
# these short names as column suffixes (``{series}__repro`` etc.).
ERROR_METRICS = ("repro", "trans", "rot")
# Short metric name → source column in the Tier-1 records / axis label (for plots).
METRIC_COLUMN = {"repro": "repro_err_px", "trans": "trans_err_m", "rot": "rot_err_deg"}
METRIC_LABEL = {
    "repro": "corner RMSE (px)",
    "trans": "translation error (m)",
    "rot": "rotation error (deg)",
}
_FAILURE_KIND = {"repro": "worse_corners", "trans": "worse_pose", "rot": "worse_pose"}


def load_instances(parquet_path: Path | str) -> pl.DataFrame:
    """Load the combined parquet, add ``series`` + ``stratum_id``, keep GT rows.

    Keeps only ``matched`` / ``missed_gt`` records — the per-series view of each
    ground-truth tag instance (Collector emits exactly one such row per GT tag).
    """
    df = pl.read_parquet(parquet_path)
    # Tier-1 records store math.nan (not null) for a missed/pose-less error. Convert
    # to null so polars aggregations skip them: NaN would otherwise poison medians/
    # quantiles (accuracy table) and be picked as a spurious "best" by min_horizontal
    # (NaN==NaN is True in polars). null is the correct "absent value" representation.
    df = df.with_columns(
        pl.col(c).fill_nan(None) for c in ("repro_err_px", "trans_err_m", "rot_err_deg")
    )
    df = df.with_columns((pl.col("binary") + pl.lit(":") + pl.col("profile")).alias("series"))
    stratum = stratum_id_series(
        df["resolution_h"].to_list(),
        df["distance_m"].cast(pl.Float64).to_list(),
        df["aoi_deg"].cast(pl.Float64).to_list(),
        df["ppm"].cast(pl.Float64).to_list(),
        velocity=None,
    )
    df = df.with_columns(pl.Series("stratum_id", stratum))
    return df.filter(pl.col("record_kind").is_in(["matched", "missed_gt"]))


def build_wide(long: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Pivot GT-instance rows to one row per instance with per-series columns.

    Columns per series ``s``: ``{s}__detected``, ``{s}__repro``, ``{s}__trans``,
    ``{s}__rot``, ``{s}__hamming``. Returns ``(wide, series_list)``.
    """
    series_list = sorted(long["series"].unique().to_list())
    base = long.group_by(KEY).agg([pl.first(a).alias(a) for a in AXES])
    for s in series_list:
        sub = long.filter(pl.col("series") == s).select(
            [
                *KEY,
                pl.col("matched").alias(f"{s}__detected"),
                pl.col("repro_err_px").alias(f"{s}__repro"),
                pl.col("trans_err_m").alias(f"{s}__trans"),
                pl.col("rot_err_deg").alias(f"{s}__rot"),
                pl.col("hamming_bits").alias(f"{s}__hamming"),
            ]
        )
        base = base.join(sub, on=KEY, how="left")
    return base, series_list


def _best_of(value_cols: list[str], names: list[str]) -> tuple[pl.Expr, pl.Expr]:
    """(best value, series-name-of-best) across ``value_cols`` (min, null-skipping).

    Ties go to the earliest series in ``names``. All-null ⇒ null value and name.
    Empty ``value_cols`` (no competitors) ⇒ null value and name — ``min_horizontal``
    of an empty list is an error, so guard it here for the single-library case.
    """
    if not value_cols:
        return pl.lit(None, dtype=pl.Float64), pl.lit(None, dtype=pl.Utf8)
    best_val = pl.min_horizontal([pl.col(c) for c in value_cols])
    name_expr: pl.Expr = pl.lit(None, dtype=pl.Utf8)
    # Build nested when/otherwise so the FIRST-listed series wins ties.
    for col, name in zip(reversed(value_cols), reversed(names), strict=True):
        name_expr = (
            pl.when(pl.col(col).is_not_null() & (pl.col(col) == best_val))
            .then(pl.lit(name))
            .otherwise(name_expr)
        )
    return best_val, name_expr


def compare_section(
    wide: pl.DataFrame, *, locus_series: str, competitors: list[str], metric: str
) -> pl.DataFrame:
    """Annotate ``wide`` with Locus-vs-best-competitor columns for one metric.

    Adds ``locus_detected, locus_value, best_competitor, best_competitor_value,
    delta`` (delta = locus − best competitor; positive = Locus worse). Used per
    report section (tuned-Locus or shipped-Locus).
    """
    best_val, best_name = _best_of([f"{c}__{metric}" for c in competitors], competitors)
    return wide.with_columns(
        pl.col(f"{locus_series}__detected").fill_null(False).alias("locus_detected"),
        pl.col(f"{locus_series}__{metric}").alias("locus_value"),
        best_val.alias("best_competitor_value"),
        best_name.alias("best_competitor"),
    ).with_columns((pl.col("locus_value") - pl.col("best_competitor_value")).alias("delta"))


def winrate_by_stratum(wide: pl.DataFrame, series_list: list[str]) -> pl.DataFrame:
    """Per (stratum, metric, series): win count, n, and win-rate.

    For error metrics a "win" is the strict-min error among *detecting* series;
    for ``detection`` the win-rate is the fraction of GT instances the series
    detected (not a head-to-head win, but the natural detection score).
    """
    frames: list[pl.DataFrame] = []
    for metric in ERROR_METRICS:
        best_name = _best_of([f"{s}__{metric}" for s in series_list], series_list)[1]
        judged = wide.with_columns(best_name.alias("series")).filter(pl.col("series").is_not_null())
        wins = judged.group_by(["stratum_id", "series"]).len().rename({"len": "wins"})
        totals = judged.group_by("stratum_id").len().rename({"len": "n"})
        frames.append(
            wins.join(totals, on="stratum_id").with_columns(
                (pl.col("wins") / pl.col("n")).alias("win_rate"),
                pl.lit(metric).alias("metric"),
            )
        )
    for s in series_list:
        det = wide.group_by("stratum_id").agg(
            pl.col(f"{s}__detected").fill_null(False).mean().alias("win_rate"),
            pl.col(f"{s}__detected").fill_null(False).sum().alias("wins"),
            pl.len().alias("n"),
        )
        frames.append(
            det.with_columns(pl.lit(s).alias("series"), pl.lit("detection").alias("metric"))
        )
    cols = ["stratum_id", "metric", "series", "wins", "n", "win_rate"]
    return pl.concat([f.select(cols) for f in frames], how="vertical")


def worst_locus(section: pl.DataFrame, *, metric: str, top_n: int) -> pl.DataFrame:
    """Rank the instances where Locus most underperforms — the improvement levers.

    Keeps rows where Locus is worse than the best competitor (``delta > 0``) or
    missed a tag a competitor found. Ranks by severity **within each stratum** and
    keeps the top ``top_n`` per stratum. ``section`` is a :func:`compare_section` frame.
    """
    tagged = section.with_columns(
        pl.when(~pl.col("locus_detected") & pl.col("best_competitor_value").is_not_null())
        .then(pl.lit("missed"))
        .when(pl.col("delta") > 0)
        .then(pl.lit(_FAILURE_KIND[metric]))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("failure_kind")
    ).filter(pl.col("failure_kind").is_not_null())
    tagged = tagged.with_columns(
        # missed cases are scored by how good the competitor was (they're the most
        # actionable gaps); worse-but-detected cases by the delta magnitude.
        pl.when(pl.col("failure_kind") == pl.lit("missed"))
        .then(pl.col("best_competitor_value"))
        .otherwise(pl.col("delta"))
        .alias("severity"),
        pl.lit(metric).alias("metric"),
    )
    ranked = tagged.with_columns(
        pl.col("severity").rank(method="ordinal", descending=True).over("stratum_id").alias("_rk")
    ).filter(pl.col("_rk") <= top_n)
    return ranked.drop("_rk").sort("severity", descending=True)


def best_locus(section: pl.DataFrame, *, metric: str, top_n: int) -> pl.DataFrame:
    """Instances where Locus most *outperforms* the best competitor (contrast set)."""
    return (
        section.filter(pl.col("locus_detected") & (pl.col("delta") < 0))
        .with_columns(pl.lit(metric).alias("metric"))
        .sort("delta")
        .head(top_n)
    )


def accuracy_by_resolution(long: pl.DataFrame, series_list: list[str]) -> pl.DataFrame:
    """Headline per (series, resolution): recall + p50 errors + trans p99."""
    return (
        long.filter(pl.col("series").is_in(series_list))
        .group_by(["series", "resolution_h"])
        .agg(
            (pl.col("matched").sum() / pl.len()).alias("recall"),
            pl.col("repro_err_px").filter(pl.col("matched")).median().alias("repro_p50"),
            pl.col("trans_err_m").filter(pl.col("matched")).median().alias("trans_p50"),
            pl.col("rot_err_deg").filter(pl.col("matched")).median().alias("rot_p50"),
            pl.col("trans_err_m").filter(pl.col("matched")).quantile(0.99).alias("trans_p99"),
        )
        .sort(["resolution_h", "series"])
    )


def detection_agreement(wide: pl.DataFrame, series_list: list[str]) -> pl.DataFrame:
    """Per-stratum detection-agreement counts across series."""
    det = [pl.col(f"{s}__detected").fill_null(False) for s in series_list]
    n_detected = pl.sum_horizontal(det).alias("n_libs_detected")
    return (
        wide.with_columns(n_detected)
        .group_by(["stratum_id", "n_libs_detected"])
        .len()
        .rename({"len": "n_instances"})
        .sort(["stratum_id", "n_libs_detected"])
    )
