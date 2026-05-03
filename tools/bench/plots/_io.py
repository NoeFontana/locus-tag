"""Load Tier-1 parquet files into a single tidy DataFrame for plotting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tools.bench.strata import AxisValues, compute_stratum_id


def load_records_df(paths: list[Path | str]) -> pd.DataFrame:
    """Concatenate one or more Tier-1 parquet files into a tidy DataFrame.

    Adds derived columns the plot modules consume:

    - ``stratum_id``: ``compute_stratum_id(AxisValues(...))`` per row
    - ``source_path``: filename of the parquet the row came from (for traceability
      when overlaying multiple runs)
    """
    if not paths:
        raise ValueError("load_records_df: at least one parquet path required")

    frames: list[pd.DataFrame] = []
    for p in paths:
        path = Path(p)
        table = pq.read_table(path)
        frame = table.to_pandas()
        frame["source_path"] = path.name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    df["stratum_id"] = [
        compute_stratum_id(
            AxisValues(
                resolution_h=int(r) if pd.notna(r) else None,
                distance_m=float(d),
                aoi_deg=float(a),
                ppm=float(p),
                velocity=None,
            )
        )
        for r, d, a, p in zip(
            df["resolution_h"], df["distance_m"], df["aoi_deg"], df["ppm"], strict=True
        )
    ]
    return df
