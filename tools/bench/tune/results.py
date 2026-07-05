"""Tidy long-form persistence for sweep results.

``tune_results.parquet`` is one row per ``(library, param_hash, dataset,
stratum_id, metric)`` — the join-friendly shape the Pareto selection and the
comparative lever analysis consume. ``tune_configs.parquet`` is the sidecar
mapping ``param_hash → param_values`` (JSON) so a metric row can be traced back
to the exact configuration. Both carry the run :class:`Provenance` in the
parquet footer, mirroring ``records.write_records``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools.bench.records import _PROVENANCE_METADATA_KEY
from tools.bench.schema import Provenance
from tools.bench.tune.executor import CellResult

# Sentinel stratum for the dataset-wide (non-stratified) block. Real stratum ids
# always start with "res=", so this never collides.
OVERALL_STRATUM = "__overall__"

# Reuse the single provenance metadata key from records.py so a reader keyed off
# that constant finds provenance in both Tier-1 and tune tables. The schema key
# is tune-specific (a different table shape), so it stays local.
_SCHEMA_KEY = b"locus.tune.results_schema_version"
_SCHEMA_VERSION = b"1.0"


def results_to_frame(results: list[CellResult]) -> pd.DataFrame:
    """Flatten cell results to the long-form ``(…, metric) → value`` table."""
    rows: list[dict[str, object]] = []
    for r in results:
        base = {
            "library": r.library,
            "param_hash": r.param_hash,
            "dataset": r.dataset,
            "n_frames": r.n_frames,
            "latency_valid": r.latency_valid,
            "error": r.error or "",
        }
        if r.error and not r.overall:
            rows.append(
                {
                    **base,
                    "stratum_id": OVERALL_STRATUM,
                    "scope": "overall",
                    "metric": "__error__",
                    "value": float("nan"),
                }
            )
            continue
        for metric, value in r.overall.items():
            rows.append(
                {
                    **base,
                    "stratum_id": OVERALL_STRATUM,
                    "scope": "overall",
                    "metric": metric,
                    "value": float(value),
                }
            )
        for stratum_id, block in r.per_stratum.items():
            for metric, value in block.items():
                rows.append(
                    {
                        **base,
                        "stratum_id": stratum_id,
                        "scope": "stratum",
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def configs_to_frame(results: list[CellResult]) -> pd.DataFrame:
    """One row per unique ``(library, param_hash)`` with the JSON param values."""
    seen: dict[tuple[str, str], dict[str, object]] = {}
    for r in results:
        seen[(r.library, r.param_hash)] = r.param_values
    rows = [
        {
            "library": lib,
            "param_hash": ph,
            "param_values": json.dumps(values, sort_keys=True, default=str),
        }
        for (lib, ph), values in seen.items()
    ]
    return pd.DataFrame(rows)


def _write_with_provenance(df: pd.DataFrame, path: Path, provenance: Provenance) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(
        {
            _PROVENANCE_METADATA_KEY: provenance.model_dump_json().encode("utf-8"),
            _SCHEMA_KEY: _SCHEMA_VERSION,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def write_results(
    results: list[CellResult], out_dir: Path | str, provenance: Provenance
) -> tuple[Path, Path]:
    """Write ``tune_results.parquet`` + ``tune_configs.parquet`` under ``out_dir``.

    Returns the two written paths.
    """
    out = Path(out_dir)
    results_path = out / "tune_results.parquet"
    configs_path = out / "tune_configs.parquet"
    _write_with_provenance(results_to_frame(results), results_path, provenance)
    _write_with_provenance(configs_to_frame(results), configs_path, provenance)
    return results_path, configs_path


def read_results(path: Path | str) -> tuple[pd.DataFrame, Provenance]:
    """Read a ``tune_results.parquet`` back with its provenance."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    prov_bytes = meta.get(_PROVENANCE_METADATA_KEY)
    if prov_bytes is None:
        raise ValueError(f"{path}: missing provenance metadata")
    provenance = Provenance.model_validate(json.loads(prov_bytes.decode("utf-8")))
    return table.to_pandas(), provenance
