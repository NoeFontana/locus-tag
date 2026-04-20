"""Baseline JSON schema v2.

Pydantic models for the stratified regression-baseline format consumed by the
A1+ benchmark reporter. This module defines the *contract only* — no I/O, no
runtime wiring into ``tools/cli.py``. A ``__main__`` entry exports the JSON
Schema to ``docs/engineering/benchmarking/baseline_schema_v2.json``.

Schema version: **2.0**. Breaking change from the v1 nested-dict baseline in
``docs/engineering/benchmarking/baseline.json``.

The ``stratum_id`` string format is specified in
``docs/engineering/stratification.md``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# stratum_id grammar (see docs/engineering/stratification.md §4)
# ---------------------------------------------------------------------------

_STRATUM_KEYS: tuple[str, ...] = ("res", "ppm", "aoi", "dist", "mot")
_SLUG_RE = re.compile(r"^[a-z0-9]+$")


def _validate_stratum_id(value: str) -> str:
    parts = value.split("|")
    if len(parts) != len(_STRATUM_KEYS):
        raise ValueError(
            f"stratum_id must have {len(_STRATUM_KEYS)} axes separated by '|', "
            f"got {len(parts)}: {value!r}"
        )
    for expected_key, pair in zip(_STRATUM_KEYS, parts, strict=True):
        if "=" not in pair:
            raise ValueError(f"stratum_id axis {pair!r} missing '=' separator")
        key, bucket = pair.split("=", 1)
        if key != expected_key:
            raise ValueError(
                f"stratum_id axis order wrong: expected key {expected_key!r}, "
                f"got {key!r} in {value!r}"
            )
        if not _SLUG_RE.match(bucket):
            raise ValueError(
                f"stratum_id bucket {bucket!r} is not a valid slug "
                f"(lowercase alphanumeric); in {value!r}"
            )
    return value


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Build-and-host metadata captured alongside the baseline."""

    model_config = ConfigDict(extra="forbid")

    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    cpu_model: str = Field(min_length=1)
    cpu_cores_physical: int = Field(ge=1)
    cpu_cores_logical: int = Field(ge=1)
    locus_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    rayon_threads: int | None = Field(default=None, ge=1)
    build_profile: Literal["release", "debug"]
    timestamp_utc: datetime


class Tolerances(BaseModel):
    """Per-entry regression thresholds. Absent key = no gate on that metric."""

    model_config = ConfigDict(extra="forbid")

    recall_abs: float | None = Field(default=None, ge=0.0)
    precision_abs: float | None = Field(default=None, ge=0.0)
    repro_rmse_rel: float | None = Field(default=None, ge=0.0)
    translation_p95_rel: float | None = Field(default=None, ge=0.0)
    rotation_p95_rel: float | None = Field(default=None, ge=0.0)
    latency_p95_rel: float | None = Field(default=None, ge=0.0)


class MetricBlock(BaseModel):
    """The metrics reported per (profile, dataset, stratum_id) entry."""

    model_config = ConfigDict(extra="forbid")

    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    repro_rmse: float = Field(ge=0.0)
    translation_p50: float = Field(ge=0.0)
    translation_p95: float = Field(ge=0.0)
    rotation_p50: float = Field(ge=0.0)
    rotation_p95: float = Field(ge=0.0)
    latency_p50: float = Field(ge=0.0)
    latency_p95: float = Field(ge=0.0)


class BaselineEntry(BaseModel):
    """One row of the baseline, keyed by (profile, dataset, stratum_id)."""

    model_config = ConfigDict(extra="forbid")

    profile: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    stratum_id: str
    metrics: MetricBlock
    tolerances: Tolerances = Field(default_factory=Tolerances)
    n_images: int = Field(ge=1)
    n_tags: int = Field(ge=0)

    @field_validator("stratum_id")
    @classmethod
    def _check_stratum_id(cls, value: str) -> str:
        return _validate_stratum_id(value)


class BaselineV2(BaseModel):
    """Top-level v2 baseline document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"]
    provenance: Provenance
    entries: list[BaselineEntry] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def _check_entry_keys_unique(cls, entries: list[BaselineEntry]) -> list[BaselineEntry]:
        seen: set[tuple[str, str, str]] = set()
        for e in entries:
            key = (e.profile, e.dataset, e.stratum_id)
            if key in seen:
                raise ValueError(
                    f"duplicate baseline entry key: profile={e.profile!r}, "
                    f"dataset={e.dataset!r}, stratum_id={e.stratum_id!r}"
                )
            seen.add(key)
        return entries


# ---------------------------------------------------------------------------
# JSON Schema export
# ---------------------------------------------------------------------------


def _export_json_schema() -> None:
    """Write ``baseline_schema_v2.json`` next to this module's docs target."""
    import json
    import pathlib

    here = pathlib.Path(__file__).resolve()
    repo_root = here.parents[2]
    out = repo_root / "docs" / "engineering" / "benchmarking" / "baseline_schema_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = BaselineV2.model_json_schema()
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    _export_json_schema()


__all__ = [
    "BaselineEntry",
    "BaselineV2",
    "MetricBlock",
    "Provenance",
    "Tolerances",
]
