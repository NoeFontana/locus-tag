"""Tier-1 observation records — per-candidate parquet rows for bench runs.

One row per ``(binary, profile, dataset, image, candidate)``. ``record_kind``
distinguishes a matched detection (``matched``), an unmatched ground-truth tag
(``missed_gt``), and a Locus quad rejected by the funnel (``rejected_quad``).

Continuous physical axes (distance, AOI, PPM, blur, occlusion, iso) are stored
as raw floats — **never bucketed**. ``stratum_id`` is computed at aggregate
time by ``tools/bench/strata.py``, so re-bucketing is a five-line edit and
re-aggregation, not a re-bench.

Provenance is written once into the parquet ``FileMetadata`` block (key-value
metadata), not replicated per row, so a 200k-record file stays in single-digit
megabytes.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from tools.bench.schema import Provenance

RecordKind = Literal["matched", "missed_gt", "rejected_quad", "false_positive"]

_PROVENANCE_METADATA_KEY = b"locus.bench.provenance"
_SCHEMA_VERSION_KEY = b"locus.bench.records_schema_version"
_SCHEMA_VERSION = b"1.0"


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """One Tier-1 row.

    Float fields use ``math.nan`` (not ``None``) when N/A. Integer fields use
    ``-1`` only when domain-meaningful — ``hamming_bits`` is ``-1`` if the
    decoder never ran. ``tag_id`` is ``None`` for unattributed rejected quads
    and false positives.
    """

    # Identity
    run_id: str
    binary: str  # "locus" | "opencv_aruco" | "apriltag3"
    profile: str
    dataset: str
    image_id: str
    record_kind: RecordKind
    tag_id: int | None

    # Continuous axes — raw, never bucketed. NaN = unknown.
    distance_m: float
    aoi_deg: float
    ppm: float
    occlusion_ratio: float
    blur_px: float
    iso: float
    resolution_h: int

    # Outcome metrics. NaN where N/A.
    matched: bool
    trans_err_m: float
    rot_err_deg: float
    repro_err_px: float
    hamming_bits: int  # -1 if decoder did not run
    rejection_reason: str  # "" unless record_kind == "rejected_quad"

    # Per-frame, replicated for joinability
    frame_latency_ms: float
    n_gt_in_frame: int
    n_det_in_frame: int


# Arrow schema — explicit dtypes avoid pandas-coercion surprises on round-trip.
# Order matches `ObservationRecord` field order so column-major iteration is
# straightforward in `write_records`.
_ARROW_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("binary", pa.string(), nullable=False),
        pa.field("profile", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("record_kind", pa.string(), nullable=False),
        pa.field("tag_id", pa.int32(), nullable=True),
        pa.field("distance_m", pa.float32(), nullable=False),
        pa.field("aoi_deg", pa.float32(), nullable=False),
        pa.field("ppm", pa.float32(), nullable=False),
        pa.field("occlusion_ratio", pa.float32(), nullable=False),
        pa.field("blur_px", pa.float32(), nullable=False),
        pa.field("iso", pa.float32(), nullable=False),
        pa.field("resolution_h", pa.int32(), nullable=False),
        pa.field("matched", pa.bool_(), nullable=False),
        pa.field("trans_err_m", pa.float32(), nullable=False),
        pa.field("rot_err_deg", pa.float32(), nullable=False),
        pa.field("repro_err_px", pa.float32(), nullable=False),
        pa.field("hamming_bits", pa.int32(), nullable=False),
        pa.field("rejection_reason", pa.string(), nullable=False),
        pa.field("frame_latency_ms", pa.float32(), nullable=False),
        pa.field("n_gt_in_frame", pa.int32(), nullable=False),
        pa.field("n_det_in_frame", pa.int32(), nullable=False),
    ]
)


def write_records(
    records: list[ObservationRecord],
    provenance: Provenance,
    path: Path | str,
) -> None:
    """Write Tier-1 records to a parquet file with provenance in the file footer.

    The parent directory is created if missing.
    """
    columns: dict[str, list[object]] = {f.name: [] for f in _ARROW_SCHEMA}
    for r in records:
        d = asdict(r)
        for name in columns:
            columns[name].append(d[name])

    table = pa.Table.from_pydict(columns, schema=_ARROW_SCHEMA)

    metadata = {
        _PROVENANCE_METADATA_KEY: provenance.model_dump_json().encode("utf-8"),
        _SCHEMA_VERSION_KEY: _SCHEMA_VERSION,
    }
    table = table.replace_schema_metadata(metadata)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")


def read_records(path: Path | str) -> tuple[list[ObservationRecord], Provenance]:
    """Read Tier-1 records back. Raises ``ValueError`` if provenance is missing."""
    table = pq.read_table(path)
    raw_metadata = table.schema.metadata or {}

    provenance_bytes = raw_metadata.get(_PROVENANCE_METADATA_KEY)
    if provenance_bytes is None:
        raise ValueError(
            f"{path}: parquet file is missing the {_PROVENANCE_METADATA_KEY!r} metadata key"
        )
    provenance = Provenance.model_validate(json.loads(provenance_bytes.decode("utf-8")))

    rows = table.to_pylist()
    records = [ObservationRecord(**r) for r in rows]
    return records, provenance


def empty_record(
    *,
    run_id: str,
    binary: str,
    profile: str,
    dataset: str,
    image_id: str,
    record_kind: RecordKind,
    n_gt_in_frame: int,
    n_det_in_frame: int,
    frame_latency_ms: float,
    resolution_h: int,
) -> ObservationRecord:
    """Construct a record with NaN/sentinel defaults for axes and metrics.

    Callers fill in the fields that apply for the given ``record_kind`` —
    e.g. ``matched`` records set ``trans_err_m`` and ``rot_err_deg``,
    ``rejected_quad`` records set ``rejection_reason``. Reduces the chance of
    silently passing zero where NaN is meant.
    """
    return ObservationRecord(
        run_id=run_id,
        binary=binary,
        profile=profile,
        dataset=dataset,
        image_id=image_id,
        record_kind=record_kind,
        tag_id=None,
        distance_m=math.nan,
        aoi_deg=math.nan,
        ppm=math.nan,
        occlusion_ratio=math.nan,
        blur_px=math.nan,
        iso=math.nan,
        resolution_h=resolution_h,
        matched=False,
        trans_err_m=math.nan,
        rot_err_deg=math.nan,
        repro_err_px=math.nan,
        hamming_bits=-1,
        rejection_reason="",
        frame_latency_ms=frame_latency_ms,
        n_gt_in_frame=n_gt_in_frame,
        n_det_in_frame=n_det_in_frame,
    )
