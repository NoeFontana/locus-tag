"""Round-trip tests for ``tools/bench/records.py``."""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from tools.bench.records import (
    ObservationRecord,
    empty_record,
    read_records,
    write_records,
)
from tools.bench.schema import Provenance


def _make_provenance() -> Provenance:
    return Provenance(
        git_sha="0" * 40,
        git_dirty=False,
        cpu_model="test-cpu",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        locus_version="0.4.0",
        dataset_version="v1.0.0",
        rayon_threads=8,
        build_profile="release",
        timestamp_utc=datetime(2026, 5, 2, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_record(**overrides: Any) -> ObservationRecord:
    base = empty_record(
        run_id="run-1",
        binary="locus",
        profile="standard",
        dataset="single_tag_locus_v1_tag36h11_1920x1080",
        image_id="scene_0000_cam_0000",
        record_kind="matched",
        n_gt_in_frame=1,
        n_det_in_frame=1,
        frame_latency_ms=2.5,
        resolution_h=1080,
    )
    return dataclasses.replace(base, **overrides)


def test_roundtrip_preserves_record_count(tmp_path: Path) -> None:
    records = [_make_record(image_id=f"img_{i}") for i in range(10)]
    out = tmp_path / "run.parquet"
    write_records(records, _make_provenance(), out)

    loaded, prov = read_records(out)
    assert len(loaded) == 10
    assert prov.cpu_model == "test-cpu"


def test_roundtrip_preserves_field_values(tmp_path: Path) -> None:
    record = _make_record(
        tag_id=42,
        distance_m=2.5,
        aoi_deg=37.0,
        ppm=950.0,
        matched=True,
        trans_err_m=0.001,
        rot_err_deg=0.5,
        repro_err_px=0.3,
        hamming_bits=0,
    )
    out = tmp_path / "run.parquet"
    write_records([record], _make_provenance(), out)

    loaded, _ = read_records(out)
    assert loaded[0].tag_id == 42
    assert loaded[0].distance_m == pytest.approx(2.5, rel=1e-6)
    assert loaded[0].matched is True
    assert loaded[0].hamming_bits == 0


def test_nan_fields_round_trip(tmp_path: Path) -> None:
    """NaN floats must survive parquet round-trip — Arrow preserves NaN distinctly."""
    record = _make_record(distance_m=math.nan, aoi_deg=math.nan, ppm=math.nan)
    out = tmp_path / "run.parquet"
    write_records([record], _make_provenance(), out)

    loaded, _ = read_records(out)
    assert math.isnan(loaded[0].distance_m)
    assert math.isnan(loaded[0].aoi_deg)
    assert math.isnan(loaded[0].ppm)


def test_tag_id_none_round_trips(tmp_path: Path) -> None:
    """Unattributed rejected quads have ``tag_id=None`` — must survive as null."""
    record = _make_record(
        record_kind="rejected_quad",
        tag_id=None,
        rejection_reason="RejectedContrast",
    )
    out = tmp_path / "run.parquet"
    write_records([record], _make_provenance(), out)

    loaded, _ = read_records(out)
    assert loaded[0].tag_id is None
    assert loaded[0].rejection_reason == "RejectedContrast"


def test_provenance_lives_in_file_metadata_not_rows(tmp_path: Path) -> None:
    """Provenance must be a single key in the parquet footer, never replicated per row."""
    records = [_make_record(image_id=f"img_{i}") for i in range(100)]
    out = tmp_path / "run.parquet"
    write_records(records, _make_provenance(), out)

    table = pq.read_table(out)
    assert "git_sha" not in table.column_names
    assert "cpu_model" not in table.column_names

    metadata = table.schema.metadata
    assert metadata is not None
    assert b"locus.bench.provenance" in metadata


def test_missing_provenance_raises(tmp_path: Path) -> None:
    """A parquet file produced outside this writer (no provenance) is rejected on read."""
    import pyarrow as pa

    table = pa.table({"run_id": ["x"]})
    out = tmp_path / "no_provenance.parquet"
    pq.write_table(table, out)

    with pytest.raises(ValueError, match="missing the .* metadata key"):
        read_records(out)


def test_empty_record_defaults_are_safe(tmp_path: Path) -> None:
    record = empty_record(
        run_id="run-1",
        binary="locus",
        profile="standard",
        dataset="ds",
        image_id="img",
        record_kind="missed_gt",
        n_gt_in_frame=1,
        n_det_in_frame=0,
        frame_latency_ms=2.0,
        resolution_h=1080,
    )
    assert math.isnan(record.distance_m)
    assert record.tag_id is None
    assert record.matched is False
    assert record.hamming_bits == -1
    assert record.rejection_reason == ""

    out = tmp_path / "run.parquet"
    write_records([record], _make_provenance(), out)
    loaded, _ = read_records(out)
    # NaN != NaN by IEEE, so compare via dict + isnan-aware diff
    assert loaded[0].tag_id is None
    assert loaded[0].matched is False
    assert loaded[0].hamming_bits == -1
    assert loaded[0].rejection_reason == ""
    assert math.isnan(loaded[0].distance_m)
