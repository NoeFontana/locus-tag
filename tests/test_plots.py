"""Smoke tests for the plot modules.

Strategy: build a tiny synthetic Tier-1 records DataFrame, drive each plot
end-to-end, and assert the PNG/HTML files are produced and non-empty. Visual
correctness is checked by eye on real runs; this suite catches regressions
in axis handling, error paths, and the orchestrator wiring.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from tools.bench.plots import pareto, rejection, sweep
from tools.bench.records import ObservationRecord, empty_record


def _record(binary: str, res: int, image_id: str, **overrides: object) -> ObservationRecord:
    """Build an ``ObservationRecord`` with the run/profile/dataset frozen."""
    base = empty_record(
        run_id="run-1",
        binary=binary,
        profile="standard",
        dataset="synthetic",
        image_id=image_id,
        record_kind="matched",
        n_gt_in_frame=1,
        n_det_in_frame=1,
        frame_latency_ms=5.0 if binary == "Locus" else 15.0,
        resolution_h=res,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def _synthetic_records() -> list[ObservationRecord]:
    """Two binaries × two resolutions × a few records of each kind."""
    records: list[ObservationRecord] = []
    for binary in ("Locus", "OpenCV"):
        for res in (720, 1080):
            for i in range(5):
                records.append(
                    _record(
                        binary,
                        res,
                        f"img_{i}",
                        tag_id=i,
                        distance_m=0.5 + 0.5 * i,
                        aoi_deg=10.0 + 10.0 * i,
                        ppm=500.0 + 200.0 * i,
                        occlusion_ratio=0.0,
                        matched=True,
                        trans_err_m=0.001 + 0.001 * i,
                        rot_err_deg=0.1 + 0.05 * i,
                        repro_err_px=0.5,
                        hamming_bits=0,
                    )
                )
            # one missed_gt per (binary, res) so recall != 100% for the test
            records.append(
                _record(
                    binary,
                    res,
                    "img_5",
                    record_kind="missed_gt",
                    tag_id=5,
                    distance_m=3.0,
                    aoi_deg=60.0,
                    ppm=300.0,
                    occlusion_ratio=0.0,
                    n_det_in_frame=0,
                )
            )
    # rejected_quad rows for Locus only (other libraries don't expose rejections)
    for res in (720, 1080):
        records.append(
            _record(
                "Locus",
                res,
                "img_0",
                record_kind="rejected_quad",
                distance_m=1.0,
                aoi_deg=20.0,
                ppm=800.0,
                occlusion_ratio=0.0,
                hamming_bits=5,
                rejection_reason="RejectedDecode",
            )
        )
    return records


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(r) for r in _synthetic_records()])


def test_pareto_plot_writes_png(tmp_path: Path) -> None:
    df = _synthetic_df()
    out = pareto.plot(df, tmp_path / "pareto.png")
    assert out.exists()
    assert out.stat().st_size > 1024  # non-empty PNG


def test_sweep_recall_writes_png(tmp_path: Path) -> None:
    df = _synthetic_df()
    out = sweep.plot(df, tmp_path / "sweep.png", axis="distance_m", metric="recall")
    assert out.exists()
    assert out.stat().st_size > 1024


def test_sweep_trans_err_writes_png(tmp_path: Path) -> None:
    df = _synthetic_df()
    out = sweep.plot(df, tmp_path / "sweep.png", axis="aoi_deg", metric="trans_err_p50_m")
    assert out.exists()
    assert out.stat().st_size > 1024


def test_sweep_raises_when_no_rows(tmp_path: Path) -> None:
    """Empty DataFrame after filtering should raise — caller handles in report.py."""
    df = pd.DataFrame(columns=_synthetic_df().columns)
    with pytest.raises(ValueError, match="no rows after binning"):
        sweep.plot(df, tmp_path / "sweep.png", axis="distance_m", metric="recall")


def test_rejection_plot_writes_png(tmp_path: Path) -> None:
    df = _synthetic_df()
    out = rejection.plot(df, tmp_path / "rejection.png", group_by="resolution_h")
    assert out.exists()
    assert out.stat().st_size > 1024


def test_rejection_plot_raises_when_no_rejections(tmp_path: Path) -> None:
    """A run with zero rejected_quad rows can't render — caller handles."""
    df = _synthetic_df()
    df = df[df["record_kind"] != "rejected_quad"]
    with pytest.raises(ValueError, match="no rejected_quad rows"):
        rejection.plot(df, tmp_path / "rejection.png")


def test_report_generate_produces_index_html(tmp_path: Path) -> None:
    """End-to-end: write a parquet, run the orchestrator, verify outputs."""
    from datetime import datetime, timezone

    from tools.bench.records import write_records
    from tools.bench.report import generate
    from tools.bench.schema import Provenance

    prov = Provenance(
        git_sha="0" * 40,
        git_dirty=False,
        cpu_model="test",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        locus_version="0.4.0",
        dataset_version="test",
        rayon_threads=8,
        build_profile="release",
        timestamp_utc=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )
    parquet = tmp_path / "run.parquet"
    write_records(_synthetic_records(), prov, parquet)

    out_dir = tmp_path / "report"
    generate([parquet], out_dir, title="test")

    assert (out_dir / "index.html").exists()
    assert (out_dir / "pareto.png").exists()
    # At least one sweep + the rejection plot
    sweep_pngs = list(out_dir.glob("sweep_*.png"))
    assert len(sweep_pngs) >= 1
    assert (out_dir / "rejection_by_resolution.png").exists()
    # Confirm index references at least one PNG
    html = (out_dir / "index.html").read_text()
    assert ".png" in html
    assert "test" in html  # title
