"""Smoke tests for the plot modules.

Strategy: build a tiny synthetic Tier-1 records DataFrame, drive each plot
end-to-end, and assert the PNG/HTML files are produced and non-empty. Visual
correctness is checked by eye on real runs; this suite catches regressions
in axis handling, error paths, and the orchestrator wiring.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from tools.bench.plots import pareto, rejection, sweep


def _synthetic_df() -> pd.DataFrame:
    """Two binaries × two resolutions × a few records of each kind."""
    rows = []
    for binary in ("Locus", "OpenCV"):
        for res in (720, 1080):
            for i in range(5):
                rows.append(
                    {
                        "run_id": "run-1",
                        "binary": binary,
                        "profile": "standard",
                        "dataset": "synthetic",
                        "image_id": f"img_{i}",
                        "record_kind": "matched",
                        "tag_id": i,
                        "distance_m": 0.5 + 0.5 * i,
                        "aoi_deg": 10.0 + 10.0 * i,
                        "ppm": 500.0 + 200.0 * i,
                        "occlusion_ratio": 0.0,
                        "blur_px": math.nan,
                        "iso": math.nan,
                        "resolution_h": res,
                        "matched": True,
                        "trans_err_m": 0.001 + 0.001 * i,
                        "rot_err_deg": 0.1 + 0.05 * i,
                        "repro_err_px": 0.5,
                        "hamming_bits": 0,
                        "rejection_reason": "",
                        "frame_latency_ms": 5.0 if binary == "Locus" else 15.0,
                        "n_gt_in_frame": 1,
                        "n_det_in_frame": 1,
                    }
                )
            # one missed_gt per (binary, res) so recall != 100% for the test
            rows.append(
                {
                    "run_id": "run-1",
                    "binary": binary,
                    "profile": "standard",
                    "dataset": "synthetic",
                    "image_id": "img_5",
                    "record_kind": "missed_gt",
                    "tag_id": 5,
                    "distance_m": 3.0,
                    "aoi_deg": 60.0,
                    "ppm": 300.0,
                    "occlusion_ratio": 0.0,
                    "blur_px": math.nan,
                    "iso": math.nan,
                    "resolution_h": res,
                    "matched": False,
                    "trans_err_m": math.nan,
                    "rot_err_deg": math.nan,
                    "repro_err_px": math.nan,
                    "hamming_bits": -1,
                    "rejection_reason": "",
                    "frame_latency_ms": 5.0 if binary == "Locus" else 15.0,
                    "n_gt_in_frame": 1,
                    "n_det_in_frame": 0,
                }
            )
    # rejected_quad rows for Locus only (other libraries don't expose rejections)
    for res in (720, 1080):
        rows.append(
            {
                "run_id": "run-1",
                "binary": "Locus",
                "profile": "standard",
                "dataset": "synthetic",
                "image_id": "img_0",
                "record_kind": "rejected_quad",
                "tag_id": None,
                "distance_m": 1.0,
                "aoi_deg": 20.0,
                "ppm": 800.0,
                "occlusion_ratio": 0.0,
                "blur_px": math.nan,
                "iso": math.nan,
                "resolution_h": res,
                "matched": False,
                "trans_err_m": math.nan,
                "rot_err_deg": math.nan,
                "repro_err_px": math.nan,
                "hamming_bits": 5,
                "rejection_reason": "RejectedDecode",
                "frame_latency_ms": 5.0,
                "n_gt_in_frame": 1,
                "n_det_in_frame": 1,
            }
        )
    return pd.DataFrame(rows)


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

    from tools.bench.records import ObservationRecord, write_records
    from tools.bench.report import generate
    from tools.bench.schema import Provenance

    df = _synthetic_df()
    records = [
        ObservationRecord(
            run_id=str(r["run_id"]),
            binary=str(r["binary"]),
            profile=str(r["profile"]),
            dataset=str(r["dataset"]),
            image_id=str(r["image_id"]),
            record_kind=r["record_kind"],
            tag_id=None if pd.isna(r["tag_id"]) else int(r["tag_id"]),
            distance_m=float(r["distance_m"]),
            aoi_deg=float(r["aoi_deg"]),
            ppm=float(r["ppm"]),
            occlusion_ratio=float(r["occlusion_ratio"]),
            blur_px=float(r["blur_px"]),
            iso=float(r["iso"]),
            resolution_h=int(r["resolution_h"]),
            matched=bool(r["matched"]),
            trans_err_m=float(r["trans_err_m"]),
            rot_err_deg=float(r["rot_err_deg"]),
            repro_err_px=float(r["repro_err_px"]),
            hamming_bits=int(r["hamming_bits"]),
            rejection_reason=str(r["rejection_reason"]),
            frame_latency_ms=float(r["frame_latency_ms"]),
            n_gt_in_frame=int(r["n_gt_in_frame"]),
            n_det_in_frame=int(r["n_det_in_frame"]),
        )
        for _, r in df.iterrows()
    ]
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
    write_records(records, prov, parquet)

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
