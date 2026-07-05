"""Unit tests for the per-instance comparative analysis (polars) + the
order-invariant corner-error fix in the collector.

Analysis tests build small polars frames directly (no dataset / no locus build);
the dataset-driven generate/report/deepdive paths are exercised via the CLI.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from tools.bench.compare import analysis as A


# --------------------------------------------------------------------------- #
# Corner error is order-PRESERVING; cross-library conventions are fixed by
# per-library adapters (not by an order-invariant metric).
# --------------------------------------------------------------------------- #
def test_collector_repro_is_order_preserving():
    from tools.bench.collect import Collector
    from tools.bench.utils import TagGroundTruth

    gt_corners = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    gt = TagGroundTruth(tag_id=7, corners=gt_corners)
    c = Collector.new("run", "locus", "tuned", "d")
    # Corner 0 of the detection compares to corner 0 of GT: aligned corners -> 0,
    # but corners rolled by 2 (a genuine wrong-orientation detection) MUST surface
    # as a large error, not be silently minimised away.
    aligned = {"id": 7, "corners": gt_corners.astype(np.float64).tolist(), "center": [5.0, 5.0]}
    assert c._errors(aligned, gt, None)[2] == pytest.approx(0.0, abs=1e-6)
    rolled = {"id": 7, "corners": np.roll(gt_corners.astype(np.float64), 2, axis=0).tolist()}
    assert c._errors(rolled, gt, None)[2] > 5.0  # wrong orientation is a real error


def test_apriltag_corner_adapter_is_a_fixed_pair_swap():
    from tools.bench.utils import _APRILTAG_CORNER_TO_GT

    perm = _APRILTAG_CORNER_TO_GT
    assert perm == [1, 0, 3, 2]  # swap corner pairs (0,1) and (2,3)
    # A fixed permutation, and its own inverse (applying twice is identity), so it
    # normalises the convention without ever hiding a real orientation error.
    assert [[1, 0, 3, 2][i] for i in perm] == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Analysis over a synthetic wide/long frame
# --------------------------------------------------------------------------- #
def _long() -> pl.DataFrame:
    # Two GT instances in one stratum; three series. locus worse on inst 1, best on inst 2.
    rows = []
    strat = "res=fhd|ppm=hi|aoi=frontal|dist=near|mot=static"

    def rec(series, image, tag, matched, repro, trans, rot):
        binary, profile = series.split(":")
        return dict(
            binary=binary,
            profile=profile,
            series=series,
            dataset="d",
            image_id=image,
            tag_id=tag,
            record_kind="matched" if matched else "missed_gt",
            stratum_id=strat,
            resolution_h=1080,
            distance_m=0.5,
            aoi_deg=10.0,
            ppm=1500.0,
            matched=matched,
            repro_err_px=repro,
            trans_err_m=trans,
            rot_err_deg=rot,
            hamming_bits=0,
        )

    rows += [
        rec("locus:tuned", "img0", 1, True, 2.0, 0.02, 1.0),
        rec("opencv_aruco:tuned", "img0", 1, True, 0.5, 0.01, 0.5),
        rec("apriltag:tuned", "img0", 1, True, 0.8, 0.03, 0.7),
        rec("locus:tuned", "img1", 2, True, 0.3, 0.005, 0.2),
        rec("opencv_aruco:tuned", "img1", 2, True, 1.0, 0.02, 0.9),
        rec("apriltag:tuned", "img1", 2, False, None, None, None),  # apriltag missed inst 2
    ]
    return pl.DataFrame(rows)


def test_build_wide_one_row_per_instance():
    wide, series = A.build_wide(_long())
    assert len(wide) == 2  # two GT instances
    assert set(series) == {"locus:tuned", "opencv_aruco:tuned", "apriltag:tuned"}
    # apriltag missed inst 2 -> detected False, error null.
    inst2 = wide.filter(pl.col("tag_id") == 2)
    assert inst2["apriltag:tuned__detected"][0] is False
    assert inst2["apriltag:tuned__repro"][0] is None


def test_compare_section_delta_orientation():
    wide, _ = A.build_wide(_long())
    sec = A.compare_section(
        wide,
        locus_series="locus:tuned",
        competitors=["opencv_aruco:tuned", "apriltag:tuned"],
        metric="repro",
    )
    inst1 = sec.filter(pl.col("tag_id") == 1)
    # locus 2.0 vs best competitor opencv 0.5 -> delta +1.5 (locus worse), best=opencv
    assert inst1["delta"][0] == pytest.approx(1.5)
    assert inst1["best_competitor"][0] == "opencv_aruco:tuned"
    inst2 = sec.filter(pl.col("tag_id") == 2)
    # locus 0.3 vs best competitor opencv 1.0 (apriltag missed) -> delta -0.7 (locus better)
    assert inst2["delta"][0] == pytest.approx(-0.7)
    assert inst2["best_competitor"][0] == "opencv_aruco:tuned"


def test_winrate_in_range_and_counts():
    wide, series = A.build_wide(_long())
    wr = A.winrate_by_stratum(wide, series)
    rates = wr["win_rate"].to_numpy()
    assert rates.min() >= 0.0 and rates.max() <= 1.0
    # repro winners: inst1 opencv (0.5), inst2 locus (0.3) -> opencv 1 win, locus 1 win of 2.
    repro = wr.filter(pl.col("metric") == "repro")
    wins = dict(zip(repro["series"].to_list(), repro["wins"].to_list(), strict=True))
    assert wins.get("opencv_aruco:tuned") == 1 and wins.get("locus:tuned") == 1


def test_worst_locus_ranks_and_flags():
    wide, _ = A.build_wide(_long())
    sec = A.compare_section(
        wide,
        locus_series="locus:tuned",
        competitors=["opencv_aruco:tuned", "apriltag:tuned"],
        metric="repro",
    )
    wl = A.worst_locus(sec, metric="repro", top_n=5)
    # Only inst1 has locus worse (delta +1.5); inst2 locus is better -> excluded.
    assert len(wl) == 1
    assert wl["tag_id"][0] == 1
    assert wl["failure_kind"][0] == "worse_corners"


def test_accuracy_by_resolution_recall():
    long = _long()
    acc = A.accuracy_by_resolution(long, sorted(long["series"].unique().to_list()))
    by = dict(zip(acc["series"].to_list(), acc["recall"].to_list(), strict=True))
    assert by["locus:tuned"] == pytest.approx(1.0)  # detected both
    assert by["apriltag:tuned"] == pytest.approx(0.5)  # missed inst 2


def test_best_of_empty_competitors_is_null_not_crash():
    # Single-library case: no competitors -> min_horizontal([]) would raise; the
    # guard must return null value + null name instead.
    val, name = A._best_of([], [])
    got = pl.DataFrame({"x": [1.0]}).select(val.alias("v"), name.alias("n"))
    assert got["v"][0] is None and got["n"][0] is None


# --------------------------------------------------------------------------- #
# Real-data NaN handling: records store math.nan (not None) for missed/pose-less
# errors. load_instances must convert NaN→null so aggregations/argmin are correct.
# --------------------------------------------------------------------------- #
def _write_records(recs, path):
    from tools.bench.collect import build_provenance
    from tools.bench.records import write_records

    write_records(recs, build_provenance(), path)


def _obs(binary, kind, tag, *, matched, repro, trans):
    import dataclasses

    from tools.bench.records import empty_record

    r = empty_record(
        run_id="r",
        binary=binary,
        profile="tuned",
        dataset="d",
        image_id="img0",
        record_kind=kind,
        n_gt_in_frame=1,
        n_det_in_frame=1,
        frame_latency_ms=1.0,
        resolution_h=1080,
    )
    return dataclasses.replace(
        r,
        tag_id=tag,
        matched=matched,
        repro_err_px=repro,
        trans_err_m=trans,
        distance_m=0.5,
        aoi_deg=10.0,
        ppm=1500.0,
    )


def test_nan_missed_rows_do_not_poison_best_competitor(tmp_path):
    p = tmp_path / "r.parquet"
    _write_records(
        [
            _obs("locus", "matched", 1, matched=True, repro=1.0, trans=0.02),
            _obs("opencv_aruco", "matched", 1, matched=True, repro=0.5, trans=0.01),
            _obs("apriltag", "missed_gt", 1, matched=False, repro=math.nan, trans=math.nan),
        ],
        p,
    )
    long = A.load_instances(p)
    # NaN error on the missed row is converted to null (not left as NaN).
    apr = long.filter(pl.col("series") == "apriltag:tuned")
    assert apr["repro_err_px"][0] is None
    wide, _ = A.build_wide(long)
    sec = A.compare_section(
        wide,
        locus_series="locus:tuned",
        competitors=["apriltag:tuned", "opencv_aruco:tuned"],
        metric="repro",
    )
    # Best competitor is opencv (0.5), NOT the missing apriltag; delta = 1.0 − 0.5.
    assert sec["best_competitor"][0] == "opencv_aruco:tuned"
    assert sec["delta"][0] == pytest.approx(0.5)


def test_generate_refuses_corrupted_parquet_on_partial_failure(tmp_path, monkeypatch):
    # If one (series×dataset) cell fails (empty records) while another succeeds,
    # writing the parquet would silently score the failed series as all-missed.
    # generate must refuse rather than alias a failure to a miss.
    from tools.bench.compare import generate as G
    from tools.bench.tune.space import SearchSpace

    cfg = G.ChosenConfig(
        library="locus",
        profile_label="tuned",
        param_hash="h",
        param_values={},
        space=SearchSpace(library="locus", base_profile="standard", params={}),
        space_name="s",
    )
    monkeypatch.setattr(G, "run_search_records", lambda cells, workers=None: [[object()], []])
    with pytest.raises(ValueError, match="corrupted comparison parquet"):
        G.generate_instance_records(
            configs=[cfg],
            datasets=["d1", "d2"],
            family=0,
            data_dir=tmp_path,
            out_path=tmp_path / "o.parquet",
        )


def test_accuracy_p99_finite_with_nan_pose(tmp_path):
    # A matched detection lacking a pose stores NaN trans; the accuracy table must
    # not return NaN p99 / biased p50 (null is skipped by median/quantile).
    p = tmp_path / "r.parquet"
    _write_records(
        [
            _obs("locus", "matched", 1, matched=True, repro=0.5, trans=0.01),
            _obs("locus", "matched", 2, matched=True, repro=0.5, trans=math.nan),
            _obs("locus", "matched", 3, matched=True, repro=0.5, trans=0.03),
        ],
        p,
    )
    long = A.load_instances(p)
    acc = A.accuracy_by_resolution(long, ["locus:tuned"])
    p99 = acc["trans_p99"][0]
    assert p99 is not None and p99 == p99  # finite, not NaN
