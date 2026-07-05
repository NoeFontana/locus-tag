"""Unit tests for the per-instance comparative analysis (polars) + the
order-invariant corner-error fix in the collector.

Analysis tests build small polars frames directly (no dataset / no locus build);
the dataset-driven generate/report/deepdive paths are exercised via the CLI.
"""

from __future__ import annotations

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
