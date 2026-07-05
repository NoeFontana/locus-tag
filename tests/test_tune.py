"""Unit tests for the tuning harness pure logic (no datasets / no locus build).

Covers the dataset-free pieces: search-space grid/random, config materialization
enum coercion, param hashing, record summarization, and Pareto selection with the
constraint gate + baseline guard. The dataset-driven executor path is exercised
end-to-end via the ``bench tune`` CLI, not here.
"""

from __future__ import annotations

import math

import pytest

from tools.bench.records import empty_record
from tools.bench.tune.aggregate import summarize
from tools.bench.tune.materialize import param_hash
from tools.bench.tune.pareto_select import (
    ConfigSummary,
    aggregate_across_datasets,
    apply_baseline_guard,
    select_frontier,
)
from tools.bench.tune.space import ParamSpec, SearchSpace, load_space


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def test_shipped_spaces_load_and_declare_expected_library():
    for name, lib in [
        ("locus_default", "locus"),
        ("opencv_default", "opencv_aruco"),
        ("apriltag_default", "apriltag"),
    ]:
        space = load_space(name)
        assert space.library == lib
        assert space.params  # non-empty


def test_grid_axis_int_and_float_inclusive_of_high():
    assert ParamSpec(kind="int", low=4, high=12, step=4).grid_axis() == [4, 8, 12]
    assert ParamSpec(kind="float", low=0.0, high=0.5, step=0.25).grid_axis() == [0.0, 0.25, 0.5]
    assert ParamSpec(kind="bool").grid_axis() == [False, True]
    assert ParamSpec(kind="categorical", choices=["a", "b"]).grid_axis() == ["a", "b"]


def test_random_draws_are_seed_reproducible():
    space = SearchSpace(
        library="locus",
        params={"decoder.min_contrast": ParamSpec(kind="float", low=5.0, high=30.0)},
    )
    a = list(space.random_draws(8, seed=1))
    b = list(space.random_draws(8, seed=1))
    c = list(space.random_draws(8, seed=2))
    assert a == b
    assert a != c


def test_grid_size_matches_enumeration():
    space = SearchSpace(
        library="locus",
        params={
            "a": ParamSpec(kind="int", low=0, high=2, step=1),
            "b": ParamSpec(kind="categorical", choices=["x", "y"]),
        },
    )
    assert space.grid_size() == len(list(space.grid())) == 3 * 2


def test_categorical_requires_choices():
    with pytest.raises(ValueError, match="requires 'choices'"):
        ParamSpec(kind="categorical")


def test_exclusions_skip_incompatible_combos_in_grid_and_random():
    space = SearchSpace(
        library="locus",
        params={
            "quad.extraction_mode": ParamSpec(
                kind="categorical", choices=["ContourRdp", "EdLines"]
            ),
            "decoder.refinement_mode": ParamSpec(
                kind="categorical", choices=["None", "Erf", "Gwlf"]
            ),
        },
        exclusions=[{"quad.extraction_mode": "EdLines", "decoder.refinement_mode": "Erf"}],
    )
    grid = list(space.grid())
    assert len(grid) == 2 * 3 - 1  # the one excluded combo is dropped
    assert space.grid_size() == 5
    assert not any(
        d["quad.extraction_mode"] == "EdLines" and d["decoder.refinement_mode"] == "Erf"
        for d in grid
    )
    # random draws also never emit the excluded pair, and still yield n valid ones.
    draws = list(space.random_draws(20, seed=0))
    assert len(draws) == 20
    assert not any(
        d["quad.extraction_mode"] == "EdLines" and d["decoder.refinement_mode"] == "Erf"
        for d in draws
    )


def test_shipped_locus_space_excludes_edlines_erf():
    space = load_space("locus_default")
    assert {"quad.extraction_mode": "EdLines", "decoder.refinement_mode": "Erf"} in space.exclusions


def test_grid_and_random_work_without_optuna_and_bayes_degrades_friendly():
    import importlib.util

    from tools.bench.tune.search import iter_param_draws

    space = SearchSpace(
        library="locus",
        params={"decoder.min_contrast": ParamSpec(kind="float", low=5.0, high=30.0)},
    )
    # grid + random must never touch optuna.
    assert len(list(iter_param_draws(space, "random", 3, seed=0))) == 3
    assert len(list(iter_param_draws(space, "grid", 0, 0))) >= 1

    if importlib.util.find_spec("optuna") is None:
        # bayes without the [tune] extra must raise a friendly RuntimeError,
        # never leak a raw ImportError.
        with pytest.raises(RuntimeError, match=r"\.\[tune\]"):
            list(iter_param_draws(space, "bayes", 3, seed=0))


# --------------------------------------------------------------------------- #
# Materialize / hashing
# --------------------------------------------------------------------------- #
def test_param_hash_is_order_independent_and_library_scoped():
    assert param_hash("locus", {"a": 1, "b": 2}) == param_hash("locus", {"b": 2, "a": 1})
    assert param_hash("locus", {"a": 1}) != param_hash("opencv_aruco", {"a": 1})
    assert len(param_hash("locus", {})) == 16


def test_family_resolver_accepts_short_aruco_aliases():
    # bench tune must resolve the same aliases bench real accepts (regression:
    # FAMILY_MAP previously omitted the short ArUco aliases).
    from tools.bench.utils import resolve_tag_family

    for alias in ["4x4_50", "4x4_100", "6x6_250", "36h11", "AprilTag36h11"]:
        assert isinstance(resolve_tag_family(alias), int)
    with pytest.raises(ValueError, match="unknown tag family"):
        resolve_tag_family("nope")


def test_select_images_limit_zero_is_zero_frames():
    from tools.bench.tune.executor import _select_images

    names = ["a", "b", "c"]
    assert _select_images(names, 0, 0) == []  # limit=0 means zero, not "all"
    assert _select_images(names, 0, None) == ["a", "b", "c"]  # None means no limit
    assert _select_images(names, 1, 1) == ["b"]


# --------------------------------------------------------------------------- #
# Aggregate / summarize
# --------------------------------------------------------------------------- #
def _rec(kind, *, trans=math.nan, resolution_h=1080, image_id="img0"):
    r = empty_record(
        run_id="r",
        binary="locus",
        profile="p",
        dataset="d",
        image_id=image_id,
        record_kind=kind,
        n_gt_in_frame=1,
        n_det_in_frame=1,
        frame_latency_ms=10.0,
        resolution_h=resolution_h,
    )
    import dataclasses

    return dataclasses.replace(r, matched=(kind == "matched"), trans_err_m=trans)


def test_summarize_recall_precision_and_pose_percentiles():
    records = [
        _rec("matched", trans=0.01),
        _rec("matched", trans=0.03),
        _rec("missed_gt"),
        _rec("false_positive"),
    ]
    overall, per_stratum = summarize(records)
    # tp=2, fn=1 -> recall 2/3 ; tp=2, fp=1 -> precision 2/3
    assert overall["recall"] == pytest.approx(2 / 3)
    assert overall["precision"] == pytest.approx(2 / 3)
    assert overall["trans_err_p50_m"] == pytest.approx(0.02)
    assert overall["pose_samples"] == 2.0
    assert per_stratum  # at least one stratum


def test_summarize_empty_returns_empty():
    assert summarize([]) == ({}, {})


# --------------------------------------------------------------------------- #
# Pareto selection
# --------------------------------------------------------------------------- #
def _summary(ph, recall, trans_p99, precision=1.0, mean=0.001, latency=None):
    return ConfigSummary(
        library="locus",
        param_hash=ph,
        param_values={},
        recall=recall,
        precision=precision,
        trans_mean_m=mean,
        trans_p99_m=trans_p99,
        rot_p99_deg=1.0,
        latency_p95_ms=latency,
        n_datasets=1,
    )


def test_select_frontier_precision_gate_and_domination():
    summaries = [
        _summary("a", recall=0.9, trans_p99=0.01),  # frontier
        _summary("b", recall=0.8, trans_p99=0.02),  # dominated by a
        _summary("c", recall=0.95, trans_p99=0.05),  # frontier (higher recall)
        _summary("d", recall=0.99, trans_p99=0.001, precision=0.5),  # infeasible (precision)
    ]
    entries = select_frontier(summaries, precision_floor=0.99, tail_metric="trans_p99")
    by = {e.summary.param_hash: e for e in entries}
    assert by["a"].on_frontier and by["c"].on_frontier
    assert not by["b"].on_frontier
    assert not by["d"].feasible and not by["d"].on_frontier  # gated out


def test_latency_budget_gate():
    summaries = [
        _summary("fast", recall=0.9, trans_p99=0.01, latency=20.0),
        _summary("slow", recall=0.95, trans_p99=0.01, latency=99.0),
    ]
    entries = select_frontier(
        summaries, precision_floor=0.0, tail_metric="trans_p99", latency_budget_ms=50.0
    )
    by = {e.summary.param_hash: e for e in entries}
    assert by["fast"].feasible and not by["slow"].feasible


def test_nan_tail_never_lands_on_frontier():
    # A pose-less / zero-recall config has NaN tail; it must be gated out, not
    # silently sit on the frontier (NaN is never dominated in pareto_mask).
    summaries = [
        _summary("real", recall=0.9, trans_p99=0.02),
        _summary("nan_tail", recall=1.0, trans_p99=float("nan")),
    ]
    entries = select_frontier(summaries, precision_floor=0.0, tail_metric="trans_p99")
    by = {e.summary.param_hash: e for e in entries}
    assert by["real"].on_frontier
    assert not by["nan_tail"].feasible and not by["nan_tail"].on_frontier


def test_baseline_guard_uses_best_tail_and_best_mean_separately():
    # standard has the lower mean; high_accuracy has the lower tail. A config that
    # regresses standard's mean must NOT be promotable even though it beats
    # high_accuracy's looser mean.
    entries = select_frontier(
        [_summary("cand", recall=1.0, trans_p99=0.004, mean=0.010)],
        precision_floor=0.0,
        tail_metric="trans_p99",
    )
    standard = _summary("standard", recall=1.0, trans_p99=0.030, mean=0.005)
    high_acc = _summary("high_accuracy", recall=1.0, trans_p99=0.005, mean=0.020)
    apply_baseline_guard(entries, [standard, high_acc])
    e = entries[0]
    assert e.no_tail_regression  # 0.004 <= best tail 0.005
    assert not e.no_mean_regression  # 0.010 > best mean 0.005 (standard's)
    assert not e.promotable  # derived: tail-ok but mean-regressed


def test_baseline_guard_flags_tail_regression():
    entries = select_frontier(
        [
            _summary("better", recall=1.0, trans_p99=0.005, mean=0.001),
            _summary("worse", recall=1.0, trans_p99=0.05, mean=0.02),
        ],
        precision_floor=0.0,
        tail_metric="trans_p99",
    )
    baseline = _summary("base", recall=1.0, trans_p99=0.01, mean=0.005)
    ref = apply_baseline_guard(entries, [baseline])
    assert ref is not None and ref.param_hash == "base"
    by = {e.summary.param_hash: e for e in entries}
    # "better" beats the baseline tail+mean -> promotable; "worse" regresses.
    if by["better"].on_frontier:
        assert by["better"].promotable
    if by["worse"].on_frontier:
        assert not by["worse"].promotable


def test_aggregate_macro_averages_across_datasets():
    from tools.bench.tune.executor import CellResult

    def cr(ph, dataset, recall, trans_p99):
        return CellResult(
            library="locus",
            param_hash=ph,
            dataset=dataset,
            param_values={},
            overall={
                "recall": recall,
                "precision": 1.0,
                "trans_err_mean_m": 0.001,
                "trans_err_p99_m": trans_p99,
                "rot_err_p99_deg": 1.0,
            },
            per_stratum={},
            latency_valid=False,
            n_frames=5,
        )

    results = [
        cr("x", "d1", 1.0, 0.02),
        cr("x", "d2", 0.8, 0.04),
    ]
    summaries = aggregate_across_datasets(results)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.recall == pytest.approx(0.9)  # macro-average
    assert s.trans_p99_m == pytest.approx(0.03)
    assert s.n_datasets == 2


# --------------------------------------------------------------------------- #
# Lever analysis
# --------------------------------------------------------------------------- #
def _tidy_results(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def test_lever_sensitivity_ranks_the_influential_knob():
    import pandas as pd

    from tools.bench.tune.levers import lever_sensitivity
    from tools.bench.tune.results import OVERALL_STRATUM

    # knob "big" flips recall (0.5 vs 1.0); knob "noise" barely moves it.
    results = _tidy_results(
        [
            {
                "library": "locus",
                "param_hash": h,
                "dataset": "d",
                "scope": "overall",
                "stratum_id": OVERALL_STRATUM,
                "metric": "recall",
                "value": v,
            }
            for h, v in [("a", 0.5), ("b", 0.5), ("c", 1.0), ("d", 1.0)]
        ]
    )
    configs = pd.DataFrame(
        [
            {"library": "locus", "param_hash": "a", "param_values": '{"big": 0, "noise": 1}'},
            {"library": "locus", "param_hash": "b", "param_values": '{"big": 0, "noise": 2}'},
            {"library": "locus", "param_hash": "c", "param_values": '{"big": 1, "noise": 1}'},
            {"library": "locus", "param_hash": "d", "param_values": '{"big": 1, "noise": 2}'},
        ]
    )
    sens = lever_sensitivity(results, configs, min_configs=3)
    recall = sens[sens["metric"] == "recall"].set_index("param_name")
    assert recall.loc["big", "effect"] > recall.loc["noise", "effect"]
    assert recall.loc["big", "effect_norm"] == pytest.approx(1.0)  # normalised top


def test_comparative_deltas_orients_negative_as_locus_worse():
    import pandas as pd

    from tools.bench.tune.levers import comparative_deltas

    sid = "res=fhd|ppm=hi|aoi=frontal|dist=near|mot=static"
    results = _tidy_results(
        [
            # recall: locus 0.8 vs opencv 0.9 -> delta negative (locus worse).
            {
                "library": "locus",
                "param_hash": "L",
                "dataset": "d",
                "scope": "stratum",
                "stratum_id": sid,
                "metric": "recall",
                "value": 0.8,
            },
            {
                "library": "opencv_aruco",
                "param_hash": "O",
                "dataset": "d",
                "scope": "stratum",
                "stratum_id": sid,
                "metric": "recall",
                "value": 0.9,
            },
            # trans tail: locus 0.01 vs opencv 0.03 -> locus better -> delta positive.
            {
                "library": "locus",
                "param_hash": "L",
                "dataset": "d",
                "scope": "stratum",
                "stratum_id": sid,
                "metric": "trans_err_p99_m",
                "value": 0.01,
            },
            {
                "library": "opencv_aruco",
                "param_hash": "O",
                "dataset": "d",
                "scope": "stratum",
                "stratum_id": sid,
                "metric": "trans_err_p99_m",
                "value": 0.03,
            },
        ]
    )
    deltas = comparative_deltas(results, {"locus": "L", "opencv_aruco": "O"})
    by_metric = deltas.set_index("metric")
    assert by_metric.loc["recall", "delta"] == pytest.approx(-0.1)  # locus worse
    assert by_metric.loc["trans_err_p99_m", "delta"] == pytest.approx(0.02)  # locus better
    assert isinstance(pd.DataFrame(), pd.DataFrame)
