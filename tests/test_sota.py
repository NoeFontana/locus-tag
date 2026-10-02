"""Unit tests for ``tools/bench/sota`` (offline: synthetic runs and ground truth only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.bench import dataset_registry as dsm
from tools.bench.liu4k import score_detections
from tools.bench.sota import report, score, spec
from tools.bench.utils import TagGroundTruth

_DATA = """
[hub]
kind = "hf-hub-subsets"
repo = "a/b"
revision = "0123456789012345678901234567890123456789"
dest = "tests/data/hub_cache"
license = "MIT"

[arc]
kind = "url-zip"
url = "https://example.invalid/x.zip"
md5 = "00000000000000000000000000000000"
ready = ["x"]
dest = "tests/data/arc"
license = "MIT"
"""

_BENCH = """
[sota.{name}]
data = "{data}"
images = "images/*.png"
family = "AprilTag36h11"
opencv_dict = "APRILTAG_36h11"
border_bits = 1
scorer = "gt-hub"
gt = "rich_truth.json"
gt_convention = "locus"
{extra}
"""


def _manifest(tmp_path: Path, *benches: str) -> Path:
    p = tmp_path / "datasets.toml"
    p.write_text(_DATA + "".join(benches))
    return p


def _bench(name: str = "b", data: str = "hub", extra: str = 'subset = "s"') -> str:
    return _BENCH.format(name=name, data=data, extra=extra)


# ── spec ─────────────────────────────────────────────────────────────────────


def test_shipped_specs_are_valid() -> None:
    specs = spec.load_specs()
    assert {"liu4k", "euroc", "icra-forward", "hub-640", "hub-tag16h5"} <= set(specs)
    assert specs["euroc"].border_bits == 2
    assert "aruco_nano" in specs["euroc"].unsupported


def test_registry_ignores_the_sota_table(tmp_path: Path) -> None:
    m = dsm.load_manifest(_manifest(tmp_path, _bench()))
    assert set(m) == {"hub", "arc"}


@pytest.mark.parametrize(
    ("bench", "msg"),
    [
        (_bench(data="nope"), "unknown dataset"),
        (_bench(extra=""), "`subset` is required"),
        (_bench(data="arc", extra='subset = "s"'), "`subset` is required"),
        (_bench().replace('"gt-hub"', '"magic"'), "unknown scorer"),
        (_bench().replace('gt = "rich_truth.json"', ""), "needs `gt`"),
        (_bench().replace('"locus"', '"matlab"'), "gt_convention"),
        (_bench().replace("border_bits = 1", "border_bits = 0"), "border_bits"),
    ],
)
def test_spec_validation_rejects(tmp_path: Path, bench: str, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        spec.load_specs(_manifest(tmp_path, bench))


def test_show_is_line_safe() -> None:
    s = spec.Spec(
        name="x",
        data="hub",
        images="*.png",
        family="F",
        opencv_dict="D",
        border_bits=2,
        scorer="euroc",
        gt_convention="opencv",
        unsupported={"aruco_nano": "multi\nline   reason"},
    )
    lines = spec.show(s).splitlines()
    assert "border_bits=2" in lines
    assert "apriltag_family=" in lines
    assert "unsupported.aruco_nano=multi line reason" in lines


# ── scoring ──────────────────────────────────────────────────────────────────


def _square(x: float, y: float, side: float = 40.0) -> np.ndarray:
    """Clockwise from top-left (OpenCV / Locus order)."""
    return np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=np.float64)


def test_first_match_reproduces_testperf() -> None:
    tags = [TagGroundTruth(1, _square(0, 0)), TagGroundTruth(1, _square(5, 0))]
    corners = np.stack([_square(4, 0), _square(100, 100), _square(1, 0)])
    ids = [1, 1, 1]
    pairs = score.first_match(ids, corners, tags, 10.0)
    # First unmatched same-id GT within range, in GT order (not the nearest one).
    assert pairs == [(0, 0), (2, 1)]
    tp, fp, fn = score_detections(ids, corners, tags)
    assert (tp, fp, fn) == (len(pairs), len(corners) - len(pairs), len(tags) - len(pairs))


def test_corner_permutation_is_fixed_per_detector_not_per_instance() -> None:
    t = score.Tally()
    gt = _square(0, 0)
    ccw = gt[[0, 3, 2, 1]]  # GT stored counter-clockwise
    t.matches[("a", 0)] = (gt + 0.1, ccw)
    t.matches[("b", 0)] = (gt + 0.1, ccw)
    # One instance with a rotated labelling: it must be penalised, not re-labelled.
    t.matches[("c", 0)] = (np.roll(gt, 1, axis=0), ccw)
    perm, errs = score._corner_errors(t)
    assert perm == [0, 3, 2, 1]
    assert errs[("a", 0)] == pytest.approx(np.sqrt(0.02))
    assert errs[("c", 0)] > 10.0


def _write_hub(root: Path, records: list[dict]) -> Path:
    p = root / "rich_truth.json"
    p.write_text(json.dumps({"records": records}))
    return p


def _record(image: str, tid: int, corners: np.ndarray, complete: bool = True) -> dict:
    return {
        "record_type": "TAG",
        "image_id": image,
        "tag_id": tid,
        "corners": corners.tolist(),
        "eval_complete": complete,
    }


def test_gt_scorer_conventions_ignores_and_counts(tmp_path: Path) -> None:
    gt_locus = _square(100.5, 100.5)  # +0.5 convention
    _write_hub(
        tmp_path,
        [
            _record("f0", 1, gt_locus),
            _record("f0", 2, _square(300.5, 100.5)),  # missed -> FN
            _record("f0", 3, _square(500.5, 100.5), complete=False),  # ignored
        ],
    )
    frames = score.load_gt_hub(tmp_path / "rich_truth.json")
    assert len(frames["f0.png"].tags) == 2 and len(frames["f0.png"].ignore) == 1

    runs = tmp_path / "runs"
    runs.mkdir()

    def rec(corners: list[np.ndarray], conv: str) -> dict:
        return {
            "image": "/x/f0.png",
            "ms": 1.0,
            "ids": [1, 3, 9],
            "corners": [c.tolist() for c in corners],
            "convention": conv,
        }

    exact_opencv = gt_locus - 0.5
    with open(runs / "ref.jsonl", "w") as f:
        dets = [exact_opencv, _square(500, 100), _square(700, 100)]
        f.write(json.dumps(rec(dets, "opencv")) + "\n")
    with open(runs / "locus_x.jsonl", "w") as f:
        dets = [gt_locus, _square(500.5, 100.5), _square(700.5, 100.5)]
        f.write(json.dumps(rec(dets, "locus")) + "\n")

    s = spec.Spec(
        name="t",
        data="hub",
        images="*.png",
        family="F",
        opencv_dict="D",
        border_bits=1,
        scorer="gt-hub",
        gt_convention="locus",
        subset="s",
        gt="rich_truth.json",
    )
    out = score.score_gt(s, runs, frames)
    for name in ("ref", "locus_x"):
        r = out[name]
        # id 1 TP, id 3 ignored (not evaluable), id 9 FP; id 2 missed.
        assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
        assert r["corner_mean"] == pytest.approx(0.0, abs=1e-9)
    assert out["_meta"]["common_tags"] == 1
    assert out["_meta"]["common_detectors"] == ["ref"]  # Locus runs never shape the set


# ── win table ────────────────────────────────────────────────────────────────


def test_win_table_compares_against_best_reference_and_ties_win() -> None:
    rows = {
        "locus_standard": {"recall": 70.0, "precision": 100.0, "f1": 82.0, "ms_mean": 9.0},
        "aruco_nano": {"recall": 66.0, "precision": 100.0, "f1": 80.0, "ms_mean": 5.0},
        "opencv_subpix": {"recall": 72.0, "precision": 99.0, "f1": 83.0, "ms_mean": 30.0},
        "apriltag3": {"recall": 99.0, "precision": 100.0, "f1": 99.0, "ms_mean": 1.0},
    }
    v = {x.metric.key: x for x in report.win_table(rows, "gt-hub", "locus_standard", True)}
    assert (v["recall"].reference, v["recall"].win) == ("opencv_subpix", False)
    assert (v["precision"].reference, v["precision"].win) == ("aruco_nano", True)  # tie
    assert (v["ms_mean"].reference, v["ms_mean"].win) == ("aruco_nano", False)
    # apriltag3 is reported but is not a reference of the win criterion.
    assert all(x.reference != "apriltag3" for x in v.values())
    # Missing corner metrics are not judged.
    assert v["corner_common_mean"].win is None


def test_win_table_does_not_judge_invalid_timing() -> None:
    rows = {"locus_standard": {"ms_mean": 1.0}, "opencv": {"ms_mean": 9.0}}
    v = {x.metric.key: x for x in report.win_table(rows, "liu4k", "locus_standard", False)}
    assert v["ms_mean"].win is None
    assert report.verdict_cell(v["ms_mean"]) == "n/a"


def test_only_current_runs_are_scored_and_crashes_are_reported(tmp_path: Path) -> None:
    for label in ("cur", "stale"):
        (tmp_path / f"{label}.jsonl").write_text(
            json.dumps({"image": "a.png", "ms": 1.0, "ids": [], "corners": []}) + "\n"
        )
    (tmp_path / "crash.jsonl.failed").write_text("partial")
    # Directory without xtask bookkeeping: every JSONL is a run.
    assert set(score._load_runs(tmp_path)) == {"cur", "stale"}
    (tmp_path / "detectors.txt").write_text("cur\n")
    (tmp_path / "failed.txt").write_text("crash\tsignal: 11 (SIGSEGV)\n")
    assert set(score._load_runs(tmp_path)) == {"cur"}
    assert score.failed_runs(tmp_path) == {"crash": "signal: 11 (SIGSEGV)"}


def test_scoreboard_takes_latency_from_a_timing_valid_tagged_run(tmp_path: Path) -> None:
    from tools.bench.sota import scoreboard  # noqa: PLC0415

    def write(run: str, timing_valid: bool, ms_locus: float) -> None:
        d = tmp_path / run
        d.mkdir()
        rows = {
            "locus_standard": {"recall": 90.0, "precision": 100.0, "f1": 95.0, "ms_mean": ms_locus},
            "aruco_nano": {"recall": 80.0, "precision": 100.0, "f1": 89.0, "ms_mean": 10.0},
        }
        (d / "score.json").write_text(json.dumps(rows))
        (d / "meta.json").write_text(json.dumps({"timing_valid": timing_valid}))

    write("hub-640", timing_valid=False, ms_locus=1.0)  # concurrent: latency not judged
    write("hub-640@t1", timing_valid=True, ms_locus=20.0)  # serial: judged, and lost
    board = scoreboard.collect(tmp_path, "locus_standard")
    cells = {v.metric.key: v for v in board["hub-640"]}
    assert cells["recall"].win is True
    assert cells["ms_mean"].champion == 20.0
    assert cells["ms_mean"].win is False
