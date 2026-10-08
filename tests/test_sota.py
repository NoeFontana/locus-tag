"""Unit tests for ``tools/bench/sota`` (offline: synthetic runs and ground truth only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.bench import dataset_registry as dsm
from tools.bench.liu4k import score_detections
from tools.bench.matching import TagGroundTruth
from tools.bench.sota import report, score, spec

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
    assert v["corner_debiased_common_mean"].win is None


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


def test_radial_debias_separates_a_photometric_offset_from_scatter() -> None:
    gt = _square(0, 0)
    u = (gt - gt.mean(axis=0)) / np.linalg.norm(gt - gt.mean(axis=0), axis=1, keepdims=True)
    jitter = np.array([[0.1, 0.0], [0.0, -0.1], [-0.1, 0.0], [0.0, 0.1]])
    # Every corner 0.6 px inward (an sRGB-like edge shift) plus a small per-corner jitter.
    pairs = [(gt - 0.6 * u + jitter, gt), (gt - 0.6 * u - jitter, gt)]
    bias, rmse = score._radial_debias(pairs, [0, 1, 2, 3])
    assert bias == pytest.approx(-0.6, abs=1e-9)
    # What is left is the jitter alone (its radial parts cancel over the two matches).
    assert rmse == pytest.approx([0.1, 0.1], abs=1e-9)
    assert score._radial_debias([], [0, 1, 2, 3]) == (None, [])


def test_scoreboard_reports_bias_without_judging_it() -> None:
    from tools.bench.sota import scoreboard  # noqa: PLC0415

    board = {"hub-640": report.win_table({"locus_x": {"recall": 1.0}}, "gt-hub", "locus_x", True)}
    md = scoreboard.render(board, "locus_x", {"hub-640": {"locus_x": -0.6, "aruco_nano": -0.68}})
    assert "| hub-640 | -0.600 | -0.680 |" in md


# ── euroc ────────────────────────────────────────────────────────────────────


def _euroc_runs(
    tmp_path: Path,
    detectors: dict[str, dict[int, np.ndarray]],
    origin: tuple[float, float] = (230.0, 110.0),
    missing: dict[str, set[int]] | None = None,
    extra: dict[str, list[int]] | None = None,
) -> None:
    """Eight identical frames per detector; ``detectors`` maps a label to per-tag corner
    offsets (px, image space) applied to the exact projection of a board. Tags falling
    outside the image are not detected; ``missing`` drops more tags per detector and
    ``extra`` adds a detection with that id on tag 0's outline."""
    # Board (m) -> undistorted pixels: 300 px/m, board origin placed at ``origin``.
    u0, v0 = origin
    h = np.array([[300.0, 0.0, u0], [0.0, 300.0, v0], [0.0, 0.0, 1.0]])
    for label, offsets in detectors.items():
        lines = []
        for k in range(8):
            ids, corners = [], []
            for t in range(36):
                c = score._redist(score._proj(h, score._board(t, score.DIHEDRAL[0])))
                if t in (missing or {}).get(label, set()) or not np.all(
                    (c >= 0) & (c <= [score.EUROC_W - 1, score.EUROC_H - 1])
                ):
                    continue
                ids.append(t)
                corners.append((c + offsets.get(t, 0.0)).tolist())
            for tid in (extra or {}).get(label, []):
                ids.append(tid)
                corners.append(
                    score._redist(score._proj(h, score._board(0, score.DIHEDRAL[0]))).tolist()
                )
            lines.append(
                json.dumps(
                    {
                        "image": f"f{k}.png",
                        "ms": 1.0,
                        "ids": ids,
                        "corners": corners,
                        "convention": "opencv",
                    }
                )
            )
        (tmp_path / f"{label}.jsonl").write_text("\n".join(lines) + "\n")


def _inset(px: float) -> dict[int, np.ndarray]:
    """Per-tag offsets moving every corner ``px`` towards its tag's centre (an inset detector)."""
    d = px / np.sqrt(2.0)
    return {t: np.array([[d, d], [-d, d], [-d, -d], [d, -d]]) for t in range(36)}


def test_euroc_reference_is_pooled_from_self_consistent_detectors(tmp_path: Path) -> None:
    exact: dict[int, np.ndarray] = {}
    _euroc_runs(tmp_path, {"exact": exact, "inset": _inset(2.5)})
    r = score.score_euroc(tmp_path)
    assert r["_meta"]["reference_pool"] == ["exact"]
    assert r["exact"]["fp"] == 0
    assert r["exact"]["loo_own_median_px"] < 0.05


def test_euroc_judges_corners_in_image_pixels_inside_the_modelled_radius(tmp_path: Path) -> None:
    # Tag 0's top-left corner lies near (230, 110) px, inside the modelled radius: moving it
    # 5 px in the image makes a false positive, moving it 3 px does not.
    off5 = {0: np.array([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])}
    off3 = {0: np.array([[3.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])}
    _euroc_runs(tmp_path, {"exact": {}, "off5": off5, "off3": off3})
    r = score.score_euroc(tmp_path)
    assert r["off5"]["fp"] == 8
    assert r["off3"]["fp"] == 0


def test_euroc_does_not_judge_tags_beyond_the_lens_model() -> None:
    c = score.EUROC_K[:2, 2]
    inside = c + np.array([[score.EUROC_VALID_RADIUS_PX - 1.0, 0.0]])
    beyond = c + np.array([[0.0, score.EUROC_VALID_RADIUS_PX + 1.0]])
    assert score._modelled(inside)
    assert not score._modelled(np.vstack([inside, beyond]))


def test_euroc_recall_counts_only_present_tags(tmp_path: Path) -> None:
    # Shifted board: tag 0 is in the image but beyond the modelled radius, tag 5 is outside
    # the image; the other 34 tags are present.
    origin = (-50.0, -40.0)
    _euroc_runs(
        tmp_path,
        {"exact": {}, "miss_beyond": {}, "miss_present": {}},
        origin=origin,
        missing={"miss_beyond": {0}, "miss_present": {20}},
    )
    r = score.score_euroc(tmp_path)
    # Tag 0's detection is neither TP nor FP, and missing it is not a miss.
    for name in ("exact", "miss_beyond"):
        assert r[name]["recall"] == pytest.approx(100.0)
        assert (r[name]["fp"], r[name]["precision"]) == (0, pytest.approx(100.0))
    assert r["miss_present"]["recall"] == pytest.approx(100.0 * 33 / 34)


def test_euroc_off_board_ids_and_duplicates_are_false_positives(tmp_path: Path) -> None:
    # Id 40 is not on the 6x6 board; a second id-0 detection makes both id-0 detections
    # ambiguous, so tag 0 is missed as well.
    _euroc_runs(
        tmp_path,
        {"exact": {}, "off_board": {}, "duplicate": {}},
        extra={"off_board": [40], "duplicate": [0]},
    )
    r = score.score_euroc(tmp_path)
    assert (r["off_board"]["fp"], r["off_board"]["recall"]) == (8, pytest.approx(100.0))
    assert (r["duplicate"]["fp"], r["duplicate"]["recall"]) == (16, pytest.approx(100.0 * 35 / 36))


# ── ICRA tags.csv ────────────────────────────────────────────────────────────


def _write_tags_csv(path: Path, rows: list[tuple[str, int, int, float, float, int]]) -> None:
    lines = ["image,tag_id,corner,ground_truth_x,ground_truth_y,tag_fully_visible"]
    lines += [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def test_load_gt_csv_ignores_partly_visible_and_incomplete_tags(tmp_path: Path) -> None:
    rows = []
    for tid, x0, visible, n_corners in [(1, 100.5, 1, 4), (2, 300.5, 0, 4), (3, 500.5, 1, 3)]:
        for k, (x, y) in enumerate(_square(x0, 100.5)[:n_corners]):
            # A single not-fully-visible row marks the whole tag.
            rows.append(("f0.png", tid, k, x, y, visible if k == 0 else 1))
    rows.append(("f0.png", 1, 7, 0.0, 0.0, 1))  # out-of-range corner index: dropped
    _write_tags_csv(tmp_path / "tags.csv", rows)
    frames = score.load_gt_csv(tmp_path / "tags.csv")
    f0 = frames["f0.png"]
    assert [g.tag_id for g in f0.tags] == [1]
    np.testing.assert_allclose(f0.tags[0].corners, _square(100.5, 100.5))
    assert sorted(g.tag_id for g in f0.ignore) == [2, 3]
    incomplete = next(g for g in f0.ignore if g.tag_id == 3)
    assert np.isnan(incomplete.corners[3]).all() and not np.isnan(incomplete.corners[:3]).any()

    # Detections of ignored tags are neither TP nor FP (an incomplete tag is located by the
    # centre of its known corners); only tag 1 counts.
    runs = tmp_path / "runs"
    runs.mkdir()
    dets = [_square(100, 100), _square(300, 100), _square(500, 100)]
    (runs / "ref.jsonl").write_text(
        json.dumps(
            {
                "image": "/x/f0.png",
                "ms": 1.0,
                "ids": [1, 2, 3],
                "corners": [d.tolist() for d in dets],
                "convention": "opencv",
            }
        )
        + "\n"
    )
    s = spec.Spec(
        name="t",
        data="icra2020-forward",
        images="*.png",
        family="F",
        opencv_dict="D",
        border_bits=1,
        scorer="gt-csv",
        gt_convention="locus",
        gt="tags.csv",
    )
    out = score.score_gt(s, runs, frames)
    assert (out["ref"]["tp"], out["ref"]["fp"], out["ref"]["fn"]) == (1, 0, 0)
    assert out["ref"]["corner_mean"] == pytest.approx(0.0, abs=1e-9)


def test_report_lists_the_locus_extension_of_each_run(tmp_path: Path) -> None:
    ext = {"path": "/venv/locus/locus.abi3.so", "mtime": "2026-10-04T00:00:00+00:00"}
    first = {"image": "a.png", "ms": 1.0, "ids": [], "corners": [], "locus_extension": ext}
    (tmp_path / "locus_standard.jsonl").write_text(json.dumps(first) + "\n")
    (tmp_path / "locus_old.jsonl").write_text(json.dumps({**first, "locus_extension": None}) + "\n")
    (tmp_path / "opencv.jsonl").write_text(json.dumps(first) + "\n")
    assert report.locus_extensions(tmp_path) == {
        f"`{ext['path']}` (mtime {ext['mtime']})": ["locus_standard"],
        "not recorded": ["locus_old"],
    }


def test_euroc_undistortion_inverse_is_exact_to_its_own_tolerance() -> None:
    """The radtan inverse must invert the forward model, not approximate it.

    `_undist` used to be `cv2.undistortPoints`, which runs a fixed, small iteration
    count with no convergence test; on this lens it stopped short by a radius-growing
    amount (0.046 px median at r in [300, 380), 0.134 px worst). Because the shortfall
    is smooth a homography absorbs part of it, so the LOO median came out 3.98 % LOW for
    Locus against 0.82 % for OpenCV APRILTAG -- it flattered whichever detector had the
    better corners and did not cancel in the head-to-head. Asserted on the achieved
    round-trip error rather than on a loose bound, so a regression to an approximate
    inverse fails here.
    """
    rng = np.random.default_rng(7)
    pts = rng.uniform([0.0, 0.0], [score.EUROC_W, score.EUROC_H], size=(4000, 2))
    inside = np.linalg.norm(pts - score.EUROC_K[:2, 2], axis=1) <= score.EUROC_VALID_RADIUS_PX
    undistorted = pts[inside]
    assert len(undistorted) > 500, "sampling must cover the disc the scorer judges"

    err = np.linalg.norm(score._undist(score._redist(undistorted)) - undistorted, axis=1)
    assert err.max() < 1e-8, f"inverse is not a fixed point: max round-trip {err.max():.3e} px"

    # And it must stay exact where cv2's inverse degraded worst: the outer annulus.
    radius = np.linalg.norm(undistorted - score.EUROC_K[:2, 2], axis=1)
    outer = radius >= 300.0
    assert outer.sum() > 100, "the outer annulus is where an inexact inverse shows up"
    assert err[outer].max() < 1e-8, f"inexact at radius: {err[outer].max():.3e} px"
