"""Unit tests for ``tools/bench/liu4k.py`` (synthetic data; no network, no dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.bench.liu4k import load_liu4k, score_detections
from tools.bench.utils import TagGroundTruth


def _square(cx: float, cy: float, half: float = 20.0) -> list[list[float]]:
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


def test_load_liu4k_parses_markers(tmp_path: Path) -> None:
    (tmp_path / "001.jpg").write_bytes(b"")
    (tmp_path / "001.json").write_text(
        json.dumps({"markers": [{"id": 7, "corners": _square(100, 100), "rot": 2}]})
    )
    (tmp_path / "002.json").write_text(json.dumps({"markers": []}))  # no image: skipped
    samples = load_liu4k(tmp_path)
    assert list(samples) == ["001.jpg"]
    tag = samples["001.jpg"].tags[0]
    assert tag.tag_id == 7
    assert tag.corners.shape == (4, 2)
    assert samples["001.jpg"].rot == [2]


def _gt(tid: int, cx: float, cy: float) -> TagGroundTruth:
    return TagGroundTruth(tag_id=tid, corners=np.asarray(_square(cx, cy), dtype=np.float32))


def test_score_matches_aruco_nano_semantics() -> None:
    gt = [_gt(1, 100, 100), _gt(2, 400, 100)]
    quads = np.asarray([_square(108, 100), _square(400, 100), _square(700, 700)])
    # id-aware: det 1 within 8 px (TP), det with wrong id 9 (FP), far det (FP)
    assert score_detections([1, 9, 2], quads, gt) == (1, 2, 1)
    # boundary is inclusive (<= 10 px), 10.5 px is a miss
    assert score_detections([1], np.asarray([_square(110, 100)]), gt) == (1, 0, 1)
    assert score_detections([1], np.asarray([_square(110.5, 100)]), gt) == (0, 1, 2)
    # id-agnostic quad variant ignores ids; one quad claims one GT only
    assert score_detections(None, quads, gt) == (2, 1, 0)
    assert score_detections(None, np.asarray([_square(100, 100)] * 2), gt) == (1, 1, 1)
