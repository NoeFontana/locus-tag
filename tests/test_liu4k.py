"""Unit tests for ``tools/bench/liu4k.py`` (synthetic data; no network, no dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.bench.liu4k import load_liu4k, quad_recall_for_image
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


def test_quad_recall_is_one_to_one_and_id_agnostic() -> None:
    gt = [
        TagGroundTruth(tag_id=1, corners=np.asarray(_square(100, 100), dtype=np.float32)),
        TagGroundTruth(tag_id=2, corners=np.asarray(_square(400, 100), dtype=np.float32)),
    ]
    quads = np.asarray([_square(102, 99)])  # near GT 1 only
    assert quad_recall_for_image(quads, gt) == (1, 2)
    # One quad cannot satisfy two GT tags.
    both_near = [
        TagGroundTruth(tag_id=1, corners=np.asarray(_square(100, 100), dtype=np.float32)),
        TagGroundTruth(tag_id=2, corners=np.asarray(_square(105, 100), dtype=np.float32)),
    ]
    assert quad_recall_for_image(quads, both_near) == (1, 2)
    assert quad_recall_for_image(np.empty((0, 4, 2)), gt) == (0, 2)
