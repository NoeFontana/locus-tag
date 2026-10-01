"""Liu4K dataset (Zenodo 10.5281/zenodo.18667018): load and score.

Provisioning (pinned URL + md5, license, citation) lives in ``xtask/datasets.toml`` and
runs through :mod:`tools.bench.dataset_registry` (``cargo xtask data fetch liu4k``). Licensed
CC-BY-4.0; the data lands in ``tests/data/liu4k/`` (gitignored) and is never committed,
packaged in wheels/sdist, or republished in converted form.

Ground truth is per-image corners + ids + a rotation code; there are no poses
and no intrinsics. The markers use the ``ARUCO_MIP_36h12`` dictionary
(``locus.TagFamily.ArUcoMip36h12``). Scoring is id-agnostic *quad* recall
(:func:`score_detections` with ``det_ids=None``) plus, when that family is selected, id-aware
decode recall/precision via the shared centre matcher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.bench.dataset_registry import fetch, load_manifest
from tools.bench.utils import TagGroundTruth

LIU4K_CACHE_DIR = Path("tests/data/liu4k")
LIU4K_SUBDIR = "liu4k_markers_1024"
# Dictionary the markers were generated from (per aruco_nano's testperf.cpp).
LIU4K_DICTIONARY = "ARUCO_MIP_36h12"

# Liu4K-specific match radius (px): aruco_nano's testperf.cpp uses `dist <= 10.0`.
# Deliberately NOT the repo-wide MATCH_DISTANCE_THRESHOLD_PX used by other datasets.
LIU4K_MATCH_THRESHOLD_PX = 10.0

# `FunnelStatus` code for "passed the geometric funnel, rejected by the decoder"
# (see tools/bench/collect.py).
_PASSED_FUNNEL = 1

_ENTRY = load_manifest()["liu4k"]
CITATION = f"{_ENTRY.citation}. {_ENTRY.license}."


def prepare_liu4k(cache_dir: Path = LIU4K_CACHE_DIR) -> Path:
    """Fetch Liu4K if needed (pinned, md5-verified; see ``xtask/datasets.toml``); return the data dir."""
    return fetch("liu4k", dest=cache_dir) / LIU4K_SUBDIR


@dataclass(frozen=True)
class Liu4kSample:
    image: str  # file name inside the data dir
    tags: list[TagGroundTruth]
    rot: list[int]  # per-tag rotation code from the annotation (unused for scoring)


def load_liu4k(data_dir: Path) -> dict[str, Liu4kSample]:
    """Parse every ``NNN.json`` next to its ``NNN.jpg`` into ground truth."""
    out: dict[str, Liu4kSample] = {}
    for jpath in sorted(data_dir.glob("*.json")):
        img = jpath.with_suffix(".jpg")
        if not img.exists():
            continue
        with open(jpath) as f:
            markers = json.load(f).get("markers", [])
        tags = [
            TagGroundTruth(tag_id=int(m["id"]), corners=np.asarray(m["corners"], dtype=np.float32))
            for m in markers
        ]
        out[img.name] = Liu4kSample(img.name, tags, [int(m.get("rot", 0)) for m in markers])
    return out


def candidate_quads(batch: Any) -> np.ndarray:
    """Accepted detections plus decoder-rejected quads, as an ``(N, 4, 2)`` array.

    Funnel-rejected quads are excluded so the count reflects quads the geometric
    detector actually stands behind. Works with any configured family because
    the decode outcome is ignored.
    """
    parts = [np.asarray(batch.corners, dtype=np.float64).reshape(-1, 4, 2)]
    if batch.rejected_corners is not None and batch.rejected_funnel_status is not None:
        keep = np.asarray(batch.rejected_funnel_status) == _PASSED_FUNNEL
        parts.append(np.asarray(batch.rejected_corners, dtype=np.float64)[keep].reshape(-1, 4, 2))
    return np.concatenate(parts, axis=0)


def _center(corners: np.ndarray) -> np.ndarray:
    return np.asarray(corners, dtype=np.float64).reshape(4, 2).mean(axis=0)


def score_detections(
    det_ids: list[int] | None,
    det_corners: np.ndarray,
    gt_tags: list[TagGroundTruth],
    threshold: float = LIU4K_MATCH_THRESHOLD_PX,
) -> tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` exactly as aruco_nano's ``evaluateDetection``.

    For each detection in order, the *first* not-yet-matched GT marker with the same id
    whose centre is within ``threshold`` pixels (``<=``) is a TP; a detection with no such
    GT is a FP; ``fn = n_gt - tp``. Pass ``det_ids=None`` for the id-agnostic quad variant,
    which drops the id condition (same first-match rule, same threshold).
    """
    matched = [False] * len(gt_tags)
    gt_centers = [_center(g.corners) for g in gt_tags]
    tp = fp = 0
    for k in range(len(det_corners)):
        c = _center(det_corners[k])
        hit = False
        for j, g in enumerate(gt_tags):
            if matched[j] or (det_ids is not None and int(det_ids[k]) != g.tag_id):
                continue
            if float(np.linalg.norm(c - gt_centers[j])) <= threshold:
                matched[j] = True
                tp += 1
                hit = True
                break
        if not hit:
            fp += 1
    return tp, fp, len(gt_tags) - tp


@dataclass
class Liu4kTally:
    """Accumulated aruco_nano-style TP/FP/FN over images."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, counts: tuple[int, int, int]) -> None:
        self.tp += counts[0]
        self.fp += counts[1]
        self.fn += counts[2]

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d * 100 if d else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d * 100 if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0
