"""Score SOTA JSONL runs.

Usage::

    python -m tools.bench.sota.score liu4k <runs_dir> <out.json>
    python -m tools.bench.sota.score euroc <runs_dir> <out.json>

Liu4K: aruco_nano ``testperf.cpp`` rule (same id, centre distance <= 10 px,
first-match TP/FP/FN) via :mod:`tools.bench.liu4k`, plus recall by marker side.

EuRoC ``cam_april`` (no per-corner GT) uses a ground-truth-free protocol:

* **Presence** — corners pooled over all detectors are undistorted with the
  published cam0 radtan calibration and a RANSAC homography board -> undistorted
  image is fitted per frame (exact for a planar board). A tag is *present* when
  its four reprojected, redistorted corners lie >= 3 px inside the image.
* **Precision** — a detection is a TP when its id is on the board and all four
  corners lie within 4 px (undistorted) of the reference projection.
* **Accuracy** — leave-one-tag-out: a tag's corners are predicted from a DLT
  homography fitted to the *same detector's* other tags in the frame; reported on
  the (frame, tag) set common to every detector with >= 20 % recall.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.bench.liu4k import (
    LIU4K_CACHE_DIR,
    LIU4K_SUBDIR,
    Liu4kTally,
    load_liu4k,
    score_detections,
)

Dets = dict[int, list[np.ndarray]]


def _load_runs(runs_dir: Path) -> dict[str, list[dict[str, Any]]]:
    runs = {}
    for p in sorted(runs_dir.glob("*.jsonl")):
        with open(p) as f:
            runs[p.stem] = [json.loads(line) for line in f if line.strip()]
    if not runs:
        raise SystemExit(f"no *.jsonl runs in {runs_dir}")
    return runs


def _to_opencv(rec: dict[str, Any]) -> np.ndarray:
    """Corners as (N, 4, 2) in OpenCV's integer-pixel-centre convention."""
    c = np.asarray(rec["corners"], dtype=np.float64).reshape(-1, 4, 2)
    return c - 0.5 if rec.get("convention") == "locus" else c


def _pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else 0.0


# ── Liu4K ────────────────────────────────────────────────────────────────────

SIDE_BINS = [0.0, 20.0, 45.0, 100.0, 250.0, float("inf")]


def score_liu4k(runs_dir: Path) -> dict[str, Any]:
    gt = load_liu4k(LIU4K_CACHE_DIR / LIU4K_SUBDIR)
    out: dict[str, Any] = {}
    for name, recs in _load_runs(runs_dir).items():
        tally = Liu4kTally()
        hit_by_bin = np.zeros(len(SIDE_BINS) - 1)
        n_by_bin = np.zeros(len(SIDE_BINS) - 1)
        for rec in recs:
            tags = gt[Path(rec["image"]).name].tags
            corners = _to_opencv(rec)
            tally.add(score_detections(rec["ids"], corners, tags))
            # Per-GT hit (same first-match rule) for the side-stratified recall.
            used = [False] * len(tags)
            for tid, c in zip(rec["ids"], corners, strict=True):
                for j, g in enumerate(tags):
                    if used[j] or g.tag_id != tid:
                        continue
                    if np.linalg.norm(c.mean(0) - g.corners.mean(0)) <= 10.0:
                        used[j] = True
                        break
            for j, g in enumerate(tags):
                side = float(np.mean(np.linalg.norm(g.corners - np.roll(g.corners, 1, 0), axis=1)))
                b = int(np.searchsorted(SIDE_BINS, side, side="right")) - 1
                n_by_bin[b] += 1
                hit_by_bin[b] += used[j]
        out[name] = {
            "tp": tally.tp,
            "fp": tally.fp,
            "fn": tally.fn,
            "recall": tally.recall,
            "precision": tally.precision,
            "f1": tally.f1,
            "ms_mean": float(np.mean([r["ms"] for r in recs])),
            "recall_by_side": {
                f"[{SIDE_BINS[i]:g},{SIDE_BINS[i + 1]:g})": _pct(hit_by_bin[i], n_by_bin[i])
                for i in range(len(n_by_bin))
            },
        }
    return out


# ── EuRoC cam_april ──────────────────────────────────────────────────────────

EUROC_K = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
EUROC_D = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
EUROC_W, EUROC_H = 752, 480
TAG, PITCH = 0.088, 0.088 * 1.3  # Kalibr april_6x6.yaml: tagSize 0.088, tagSpacing 0.3
_LOCAL = np.array([[0.0, 0.0], [TAG, 0.0], [TAG, TAG], [0.0, TAG]])
# Corner order on the board is detector-convention dependent; the right dihedral
# permutation is selected empirically per detector (all agree in practice).
_PERMS = [list(np.roll(range(4), k)) for k in range(4)] + [
    list(np.roll([0, 3, 2, 1], k)) for k in range(4)
]


def _board(tid: int, perm: list[int]) -> np.ndarray:
    r, c = divmod(tid, 6)
    return _LOCAL[perm] + np.array([c * PITCH, r * PITCH])


def _undist(px: np.ndarray) -> np.ndarray:
    pts = px.reshape(-1, 1, 2).astype(np.float64)
    return cv2.undistortPoints(pts, EUROC_K, EUROC_D, P=EUROC_K).reshape(-1, 2)


def _redist(pu: np.ndarray) -> np.ndarray:
    n = np.c_[
        (pu[:, 0] - EUROC_K[0, 2]) / EUROC_K[0, 0],
        (pu[:, 1] - EUROC_K[1, 2]) / EUROC_K[1, 1],
        np.ones(len(pu)),
    ]
    out, _ = cv2.projectPoints(n, np.zeros(3), np.zeros(3), EUROC_K, EUROC_D)
    return out.reshape(-1, 2)


def _proj(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2)


def _frames(recs: list[dict[str, Any]]) -> dict[str, tuple[float, Dets]]:
    frames = {}
    for rec in recs:
        dets: Dets = defaultdict(list)
        for tid, c in zip(rec["ids"], _to_opencv(rec), strict=True):
            dets[int(tid)].append(c)
        frames[Path(rec["image"]).name] = (float(rec["ms"]), dict(dets))
    return frames


def _unique(dets: Dets) -> dict[int, np.ndarray]:
    return {i: cs[0] for i, cs in dets.items() if 0 <= i < 36 and len(cs) == 1}


def _find_perm(frames: dict[str, tuple[float, Dets]]) -> list[int]:
    best = (float("inf"), _PERMS[0])
    for perm in _PERMS:
        errs = []
        for _, dets in list(frames.values())[::7]:
            ok = _unique(dets)
            if len(ok) < 6:
                continue
            src = np.concatenate([_board(i, perm) for i in ok])
            dst = np.concatenate([_undist(c) for c in ok.values()])
            h, _ = cv2.findHomography(src, dst, 0)
            if h is not None:
                errs.append(np.median(np.linalg.norm(_proj(h, src) - dst, axis=1)))
        if errs and float(np.median(errs)) < best[0]:
            best = (float(np.median(errs)), perm)
    return best[1]


def score_euroc(runs_dir: Path) -> dict[str, Any]:
    runs = {n: _frames(r) for n, r in _load_runs(runs_dir).items()}
    names = sorted(runs)
    perms = {n: _find_perm(runs[n]) for n in names}
    st: dict[str, dict[str, float]] = {n: defaultdict(float) for n in names}
    loo: dict[str, dict[tuple[str, int], np.ndarray]] = {n: {} for n in names}
    n_ref = 0
    for fr in sorted(runs[names[0]]):
        src, dst = [], []
        for n in names:
            for i, c in _unique(runs[n][fr][1]).items():
                src.append(_board(i, perms[n]))
                dst.append(_undist(c))
        if len(src) < 8:
            continue
        href, mask = cv2.findHomography(np.concatenate(src), np.concatenate(dst), cv2.RANSAC, 3.0)
        if href is None or mask is None or mask.sum() < 16:
            continue
        n_ref += 1
        present = set()
        for t in range(36):
            pu = _proj(href, _board(t, _PERMS[0]))
            if np.any(np.abs(pu - EUROC_K[:2, 2]) > 1500):
                continue
            pd = _redist(pu)
            if np.all((pd >= 3.0) & (pd <= [EUROC_W - 4.0, EUROC_H - 4.0])):
                present.add(t)
        for n in names:
            ms, dets = runs[n][fr]
            good: dict[int, np.ndarray] = {}
            for i, cs in dets.items():
                for c in cs:
                    ok = 0 <= i < 36 and len(cs) == 1
                    if ok:
                        err = np.linalg.norm(_undist(c) - _proj(href, _board(i, perms[n])), axis=1)
                        ok = bool(err.max() <= 4.0)
                    if ok:
                        good[i] = _undist(c)
                        st[n]["tp"] += 1
                    else:
                        st[n]["fp"] += 1
            st[n]["frames"] += 1
            st[n]["ms"] += ms
            st[n]["present"] += len(present)
            st[n]["hit"] += len(present & set(good))
            if len(good) >= 5:
                for t in good:
                    others = [i for i in good if i != t]
                    h, _ = cv2.findHomography(
                        np.concatenate([_board(i, perms[n]) for i in others]),
                        np.concatenate([good[i] for i in others]),
                        0,
                    )
                    if h is not None:
                        e = np.linalg.norm(_proj(h, _board(t, perms[n])) - good[t], axis=1)
                        loo[n][(fr, t)] = e
    recall = {n: _pct(st[n]["hit"], st[n]["present"]) for n in names}
    eligible = [n for n in names if recall[n] >= 20.0]
    common = set.intersection(*(set(loo[n]) for n in eligible)) if eligible else set()
    out: dict[str, Any] = {"_meta": {"reference_frames": n_ref, "common_loo_tags": len(common)}}
    for n in names:
        own = np.concatenate(list(loo[n].values())) if loo[n] else np.array([np.nan])
        com = np.concatenate([loo[n][k] for k in common]) if (common and n in eligible) else None
        out[n] = {
            "recall": recall[n],
            "fp": int(st[n]["fp"]),
            "precision": _pct(st[n]["tp"], st[n]["tp"] + st[n]["fp"]),
            "ms_mean": st[n]["ms"] / max(1.0, st[n]["frames"]),
            "loo_own_median_px": float(np.median(own)),
            "loo_common_median_px": float(np.median(com)) if com is not None else None,
            "loo_common_p90_px": float(np.percentile(com, 90)) if com is not None else None,
        }
    return out


def main(argv: list[str]) -> None:
    dataset, runs_dir, out = argv
    scorer = {"liu4k": score_liu4k, "euroc": score_euroc}[dataset]
    result = scorer(Path(runs_dir))
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
