"""Score SOTA JSONL runs.

Usage::

    python -m tools.bench.sota.score <benchmark> <runs_dir> <out.json>

The scorer is chosen by the benchmark's ``[sota.<name>]`` table (``xtask/datasets.toml``).
Every scorer compares corners in OpenCV's integer-pixel-centre convention.

**liu4k** — aruco_nano ``testperf.cpp`` rule (same id, centre distance <= 10 px,
first-match TP/FP/FN) via :mod:`tools.bench.liu4k`, plus recall by marker side and
corner error on the matched markers.

**gt-csv / gt-hub** (ICRA 2020 ``tags.csv`` / render-tag ``rich_truth.json``) — a detection
is a TP when it pairs with a same-id GT tag whose centre is within
``MATCH_DISTANCE_THRESHOLD_PX`` (:func:`tools.bench.matching.match_detections_to_gt`).
Detections of tags the GT marks as not fully visible / not evaluable are ignored (neither TP
nor FP); such tags are not counted as misses.

**Corner error** (liu4k, gt-*) is order-preserving: one dihedral relabelling of the GT corners
is selected per (benchmark, detector) — the one minimising the median error over all its
matches — and then applied to every match; it is never chosen per instance. Per-tag error is
the RMSE over the four corners. ``corner_common_*`` restricts to the GT tags matched by every
detector with >= 20 % recall, so detectors are compared on the same tags.

**euroc** (``cam_april``, no per-corner GT) uses a ground-truth-free protocol:

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

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.bench.liu4k import LIU4K_MATCH_THRESHOLD_PX, load_liu4k
from tools.bench.matching import MATCH_DISTANCE_THRESHOLD_PX, match_detections_to_gt
from tools.bench.sota.spec import Spec, load_specs
from tools.bench.utils import TagGroundTruth

Dets = dict[int, list[np.ndarray]]
# Detectors below this recall do not restrict the common-tag set (they would empty it).
COMMON_MIN_RECALL = 20.0


def run_labels(runs_dir: Path) -> list[str] | None:
    """Detectors whose JSONL belongs to the current image list (xtask's ``detectors.txt``);
    ``None`` for a directory written without one (every ``*.jsonl`` is then a run)."""
    path = runs_dir / "detectors.txt"
    return path.read_text().split() if path.exists() else None


def failed_runs(runs_dir: Path) -> dict[str, str]:
    """Detectors that crashed (xtask's ``failed.txt``: ``label<TAB>exit status``)."""
    path = runs_dir / "failed.txt"
    lines = path.read_text().splitlines() if path.exists() else []
    return dict(line.split("\t", 1) for line in lines if "\t" in line)


def _load_runs(runs_dir: Path) -> dict[str, list[dict[str, Any]]]:
    labels = run_labels(runs_dir)
    runs = {}
    for p in sorted(runs_dir.glob("*.jsonl")):
        if labels is not None and p.stem not in labels:
            continue
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


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


# ── shared GT scoring ────────────────────────────────────────────────────────

SIDE_BINS = [0.0, 20.0, 45.0, 100.0, 250.0, float("inf")]
# The 8 dihedral relabellings of a quad's corners (4 rotations x 2 windings).
DIHEDRAL = [np.roll(np.arange(4), k) for k in range(4)] + [
    np.roll(np.array([0, 3, 2, 1]), k) for k in range(4)
]


@dataclass
class Frame:
    """GT of one image: evaluated tags, plus tags to ignore (partly visible / not evaluable)."""

    tags: list[TagGroundTruth] = field(default_factory=list)
    ignore: list[TagGroundTruth] = field(default_factory=list)


@dataclass
class Tally:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ms: list[float] = field(default_factory=list)
    hit_by_bin: np.ndarray = field(default_factory=lambda: np.zeros(len(SIDE_BINS) - 1))
    n_by_bin: np.ndarray = field(default_factory=lambda: np.zeros(len(SIDE_BINS) - 1))
    # (image, gt index) -> (detected corners, GT corners), OpenCV convention.
    matches: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        return _pct(self.tp, self.tp + self.fn)

    @property
    def precision(self) -> float:
        return _pct(self.tp, self.tp + self.fp)


def _side(corners: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, 0), axis=1)))


def _tally_frame(
    t: Tally,
    image: str,
    frame: Frame,
    pairs: list[tuple[int, int]],
    corners: np.ndarray,
    ignored: int,
    to_opencv_gt: float,
) -> None:
    t.tp += len(pairs)
    t.fp += len(corners) - len(pairs) - ignored
    t.fn += len(frame.tags) - len(pairs)
    hit = {g for _, g in pairs}
    for j, g in enumerate(frame.tags):
        b = int(np.searchsorted(SIDE_BINS, _side(g.corners), side="right")) - 1
        t.n_by_bin[b] += 1
        t.hit_by_bin[b] += j in hit
    for d, g in pairs:
        gt = np.asarray(frame.tags[g].corners, dtype=np.float64) - to_opencv_gt
        t.matches[(image, g)] = (corners[d], gt)


def _rmse(det: np.ndarray, gt: np.ndarray, perm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((det - gt[perm]) ** 2, axis=1))))


def _corner_errors(t: Tally) -> tuple[list[int], dict[tuple[str, int], float]]:
    """Per-match RMSE under the single dihedral relabelling that fits this detector best."""
    if not t.matches:
        return list(range(4)), {}
    pairs = list(t.matches.values())
    medians = [float(np.median([_rmse(d, g, perm) for d, g in pairs])) for perm in DIHEDRAL]
    perm = DIHEDRAL[int(np.argmin(medians))]
    return perm.tolist(), {k: _rmse(d, g, perm) for k, (d, g) in t.matches.items()}


def _stats(errs: list[float], prefix: str) -> dict[str, float | None]:
    if not errs:
        return {f"{prefix}_{k}": None for k in ("mean", "median", "p90", "p99")}
    a = np.asarray(errs)
    return {
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_p90": float(np.percentile(a, 90)),
        f"{prefix}_p99": float(np.percentile(a, 99)),
    }


def _summarise(tallies: dict[str, Tally]) -> dict[str, Any]:
    errors = {n: _corner_errors(t) for n, t in tallies.items()}
    eligible = [n for n, t in tallies.items() if t.recall >= COMMON_MIN_RECALL]
    common = set.intersection(*(set(errors[n][1]) for n in eligible)) if eligible else set[Any]()
    out: dict[str, Any] = {"_meta": {"common_tags": len(common), "common_detectors": eligible}}
    for n, t in tallies.items():
        perm, errs = errors[n]
        out[n] = {
            "tp": t.tp,
            "fp": t.fp,
            "fn": t.fn,
            "recall": t.recall,
            "precision": t.precision,
            "f1": _f1(t.precision, t.recall),
            "ms_mean": float(np.mean(t.ms)) if t.ms else None,
            "recall_by_side": {
                f"[{SIDE_BINS[i]:g},{SIDE_BINS[i + 1]:g})": _pct(t.hit_by_bin[i], t.n_by_bin[i])
                for i in range(len(SIDE_BINS) - 1)
            },
            "corner_perm": perm,
            **_stats(list(errs.values()), "corner"),
            **_stats([errs[k] for k in common] if n in eligible else [], "corner_common"),
        }
    return out


# ── Liu4K ────────────────────────────────────────────────────────────────────


def first_match(
    ids: list[int], corners: np.ndarray, tags: list[TagGroundTruth], threshold: float
) -> list[tuple[int, int]]:
    """aruco_nano ``evaluateDetection``: each detection, in order, takes the *first*
    unmatched same-id GT marker whose centre is within ``threshold`` (``<=``)."""
    used = [False] * len(tags)
    centres = [np.asarray(g.corners, dtype=np.float64).mean(0) for g in tags]
    pairs = []
    for d, (tid, c) in enumerate(zip(ids, corners, strict=True)):
        for j, g in enumerate(tags):
            if used[j] or g.tag_id != tid:
                continue
            if np.linalg.norm(c.mean(0) - centres[j]) <= threshold:
                used[j] = True
                pairs.append((d, j))
                break
    return pairs


def score_liu4k(spec: Spec, runs_dir: Path) -> dict[str, Any]:
    gt = load_liu4k(spec.root() / Path(spec.images).parent)
    tallies: dict[str, Tally] = {}
    for name, recs in _load_runs(runs_dir).items():
        t = tallies[name] = Tally()
        for rec in recs:
            image = Path(rec["image"]).name
            frame = Frame(tags=gt[image].tags)
            corners = _to_opencv(rec)
            pairs = first_match(rec["ids"], corners, frame.tags, LIU4K_MATCH_THRESHOLD_PX)
            _tally_frame(t, image, frame, pairs, corners, 0, 0.0)
            t.ms.append(float(rec["ms"]))
    return _summarise(tallies)


# ── ICRA 2020 / render-tag hub ───────────────────────────────────────────────


def load_gt_csv(path: Path) -> dict[str, Frame]:
    """ICRA 2020 ``tags.csv``: one row per corner; partly visible tags are ignored."""
    rows: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            tag = rows[r["image"]].setdefault(
                int(r["tag_id"]), {"corners": [None] * 4, "visible": True}
            )
            k = int(r["corner"])
            if 0 <= k < 4:
                tag["corners"][k] = [float(r["ground_truth_x"]), float(r["ground_truth_y"])]
            if int(r.get("tag_fully_visible", 1)) == 0:
                tag["visible"] = False
    out: dict[str, Frame] = defaultdict(Frame)
    for image, tags in rows.items():
        for tid, tag in tags.items():
            complete = all(c is not None for c in tag["corners"])
            g = TagGroundTruth(
                tag_id=tid,
                corners=np.asarray(
                    [c if c is not None else [np.nan, np.nan] for c in tag["corners"]],
                    dtype=np.float64,
                ),
            )
            (out[image].tags if tag["visible"] and complete else out[image].ignore).append(g)
    return dict(out)


def load_gt_hub(path: Path) -> dict[str, Frame]:
    """render-tag ``rich_truth.json`` (v1 list or v2 ``{records: [...]}``), TAG records only."""
    data = json.loads(path.read_text())
    entries = data["records"] if isinstance(data, dict) and "records" in data else data
    out: dict[str, Frame] = defaultdict(Frame)
    for e in entries:
        if e.get("record_type", "TAG") != "TAG":
            continue
        image = e.get("image_filename") or e.get("image_id")
        if image is None:
            continue
        image = image if image.endswith(".png") else f"{image}.png"
        g = TagGroundTruth(
            tag_id=int(e["tag_id"]), corners=np.asarray(e["corners"], dtype=np.float64)
        )
        (out[image].tags if e.get("eval_complete", True) else out[image].ignore).append(g)
    return dict(out)


def _ignored(
    ids: list[int], corners: np.ndarray, matched: set[int], ignore: list[TagGroundTruth]
) -> int:
    """Unmatched detections that land on an ignored GT tag (same id, centre in range)."""
    n = 0
    for d, (tid, c) in enumerate(zip(ids, corners, strict=True)):
        if d in matched:
            continue
        for g in ignore:
            centre = np.nanmean(np.asarray(g.corners, dtype=np.float64), axis=0)
            if g.tag_id == tid and np.linalg.norm(c.mean(0) - centre) < MATCH_DISTANCE_THRESHOLD_PX:
                n += 1
                break
    return n


def score_gt(spec: Spec, runs_dir: Path, gt: dict[str, Frame] | None = None) -> dict[str, Any]:
    """Score every run against ``gt`` (default: the spec's GT file)."""
    if gt is None:
        gt = (load_gt_csv if spec.scorer == "gt-csv" else load_gt_hub)(spec.gt_path())
    shift = 0.5 if spec.gt_convention == "locus" else 0.0
    tallies: dict[str, Tally] = {}
    for name, recs in _load_runs(runs_dir).items():
        t = tallies[name] = Tally()
        for rec in recs:
            image = Path(rec["image"]).name
            frame = gt.get(image, Frame())
            corners = _to_opencv(rec)
            # Matching is convention-free at the 20 px scale; corner error is not.
            dets = [
                {"id": int(i), "center": (c.mean(0) + shift).tolist()}
                for i, c in zip(rec["ids"], corners, strict=True)
            ]
            pairs = match_detections_to_gt(dets, frame.tags).pairs
            ign = _ignored(rec["ids"], corners + shift, {d for d, _ in pairs}, frame.ignore)
            _tally_frame(t, image, frame, pairs, corners, ign, shift)
            t.ms.append(float(rec["ms"]))
    return _summarise(tallies)


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


def score(name: str, runs_dir: Path) -> dict[str, Any]:
    spec = load_specs()[name]
    if spec.scorer == "liu4k":
        return score_liu4k(spec, runs_dir)
    if spec.scorer == "euroc":
        return score_euroc(runs_dir)
    return score_gt(spec, runs_dir)


def main(argv: list[str]) -> None:
    name, runs_dir, out = argv
    result = score(name, Path(runs_dir))
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
