"""Liu4K dataset (Zenodo 10.5281/zenodo.18667018): download, verify, load, score.

Licensed CC-BY-4.0 (see ``docs/engineering/benchmarking.md`` for attribution).
The data is fetched at runtime into ``tests/data/liu4k/`` (gitignored) and is
never committed, packaged in wheels/sdist, or republished in converted form.

Ground truth is per-image corners + ids + a rotation code; there are no poses
and no intrinsics. The markers use the ``ARUCO_MIP_36h12`` dictionary
(``locus.TagFamily.ArUcoMip36h12``). Scoring is id-agnostic *quad* recall
(:func:`quad_recall_for_image`) plus, when that family is selected, id-aware
decode recall/precision via the shared centre matcher.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.bench.matching import MATCH_DISTANCE_THRESHOLD_PX
from tools.bench.utils import TagGroundTruth

ZENODO_RECORD = 18667018
ZENODO_DOI = "10.5281/zenodo.18667018"
LIU4K_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/liu4k.zip/content"
LIU4K_ZIP_MD5 = "e8fafe5444a9e346f25123151ef1a699"  # from the Zenodo record metadata
LIU4K_ZIP_SIZE = 4_015_411_869
LIU4K_CACHE_DIR = Path("tests/data/liu4k")
LIU4K_SUBDIR = "liu4k_markers_1024"
# Dictionary the markers were generated from (per aruco_nano's testperf.cpp).
LIU4K_DICTIONARY = "ARUCO_MIP_36h12"

# `FunnelStatus` code for "passed the geometric funnel, rejected by the decoder"
# (see tools/bench/collect.py).
_PASSED_FUNNEL = 1

CITATION = (
    "Muñoz-Salinas, R. Liu4K dataset employed for Aruco_Nano paper. Zenodo "
    f"(2026). https://doi.org/{ZENODO_DOI}. CC-BY-4.0."
)


def _md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_liu4k(cache_dir: Path = LIU4K_CACHE_DIR) -> Path:
    """Download (if needed), md5-verify and extract Liu4K; return the data dir.

    Idempotent: returns immediately when the extracted directory holds images.
    The zip is streamed to ``liu4k.zip.part``, verified, extracted, then deleted.
    """
    data_dir = cache_dir / LIU4K_SUBDIR
    if data_dir.is_dir() and any(data_dir.glob("*.jpg")):
        return data_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "liu4k.zip"
    if not zip_path.exists() or _md5(zip_path) != LIU4K_ZIP_MD5:
        part = cache_dir / "liu4k.zip.part"
        h = hashlib.md5(usedforsecurity=False)
        print(f"Downloading Liu4K ({LIU4K_ZIP_SIZE / 1e9:.1f} GB) from Zenodo, CC-BY-4.0.")
        with urllib.request.urlopen(LIU4K_URL) as resp, open(part, "wb") as out:  # noqa: S310
            while chunk := resp.read(1 << 20):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != LIU4K_ZIP_MD5:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Liu4K md5 mismatch: got {h.hexdigest()}, expected {LIU4K_ZIP_MD5}")
        part.replace(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        root = cache_dir.resolve()
        for member in zf.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise RuntimeError(f"Unsafe path in liu4k.zip: {member}")
        zf.extractall(cache_dir)  # noqa: S202 - member paths validated above
    zip_path.unlink()
    return data_dir


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


def quad_recall_for_image(
    quads: np.ndarray,
    gt_tags: list[TagGroundTruth],
    threshold: float = MATCH_DISTANCE_THRESHOLD_PX,
) -> tuple[int, int]:
    """Id-agnostic greedy one-to-one centre matching; returns ``(matched, n_gt)``.

    Corner error is deliberately not reported: without a decoded id the corner
    order (rotation) is unknown, and corner error must stay order-preserving.
    """
    used: set[int] = set()
    matched = 0
    centers = quads.mean(axis=1) if len(quads) else np.empty((0, 2))
    for gt in gt_tags:
        gc = gt.corners.astype(np.float64).mean(axis=0)
        best, best_d = -1, threshold
        for i, c in enumerate(centers):
            if i in used:
                continue
            d = float(np.linalg.norm(c - gc))
            if d < best_d:
                best, best_d = i, d
        if best >= 0:
            used.add(best)
            matched += 1
    return matched, len(gt_tags)
