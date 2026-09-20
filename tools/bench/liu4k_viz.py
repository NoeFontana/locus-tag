"""Liu4K failure-mode visualisation: annotated PNGs + Rerun ``.rrd`` recordings.

Purpose: make the Liu4K recall losses inspectable *by eye*. For every selected
image and every detector configuration the tool emits the ground truth, the
accepted detections (TP/FP), the missed markers (FN) tagged with a diagnosed
**stage of loss**, the funnel-rejected candidate quads, and the intermediate
pipeline views (sharpened image, threshold map, CCL foreground mask).

Data: Liu4K (Muñoz-Salinas, Zenodo DOI 10.5281/zenodo.18667018, CC-BY-4.0) —
see :mod:`tools.bench.liu4k`. The renders are derivative images of CC-BY-4.0
data: they stay in the local output directory, are never committed, and the
output directory carries the attribution (``README.md``).

Loss classes follow the Liu4K root-cause analysis (2026-09-19). Two levels:

* **coarse** — from the threshold map + foreground mask alone (cheap, used for
  the full-dataset scan that drives image selection);
* **detailed** — adds a connected-component analysis around the marker
  (dominant-component bbox, Douglas-Peucker vertex count) so a miss can be
  split into "merged into a giant component" vs "quad-gate reject".

The detailed level reproduces the detector's quad gates *in Python* (OpenCV
contour + ``approxPolyDP``) because the Rust quad stage does not expose its
reject reason through the FFI telemetry. Labels derived that way are marked
``~`` in the rendered label text.

No detector code is modified or required: the foreground mask is recomputed as
``sharpened < telemetry.threshold_map``, which is exactly what
``simd_ccl_fusion::label_components_lsl`` consumes, and the sharpening
replication is validated per image against ``telemetry.threshold_map``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# --- Loss classes -----------------------------------------------------------

LOSS_DECODED_WRONG_ID = "decoded_wrong_id"
LOSS_DECODE_FAIL = "decode_fail"
LOSS_CONTRAST_REJECT = "contrast_reject"
LOSS_CANDIDATE_OFFSET = "candidate_offset"
LOSS_HOLLOW_BORDER = "hollow_border"
LOSS_LOW_CONTRAST = "low_contrast"
LOSS_MERGED_GIANT = "merged_giant_component"
LOSS_MERGED = "merged_component"
LOSS_QUAD_GATE = "quad_gate_reject"
LOSS_MERGED_OR_GATE = "merged_or_gate"  # coarse level: not separated yet

#: Human-readable one-liners, used in the contact sheets and ``index.html``.
LOSS_DESCRIPTIONS: dict[str, str] = {
    LOSS_DECODED_WRONG_ID: "a detection sits on the marker but carries a different id",
    LOSS_DECODE_FAIL: "quad reached the decoder and was rejected (hamming/bit errors)",
    LOSS_CONTRAST_REJECT: "quad rejected by the decoder funnel's contrast gate",
    LOSS_CANDIDATE_OFFSET: "a candidate quad exists but its centre is >10 px off",
    LOSS_HOLLOW_BORDER: "flat (low-range) tiles inside the black border: t=(min+max)/2 never fires, the border is hollow",
    LOSS_LOW_CONTRAST: "border is not below the local mid-extreme threshold (too little contrast)",
    LOSS_MERGED_GIANT: "border fused into a component whose bbox covers >90% of the image",
    LOSS_MERGED: "border fused with leaking background into one oversized component",
    LOSS_QUAD_GATE: "border is a clean own component but the quad gates rejected it",
    LOSS_MERGED_OR_GATE: "border is foreground and single-piece, lost after CCL (merged or quad-gate)",
}

#: Ring geometry (fractions of the way from the outer corner to the centre).
#: The ArUco-MIP 36h12 black border is the outer 1/8 of the marker side, so
#: 3/6/9 % stays inside it for every marker size in the dataset.
RING_INSETS: tuple[float, ...] = (0.03, 0.06, 0.09)
RING_SAMPLES_PER_EDGE = 100

#: Heuristic cut-offs, taken from the root-cause analysis (2026-09-19).
RING_FOREGROUND_MIN = 0.70  # below this the border did not binarise
LOW_RANGE_MIN_FRAC = 0.50  # at least half the ring in flat tiles => hollow
MERGED_BBOX_RATIO = 2.5  # dominant component bbox vs marker bbox
GIANT_BBOX_IMAGE_FRAC = 0.90  # component bbox vs image area
RDP_MAX_VERTICES = 11  # quad.rs rejects `simplified.len() > 11`
RDP_MIN_VERTICES = 4

#: Seed for the reproducible random fill of the image selection.
SELECTION_SEED = 20260920

# `FunnelStatus` codes (see crates/locus-py: 0 = none, 1 = passed funnel,
# 2 = rejected by sampling/decoding, 3 = rejected by the contrast gate).
FUNNEL_PASSED = 1
FUNNEL_REJECTED_SAMPLING = 2
FUNNEL_REJECTED_CONTRAST = 3


class UnsupportedConfigError(RuntimeError):
    """The running build does not expose a knob the requested config needs."""


# --- Detector configurations to compare -------------------------------------


@dataclass(frozen=True)
class ConfigSpec:
    """One detector configuration column of the comparison."""

    name: str
    label: str
    #: ``{group: {key: value}}`` overrides applied to the ``standard`` profile dump.
    overrides: Mapping[str, Mapping[str, Any]]
    #: Where the knob comes from, for the report.
    provenance: str
    #: Sharpening kernel used by this config, for the Python replication:
    #: ``"none"``, ``"standard"`` or ``"shoot_limited"``.
    sharpening: str = "standard"


CONFIGS: dict[str, ConfigSpec] = {
    "standard": ConfigSpec(
        name="standard",
        label="standard (shipped)",
        overrides={},
        provenance="shipped `standard` profile",
        sharpening="standard",
    ),
    "no_sharpen": ConfigSpec(
        name="no_sharpen",
        label="standard, sharpening off",
        overrides={"threshold": {"enable_sharpening": False}},
        provenance="shipped `standard` profile with threshold.enable_sharpening=false",
        sharpening="none",
    ),
    "local_mean": ConfigSpec(
        name="local_mean",
        label="LocalMean threshold (PR #383)",
        overrides={"threshold": {"mode": "LocalMean"}},
        provenance="branch feat/robust-foreground-threshold (PR #383), threshold.mode=LocalMean",
        sharpening="standard",
    ),
    "shoot_limited": ConfigSpec(
        name="shoot_limited",
        label="ShootLimited sharpening (PR #384)",
        overrides={"threshold": {"sharpening_mode": "ShootLimited"}},
        provenance="branch feat/shoot-limited-sharpening (PR #384), threshold.sharpening_mode=ShootLimited",
        sharpening="shoot_limited",
    ),
}

#: Display order of the comparison columns.
CONFIG_ORDER: tuple[str, ...] = ("standard", "no_sharpen", "local_mean", "shoot_limited")


def build_config_dump(base: dict[str, Any], spec: ConfigSpec) -> dict[str, Any]:
    """Apply ``spec``'s overrides to a ``DetectorConfig.model_dump()``.

    Raises :class:`UnsupportedConfigError` when a key is absent from the dump,
    which is how a build without the corresponding open PR announces itself.

    The copy is shallow per group on purpose: the dump holds PyO3 enum
    instances that must round-trip back into ``model_validate`` unchanged.
    """
    out: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for group, kvs in spec.overrides.items():
        if group not in out:
            raise UnsupportedConfigError(f"config '{spec.name}': no '{group}' group in this build")
        for key, value in kvs.items():
            if key not in out[group]:
                raise UnsupportedConfigError(
                    f"config '{spec.name}' needs `{group}.{key}`, absent from this build "
                    f"({spec.provenance})"
                )
            out[group][key] = value
    return out


# --- Loss classification ----------------------------------------------------


@dataclass(frozen=True)
class MissFeatures:
    """Measurements around one missed (FN) ground-truth marker.

    ``None`` means "not measured": :func:`classify_miss` then stays at the
    coarse level instead of guessing.
    """

    #: Fraction of the border-ring samples that are CCL foreground.
    ring_foreground: float
    #: Fraction of the border-ring samples sitting in flat (range < min_range) tiles.
    low_range_frac: float
    #: Nearest candidate quad: ``None``, ``"detection"`` (accepted, wrong id) or
    #: ``"rejected"`` (funnel-rejected).
    candidate_kind: str | None = None
    #: ``FunnelStatus`` of that candidate, when it is a rejected one.
    candidate_funnel_status: int | None = None
    #: Decoder error rate / bit errors reported for that candidate.
    candidate_error: float | None = None
    #: Centre distance (px) of the nearest candidate quad, any distance.
    nearest_candidate_px: float | None = None
    #: bbox area of the dominant foreground component over the marker bbox area.
    dominant_bbox_ratio: float | None = None
    #: bbox area of the dominant foreground component over the image area.
    dominant_image_frac: float | None = None
    #: Vertices of ``approxPolyDP`` on that component's outer contour.
    rdp_vertices: int | None = None


def classify_miss(f: MissFeatures, match_threshold_px: float = 10.0) -> tuple[str, str]:
    """Return ``(loss_class, label)`` for one missed ground-truth marker.

    Pure function of the measured features — the unit tests pin its decisions.
    The order mirrors the pipeline: decoder outcome first (a candidate quad
    reached it), then the threshold stage, then segmentation/quad gates.
    """
    near = f.nearest_candidate_px is not None and f.nearest_candidate_px <= match_threshold_px
    if near and f.candidate_kind == "detection":
        return LOSS_DECODED_WRONG_ID, "decoded, wrong id"
    if near and f.candidate_kind == "rejected":
        if f.candidate_funnel_status == FUNNEL_REJECTED_CONTRAST:
            return LOSS_CONTRAST_REJECT, "funnel: contrast reject"
        bits = "" if f.candidate_error is None else f" ({f.candidate_error:.0f} bits)"
        return LOSS_DECODE_FAIL, f"decode fail{bits}"
    if (
        f.nearest_candidate_px is not None
        and f.nearest_candidate_px <= 3 * match_threshold_px
        and not near
    ):
        return LOSS_CANDIDATE_OFFSET, f"candidate {f.nearest_candidate_px:.0f} px off"

    if f.ring_foreground < RING_FOREGROUND_MIN:
        if f.low_range_frac >= LOW_RANGE_MIN_FRAC:
            return (
                LOSS_HOLLOW_BORDER,
                f"hollow border (ring fg {f.ring_foreground:.0%}, flat tiles {f.low_range_frac:.0%})",
            )
        return LOSS_LOW_CONTRAST, f"low contrast (ring fg {f.ring_foreground:.0%})"

    if f.dominant_bbox_ratio is None:
        return LOSS_MERGED_OR_GATE, f"merged/quad-gate (ring fg {f.ring_foreground:.0%})"
    if f.dominant_image_frac is not None and f.dominant_image_frac >= GIANT_BBOX_IMAGE_FRAC:
        return (
            LOSS_MERGED_GIANT,
            f"~merged into giant component ({f.dominant_image_frac:.0%} of image)",
        )
    if f.dominant_bbox_ratio >= MERGED_BBOX_RATIO:
        return LOSS_MERGED, f"~merged component (bbox {f.dominant_bbox_ratio:.1f}x marker)"
    if f.rdp_vertices is not None and not (RDP_MIN_VERTICES <= f.rdp_vertices <= RDP_MAX_VERTICES):
        return LOSS_QUAD_GATE, f"~quad gate: RDP {f.rdp_vertices} vertices"
    return LOSS_QUAD_GATE, f"~quad gate (bbox {f.dominant_bbox_ratio:.1f}x marker)"


# --- Image selection --------------------------------------------------------


@dataclass(frozen=True)
class ImageScan:
    """Per-image outcome of one configuration over the whole dataset."""

    image: str
    n_gt: int
    tp: int
    fp: int
    fn: int
    #: Coarse loss class -> count, only filled by a ``--classify`` scan.
    classes: Mapping[str, int] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0


@dataclass(frozen=True)
class Selection:
    """One selected image plus every rule that selected it."""

    image: str
    reasons: tuple[str, ...]


def select_images(
    scans: Mapping[str, Mapping[str, ImageScan]],
    *,
    baseline: str = "standard",
    target: int = 30,
    min_gt_for_recall: int = 5,
    pinned: Sequence[tuple[str, str]] = (
        ("004.jpg", "RCA case: large clean marker id 229 lost"),
        ("001.jpg", "RCA case: dark marker id 155 lost to sharpening halo"),
    ),
    seed: int = SELECTION_SEED,
) -> list[Selection]:
    """Choose ~``target`` images from the full-dataset scans, deterministically.

    Rules, in order: the pinned root-cause cases, best/worst per-image recall,
    one exemplar per loss class, the images each open-PR config fixes and
    regresses the most, the images with the most false positives, then a
    seeded random fill. Every rule records *why* the image was taken, and the
    result is stable for a given set of scans (ties break on the file name).
    """
    base = scans[baseline]
    picked: dict[str, list[str]] = {}

    def take(image: str, reason: str) -> None:
        if image not in base:
            return
        picked.setdefault(image, [])
        if reason not in picked[image]:
            picked[image].append(reason)

    for image, reason in pinned:
        take(image, reason)

    eligible = [s for s in base.values() if s.n_gt >= min_gt_for_recall]
    by_recall = sorted(eligible, key=lambda s: (-s.recall, -s.n_gt, s.image))
    for s in by_recall[:2]:
        take(s.image, f"best per-image recall ({baseline}: {s.tp}/{s.n_gt})")
    for s in sorted(eligible, key=lambda s: (s.recall, -s.n_gt, s.image))[:2]:
        take(s.image, f"worst per-image recall ({baseline}: {s.tp}/{s.n_gt})")

    classes = sorted({c for s in base.values() for c in s.classes})
    for cls in classes:
        best = sorted(base.values(), key=lambda s: (-s.classes.get(cls, 0), s.image))[0]
        if best.classes.get(cls, 0) > 0:
            take(best.image, f"most '{cls}' misses ({best.classes[cls]})")

    for name in CONFIG_ORDER:
        if name == baseline or name not in scans:
            continue
        other = scans[name]
        deltas = [
            (other[img].tp - s.tp, img) for img, s in base.items() if img in other and s.n_gt > 0
        ]
        for delta, img in sorted(deltas, key=lambda d: (-d[0], d[1]))[:2]:
            if delta > 0:
                take(img, f"{name} fixes {delta} marker(s) vs {baseline}")
        for delta, img in sorted(deltas, key=lambda d: (d[0], d[1]))[:2]:
            if delta < 0:
                take(img, f"{name} regresses {-delta} marker(s) vs {baseline}")

    for s in sorted(base.values(), key=lambda s: (-s.fp, s.image))[:2]:
        if s.fp > 0:
            take(s.image, f"most false positives ({baseline}: {s.fp})")

    remaining = sorted(img for img in base if img not in picked)
    rng = random.Random(seed)
    n_fill = max(0, target - len(picked))
    for img in sorted(rng.sample(remaining, min(n_fill, len(remaining)))):
        take(img, f"random sample (seed {seed})")

    return [Selection(image=img, reasons=tuple(picked[img])) for img in sorted(picked)]


# --- Pixel-level reproduction of the detector's pre-processing --------------


def laplacian_sharpen(img: NDArray[np.uint8], *, shoot_limited: bool = False) -> NDArray[np.uint8]:
    """Replicate ``filter.rs::laplacian_sharpen`` (and PR #384's limiter).

    ``5·centre − (up + down + left + right)``, edge-clamped, saturating to
    ``0..=255``. With ``shoot_limited`` the result is additionally clamped into
    the min/max of the five samples, which is what PR #384 ships.
    """
    src = img.astype(np.int32)
    up = np.vstack([src[:1], src[:-1]])
    down = np.vstack([src[1:], src[-1:]])
    left = np.hstack([src[:, :1], src[:, :-1]])
    right = np.hstack([src[:, 1:], src[:, -1:]])
    out = 5 * src - (up + down + left + right)
    if shoot_limited:
        stack = np.stack([src, up, down, left, right])
        out = np.clip(out, stack.min(axis=0), stack.max(axis=0))
    return np.clip(out, 0, 255).astype(np.uint8)


def tile_neighbourhood_stats(
    img: NDArray[np.uint8], tile_size: int
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Per-pixel min/max over the 3x3 *tile* neighbourhood (``threshold.rs``).

    Mirrors ``ThresholdEngine::{compute_tile_stats,apply_threshold_with_map}``:
    tiles are ``tile_size²``, the image remainder (``w % tile_size`` columns and
    ``h % tile_size`` rows) is not covered by any tile and is returned as
    ``min=255, max=0`` so callers see it as "never foreground", exactly like the
    zero-filled tail of the Rust threshold map.
    """
    h, w = img.shape
    th, tw = h // tile_size, w // tile_size
    tiles = img[: th * tile_size, : tw * tile_size].reshape(th, tile_size, tw, tile_size)
    tmin = tiles.min(axis=(1, 3))
    tmax = tiles.max(axis=(1, 3))
    pmin = np.pad(tmin, 1, mode="edge")
    pmax = np.pad(tmax, 1, mode="edge")
    nmin = np.full_like(tmin, 255)
    nmax = np.zeros_like(tmax)
    for dy in range(3):
        for dx in range(3):
            nmin = np.minimum(nmin, pmin[dy : dy + th, dx : dx + tw])
            nmax = np.maximum(nmax, pmax[dy : dy + th, dx : dx + tw])
    full_min = np.full((h, w), 255, dtype=np.uint8)
    full_max = np.zeros((h, w), dtype=np.uint8)
    full_min[: th * tile_size, : tw * tile_size] = np.repeat(
        np.repeat(nmin, tile_size, axis=0), tile_size, axis=1
    )
    full_max[: th * tile_size, : tw * tile_size] = np.repeat(
        np.repeat(nmax, tile_size, axis=0), tile_size, axis=1
    )
    return full_min, full_max


def mid_extreme_threshold(nmin: NDArray[np.uint8], nmax: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """``(min + max) >> 1`` of the 3x3-tile neighbourhood, as the detector does."""
    mid = (nmin.astype(np.uint16) + nmax.astype(np.uint16)) >> 1
    mid[nmax < nmin] = 0  # uncovered remainder rows/columns
    return mid.astype(np.uint8)


@dataclass(frozen=True)
class Preprocessing:
    """The detector's pre-segmentation state, reproduced for one frame."""

    work: NDArray[np.uint8]  # image the threshold/CCL stages actually see
    threshold_map: NDArray[np.uint8]  # from telemetry (authoritative)
    foreground: NDArray[np.bool_]  # work < threshold_map, i.e. the CCL input
    tile_range: NDArray[np.uint8]  # 3x3-tile neighbourhood range of `work`
    #: Fraction of pixels where the locally recomputed mid-extreme threshold
    #: equals ``telemetry.threshold_map`` — the self-check of this replication.
    #: ``None`` when the config does not use the mid-extreme rule.
    threshold_agreement: float | None


def reproduce_preprocessing(
    img: NDArray[np.uint8],
    threshold_map: NDArray[np.uint8],
    spec: ConfigSpec,
    *,
    tile_size: int = 8,
) -> Preprocessing:
    """Rebuild the sharpened image, the CCL foreground and the tile-range map.

    ``threshold_map`` comes from the detector telemetry, so the foreground mask
    is exact as long as the sharpening replication is: the returned
    ``threshold_agreement`` is that check (mid-extreme configs only).
    """
    if spec.sharpening == "none":
        work = img
    else:
        work = laplacian_sharpen(img, shoot_limited=spec.sharpening == "shoot_limited")
    nmin, nmax = tile_neighbourhood_stats(work, tile_size)
    tile_range = np.where(nmax >= nmin, nmax.astype(np.int16) - nmin.astype(np.int16), 0).astype(
        np.uint8
    )
    agreement: float | None = None
    if spec.overrides.get("threshold", {}).get("mode") != "LocalMean":
        recomputed = mid_extreme_threshold(nmin, nmax)
        agreement = float(np.mean(recomputed == threshold_map))
    return Preprocessing(
        work=work,
        threshold_map=threshold_map,
        foreground=work < threshold_map,
        tile_range=tile_range,
        threshold_agreement=agreement,
    )


# --- Geometry helpers -------------------------------------------------------


def ring_points(
    corners: NDArray[np.float64], insets: Sequence[float] = RING_INSETS
) -> NDArray[np.float64]:
    """Sample points along the marker's black border, inset towards the centre."""
    quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    centre = quad.mean(axis=0)
    t = np.linspace(0.0, 1.0, RING_SAMPLES_PER_EDGE, endpoint=False)[:, None]
    edges = [quad[i] * (1.0 - t) + quad[(i + 1) % 4] * t for i in range(4)]
    border = np.concatenate(edges, axis=0)
    return np.concatenate([border + (centre - border) * f for f in insets], axis=0)


def sample_mask(mask: NDArray[Any], pts: NDArray[np.float64]) -> NDArray[Any]:
    """Nearest-neighbour sample of ``mask`` at ``pts`` (out-of-frame -> clamped)."""
    h, w = mask.shape[:2]
    xs = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, h - 1)
    return mask[ys, xs]


def quad_bbox(corners: NDArray[np.float64]) -> tuple[int, int, int, int]:
    """Integer ``(x0, y0, x1, y1)`` bounding box of a quad."""
    quad = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    return (
        int(np.floor(quad[:, 0].min())),
        int(np.floor(quad[:, 1].min())),
        int(np.ceil(quad[:, 0].max())),
        int(np.ceil(quad[:, 1].max())),
    )


def quad_centres(quads: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(N, 2)`` centres of ``(N, 4, 2)`` quads (empty-safe)."""
    arr = np.asarray(quads, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return arr.reshape(-1, 4, 2).mean(axis=1)


def nearest_index(centres: NDArray[np.float64], point: NDArray[np.float64]) -> tuple[int, float]:
    """Index of and distance to the centre closest to ``point`` (``(-1, inf)`` if none)."""
    if centres.size == 0:
        return -1, float("inf")
    d = np.linalg.norm(centres - np.asarray(point, dtype=np.float64)[None, :], axis=1)
    k = int(np.argmin(d))
    return k, float(d[k])


# --- Per-marker diagnosis ---------------------------------------------------


@dataclass(frozen=True)
class MarkerDiagnosis:
    """Everything the renderer needs about one ground-truth marker."""

    tag_id: int
    corners: NDArray[np.float64]
    matched: bool  # id-aware TP
    loss_class: str | None
    label: str
    features: MissFeatures | None


def diagnose_markers(
    gt_ids: Sequence[int],
    gt_corners: Sequence[NDArray[np.float64]],
    det_ids: Sequence[int],
    det_corners: NDArray[np.float64],
    rejected_corners: NDArray[np.float64] | None,
    rejected_status: NDArray[np.int64] | None,
    rejected_error: NDArray[np.float64] | None,
    pre: Preprocessing,
    *,
    match_threshold_px: float = 10.0,
    min_range: int = 10,
    detailed: bool = False,
) -> list[MarkerDiagnosis]:
    """Diagnose every GT marker: TP, or FN with a loss class.

    ``detailed`` runs the connected-component/RDP reproduction (OpenCV) and is
    what separates "merged" from "quad-gate reject"; without it the classes
    stay at the coarse level.
    """
    det_centres = quad_centres(np.asarray(det_corners, dtype=np.float64))
    rej = np.asarray(rejected_corners, dtype=np.float64) if rejected_corners is not None else None
    rej_centres = quad_centres(rej) if rej is not None else np.zeros((0, 2))
    components = label_foreground(pre.foreground) if detailed else None

    used: set[int] = set()
    out: list[MarkerDiagnosis] = []
    for i, (tag_id, corners) in enumerate(zip(gt_ids, gt_corners, strict=True)):
        quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        centre = quad.mean(axis=0)
        matched = False
        for k, det_id in enumerate(det_ids):
            if k in used or int(det_id) != int(tag_id):
                continue
            if float(np.linalg.norm(det_centres[k] - centre)) <= match_threshold_px:
                used.add(k)
                matched = True
                break
        if matched:
            out.append(MarkerDiagnosis(int(tag_id), quad, True, None, f"TP id {int(tag_id)}", None))
            continue

        det_k, det_d = nearest_index(det_centres, centre)
        rej_k, rej_d = nearest_index(rej_centres, centre)
        if det_d <= rej_d:
            kind, dist, status, err = "detection", det_d, None, None
            if det_k >= 0 and det_d < float("inf"):
                err = float(det_ids[det_k])
        else:
            kind, dist = "rejected", rej_d
            status = int(rejected_status[rej_k]) if rejected_status is not None else None
            err = float(rejected_error[rej_k]) if rejected_error is not None else None

        pts = ring_points(quad)
        ring_fg = float(np.mean(sample_mask(pre.foreground, pts)))
        low_range = float(np.mean(sample_mask(pre.tile_range, pts) < min_range))
        extra: dict[str, float | int | None] = {}
        if components is not None:
            extra = _component_features(components[0], components[1], quad, pts)
        features = MissFeatures(
            ring_foreground=ring_fg,
            low_range_frac=low_range,
            candidate_kind=kind if np.isfinite(dist) else None,
            candidate_funnel_status=status,
            candidate_error=err if kind == "rejected" else None,
            nearest_candidate_px=dist if np.isfinite(dist) else None,
            dominant_bbox_ratio=extra.get("bbox_ratio"),
            dominant_image_frac=extra.get("image_frac"),
            rdp_vertices=extra.get("rdp_vertices"),  # pyright: ignore[reportArgumentType]
        )
        loss_class, label = classify_miss(features, match_threshold_px)
        out.append(
            MarkerDiagnosis(
                int(tag_id), quad, False, loss_class, f"{int(tag_id)}: {label}", features
            )
        )
        _ = i
    return out


def label_foreground(foreground: NDArray[np.bool_]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """8-connected labelling of the whole foreground, as the detector's CCL sees it."""
    import cv2

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(foreground.astype(np.uint8)), connectivity=8
    )
    return labels.astype(np.int32), stats.astype(np.int32)


def _component_features(
    labels: NDArray[np.int32],
    stats: NDArray[np.int32],
    quad: NDArray[np.float64],
    ring: NDArray[np.float64],
) -> dict[str, float | int | None]:
    """Connected-component + Douglas-Peucker reproduction for one marker.

    The detector's quad stage does not export its reject reason, so this
    recomputes the geometry it would have seen: the *whole-image* foreground
    component that carries most of the marker's border ring, its bbox relative
    to the marker and to the image, and the vertex count ``approxPolyDP``
    yields on its outer contour with the very epsilon ``quad.rs`` uses
    (``max(0.02 · perimeter, 1)``; the gate rejects ``> 11`` vertices).
    """
    import cv2

    none: dict[str, float | int | None] = {
        "bbox_ratio": None,
        "image_frac": None,
        "rdp_vertices": None,
    }
    h, w = labels.shape
    lab = sample_mask(labels, ring)
    lab = lab[lab > 0]
    if lab.size == 0:
        return none
    dominant = int(np.bincount(lab).argmax())
    bx, by = int(stats[dominant, cv2.CC_STAT_LEFT]), int(stats[dominant, cv2.CC_STAT_TOP])
    bw, bh = int(stats[dominant, cv2.CC_STAT_WIDTH]), int(stats[dominant, cv2.CC_STAT_HEIGHT])
    x0, y0, x1, y1 = quad_bbox(quad)
    marker_area = max(1.0, float((x1 - x0) * (y1 - y0)))
    image_frac = float(bw * bh) / float(w * h)
    rdp: int | None = None
    if image_frac < GIANT_BBOX_IMAGE_FRAC:  # a giant component needs no vertex count
        mask = np.ascontiguousarray(
            (labels[by : by + bh, bx : bx + bw] == dominant).astype(np.uint8)
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            rdp = int(len(cv2.approxPolyDP(c, max(0.02 * cv2.arcLength(c, True), 1.0), True)))
    return {
        "bbox_ratio": float(bw * bh) / marker_area,
        "image_frac": image_frac,
        "rdp_vertices": rdp,
    }


# --- Counting ---------------------------------------------------------------


def class_histogram(diagnoses: Iterable[MarkerDiagnosis]) -> dict[str, int]:
    """Loss class -> count over the missed markers of one image."""
    hist: dict[str, int] = {}
    for d in diagnoses:
        if d.loss_class is not None:
            hist[d.loss_class] = hist.get(d.loss_class, 0) + 1
    return dict(sorted(hist.items()))


def scan_to_json(scan: ImageScan) -> str:
    """One JSONL row of a dataset scan."""
    return json.dumps(
        {
            "image": scan.image,
            "n_gt": scan.n_gt,
            "tp": scan.tp,
            "fp": scan.fp,
            "fn": scan.fn,
            "classes": dict(scan.classes),
        }
    )


def load_scan(path: Path) -> dict[str, ImageScan]:
    """Read a ``scan_<config>.jsonl`` written by ``scan_to_json``."""
    out: dict[str, ImageScan] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["image"]] = ImageScan(
                image=row["image"],
                n_gt=int(row["n_gt"]),
                tp=int(row["tp"]),
                fp=int(row["fp"]),
                fn=int(row["fn"]),
                classes={str(k): int(v) for k, v in row.get("classes", {}).items()},
            )
    return out


# --- Rendering: colours and drawing ----------------------------------------

#: BGR colours (OpenCV) shared by the PNGs; the Rerun layers use the RGB flip.
COLOR_GT = (255, 200, 0)  # cyan-blue: ground truth
COLOR_TP = (0, 220, 0)  # green: accepted detection matching GT
COLOR_FP = (0, 0, 255)  # red: accepted detection with no GT
COLOR_FN = (255, 0, 255)  # magenta: missed ground truth
COLOR_REJ = (0, 165, 255)  # orange: funnel-rejected candidate quad

PNG_LONG_SIDE = 1600
CROP_LONG_SIDE = 512


def _u8(array: Any) -> NDArray[np.uint8]:
    """Narrow an OpenCV return value (``MatLike``) to a typed uint8 array."""
    return np.asarray(array, dtype=np.uint8)


def _rgb(bgr: tuple[int, int, int]) -> list[int]:
    return [bgr[2], bgr[1], bgr[0]]


def _draw_quad(
    canvas: NDArray[np.uint8],
    quad: NDArray[np.float64],
    color: tuple[int, int, int],
    *,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    thickness: int = 2,
    label: str | None = None,
) -> None:
    import cv2

    pts = (np.asarray(quad, dtype=np.float64).reshape(-1, 2) - np.asarray(offset)) * scale
    cv2.polylines(canvas, [np.rint(pts).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
    if label:
        anchor = pts.min(axis=0)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Keep the label inside the canvas: long loss-class labels on markers
        # near the right/top edge would otherwise be cut off.
        x = int(min(max(0, anchor[0]), max(0, canvas.shape[1] - tw - 2)))
        y = int(min(max(th + 2, anchor[1] - 6), canvas.shape[0] - 2))
        org = (x, y)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _legend(canvas: NDArray[np.uint8], lines: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
    import cv2

    y = 24
    for text, color in lines:
        cv2.putText(
            canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 24


@dataclass
class FrameResult:
    """Everything one (image, config) render produced."""

    image: str
    config: str
    n_gt: int
    tp: int
    fp: int
    fn: int
    classes: dict[str, int]
    threshold_agreement: float | None
    misses: list[dict[str, Any]]
    fp_centres: list[list[float]]


def annotate_overview(
    img: NDArray[np.uint8],
    diagnoses: Sequence[MarkerDiagnosis],
    det_corners: NDArray[np.float64],
    det_ids: Sequence[int],
    fp_mask: Sequence[bool],
    rejected: NDArray[np.float64] | None,
    *,
    title: str,
    long_side: int = PNG_LONG_SIDE,
) -> NDArray[np.uint8]:
    """Downscaled BGR overview: GT, TP, FP, FN (with loss class) and rejects."""
    import cv2

    h, w = img.shape
    scale = min(1.0, long_side / max(h, w))
    canvas = _u8(
        cv2.cvtColor(
            cv2.resize(
                img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            ),
            cv2.COLOR_GRAY2BGR,
        )
    )
    if rejected is not None and len(rejected) > 0:
        for quad in np.asarray(rejected, dtype=np.float64).reshape(-1, 4, 2):
            _draw_quad(canvas, quad, COLOR_REJ, scale=scale, thickness=1)
    for d in diagnoses:
        _draw_quad(canvas, d.corners, COLOR_GT, scale=scale, thickness=1)
    for k, quad in enumerate(np.asarray(det_corners, dtype=np.float64).reshape(-1, 4, 2)):
        color = COLOR_FP if fp_mask[k] else COLOR_TP
        _draw_quad(canvas, quad, color, scale=scale, thickness=2, label=f"{int(det_ids[k])}")
    for d in diagnoses:
        if not d.matched:
            _draw_quad(canvas, d.corners, COLOR_FN, scale=scale, thickness=2, label=d.label)
    _legend(
        canvas,
        [
            (title, (255, 255, 255)),
            ("GT", COLOR_GT),
            ("TP (id)", COLOR_TP),
            ("FP", COLOR_FP),
            ("FN + loss class", COLOR_FN),
            ("funnel-rejected quad", COLOR_REJ),
        ],
    )
    return canvas


def crop_around(
    img: NDArray[np.uint8],
    quad: NDArray[np.float64],
    *,
    pad_factor: float = 0.8,
    long_side: int = CROP_LONG_SIDE,
) -> tuple[NDArray[np.uint8], float, tuple[float, float]]:
    """Zoomed BGR crop around one marker; returns ``(canvas, scale, offset)``."""
    import cv2

    h, w = img.shape
    x0, y0, x1, y1 = quad_bbox(quad)
    pad = int(pad_factor * max(x1 - x0, y1 - y0)) + 16
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
    sub = img[cy0:cy1, cx0:cx1]
    scale = min(4.0, long_side / max(1, max(sub.shape)))
    resized = cv2.resize(
        sub,
        (max(1, int(sub.shape[1] * scale)), max(1, int(sub.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA,
    )
    return _u8(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)), scale, (float(cx0), float(cy0))


def intermediates_strip(
    img: NDArray[np.uint8], pre: Preprocessing, *, long_side: int = 700
) -> NDArray[np.uint8]:
    """``raw | sharpened | threshold map | CCL foreground`` side by side (BGR)."""
    import cv2

    panels = [
        ("raw", img),
        ("work (sharpened)", pre.work),
        ("threshold map", pre.threshold_map),
        ("CCL foreground", (pre.foreground * 255).astype(np.uint8)),
    ]
    h, w = img.shape
    scale = long_side / max(h, w)
    out: list[NDArray[np.uint8]] = []
    for name, panel in panels:
        small = cv2.resize(
            panel, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
        bgr = _u8(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR))
        cv2.putText(bgr, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            bgr, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        out.append(bgr)
    return _u8(np.hstack(out))


def grid(panels: Sequence[NDArray[np.uint8]], cols: int) -> NDArray[np.uint8]:
    """Tile equally-sized-ish BGR panels into a ``cols``-wide grid."""
    import cv2

    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    ph = max(p.shape[0] for p in panels)
    pw = max(p.shape[1] for p in panels)
    padded = [
        _u8(
            cv2.copyMakeBorder(
                p, 0, ph - p.shape[0], 0, pw - p.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20)
            )
        )
        for p in panels
    ]
    rows: list[NDArray[np.uint8]] = []
    for i in range(0, len(padded), cols):
        row = padded[i : i + cols]
        while len(row) < cols:
            row.append(np.full((ph, pw, 3), 20, dtype=np.uint8))
        rows.append(_u8(np.hstack(row)))
    return _u8(np.vstack(rows))


# --- Driving the detector ---------------------------------------------------

RERUN_APP_ID = "locus_liu4k_viz"


def make_detector(spec: ConfigSpec, family: Any) -> tuple[Any, dict[str, Any]]:
    """Build the detector for ``spec`` from the shipped ``standard`` profile."""
    import locus

    base: dict[str, Any] = locus.DetectorConfig.from_profile("standard").model_dump()
    dump = build_config_dump(base, spec)
    detector = locus.Detector(config=locus.DetectorConfig.model_validate(dump), families=[family])
    return detector, dump


def detect_frame(
    detector: Any,
    img: NDArray[np.uint8],
    gt_ids: Sequence[int],
    gt_corners: Sequence[NDArray[np.float64]],
    spec: ConfigSpec,
    *,
    tile_size: int,
    min_range: int,
    match_threshold_px: float,
    detailed: bool,
) -> tuple[FrameResult, list[MarkerDiagnosis], Preprocessing, dict[str, Any]]:
    """Run one frame end to end and diagnose every ground-truth marker."""
    from tools.bench.liu4k import score_detections
    from tools.bench.utils import TagGroundTruth

    batch = detector.detect(img, debug_telemetry=True)
    det_ids = [int(i) for i in batch.ids]
    det_corners = np.asarray(batch.corners, dtype=np.float64).reshape(-1, 4, 2)
    rejected = (
        np.asarray(batch.rejected_corners, dtype=np.float64).reshape(-1, 4, 2)
        if batch.rejected_corners is not None and len(batch.rejected_corners)
        else None
    )
    rej_status = (
        np.asarray(batch.rejected_funnel_status, dtype=np.int64)
        if batch.rejected_funnel_status is not None and rejected is not None
        else None
    )
    rej_error = (
        np.asarray(batch.rejected_error_rates, dtype=np.float64)
        if batch.rejected_error_rates is not None and rejected is not None
        else None
    )
    telemetry = batch.telemetry
    if telemetry is None:
        raise RuntimeError("debug_telemetry=True produced no telemetry")
    pre = reproduce_preprocessing(
        img, np.asarray(telemetry.threshold_map), spec, tile_size=tile_size
    )
    diagnoses = diagnose_markers(
        gt_ids,
        gt_corners,
        det_ids,
        det_corners,
        rejected,
        rej_status,
        rej_error,
        pre,
        match_threshold_px=match_threshold_px,
        min_range=min_range,
        detailed=detailed,
    )
    # aruco_nano scoring (`tools.bench.liu4k.score_detections`) supplies the
    # counts, so this tool's TP/FP/FN agree with `bench real --dataset liu4k`.
    gts = [
        TagGroundTruth(tag_id=int(i), corners=np.asarray(c, dtype=np.float32))
        for i, c in zip(gt_ids, gt_corners, strict=True)
    ]
    tp, fp, fn = score_detections(det_ids, det_corners, gts, threshold=match_threshold_px)
    fp_mask = _false_positive_mask(det_ids, det_corners, gt_ids, gt_corners, match_threshold_px)
    result = FrameResult(
        image="",
        config=spec.name,
        n_gt=len(gt_ids),
        tp=tp,
        fp=fp,
        fn=fn,
        classes=class_histogram(diagnoses),
        threshold_agreement=pre.threshold_agreement,
        misses=[
            {
                "tag_id": d.tag_id,
                "loss_class": d.loss_class,
                "label": d.label,
                "centre": [float(v) for v in d.corners.mean(axis=0)],
                "edge_px": float(np.linalg.norm(d.corners[0] - d.corners[1])),
                "features": {} if d.features is None else _features_json(d.features),
            }
            for d in diagnoses
            if not d.matched
        ],
        fp_centres=[
            [float(c[0]), float(c[1])]
            for c, is_fp in zip(quad_centres(det_corners), fp_mask, strict=True)
            if is_fp
        ],
    )
    extra: dict[str, Any] = {
        "det_ids": det_ids,
        "det_corners": det_corners,
        "fp_mask": fp_mask,
        "rejected": rejected,
        "rejected_status": rej_status,
    }
    return result, diagnoses, pre, extra


def _features_json(f: MissFeatures) -> dict[str, Any]:
    return {
        "ring_foreground": round(f.ring_foreground, 4),
        "low_range_frac": round(f.low_range_frac, 4),
        "nearest_candidate_px": None
        if f.nearest_candidate_px is None
        else round(f.nearest_candidate_px, 2),
        "candidate_kind": f.candidate_kind,
        "dominant_bbox_ratio": None
        if f.dominant_bbox_ratio is None
        else round(f.dominant_bbox_ratio, 3),
        "dominant_image_frac": None
        if f.dominant_image_frac is None
        else round(f.dominant_image_frac, 4),
        "rdp_vertices": f.rdp_vertices,
    }


def _false_positive_mask(
    det_ids: Sequence[int],
    det_corners: NDArray[np.float64],
    gt_ids: Sequence[int],
    gt_corners: Sequence[NDArray[np.float64]],
    threshold: float,
) -> list[bool]:
    """``True`` for detections the aruco_nano scorer counts as false positives."""
    gt_centres = [np.asarray(c, dtype=np.float64).reshape(4, 2).mean(axis=0) for c in gt_corners]
    matched = [False] * len(gt_ids)
    out: list[bool] = []
    for k, centre in enumerate(quad_centres(det_corners)):
        hit = False
        for j, gid in enumerate(gt_ids):
            if matched[j] or int(det_ids[k]) != int(gid):
                continue
            if float(np.linalg.norm(centre - gt_centres[j])) <= threshold:
                matched[j] = True
                hit = True
                break
        out.append(not hit)
    return out


# --- Rerun logging ----------------------------------------------------------


def log_frame_to_rerun(
    spec: ConfigSpec,
    img: NDArray[np.uint8],
    pre: Preprocessing,
    diagnoses: Sequence[MarkerDiagnosis],
    extra: Mapping[str, Any],
    result: FrameResult,
    *,
    intermediates: bool,
) -> None:
    """Log one (image, config) as ``<config>/…`` entities of the open recording."""
    import rerun as rr

    root = spec.name
    rr.log(f"{root}/0_input", _rr_image(img))
    if intermediates:
        rr.log(f"{root}/1_threshold_map", _rr_image(pre.threshold_map))
        rr.log(f"{root}/2_foreground", rr.Image((pre.foreground * 255).astype(np.uint8)))
        if spec.sharpening != "none":
            rr.log(f"{root}/3_work_sharpened", _rr_image(pre.work))

    gt_strips = [np.vstack([d.corners, d.corners[:1]]) for d in diagnoses]
    if gt_strips:
        rr.log(
            f"{root}/0_input/ground_truth",
            rr.LineStrips2D(
                gt_strips,
                colors=[[*_rgb(COLOR_GT), 140]] * len(gt_strips),
                radii=1.0,
                labels=[f"GT:{d.tag_id}" for d in diagnoses],
            ),
        )
    det_corners = np.asarray(extra["det_corners"], dtype=np.float64).reshape(-1, 4, 2)
    fp_mask = list(extra["fp_mask"])
    det_ids = list(extra["det_ids"])
    for name, want_fp, color in (
        ("detections", False, COLOR_TP),
        ("false_positives", True, COLOR_FP),
    ):
        idx = [k for k in range(len(det_corners)) if fp_mask[k] == want_fp]
        path = f"{root}/0_input/{name}"
        if not idx:
            rr.log(path, rr.Clear(recursive=False))
            continue
        rr.log(
            path,
            rr.LineStrips2D(
                [np.vstack([det_corners[k], det_corners[k][:1]]) for k in idx],
                colors=[_rgb(color)] * len(idx),
                radii=1.5,
                labels=[f"ID:{int(det_ids[k])}" for k in idx],
            ),
        )
    missed = [d for d in diagnoses if not d.matched]
    path = f"{root}/0_input/missed_gt"
    if missed:
        rr.log(
            path,
            rr.LineStrips2D(
                [np.vstack([d.corners, d.corners[:1]]) for d in missed],
                colors=[_rgb(COLOR_FN)] * len(missed),
                radii=2.0,
                labels=[d.label for d in missed],
            ),
        )
    else:
        rr.log(path, rr.Clear(recursive=False))
    rejected = extra.get("rejected")
    path = f"{root}/0_input/rejected_candidates"
    if rejected is not None and len(rejected) > 0:
        arr = np.asarray(rejected, dtype=np.float64).reshape(-1, 4, 2)
        status = extra.get("rejected_status")
        labels = (
            [_funnel_label(int(s)) for s in status]
            if status is not None
            else ["rejected"] * len(arr)
        )
        rr.log(
            path,
            rr.LineStrips2D(
                [np.vstack([q, q[:1]]) for q in arr],
                colors=[[*_rgb(COLOR_REJ), 160]] * len(arr),
                radii=0.8,
                labels=labels,
            ),
        )
    else:
        rr.log(path, rr.Clear(recursive=False))
    agreement = "n/a" if result.threshold_agreement is None else f"{result.threshold_agreement:.4%}"
    rr.log(
        f"{root}/summary",
        rr.TextDocument(
            f"## {spec.label}\n\n"
            f"- provenance: {spec.provenance}\n"
            f"- TP {result.tp} / FP {result.fp} / FN {result.fn} of {result.n_gt} GT\n"
            f"- loss classes: {json.dumps(result.classes)}\n"
            f"- foreground replication agreement vs telemetry threshold map: {agreement}\n",
            media_type="text/markdown",
        ),
    )


def _rr_image(img: NDArray[np.uint8]) -> Any:
    """Grayscale image, JPEG-compressed when the SDK supports it (rrd size)."""
    import rerun as rr

    image = rr.Image(img)
    try:
        return image.compress(jpeg_quality=92)
    except Exception:  # pragma: no cover - depends on the SDK build
        return image


def _funnel_label(status: int) -> str:
    return {
        FUNNEL_PASSED: "passed funnel, decoder rejected",
        FUNNEL_REJECTED_SAMPLING: "funnel: sampling/decode reject",
        FUNNEL_REJECTED_CONTRAST: "funnel: contrast reject",
    }.get(status, "rejected quad")


# --- Output-directory layout ------------------------------------------------

README_TEMPLATE = """# Liu4K detection visualisation (local, not for redistribution)

Generated by `uv run tools/cli.py bench liu4k-viz …` (`tools/bench/liu4k_viz.py`).

## Attribution (required)

The source images and ground truth are the **Liu4K dataset**:

> Muñoz-Salinas, R. *Liu4K dataset employed for Aruco_Nano paper.* Zenodo (2026).
> <https://doi.org/10.5281/zenodo.18667018>. Licensed **CC-BY-4.0**.

Every PNG, contact sheet and `.rrd` in this directory is a **derivative work**
of that CC-BY-4.0 data (annotated crops, downscales, threshold maps and
foreground masks). Keep them local: they are not committed to the repository,
not published, and not redistributed. If you do share any of them, the
attribution above and the CC-BY-4.0 licence must travel with them.

The dataset itself stays in its read-only cache and is never modified.

## What is here

| Path | Contents |
| :--- | :--- |
| `index.html` | entry point — open with `file://`, links everything |
| `images/<image>/` | per-config overview PNGs, intermediates strips, FN crops, per-config JSON |
| `compare/<image>.png` | the configs side by side (one column per config) |
| `classes/<class>.png` | contact sheet of one stage-of-loss class |
| `rrd/<image>.rrd` | Rerun recording, full resolution, all configs as entity paths |
| `scan_<config>.jsonl` | per-image TP/FP/FN over the whole dataset |
| `selection.json` | the selected images and the rule that selected each |

## Viewing

```bash
# annotated PNGs + summary tables
xdg-open {index_path}          # or: firefox {index_path}

# one Rerun recording (full resolution, every config as its own entity path)
rerun {rrd_example}

# all selected recordings at once
rerun {rrd_glob}
```

In the Rerun viewer each config is a top-level entity (`standard`,
`no_sharpen`, `local_mean`, `shoot_limited`); under it `0_input` carries the
image plus the `ground_truth` / `detections` / `false_positives` /
`missed_gt` / `rejected_candidates` overlays, and `1_threshold_map`,
`2_foreground`, `3_work_sharpened` are the intermediate views.
"""


def write_readme(out_dir: Path) -> Path:
    """Write the attribution/how-to-view README into the output directory."""
    rrds = sorted((out_dir / "rrd").glob("*.rrd"))
    example = str(rrds[0]) if rrds else str(out_dir / "rrd" / "004.rrd")
    path = out_dir / "README.md"
    path.write_text(
        README_TEMPLATE.format(
            index_path=out_dir / "index.html",
            rrd_example=example,
            rrd_glob=str(out_dir / "rrd" / "*.rrd"),
        )
    )
    return path


# --- Drivers ----------------------------------------------------------------


def run_scan(
    *,
    data_dir: Path,
    out_dir: Path,
    config: str,
    family: Any,
    classify: bool,
    limit: int | None,
    skip: int,
    tile_size: int,
    min_range: int,
    match_threshold_px: float,
    progress: bool = True,
) -> Path:
    """Score (and optionally coarse-classify) every image for one config."""
    import cv2
    from tqdm import tqdm

    from tools.bench.liu4k import load_liu4k

    spec = CONFIGS[config]
    detector, dump = make_detector(spec, family)
    samples = load_liu4k(data_dir)
    names = sorted(samples)[skip:]
    if limit:
        names = names[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scan_{config}.jsonl"
    meta = out_dir / f"scan_{config}.meta.json"
    meta.write_text(
        json.dumps(
            {
                "config": config,
                "label": spec.label,
                "provenance": spec.provenance,
                "classified": classify,
                "images": len(names),
                "config_dump": dump,
            },
            indent=1,
            default=str,
        )
    )
    agreements: list[float] = []
    with open(path, "w") as fh:
        iterator = tqdm(names, desc=f"scan {config}") if progress else names
        for name in iterator:
            raw = cv2.imread(str(data_dir / name), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            img = _u8(raw)
            sample = samples[name]
            gt_ids = [t.tag_id for t in sample.tags]
            gt_corners = [np.asarray(t.corners, dtype=np.float64) for t in sample.tags]
            if classify:
                result, _, _, _ = detect_frame(
                    detector,
                    img,
                    gt_ids,
                    gt_corners,
                    spec,
                    tile_size=tile_size,
                    min_range=min_range,
                    match_threshold_px=match_threshold_px,
                    detailed=False,
                )
                if result.threshold_agreement is not None:
                    agreements.append(result.threshold_agreement)
                scan = ImageScan(name, result.n_gt, result.tp, result.fp, result.fn, result.classes)
            else:
                scan = _score_only(detector, img, sample, match_threshold_px, name)
            fh.write(scan_to_json(scan) + "\n")
            fh.flush()  # a 924-image scan is long: keep the file readable live
    if agreements:
        data = json.loads(meta.read_text())
        data["threshold_agreement_min"] = min(agreements)
        data["threshold_agreement_mean"] = float(np.mean(agreements))
        meta.write_text(json.dumps(data, indent=1, default=str))
    return path


def _score_only(
    detector: Any, img: NDArray[np.uint8], sample: Any, threshold: float, name: str
) -> ImageScan:
    from tools.bench.liu4k import score_detections

    batch = detector.detect(img)
    det_ids = [int(i) for i in batch.ids]
    det_corners = np.asarray(batch.corners, dtype=np.float64).reshape(-1, 4, 2)
    tp, fp, fn = score_detections(det_ids, det_corners, sample.tags, threshold=threshold)
    return ImageScan(name, len(sample.tags), tp, fp, fn, {})


def run_select(*, out_dir: Path, baseline: str, target: int, seed: int = SELECTION_SEED) -> Path:
    """Turn the dataset scans into ``selection.json`` (deterministic)."""
    scans: dict[str, dict[str, ImageScan]] = {}
    for name in CONFIG_ORDER:
        path = out_dir / f"scan_{name}.jsonl"
        if path.exists():
            scans[name] = load_scan(path)
    if baseline not in scans:
        raise FileNotFoundError(f"missing {out_dir / f'scan_{baseline}.jsonl'}; run `scan` first")
    selection = select_images(scans, baseline=baseline, target=target, seed=seed)
    path = out_dir / "selection.json"
    path.write_text(
        json.dumps(
            {
                "baseline": baseline,
                "seed": seed,
                "target": target,
                "configs_scanned": sorted(scans),
                "images": [{"image": s.image, "reasons": list(s.reasons)} for s in selection],
            },
            indent=1,
        )
    )
    return path


def load_selection(out_dir: Path) -> list[Selection]:
    """Read back ``selection.json``."""
    data = json.loads((out_dir / "selection.json").read_text())
    return [Selection(image=row["image"], reasons=tuple(row["reasons"])) for row in data["images"]]


def run_render(
    *,
    data_dir: Path,
    out_dir: Path,
    configs: Sequence[str],
    images: Sequence[str],
    family: Any,
    rrd_tag: str,
    tile_size: int,
    min_range: int,
    match_threshold_px: float,
    intermediates: bool,
    emit_rrd: bool,
    png_long_side: int = PNG_LONG_SIDE,
    progress: bool = True,
) -> list[Path]:
    """Render annotated PNGs, crops, intermediates and an ``.rrd`` per image."""
    import cv2
    from tqdm import tqdm

    from tools.bench.liu4k import load_liu4k

    samples = load_liu4k(data_dir)
    specs = [CONFIGS[c] for c in configs]
    detectors = {spec.name: make_detector(spec, family)[0] for spec in specs}

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "rrd" / "parts"
    if emit_rrd:
        parts_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    iterator = tqdm(list(images), desc=f"render {rrd_tag}") if progress else list(images)
    for name in iterator:
        if name not in samples:
            continue
        raw = cv2.imread(str(data_dir / name), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        img = _u8(raw)
        stem = Path(name).stem
        img_dir = out_dir / "images" / stem
        (img_dir / "crops").mkdir(parents=True, exist_ok=True)
        sample = samples[name]
        gt_ids = [t.tag_id for t in sample.tags]
        gt_corners = [np.asarray(t.corners, dtype=np.float64) for t in sample.tags]

        rec = None
        if emit_rrd:
            rec = _open_recording(stem, parts_dir / f"{stem}__{rrd_tag}.rrd")
        for spec in specs:
            result, diagnoses, pre, extra = detect_frame(
                detectors[spec.name],
                img,
                gt_ids,
                gt_corners,
                spec,
                tile_size=tile_size,
                min_range=min_range,
                match_threshold_px=match_threshold_px,
                detailed=True,
            )
            result.image = name
            overview = annotate_overview(
                img,
                diagnoses,
                extra["det_corners"],
                extra["det_ids"],
                extra["fp_mask"],
                extra["rejected"],
                title=f"{name} - {spec.label}  TP {result.tp} / FP {result.fp} / FN {result.fn}",
                long_side=png_long_side,
            )
            overview_path = img_dir / f"{spec.name}_overview.png"
            cv2.imwrite(str(overview_path), overview)
            written.append(overview_path)

            if intermediates:
                strip_path = img_dir / f"{spec.name}_intermediates.png"
                cv2.imwrite(str(strip_path), intermediates_strip(img, pre))
                written.append(strip_path)

            for k, (d, record) in enumerate(
                zip([d for d in diagnoses if not d.matched], result.misses, strict=True)
            ):
                canvas, scale, offset = crop_around(img, d.corners)
                _draw_quad(canvas, d.corners, COLOR_FN, scale=scale, offset=offset, thickness=2)
                rejected = extra["rejected"]
                if rejected is not None:
                    for quad in np.asarray(rejected, dtype=np.float64).reshape(-1, 4, 2):
                        _draw_quad(canvas, quad, COLOR_REJ, scale=scale, offset=offset, thickness=1)
                for quad in np.asarray(extra["det_corners"], dtype=np.float64).reshape(-1, 4, 2):
                    _draw_quad(canvas, quad, COLOR_TP, scale=scale, offset=offset, thickness=1)
                _legend(canvas, [(f"{name} {spec.name}", (255, 255, 255)), (d.label, COLOR_FN)])
                crop_path = img_dir / "crops" / f"{spec.name}_fn{k:02d}_{d.loss_class}.png"
                cv2.imwrite(str(crop_path), canvas)
                record["crop"] = str(crop_path.relative_to(out_dir))
                written.append(crop_path)

            (img_dir / f"{spec.name}.json").write_text(
                json.dumps(
                    {
                        "image": name,
                        "config": spec.name,
                        "label": spec.label,
                        "provenance": spec.provenance,
                        "n_gt": result.n_gt,
                        "tp": result.tp,
                        "fp": result.fp,
                        "fn": result.fn,
                        "classes": result.classes,
                        "threshold_agreement": result.threshold_agreement,
                        "misses": result.misses,
                        "false_positive_centres": result.fp_centres,
                    },
                    indent=1,
                )
            )
            if rec is not None:
                log_frame_to_rerun(
                    spec, img, pre, diagnoses, extra, result, intermediates=intermediates
                )
        if rec is not None:
            _close_recording(rec)
            written.append(parts_dir / f"{stem}__{rrd_tag}.rrd")
    return written


def _open_recording(stem: str, path: Path) -> Any:
    """Start a per-image recording that later merges with the other build's part."""
    import rerun as rr

    rr.init(RERUN_APP_ID, recording_id=f"liu4k-{stem}")
    rr.set_time("frame", sequence=0)
    rr.save(str(path))
    return path


def _close_recording(path: Any) -> None:
    import rerun as rr

    # Dropping the global stream flushes it; `rr.init` of the next image would
    # do the same, but an explicit flush keeps the file valid if we stop here.
    flush = getattr(rr, "flush", None)
    if callable(flush):
        flush(blocking=True)


def merge_rrd_parts(out_dir: Path, *, rerun_bin: str = "rerun") -> list[Path]:
    """Merge the per-build ``.rrd`` parts of each image into one recording.

    Falls back to copying when only one part exists or the CLI is unavailable;
    the parts stay on disk either way.
    """
    import shutil
    import subprocess

    parts_dir = out_dir / "rrd" / "parts"
    merged: list[Path] = []
    groups: dict[str, list[Path]] = {}
    for part in sorted(parts_dir.glob("*.rrd")):
        groups.setdefault(part.name.split("__")[0], []).append(part)
    for stem, parts in sorted(groups.items()):
        target = out_dir / "rrd" / f"{stem}.rrd"
        if len(parts) == 1 or shutil.which(rerun_bin) is None:
            shutil.copyfile(parts[0], target)
        else:
            with open(target, "wb") as fh:
                subprocess.run(
                    [rerun_bin, "rrd", "merge", *[str(p) for p in parts]],
                    stdout=fh,
                    check=True,
                )
        merged.append(target)
    return merged


def verify_rrd(paths: Sequence[Path], *, rerun_bin: str = "rerun") -> dict[str, str]:
    """Headless check that each recording loads: ``rerun rrd verify`` + SDK read."""
    import shutil
    import subprocess

    out: dict[str, str] = {}
    have_cli = shutil.which(rerun_bin) is not None
    for path in paths:
        notes: list[str] = []
        if have_cli:
            proc = subprocess.run(
                [rerun_bin, "rrd", "verify", str(path)], capture_output=True, text=True
            )
            notes.append(
                "rrd verify: ok"
                if proc.returncode == 0
                else f"rrd verify: {proc.stderr.strip()[:200]}"
            )
        try:
            import rerun as rr

            recording = rr.dataframe.load_recording(str(path))
            notes.append(f"sdk load: ok ({len(recording.schema().component_columns())} columns)")
        except Exception as exc:  # pragma: no cover - SDK feature probe
            notes.append(f"sdk load: {exc}")
        out[path.name] = "; ".join(notes)
    return out


# --- Index: comparison grids, contact sheets, index.html --------------------

CONTACT_SHEET_MAX = 12


def _frame_jsons(out_dir: Path, stem: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in CONFIG_ORDER:
        path = out_dir / "images" / stem / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def build_comparison(out_dir: Path, stem: str) -> Path | None:
    """One PNG with the configs side by side (a column per config)."""
    import cv2

    panels: list[NDArray[np.uint8]] = []
    for name in CONFIG_ORDER:
        path = out_dir / "images" / stem / f"{name}_overview.png"
        if path.exists():
            panel = cv2.imread(str(path))
            if panel is not None:
                panels.append(_u8(panel))
    if not panels:
        return None
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)
    target = out_dir / "compare" / f"{stem}.png"
    cv2.imwrite(str(target), grid(panels, 2 if len(panels) > 2 else len(panels)))
    return target


def build_contact_sheets(out_dir: Path) -> dict[str, Path]:
    """One contact sheet per loss class, from the rendered FN crops."""
    import cv2

    per_class: dict[str, list[tuple[str, str, Path]]] = {}
    for frame_dir in sorted((out_dir / "images").glob("*")):
        for name in CONFIG_ORDER:
            path = frame_dir / f"{name}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for miss in data["misses"]:
                crop = miss.get("crop")
                if crop is None or miss["loss_class"] is None:
                    continue
                per_class.setdefault(miss["loss_class"], []).append(
                    (data["image"], name, out_dir / crop)
                )
    (out_dir / "classes").mkdir(parents=True, exist_ok=True)
    sheets: dict[str, Path] = {}
    for cls, entries in sorted(per_class.items()):
        # Deterministic and spread over images: one crop per (image, config)
        # first, in name order, then fill.
        entries.sort(key=lambda e: (e[0], e[1], e[2].name))
        seen: set[tuple[str, str]] = set()
        chosen: list[tuple[str, str, Path]] = []
        for entry in entries:
            key = (entry[0], entry[1])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(entry)
            if len(chosen) >= CONTACT_SHEET_MAX:
                break
        panels: list[NDArray[np.uint8]] = []
        for image, config, path in chosen:
            panel = cv2.imread(str(path))
            if panel is None:
                continue
            cv2.putText(
                panel,
                f"{image} [{config}]",
                (8, panel.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"{image} [{config}]",
                (8, panel.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(_u8(panel))
        if not panels:
            continue
        target = out_dir / "classes" / f"{cls}.png"
        cv2.imwrite(str(target), grid(panels, 4))
        sheets[cls] = target
    return sheets


def _rel(out_dir: Path, path: Path) -> str:
    return str(path.relative_to(out_dir))


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def run_index(*, out_dir: Path, rrd_notes: Mapping[str, str] | None = None) -> Path:
    """Build the comparison grids, contact sheets and ``index.html``."""
    selection = load_selection(out_dir) if (out_dir / "selection.json").exists() else []
    if not selection:
        selection = [
            Selection(image=f"{p.name}.jpg", reasons=())
            for p in sorted((out_dir / "images").glob("*"))
            if p.is_dir()
        ]
    sheets = build_contact_sheets(out_dir)

    totals: dict[str, dict[str, int]] = {}
    for name in CONFIG_ORDER:
        scan = out_dir / f"scan_{name}.jsonl"
        if not scan.exists():
            continue
        rows = load_scan(scan)
        totals[name] = {
            "images": len(rows),
            "gt": sum(r.n_gt for r in rows.values()),
            "tp": sum(r.tp for r in rows.values()),
            "fp": sum(r.fp for r in rows.values()),
            "fn": sum(r.fn for r in rows.values()),
        }

    parts: list[str] = []
    parts.append(_index_head())
    parts.append(_index_intro(out_dir, totals))
    if sheets:
        parts.append("<h2>Stage-of-loss contact sheets</h2>")
        for cls, path in sheets.items():
            parts.append(
                f'<h3 id="class-{cls}">{cls}</h3><p class="muted">{_html_escape(LOSS_DESCRIPTIONS.get(cls, ""))}</p>'
                f'<a href="{_rel(out_dir, path)}"><img src="{_rel(out_dir, path)}" alt="{cls}"></a>'
            )

    parts.append("<h2>Selected images</h2>")
    for sel in selection:
        stem = Path(sel.image).stem
        frames = _frame_jsons(out_dir, stem)
        if not frames:
            continue
        compare = build_comparison(out_dir, stem)
        parts.append(f'<h3 id="img-{stem}">{_html_escape(sel.image)}</h3>')
        if sel.reasons:
            parts.append(
                "<ul>" + "".join(f"<li>{_html_escape(r)}</li>" for r in sel.reasons) + "</ul>"
            )
        parts.append(_frame_table(out_dir, stem, frames))
        links: list[str] = []
        if compare is not None:
            links.append(f'<a href="{_rel(out_dir, compare)}">side-by-side PNG</a>')
        rrd = out_dir / "rrd" / f"{stem}.rrd"
        if rrd.exists():
            links.append(
                f'<a href="{_rel(out_dir, rrd)}">{stem}.rrd</a> (<code>rerun {rrd}</code>)'
            )
        for name in CONFIG_ORDER:
            strip = out_dir / "images" / stem / f"{name}_intermediates.png"
            if strip.exists():
                links.append(
                    f'<a href="{_rel(out_dir, strip)}">{name}: raw | sharpened | threshold | foreground</a>'
                )
        if links:
            parts.append("<p>" + " &middot; ".join(links) + "</p>")
        if compare is not None:
            parts.append(
                f'<a href="{_rel(out_dir, compare)}"><img src="{_rel(out_dir, compare)}" alt="{stem}"></a>'
            )
        parts.append(_miss_list(out_dir, frames))

    if rrd_notes:
        parts.append("<h2>Recording verification</h2><pre>")
        for key, value in sorted(rrd_notes.items()):
            parts.append(_html_escape(f"{key}: {value}") + "\n")
        parts.append("</pre>")
    parts.append("</body></html>")

    path = out_dir / "index.html"
    path.write_text("\n".join(parts))
    write_readme(out_dir)
    return path


def _index_head() -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Liu4K detection visualisation</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1400px;padding:0 1rem;"
        "background:#111;color:#eee}"
        "img{max-width:100%;border:1px solid #444}"
        "table{border-collapse:collapse;margin:0.5rem 0}"
        "th,td{border:1px solid #555;padding:0.25rem 0.6rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "a{color:#7fd1ff}.muted{color:#aaa}"
        "code{background:#222;padding:0.1rem 0.3rem}"
        "</style></head><body>"
    )


def _index_intro(out_dir: Path, totals: Mapping[str, Mapping[str, int]]) -> str:
    rows = [
        "<h1>Liu4K detection visualisation</h1>",
        "<p class='muted'>Images and ground truth: Liu4K, Mu&ntilde;oz-Salinas, "
        "<a href='https://doi.org/10.5281/zenodo.18667018'>DOI 10.5281/zenodo.18667018</a>, "
        "CC-BY-4.0. Everything on this page is a derivative image of that data and stays "
        "local (see <a href='README.md'>README.md</a>).</p>",
        "<h2>Configurations</h2><table><tr><th>config</th><th>what it is</th></tr>",
    ]
    for name in CONFIG_ORDER:
        spec = CONFIGS[name]
        rows.append(
            f"<tr><td>{name}</td><td style='text-align:left'>{_html_escape(spec.provenance)}</td></tr>"
        )
    rows.append("</table>")
    if totals:
        rows.append(
            "<h2>Whole-dataset scan (id-aware, aruco_nano scorer, 10 px)</h2>"
            "<table><tr><th>config</th><th>images</th><th>GT</th><th>TP</th><th>FP</th>"
            "<th>FN</th><th>recall</th><th>precision</th></tr>"
        )
        for name, t in totals.items():
            recall = 100.0 * t["tp"] / t["gt"] if t["gt"] else 0.0
            prec = 100.0 * t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else 0.0
            rows.append(
                f"<tr><td>{name}</td><td>{t['images']}</td><td>{t['gt']}</td><td>{t['tp']}</td>"
                f"<td>{t['fp']}</td><td>{t['fn']}</td><td>{recall:.2f}%</td><td>{prec:.2f}%</td></tr>"
            )
        rows.append("</table>")
    rows.append(
        "<p class='muted'>Colours: <span style='color:#00c8ff'>ground truth</span>, "
        "<span style='color:#00dc00'>true positive (id)</span>, "
        "<span style='color:#ff4040'>false positive</span>, "
        "<span style='color:#ff00ff'>missed ground truth + diagnosed loss class</span>, "
        "<span style='color:#ffa500'>funnel-rejected candidate quad</span>. "
        "Labels prefixed <code>~</code> come from the Python reproduction of the quad stage "
        "(the Rust quad gates do not export their reject reason).</p>"
    )
    _ = out_dir
    return "\n".join(rows)


def _frame_table(out_dir: Path, stem: str, frames: Mapping[str, Mapping[str, Any]]) -> str:
    rows = [
        "<table><tr><th>config</th><th>GT</th><th>TP</th><th>FP</th><th>FN</th>"
        "<th>loss classes</th><th>overview</th></tr>"
    ]
    for name, data in frames.items():
        classes = ", ".join(f"{k}&times;{v}" for k, v in data["classes"].items()) or "&mdash;"
        overview = out_dir / "images" / stem / f"{name}_overview.png"
        link = f'<a href="{_rel(out_dir, overview)}">PNG</a>' if overview.exists() else "&mdash;"
        rows.append(
            f"<tr><td>{name}</td><td>{data['n_gt']}</td><td>{data['tp']}</td><td>{data['fp']}</td>"
            f"<td>{data['fn']}</td><td style='text-align:left'>{classes}</td><td>{link}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _miss_list(out_dir: Path, frames: Mapping[str, Mapping[str, Any]]) -> str:
    parts = ["<details><summary>missed markers (crops)</summary>"]
    for name, data in frames.items():
        if not data["misses"]:
            continue
        parts.append(f"<p><b>{name}</b></p><p>")
        for miss in data["misses"]:
            crop = miss.get("crop")
            label = _html_escape(f"{miss['tag_id']}: {miss['label']}")
            if crop:
                parts.append(
                    f'<a href="{crop}" title="{label}"><img src="{crop}" alt="{label}" '
                    'style="max-width:260px;margin:2px"></a>'
                )
        parts.append("</p>")
    parts.append("</details>")
    return "\n".join(parts)
