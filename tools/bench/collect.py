"""Tier-1 record collector — fills :class:`ObservationRecord` rows from a bench loop.

Replaces the inline ``stats`` accumulator in ``tools/cli.py`` for callers that
opt into ``bench real --record-out PATH``. The aggregated print output stays
verbatim; the collector is a sidecar that emits a parquet file when enabled.

Four ``record_kind`` values are produced per image:

- ``matched``        — detection ↔ GT pair (GT axes carried through, errors filled)
- ``missed_gt``      — GT tag with no matching detection (errors NaN)
- ``false_positive`` — detection that no GT accepted (the ``1 - precision``
                       term; permissive nearest-GT attribution for stratum
                       derivation only, ``tag_id`` stays null)
- ``rejected_quad``  — Locus quad rejected by the funnel/decoder (attributed
                       to the nearest GT when within
                       ``REJECTED_NEAREST_GT_FACTOR × max_edge_px``)
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from tools.bench.records import ObservationRecord, RecordKind, empty_record, write_records
from tools.bench.schema import Provenance
from tools.bench.utils import (
    RejectedQuads,
    TagAxes,
    TagGroundTruth,
    max_edge_px,
    rotation_error_deg,
)

# Detection ↔ GT pairing threshold in pixels. Mirrors
# ``Metrics.match_detections``' default at ``tools/bench/utils.py:704`` so the
# collector and the headline aggregator agree on what counts as a TP.
MATCH_DISTANCE_THRESHOLD_PX = 20.0

# Center-distance threshold for attributing a rejected quad to a GT tag, as a
# multiple of the rejected quad's longest edge. Quads further than this from
# any GT center get ``tag_id=None`` and contribute to "unattributed" rejection
# composition rows. 1.5 was the value the plan specified — generous enough to
# attribute borderline fragments, tight enough to avoid pathological matches.
REJECTED_NEAREST_GT_FACTOR = 1.5

# False-positive attribution threshold as a fraction of image height. Uses a
# permissive bound (half the height) because we attribute *for stratum
# derivation only* — the FP keeps ``tag_id=None`` regardless. Tighter would
# leave too many FPs in unk strata; looser would smear FPs across regimes
# they didn't physically come from.
FP_ATTRIBUTION_FACTOR = 0.5

# Rejection-reason labels for records emitted into ``rejected_funnel_status``.
# The Rust ``FunnelStatus`` enum only captures gates *up to* the funnel; quads
# that pass the funnel but fail downstream (decoder/Hamming) keep
# ``PassedContrast`` as their stored status. Since they're nonetheless in the
# rejected list, we re-label them ``RejectedDecode`` for plot fidelity.
_FUNNEL_STATUS_NAMES = {
    0: "Unknown",  # funnel never ran — shouldn't appear among rejected quads
    1: "RejectedDecode",  # passed funnel, rejected downstream (Hamming et al.)
    2: "RejectedContrast",  # geometry-only failure
    3: "RejectedSampling",  # sampling / homography DDA failure
}


@dataclass
class Collector:
    """Per-(binary, profile, dataset) record accumulator.

    One ``Collector`` instance is created per unique combination; rows from
    multiple collectors are concatenated by ``write_records`` at flush time.
    """

    run_id: str
    binary: str
    profile: str
    dataset: str
    records: list[ObservationRecord]

    @classmethod
    def new(cls, run_id: str, binary: str, profile: str, dataset: str) -> Collector:
        return cls(run_id=run_id, binary=binary, profile=profile, dataset=dataset, records=[])

    def observe(
        self,
        *,
        image_id: str,
        detections: list[dict[str, Any]],
        gt_tags: list[TagGroundTruth],
        axes_lookup: dict[int, TagAxes],
        frame_latency_ms: float,
        rejected: RejectedQuads | None,
        resolution_h: int,
        intrinsics: Any | None = None,
    ) -> None:
        """Append one frame's worth of records.

        ``axes_lookup`` is ``DatasetLoader.load_axes()[image_id]`` — empty
        ``{}`` for ICRA / unknown corpora; per-tag ``TagAxes`` for hub corpora.
        """
        n_gt = len(gt_tags)
        n_det = len(detections)

        # Phase 1: match detections to GT (mirrors Metrics.match_detections so
        # the records align with the headline recall numbers).
        matched_gt: dict[int, dict[str, Any]] = {}  # gt_index -> det dict
        matched_det: set[int] = set()  # detection indices that paired
        used_gt: set[int] = set()
        for det_idx, det in enumerate(detections):
            best_idx = -1
            min_dist = float("inf")
            det_center = np.asarray(det["center"], dtype=np.float64)
            for idx, gt in enumerate(gt_tags):
                if idx in used_gt or gt.tag_id != det["id"]:
                    continue
                gt_center = np.mean(gt.corners, axis=0)
                dist = float(np.linalg.norm(det_center - gt_center))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            if best_idx != -1 and min_dist < MATCH_DISTANCE_THRESHOLD_PX:
                used_gt.add(best_idx)
                matched_gt[best_idx] = det
                matched_det.add(det_idx)

        # Phase 2: emit one row per GT tag (matched or missed).
        for idx, gt in enumerate(gt_tags):
            axes = axes_lookup.get(gt.tag_id)
            base = self._empty(
                image_id,
                "matched" if idx in matched_gt else "missed_gt",
                n_gt,
                n_det,
                frame_latency_ms,
                resolution_h,
            )
            base = self._with_axes(base, axes, gt.tag_id)
            if idx in matched_gt:
                det = matched_gt[idx]
                trans_err, rot_err, repro_err = self._errors(det, gt, intrinsics)
                self.records.append(
                    self._with_outcome(
                        base,
                        matched=True,
                        trans_err_m=trans_err,
                        rot_err_deg=rot_err,
                        repro_err_px=repro_err,
                        hamming_bits=int(det.get("hamming", 0)),
                    )
                )
            else:
                self.records.append(base)

        # Phase 3: emit one false-positive row per unmatched detection. These
        # are detections the wrapper returned but that no GT accepted — the
        # numerator term in (1 - precision). Cross-reference to nearest GT
        # for stratum derivation, but tag_id stays null because the detector's
        # claim doesn't correspond to a real tag.
        if len(matched_det) < n_det:
            self._emit_false_positives(
                detections,
                matched_det,
                gt_tags,
                axes_lookup,
                image_id,
                n_gt,
                n_det,
                frame_latency_ms,
                resolution_h,
            )

        # Phase 4: rejected quads (Locus only). Match each to nearest GT by
        # center distance with a sanity threshold; unattributed → tag_id=None.
        if rejected is not None and len(rejected) > 0:
            self._emit_rejected(
                rejected,
                gt_tags,
                axes_lookup,
                image_id,
                n_gt,
                n_det,
                frame_latency_ms,
                resolution_h,
            )

    def _emit_false_positives(
        self,
        detections: list[dict[str, Any]],
        matched_det: set[int],
        gt_tags: list[TagGroundTruth],
        axes_lookup: dict[int, TagAxes],
        image_id: str,
        n_gt: int,
        n_det: int,
        frame_latency_ms: float,
        resolution_h: int,
    ) -> None:
        gt_centers = (
            np.array([np.mean(gt.corners, axis=0) for gt in gt_tags], dtype=np.float64)
            if gt_tags
            else np.zeros((0, 2), dtype=np.float64)
        )
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det:
                continue
            det_center = np.asarray(det["center"], dtype=np.float64)
            attribution_axes: TagAxes | None = None
            if len(gt_centers) > 0:
                dists = np.linalg.norm(gt_centers - det_center, axis=1)
                best = int(np.argmin(dists))
                # Permissive attribution for stratum derivation only — half
                # the image height is enough to land in the right axis bucket
                # without claiming the FP is "near" a real tag.
                if dists[best] <= resolution_h * FP_ATTRIBUTION_FACTOR:
                    attribution_axes = axes_lookup.get(gt_tags[best].tag_id)
            base = self._empty(
                image_id, "false_positive", n_gt, n_det, frame_latency_ms, resolution_h
            )
            base = self._with_axes(base, attribution_axes, None)
            self.records.append(
                self._with_outcome(
                    base,
                    matched=False,
                    hamming_bits=int(det.get("hamming", 0)),
                )
            )

    def _emit_rejected(
        self,
        rejected: RejectedQuads,
        gt_tags: list[TagGroundTruth],
        axes_lookup: dict[int, TagAxes],
        image_id: str,
        n_gt: int,
        n_det: int,
        frame_latency_ms: float,
        resolution_h: int,
    ) -> None:
        gt_centers = (
            np.array([np.mean(gt.corners, axis=0) for gt in gt_tags], dtype=np.float64)
            if gt_tags
            else np.zeros((0, 2), dtype=np.float64)
        )
        for j in range(len(rejected)):
            quad = np.asarray(rejected.corners[j], dtype=np.float64)
            quad_center = np.mean(quad, axis=0)
            quad_max_edge = max_edge_px(quad)
            threshold_px = REJECTED_NEAREST_GT_FACTOR * quad_max_edge

            matched_tag_id: int | None = None
            if len(gt_centers) > 0:
                dists = np.linalg.norm(gt_centers - quad_center, axis=1)
                best = int(np.argmin(dists))
                if dists[best] <= threshold_px:
                    matched_tag_id = gt_tags[best].tag_id

            axes = axes_lookup.get(matched_tag_id) if matched_tag_id is not None else None
            base = self._empty(
                image_id, "rejected_quad", n_gt, n_det, frame_latency_ms, resolution_h
            )
            base = self._with_axes(base, axes, matched_tag_id)
            code = int(rejected.funnel_status[j])
            err = float(rejected.error_rates[j])
            # error_rate>0 indicates the decoder ran and produced a Hamming-bits
            # estimate; geometry-only rejections leave it at 0.
            hamming = int(err) if err > 0.0 else -1
            self.records.append(
                self._with_outcome(
                    base,
                    matched=False,
                    hamming_bits=hamming,
                    rejection_reason=_FUNNEL_STATUS_NAMES.get(code, f"Unknown({code})"),
                )
            )

    def _empty(
        self,
        image_id: str,
        kind: RecordKind,
        n_gt: int,
        n_det: int,
        latency_ms: float,
        resolution_h: int,
    ) -> ObservationRecord:
        return empty_record(
            run_id=self.run_id,
            binary=self.binary,
            profile=self.profile,
            dataset=self.dataset,
            image_id=image_id,
            record_kind=kind,
            n_gt_in_frame=n_gt,
            n_det_in_frame=n_det,
            frame_latency_ms=latency_ms,
            resolution_h=resolution_h,
        )

    def _with_axes(
        self, record: ObservationRecord, axes: TagAxes | None, tag_id: int | None
    ) -> ObservationRecord:
        if axes is None:
            # ICRA / unattributed rejected quad: keep NaN axes, set tag_id only.
            return _replace(record, tag_id=tag_id)
        blur = (
            axes.shutter_time_ms * abs(axes.velocity) * axes.ppm
            if axes.velocity is not None and not math.isnan(axes.shutter_time_ms)
            else math.nan
        )
        return _replace(
            record,
            tag_id=tag_id,
            distance_m=axes.distance_m,
            aoi_deg=axes.aoi_deg,
            ppm=axes.ppm,
            occlusion_ratio=axes.occlusion_ratio,
            blur_px=blur,
            resolution_h=axes.resolution_h,
        )

    def _with_outcome(
        self,
        record: ObservationRecord,
        *,
        matched: bool,
        trans_err_m: float = math.nan,
        rot_err_deg: float = math.nan,
        repro_err_px: float = math.nan,
        hamming_bits: int = -1,
        rejection_reason: str = "",
    ) -> ObservationRecord:
        return _replace(
            record,
            matched=matched,
            trans_err_m=trans_err_m,
            rot_err_deg=rot_err_deg,
            repro_err_px=repro_err_px,
            hamming_bits=hamming_bits,
            rejection_reason=rejection_reason,
        )

    def _errors(
        self, det: dict[str, Any], gt: TagGroundTruth, intrinsics: Any | None
    ) -> tuple[float, float, float]:
        # Reprojection error: per-corner RMS in pixels.
        det_corners = np.asarray(det["corners"], dtype=np.float64)
        gt_corners = np.asarray(gt.corners, dtype=np.float64)
        repro = float(np.sqrt(np.mean(np.sum((det_corners - gt_corners) ** 2, axis=1))))

        # Pose errors: only when both sides have a 7-vector pose.
        det_pose = det.get("pose")
        if det_pose is None or gt.pose is None:
            return math.nan, math.nan, repro

        det_pose = np.asarray(det_pose, dtype=np.float64)
        gt_pose = np.asarray(gt.pose, dtype=np.float64)
        trans_err = float(np.linalg.norm(det_pose[:3] - gt_pose[:3]))
        rot_err = rotation_error_deg(det_pose, gt_pose[3:7])
        return trans_err, rot_err, repro


def _replace(record: ObservationRecord, **kwargs: Any) -> ObservationRecord:
    """Thin alias for ``dataclasses.replace`` — exists so callers don't have
    to keep the import in line of sight. ``Any`` on kwargs because the
    dataclass has heterogeneous fields and we trust the call sites.
    """
    import dataclasses

    return dataclasses.replace(record, **kwargs)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_provenance(
    *,
    dataset_version: str = "v1.0.0",
    build_profile: str = "release",
) -> Provenance:
    """Construct a :class:`Provenance` snapshot from runtime introspection."""
    git_sha = _git_sha()
    git_dirty = _git_dirty()
    return Provenance(
        git_sha=git_sha,
        git_dirty=git_dirty,
        cpu_model=_cpu_model(),
        cpu_cores_physical=_cpu_cores_physical(),
        cpu_cores_logical=os.cpu_count() or 1,
        locus_version=_locus_version(),
        dataset_version=dataset_version,
        rayon_threads=_rayon_threads(),
        build_profile=build_profile,  # type: ignore[arg-type]
        timestamp_utc=datetime.now(timezone.utc),
    )


def new_run_id() -> str:
    return uuid.uuid4().hex


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        if len(out) == 40 and all(c in "0123456789abcdef" for c in out):
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return "0" * 40  # detached / non-git environment


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _cpu_cores_physical() -> int:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cores: set[tuple[str, str]] = set()
        physical_id = ""
        core_id = ""
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
                if physical_id and core_id:
                    cores.add((physical_id, core_id))
        if cores:
            return len(cores)
    return os.cpu_count() or 1


def _locus_version() -> str:
    try:
        return metadata.version("locus-tag")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def _rayon_threads() -> int | None:
    val = os.environ.get("RAYON_NUM_THREADS")
    if val is None:
        return None
    try:
        n = int(val)
        return n if n > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Flush helper
# ---------------------------------------------------------------------------


def flush_collectors(collectors: list[Collector], provenance: Provenance, path: Path | str) -> int:
    """Concatenate records from all collectors and write a single parquet file.

    Returns the total number of rows written.
    """
    all_records: list[ObservationRecord] = []
    for c in collectors:
        all_records.extend(c.records)
    write_records(all_records, provenance, path)
    return len(all_records)
