"""Unit tests for ``tools/bench/collect.py``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tools.bench.collect import (
    Collector,
    _quat_geodesic_deg,
    build_provenance,
    flush_collectors,
    new_run_id,
)
from tools.bench.records import read_records
from tools.bench.utils import RejectedQuads, TagAxes, TagGroundTruth


def _make_gt(tag_id: int, x: float, y: float) -> TagGroundTruth:
    """Square GT centered at (x, y) with edge length 20px."""
    half = 10.0
    return TagGroundTruth(
        tag_id=tag_id,
        corners=np.array(
            [
                [x - half, y - half],
                [x + half, y - half],
                [x + half, y + half],
                [x - half, y + half],
            ],
            dtype=np.float32,
        ),
    )


def _make_axes(tag_id: int, **overrides: object) -> TagAxes:
    base = dict(
        tag_id=tag_id,
        distance_m=1.5,
        aoi_deg=30.0,
        ppm=1000.0,
        velocity=None,
        shutter_time_ms=10.0,
        resolution_h=1080,
        occlusion_ratio=0.0,
        tag_size_mm=160.0,
    )
    base.update(overrides)
    return TagAxes(**base)  # type: ignore[arg-type]


class TestQuatGeodesic:
    def test_identity_is_zero(self) -> None:
        q = np.array([0.0, 0.0, 0.0, 1.0])
        assert _quat_geodesic_deg(q, q) == pytest.approx(0.0, abs=1e-9)

    def test_180_deg_about_z(self) -> None:
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.0, 0.0, 1.0, 0.0])  # 180° rotation about Z
        assert _quat_geodesic_deg(q1, q2) == pytest.approx(180.0, abs=1e-6)

    def test_sign_flip_is_zero(self) -> None:
        """Quaternions q and -q represent the same rotation."""
        q1 = np.array([0.1, 0.2, 0.3, 0.927])
        q1 = q1 / np.linalg.norm(q1)
        q2 = -q1
        assert _quat_geodesic_deg(q1, q2) == pytest.approx(0.0, abs=1e-6)


class TestCollectorObserve:
    def test_matched_record_carries_axes(self) -> None:
        c = Collector.new("run-1", "Locus", "standard", "ds")
        gt = _make_gt(tag_id=42, x=100.0, y=100.0)
        det = {
            "id": 42,
            "center": [100.5, 99.7],
            "corners": [[90, 90], [110, 90], [110, 110], [90, 110]],
            "hamming": 0,
        }
        c.observe(
            image_id="img_0",
            detections=[det],
            gt_tags=[gt],
            axes_lookup={42: _make_axes(42, distance_m=2.5, aoi_deg=37.0)},
            frame_latency_ms=2.5,
            rejected=None,
            resolution_h=1080,
        )
        assert len(c.records) == 1
        r = c.records[0]
        assert r.record_kind == "matched"
        assert r.matched is True
        assert r.tag_id == 42
        assert r.distance_m == pytest.approx(2.5)
        assert r.aoi_deg == pytest.approx(37.0)
        assert r.hamming_bits == 0

    def test_missed_gt_record_when_no_detection(self) -> None:
        c = Collector.new("run-1", "Locus", "standard", "ds")
        gt = _make_gt(tag_id=42, x=100.0, y=100.0)
        c.observe(
            image_id="img_0",
            detections=[],
            gt_tags=[gt],
            axes_lookup={42: _make_axes(42)},
            frame_latency_ms=2.5,
            rejected=None,
            resolution_h=1080,
        )
        assert len(c.records) == 1
        r = c.records[0]
        assert r.record_kind == "missed_gt"
        assert r.matched is False
        assert math.isnan(r.trans_err_m)

    def test_rejected_quad_attributed_to_nearest_gt(self) -> None:
        """A rejected quad whose center sits near a GT center → tag_id set."""
        c = Collector.new("run-1", "Locus", "standard", "ds")
        gt = _make_gt(tag_id=42, x=100.0, y=100.0)
        # Quad centered at (102, 101) — 2.2 px from GT center, well inside the
        # 1.5 × max_edge_px (=30 px) attribution threshold.
        rejected = RejectedQuads(
            corners=np.array(
                [[[92, 91], [112, 91], [112, 111], [92, 111]]],
                dtype=np.float32,
            ),
            funnel_status=np.array([2], dtype=np.uint8),  # RejectedContrast
            error_rates=np.array([0.0], dtype=np.float32),
        )
        c.observe(
            image_id="img_0",
            detections=[],
            gt_tags=[gt],
            axes_lookup={42: _make_axes(42)},
            frame_latency_ms=2.5,
            rejected=rejected,
            resolution_h=1080,
        )
        # Two records: missed_gt + rejected_quad
        kinds = [r.record_kind for r in c.records]
        assert kinds == ["missed_gt", "rejected_quad"]
        rq = c.records[1]
        assert rq.tag_id == 42
        assert rq.rejection_reason == "RejectedContrast"

    def test_rejected_quad_unattributed_when_far_from_gt(self) -> None:
        c = Collector.new("run-1", "Locus", "standard", "ds")
        gt = _make_gt(tag_id=42, x=100.0, y=100.0)
        # Quad far away — center at (500, 500), > 30 px from GT center.
        rejected = RejectedQuads(
            corners=np.array(
                [[[495, 495], [505, 495], [505, 505], [495, 505]]],
                dtype=np.float32,
            ),
            funnel_status=np.array([3], dtype=np.uint8),  # RejectedSampling
            error_rates=np.array([7.0], dtype=np.float32),
        )
        c.observe(
            image_id="img_0",
            detections=[],
            gt_tags=[gt],
            axes_lookup={42: _make_axes(42)},
            frame_latency_ms=2.5,
            rejected=rejected,
            resolution_h=1080,
        )
        rq = c.records[1]
        assert rq.tag_id is None
        assert rq.rejection_reason == "RejectedSampling"
        assert rq.hamming_bits == 7  # error_rate forwarded when decoder ran

    def test_false_positive_emitted_when_detection_does_not_match_gt(self) -> None:
        """A detection with the right ID at the wrong location, or the wrong
        ID, contributes to (1 - precision). Must surface as a record."""
        c = Collector.new("run-1", "Locus", "standard", "ds")
        gt = _make_gt(tag_id=42, x=100.0, y=100.0)
        # Detection at right ID but >20 px from GT center → no match.
        det_far = {
            "id": 42,
            "center": [500.0, 500.0],
            "corners": [[490, 490], [510, 490], [510, 510], [490, 510]],
            "hamming": 0,
        }
        # Detection with wrong ID → no match regardless of position.
        det_wrong_id = {
            "id": 99,
            "center": [100.0, 100.0],
            "corners": [[90, 90], [110, 90], [110, 110], [90, 110]],
            "hamming": 0,
        }
        c.observe(
            image_id="img_0",
            detections=[det_far, det_wrong_id],
            gt_tags=[gt],
            axes_lookup={42: _make_axes(42)},
            frame_latency_ms=2.5,
            rejected=None,
            resolution_h=1080,
        )
        kinds = sorted(r.record_kind for r in c.records)
        # 1 missed_gt + 2 false_positives
        assert kinds == ["false_positive", "false_positive", "missed_gt"]
        for fp in (r for r in c.records if r.record_kind == "false_positive"):
            assert fp.tag_id is None
            assert fp.matched is False

    def test_passed_contrast_relabels_to_rejected_decode(self) -> None:
        """A quad in the rejected list with ``PassedContrast`` (=1) actually
        failed downstream (Hamming) — relabel for plot fidelity.
        """
        c = Collector.new("run-1", "Locus", "standard", "ds")
        gt = _make_gt(tag_id=42, x=100.0, y=100.0)
        rejected = RejectedQuads(
            corners=np.array(
                [[[92, 91], [112, 91], [112, 111], [92, 111]]],
                dtype=np.float32,
            ),
            funnel_status=np.array([1], dtype=np.uint8),  # PassedContrast
            error_rates=np.array([5.0], dtype=np.float32),
        )
        c.observe(
            image_id="img_0",
            detections=[],
            gt_tags=[gt],
            axes_lookup={42: _make_axes(42)},
            frame_latency_ms=2.5,
            rejected=rejected,
            resolution_h=1080,
        )
        rq = c.records[1]
        assert rq.rejection_reason == "RejectedDecode"
        assert rq.hamming_bits == 5


class TestFlushAndRoundTrip:
    def test_flush_writes_parquet_with_provenance(self, tmp_path: Path) -> None:
        c = Collector.new(new_run_id(), "Locus", "standard", "ds")
        gt = _make_gt(tag_id=42, x=100.0, y=100.0)
        c.observe(
            image_id="img_0",
            detections=[],
            gt_tags=[gt],
            axes_lookup={42: _make_axes(42)},
            frame_latency_ms=2.5,
            rejected=None,
            resolution_h=1080,
        )

        out = tmp_path / "run.parquet"
        prov = build_provenance(dataset_version="test")
        n_rows = flush_collectors([c], prov, out)
        assert n_rows == 1

        loaded, prov_back = read_records(out)
        assert len(loaded) == 1
        assert loaded[0].record_kind == "missed_gt"
        # Provenance round-trips
        assert prov_back.dataset_version == "test"
        assert prov_back.locus_version  # non-empty
        assert prov_back.cpu_cores_logical >= 1


class TestBuildProvenance:
    def test_provenance_passes_schema_validation(self) -> None:
        prov = build_provenance(dataset_version="test-v1")
        # Implicit: build_provenance returns a Provenance which is itself a
        # Pydantic model; field validators run at construction. If we got
        # this far, schema validation passed.
        assert prov.dataset_version == "test-v1"
        assert prov.build_profile in ("release", "debug")
        assert len(prov.git_sha) == 40
        assert prov.cpu_cores_physical >= 1
        assert prov.cpu_cores_logical >= prov.cpu_cores_physical
