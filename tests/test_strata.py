"""Unit tests for ``tools/bench/strata.py``.

The output of ``compute_stratum_id`` must round-trip through the
``stratum_id`` grammar validator in ``tools/bench/schema.py`` for every
combination of bucket slugs, including ``unk``.
"""

from __future__ import annotations

import math

import pytest

from tools.bench.schema import _validate_stratum_id
from tools.bench.strata import (
    AxisValues,
    _bucket_aoi,
    _bucket_dist,
    _bucket_mot,
    _bucket_ppm,
    _bucket_res,
    compute_stratum_id,
)


class TestBucketEdges:
    """Boundaries are inclusive on the upper edge: e.g. d=0.7 is `near`, d>0.7 is `mid`."""

    @pytest.mark.parametrize(
        ("h", "expected"),
        [
            (None, "unk"),
            (240, "sd"),
            (480, "sd"),
            (481, "hd"),
            (720, "hd"),
            (721, "fhd"),
            (1080, "fhd"),
            (1081, "uhd"),
            (2160, "uhd"),
        ],
    )
    def test_res(self, h: int | None, expected: str) -> None:
        assert _bucket_res(h) == expected

    @pytest.mark.parametrize(
        ("d", "expected"),
        [
            (math.nan, "unk"),
            (0.5, "near"),
            (0.7, "near"),
            (0.71, "mid"),
            (1.5, "mid"),
            (1.51, "far"),
            (10.0, "far"),
        ],
    )
    def test_dist(self, d: float, expected: str) -> None:
        assert _bucket_dist(d) == expected

    @pytest.mark.parametrize(
        ("a", "expected"),
        [
            (math.nan, "unk"),
            (0.0, "frontal"),
            (35.0, "frontal"),
            (35.01, "oblique"),
            (50.0, "oblique"),
            (50.01, "grazing"),
            (89.0, "grazing"),
        ],
    )
    def test_aoi(self, a: float, expected: str) -> None:
        assert _bucket_aoi(a) == expected

    @pytest.mark.parametrize(
        ("p", "expected"),
        [
            (math.nan, "unk"),
            (100.0, "lo"),
            (800.0, "lo"),
            (800.01, "mid"),
            (1300.0, "mid"),
            (1300.01, "hi"),
            (5000.0, "hi"),
        ],
    )
    def test_ppm(self, p: float, expected: str) -> None:
        assert _bucket_ppm(p) == expected

    @pytest.mark.parametrize(
        ("v", "expected"),
        [
            (None, "static"),
            (math.nan, "unk"),
            (0.0, "static"),
            (0.01, "motion"),
            (5.0, "motion"),
        ],
    )
    def test_mot(self, v: float | None, expected: str) -> None:
        assert _bucket_mot(v) == expected


class TestStratumIdGrammar:
    """Every produced ``stratum_id`` must validate against the schema grammar."""

    def test_fully_populated_record(self) -> None:
        sid = compute_stratum_id(
            AxisValues(resolution_h=1080, distance_m=1.0, aoi_deg=20.0, ppm=1000.0, velocity=None)
        )
        assert sid == "res=fhd|ppm=mid|aoi=frontal|dist=mid|mot=static"
        assert _validate_stratum_id(sid) == sid

    def test_icra_record_collapses_to_unk(self) -> None:
        """ICRA 2020 has no per-tag metadata — every axis except resolution is `unk`."""
        sid = compute_stratum_id(
            AxisValues(
                resolution_h=1080,
                distance_m=math.nan,
                aoi_deg=math.nan,
                ppm=math.nan,
                velocity=None,
            )
        )
        assert sid == "res=fhd|ppm=unk|aoi=unk|dist=unk|mot=static"
        assert _validate_stratum_id(sid) == sid

    @pytest.mark.parametrize("res_h", [None, 240, 720, 1080, 2160])
    @pytest.mark.parametrize("dist", [math.nan, 0.5, 1.0, 5.0])
    @pytest.mark.parametrize("aoi", [math.nan, 10.0, 40.0, 60.0])
    @pytest.mark.parametrize("ppm", [math.nan, 500.0, 1000.0, 2000.0])
    @pytest.mark.parametrize("vel", [None, math.nan, 0.0, 1.0])
    def test_grammar_holds_for_full_axis_grid(
        self,
        res_h: int | None,
        dist: float,
        aoi: float,
        ppm: float,
        vel: float | None,
    ) -> None:
        sid = compute_stratum_id(
            AxisValues(resolution_h=res_h, distance_m=dist, aoi_deg=aoi, ppm=ppm, velocity=vel)
        )
        # Returns the input unchanged on success; raises ValueError otherwise.
        assert _validate_stratum_id(sid) == sid
