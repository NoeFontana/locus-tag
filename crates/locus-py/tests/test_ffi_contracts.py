"""FFI contract test suite.

Source of truth: ``docs/engineering/ffi_contracts.md``. Every invariant
enumerated there gets at least one parametrized case here; every follow-up in
§7 becomes a ``pytest.mark.xfail(strict=True)`` placeholder that flips to
passing when A1.2+ adds the hard gate.

The error messages asserted below are part of the public FFI contract — if a
developer reworks them, this suite fails and the doc and downstream callers
must be updated in the same commit.
"""

from __future__ import annotations

import locus
import numpy as np
import pytest

# A 64x64 zero image is large enough to satisfy every pipeline stage's minimum
# sizing assumptions without doing real work — SIMD kernels still sweep, but
# there are no contours so decoding short-circuits.
_VALID_SHAPE = (64, 64)


def _valid_img() -> np.ndarray:
    return np.zeros(_VALID_SHAPE, dtype=np.uint8)


@pytest.fixture
def detector() -> locus.Detector:
    return locus.Detector()


# ---------------------------------------------------------------------------
# §1 Image buffer invariants
# ---------------------------------------------------------------------------


_INVALID_DTYPES = [np.float32, np.float64, np.int8, np.int16, np.uint16]


class TestImageBuffer:
    """A0.4 §1 — image ingestion invariants.

    Each invariant is exercised against both ``detect()`` and
    ``detect_concurrent([img])`` so the two paths can't silently drift.
    """

    @pytest.mark.parametrize("dtype", _INVALID_DTYPES)
    def test_reject_non_uint8_single(self, detector: locus.Detector, dtype: type) -> None:
        img: np.ndarray = np.zeros(_VALID_SHAPE, dtype=dtype)
        with pytest.raises(ValueError, match=r"Input image must be uint8"):
            detector.detect(img)

    @pytest.mark.parametrize("dtype", _INVALID_DTYPES)
    def test_reject_non_uint8_concurrent(self, detector: locus.Detector, dtype: type) -> None:
        img: np.ndarray = np.zeros(_VALID_SHAPE, dtype=dtype)
        with pytest.raises(ValueError, match=r"Frame 0 must be uint8"):
            detector.detect_concurrent([img])

    @pytest.mark.parametrize(
        "bad_shape",
        [
            (64,),
            (64, 64, 3),
            (1, 64, 64),
            (),
        ],
        ids=["1d", "3d_hwc", "3d_chw", "0d"],
    )
    def test_reject_wrong_rank_single(
        self, detector: locus.Detector, bad_shape: tuple[int, ...]
    ) -> None:
        img = np.zeros(bad_shape, dtype=np.uint8)
        # PyO3's `PyReadonlyArray2` signature rejects wrong-rank arrays with
        # TypeError; the path never reaches our own checks.
        with pytest.raises((TypeError, ValueError)):
            detector.detect(img)

    @pytest.mark.parametrize(
        "bad_shape",
        [
            (64,),
            (64, 64, 3),
            (1, 64, 64),
        ],
        ids=["1d", "3d_hwc", "3d_chw"],
    )
    def test_reject_wrong_rank_concurrent(
        self, detector: locus.Detector, bad_shape: tuple[int, ...]
    ) -> None:
        img = np.zeros(bad_shape, dtype=np.uint8)
        with pytest.raises((TypeError, ValueError)):
            detector.detect_concurrent([img])

    def test_reject_sliced_non_contiguous(self, detector: locus.Detector) -> None:
        img = _valid_img()[:, ::2]
        assert not img.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match=r"C-contiguous"):
            detector.detect(img)

    def test_reject_fortran_order(self, detector: locus.Detector) -> None:
        img = np.asfortranarray(_valid_img())
        assert not img.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match=r"C-contiguous"):
            detector.detect(img)


# ---------------------------------------------------------------------------
# §1 follow-ups — xfail placeholders for A1.2+ gates (see ffi_contracts.md §7)
# ---------------------------------------------------------------------------


class TestSimdPaddingGate:
    """§7 follow-up: ``has_simd_padding()`` is not asserted at ``prepare_image_view``.

    ``prepare_image_view`` constructs a slice of exactly ``required_size``
    bytes — by construction ``ImageView::has_simd_padding()`` returns false
    for every caller today. SIMD kernels nonetheless run and may read up to
    3 bytes past the end via raw-pointer arithmetic on the underlying NumPy
    buffer. A1.2 will either reject or copy-into-padded-arena; this test
    flips to passing when that gate lands.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="A1.2 will assert has_simd_padding() at prepare_image_view",
    )
    def test_missing_padding_is_rejected(self, detector: locus.Detector) -> None:
        img = np.ascontiguousarray(_valid_img())
        with pytest.raises(ValueError, match=r"padding"):
            detector.detect(img)


class TestIntrinsicsShapeCouplingGate:
    """A1.2 enforced gates — previously xfail placeholders from ffi_contracts.md §7.

    Finiteness and positivity are validated at ``CameraIntrinsics`` construction;
    the principal-point bounds check lives at ``Detector.detect`` /
    ``detect_concurrent`` (it needs image dimensions).
    """

    def test_principal_point_outside_image_single(self, detector: locus.Detector) -> None:
        bad = locus.CameraIntrinsics(fx=500.0, fy=500.0, cx=5000.0, cy=5000.0)
        with pytest.raises(ValueError, match=r"CameraIntrinsics\.cx"):
            detector.detect(_valid_img(), intrinsics=bad, tag_size=0.05)

    def test_principal_point_outside_image_concurrent(self, detector: locus.Detector) -> None:
        bad = locus.CameraIntrinsics(fx=500.0, fy=500.0, cx=5000.0, cy=5000.0)
        with pytest.raises(ValueError, match=r"CameraIntrinsics\.cx"):
            detector.detect_concurrent([_valid_img()], intrinsics=bad, tag_size=0.05)

    @pytest.mark.parametrize(
        ("fx", "fy"),
        [(-1.0, 500.0), (0.0, 500.0), (500.0, -1.0), (500.0, 0.0)],
        ids=["fx_negative", "fx_zero", "fy_negative", "fy_zero"],
    )
    def test_non_positive_focal_length(self, fx: float, fy: float) -> None:
        with pytest.raises(ValueError, match=r"CameraIntrinsics\.(fx|fy) must be > 0"):
            locus.CameraIntrinsics(fx=fx, fy=fy, cx=32.0, cy=32.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("fx", float("nan")),
            ("fy", float("inf")),
            ("cx", float("-inf")),
            ("cy", float("nan")),
        ],
    )
    def test_non_finite_intrinsics(self, field: str, value: float) -> None:
        values = {"fx": 500.0, "fy": 500.0, "cx": 32.0, "cy": 32.0}
        values[field] = value
        with pytest.raises(ValueError, match=rf"CameraIntrinsics\.{field} must be finite"):
            locus.CameraIntrinsics(
                fx=values["fx"], fy=values["fy"], cx=values["cx"], cy=values["cy"]
            )


# ---------------------------------------------------------------------------
# §2 CameraIntrinsics construction
# ---------------------------------------------------------------------------


class TestCameraIntrinsicsPinhole:
    """Pinhole accepts any ``dist_coeffs`` length (including empty) — always available."""

    @pytest.mark.parametrize("n", [0, 5, 100])
    def test_pinhole_accepts_any_length(self, n: int) -> None:
        k = locus.CameraIntrinsics(
            fx=500.0,
            fy=500.0,
            cx=32.0,
            cy=32.0,
            distortion_model=locus.DistortionModel.Pinhole,
            dist_coeffs=[0.0] * n,
        )
        assert k.distortion_model == locus.DistortionModel.Pinhole

    def test_pinhole_accepts_none(self) -> None:
        k = locus.CameraIntrinsics(
            fx=500.0,
            fy=500.0,
            cx=32.0,
            cy=32.0,
            distortion_model=locus.DistortionModel.Pinhole,
        )
        assert k.distortion_model == locus.DistortionModel.Pinhole


@pytest.mark.skipif(
    not locus.HAS_NON_RECTIFIED,
    reason="requires `non_rectified` Cargo feature",
)
class TestCameraIntrinsicsBrownConrady:
    """BrownConrady requires exactly 5 dist_coeffs."""

    @pytest.mark.parametrize("n", [0, 3, 4, 6])
    def test_wrong_length_rejected(self, n: int) -> None:
        with pytest.raises(ValueError, match=r"BrownConrady requires exactly 5 dist_coeffs"):
            locus.CameraIntrinsics(
                fx=500.0,
                fy=500.0,
                cx=32.0,
                cy=32.0,
                distortion_model=locus.DistortionModel.BrownConrady,
                dist_coeffs=[0.0] * n,
            )

    def test_valid_length_accepted(self) -> None:
        k = locus.CameraIntrinsics(
            fx=500.0,
            fy=500.0,
            cx=32.0,
            cy=32.0,
            distortion_model=locus.DistortionModel.BrownConrady,
            dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
        assert len(k.dist_coeffs) == 5


@pytest.mark.skipif(
    not locus.HAS_NON_RECTIFIED,
    reason="requires `non_rectified` Cargo feature",
)
class TestCameraIntrinsicsKannalaBrandt:
    """KannalaBrandt requires exactly 4 dist_coeffs."""

    @pytest.mark.parametrize("n", [0, 3, 5, 6])
    def test_wrong_length_rejected(self, n: int) -> None:
        with pytest.raises(ValueError, match=r"KannalaBrandt requires exactly 4 dist_coeffs"):
            locus.CameraIntrinsics(
                fx=500.0,
                fy=500.0,
                cx=32.0,
                cy=32.0,
                distortion_model=locus.DistortionModel.KannalaBrandt,
                dist_coeffs=[0.0] * n,
            )

    def test_valid_length_accepted(self) -> None:
        k = locus.CameraIntrinsics(
            fx=500.0,
            fy=500.0,
            cx=32.0,
            cy=32.0,
            distortion_model=locus.DistortionModel.KannalaBrandt,
            dist_coeffs=[0.0, 0.0, 0.0, 0.0],
        )
        assert len(k.dist_coeffs) == 4


# ---------------------------------------------------------------------------
# §3 DetectorConfig validation (ConfigError variants reachable from Python)
# ---------------------------------------------------------------------------


class TestDetectorConfigValidation:
    """A0.4 §3 — ``DetectorConfig`` validation reachable from Python.

    Validation now lives in the Pydantic model (``locus._config``); invalid
    values raise :class:`pydantic.ValidationError` before the FFI boundary.
    Rust's ``DetectorConfig::validate`` still acts as a defence-in-depth gate
    and is exercised by the ``locus-core`` unit tests.
    """

    @staticmethod
    def _override(group: str, field: str, value: object) -> dict:
        base = locus.DetectorConfig.from_profile("standard").model_dump()
        base[group][field] = value
        return base

    def test_tile_size_too_small(self) -> None:
        bad = self._override("threshold", "tile_size", 1)
        with pytest.raises(Exception, match=r"tile_size"):
            locus.DetectorConfig.model_validate(bad)

    def test_decimation_zero(self) -> None:
        with pytest.raises(ValueError, match=r"decimation"):
            locus.Detector(decimation=0)

    def test_upscale_factor_zero(self) -> None:
        bad = self._override("quad", "upscale_factor", 0)
        with pytest.raises(Exception, match=r"upscale_factor"):
            locus.DetectorConfig.model_validate(bad)

    @pytest.mark.parametrize(
        "min_ratio",
        [-0.1, 0.99],
        ids=["below_zero", "above_default_max"],
    )
    def test_invalid_fill_ratio(self, min_ratio: float) -> None:
        bad = self._override("quad", "min_fill_ratio", min_ratio)
        with pytest.raises(Exception, match=r"(min_fill_ratio|greater than)"):
            locus.DetectorConfig.model_validate(bad)

    def test_edlines_erf_incompatible(self) -> None:
        base = locus.DetectorConfig.from_profile("standard").model_dump()
        base["quad"]["extraction_mode"] = "EdLines"
        base["decoder"]["refinement_mode"] = "Erf"
        with pytest.raises(Exception, match=r"EdLines"):
            locus.DetectorConfig.model_validate(base)

    def test_edlines_soft_incompatible(self) -> None:
        base = locus.DetectorConfig.from_profile("standard").model_dump()
        base["quad"]["extraction_mode"] = "EdLines"
        base["decoder"]["refinement_mode"] = "Gwlf"
        base["decoder"]["decode_mode"] = "Soft"
        with pytest.raises(Exception, match=r"EdLines"):
            locus.DetectorConfig.model_validate(base)


# ---------------------------------------------------------------------------
# §4 Detector constructor
# ---------------------------------------------------------------------------


class TestTagFamilyCoercion:
    """A0.4 §4 row 1 — ``tag_family_from_i32`` rejects out-of-range discriminants."""

    @pytest.mark.parametrize("bad", [5, -1, 999])
    def test_invalid_discriminant_rejected(self, bad: int) -> None:
        # Intentionally bypassing the enum to exercise the i32-coercion gate.
        with pytest.raises(ValueError, match=r"Invalid TagFamily value"):
            locus.Detector(families=[bad])  # type: ignore[list-item]

    def test_valid_enum_accepted(self) -> None:
        det = locus.Detector(families=[locus.TagFamily.AprilTag36h11])
        # Sanity: detector is live.
        batch = det.detect(_valid_img())
        assert hasattr(batch, "ids")


# ---------------------------------------------------------------------------
# §5 detect() vs detect_concurrent() symmetry
# ---------------------------------------------------------------------------


class TestDetectVsConcurrentSymmetry:
    """A0.4 §5 — the two paths must differ only where A0.4 documents they do."""

    def test_dtype_messages_diverge(self, detector: locus.Detector) -> None:
        """Single-frame and concurrent paths emit distinct messages — tested here
        so a future path-convergence refactor is a visible, deliberate change.
        """
        bad = np.zeros(_VALID_SHAPE, dtype=np.float32)

        with pytest.raises(ValueError, match=r"Input image") as single_exc:
            detector.detect(bad)
        with pytest.raises(ValueError, match=r"Frame 0") as concurrent_exc:
            detector.detect_concurrent([bad])

        assert "Input image" in str(single_exc.value)
        assert "Frame 0" in str(concurrent_exc.value)

    def test_telemetry_asymmetry(self, detector: locus.Detector) -> None:
        """``detect()`` may carry telemetry; ``detect_concurrent()`` never does."""
        img = _valid_img()

        single = detector.detect(img)
        assert hasattr(single, "telemetry")  # field exists (may be None)

        batch = detector.detect_concurrent([img])
        assert len(batch) == 1
        assert batch[0].telemetry is None  # documented invariant


class TestRejectedFunnelStatus:
    """Per-rejected-quad funnel status crosses the FFI as a uint8 array
    aligned with ``rejected_corners`` along axis 0. Codes are defined by
    :class:`locus.FunnelStatus`.
    """

    _VALID_CODES = {int(r) for r in locus.FunnelStatus}

    def test_shape_matches_rejected_corners_single(self, detector: locus.Detector) -> None:
        batch = detector.detect(_valid_img())
        assert batch.rejected_corners is not None
        assert batch.rejected_funnel_status is not None
        assert batch.rejected_funnel_status.dtype == np.uint8
        assert batch.rejected_funnel_status.shape == (batch.rejected_corners.shape[0],)

    def test_shape_matches_rejected_corners_concurrent(self, detector: locus.Detector) -> None:
        batches = detector.detect_concurrent([_valid_img(), _valid_img()])
        assert len(batches) == 2
        for b in batches:
            assert b.rejected_funnel_status is not None
            assert b.rejected_funnel_status.dtype == np.uint8
            assert b.rejected_corners is not None
            assert b.rejected_funnel_status.shape == (b.rejected_corners.shape[0],)

    def test_codes_are_in_enum_range(self, detector: locus.Detector) -> None:
        """Every emitted code maps to a known ``FunnelStatus`` variant.

        The fixture (random noise on a 64×64 image) is not guaranteed to
        produce any rejections — the assertion is then vacuously true. The
        defense fires when a future Rust variant is added without a Python
        mirror, in which case an unknown code would surface here.
        """
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=_VALID_SHAPE, dtype=np.uint8)
        batch = detector.detect(img)
        assert batch.rejected_funnel_status is not None
        for code in batch.rejected_funnel_status.tolist():
            assert code in self._VALID_CODES, f"unknown funnel status code: {code}"

    def test_enum_values_match_rust(self) -> None:
        """Numeric values must match ``locus_core::batch::FunnelStatus`` (#[repr(u8)])."""
        assert int(locus.FunnelStatus.NoneReason) == 0
        assert int(locus.FunnelStatus.PassedContrast) == 1
        assert int(locus.FunnelStatus.RejectedContrast) == 2
        assert int(locus.FunnelStatus.RejectedSampling) == 3
