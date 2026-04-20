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
    """§7 follow-up: no runtime check that intrinsics are consistent with the image.

    All three cases accept silently today and silently produce garbage poses.
    A1.2 will raise at ``Detector.detect`` when intrinsics are invalid.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="A1.2 will validate (cx, cy) against image bounds",
    )
    def test_principal_point_outside_image(self, detector: locus.Detector) -> None:
        img = _valid_img()
        bad = locus.CameraIntrinsics(fx=500.0, fy=500.0, cx=5000.0, cy=5000.0)
        with pytest.raises(ValueError, match=r"principal point|cx|cy"):
            detector.detect(img, intrinsics=bad, tag_size=0.05)

    @pytest.mark.xfail(
        strict=True,
        reason="A1.2 will require fx > 0 and fy > 0",
    )
    def test_non_positive_focal_length(self, detector: locus.Detector) -> None:
        img = _valid_img()
        bad = locus.CameraIntrinsics(fx=-1.0, fy=500.0, cx=32.0, cy=32.0)
        with pytest.raises(ValueError, match=r"fx|focal"):
            detector.detect(img, intrinsics=bad, tag_size=0.05)

    @pytest.mark.xfail(
        strict=True,
        reason="A1.2 will reject non-finite intrinsics",
    )
    def test_non_finite_intrinsics(self, detector: locus.Detector) -> None:
        img = _valid_img()
        bad = locus.CameraIntrinsics(fx=float("nan"), fy=500.0, cx=32.0, cy=32.0)
        with pytest.raises(ValueError, match=r"finite|nan|inf"):
            detector.detect(img, intrinsics=bad, tag_size=0.05)


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
    """A0.4 §3 — every ``ConfigError`` reachable from ``Detector(**kwargs)``.

    Two variants are NOT reachable from the Python kwargs path today —
    ``InvalidEdgeLength`` (``quad_min_edge_length``) and
    ``InvalidStructureTensorRadius`` (``structure_tensor_radius``) — because
    ``create_detector`` in ``lib.rs`` doesn't forward those kwargs. They are
    covered by the Rust-side unit tests in ``locus-core``.
    """

    def test_tile_size_too_small(self) -> None:
        with pytest.raises(ValueError, match=r"threshold_tile_size must be >= 2"):
            locus.Detector(threshold_tile_size=1)

    def test_decimation_zero(self) -> None:
        with pytest.raises(ValueError, match=r"decimation factor must be >= 1"):
            locus.Detector(decimation=0)

    def test_upscale_factor_zero(self) -> None:
        with pytest.raises(ValueError, match=r"upscale_factor must be >= 1"):
            locus.Detector(upscale_factor=0)

    @pytest.mark.parametrize(
        "min_ratio",
        [-0.1, 0.99],
        ids=["below_zero", "above_default_max"],
    )
    def test_invalid_fill_ratio(self, min_ratio: float) -> None:
        with pytest.raises(ValueError, match=r"fill ratio range invalid"):
            locus.Detector(quad_min_fill_ratio=min_ratio)

    def test_edlines_erf_incompatible(self) -> None:
        with pytest.raises(ValueError, match=r"EdLines \+ Erf refinement are incompatible"):
            locus.Detector(
                quad_extraction_mode=locus.QuadExtractionMode.EdLines,
                refinement_mode=locus.CornerRefinementMode.Erf,
            )

    def test_edlines_soft_incompatible(self) -> None:
        # Must also disable Erf refinement — it is the default, and EdLines+Erf
        # is checked first, so testing Soft alone would hit the Erf gate.
        with pytest.raises(ValueError, match=r"EdLines \+ Soft decoding are incompatible"):
            locus.Detector(
                quad_extraction_mode=locus.QuadExtractionMode.EdLines,
                refinement_mode=locus.CornerRefinementMode.Gwlf,
                decode_mode=locus.DecodeMode.Soft,
            )


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


class TestPresetConnectivity:
    """A0.4 §4 row 2 — Grid preset + 8-connectivity is a soft warning.

    This overlaps ``tests/test_presets.py::test_preset_warnings`` intentionally:
    the FFI contract suite mirrors A0.4 in full so the matrix is
    self-contained for future audits.
    """

    def test_grid_plus_eight_warns(self) -> None:
        with pytest.warns(UserWarning, match=r"Grid preset relies on 4-connectivity"):
            locus.Detector(
                preset=locus.DetectorPreset.Grid,
                segmentation_connectivity=locus.SegmentationConnectivity.Eight,
            )


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
