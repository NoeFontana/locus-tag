from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from ._config import (
    AdaptivePpbConfig,
    DetectOptions,
    DetectorConfig,
    ProfileName,
    QuadExtractionPolicy,
)
from .locus import (
    AprilGrid,
    BoardEstimateResult,
    CameraIntrinsics,
    CharucoBoard,
    CharucoEstimateResult,
    CharucoTelemetryResult,
    CornerRefinementMode,
    DetectionResult,
    DetectorBuilder,
    EdLinesImbalanceGatePolicy,
    PipelineTelemetryResult,
    QuadExtractionMode,
    SegmentationConnectivity,
    TagFamily,
    init_tracy,
)
from .locus import BoardEstimator as _BoardEstimator
from .locus import CharucoRefiner as _CharucoRefiner
from .locus import DistortionModel as _RustDistortionModel
from .locus import (
    PyPose as Pose,
)
from .locus import (
    _create_detector_from_config as _create_detector_from_config,
)

HAS_NON_RECTIFIED = hasattr(_RustDistortionModel, "BrownConrady")
"""True when this wheel was built with the `non_rectified` Cargo feature."""


class LocusFeatureError(RuntimeError):
    """Raised when an operation requires a Cargo feature this wheel was not built with."""


_DISTORTION_REMEDIATION = (
    "Distortion models require the `non_rectified` Cargo feature, which is not "
    "compiled into this wheel. Reinstall from source:\n"
    '    MATURIN_PEP517_ARGS="--features locus-py/non_rectified" \\\n'
    "        pip install --no-binary=locus-tag --force-reinstall locus-tag\n"
    "See the 'Install with distortion support' how-to for details."
)


if HAS_NON_RECTIFIED:
    DistortionModel = _RustDistortionModel  # pyright: ignore[reportAssignmentType]
else:

    class _LeanDistortionModelMeta(type):
        _STRIPPED = ("BrownConrady", "KannalaBrandt")

        def __getattr__(cls, name: str) -> Any:
            if name in cls._STRIPPED:
                raise LocusFeatureError(
                    f"DistortionModel.{name} is unavailable.\n\n{_DISTORTION_REMEDIATION}"
                )
            raise AttributeError(name)

    class DistortionModel(metaclass=_LeanDistortionModelMeta):
        """Lean-build placeholder for the compiled `DistortionModel` enum.

        Exposes only the variants compiled into this wheel. Accessing a variant
        stripped by the lean build (`BrownConrady`, `KannalaBrandt`) raises
        `LocusFeatureError` with a source-install recipe.
        """

        Pinhole = _RustDistortionModel.Pinhole


class BoardEstimator:
    """Estimator for multi-tag board poses (AprilGrid)."""

    def __init__(self, board: AprilGrid) -> None:
        self._inner = _BoardEstimator(board)

    @classmethod
    def from_charuco(cls, board: CharucoBoard) -> "BoardEstimator":
        instance = cls.__new__(cls)
        instance._inner = _BoardEstimator.from_charuco(board)
        return instance

    def estimate(
        self,
        detector: "Detector",
        img: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> BoardEstimateResult:
        return self._inner.estimate(detector._inner, img, intrinsics)


class CharucoRefiner:
    """Extracts ChAruco saddle points and estimates board pose."""

    def __init__(self, board: CharucoBoard) -> None:
        self._inner = _CharucoRefiner(board)

    def estimate(
        self,
        detector: "Detector",
        img: np.ndarray,
        intrinsics: CameraIntrinsics,
        debug_telemetry: bool = False,
    ) -> CharucoEstimateResult:
        return self._inner.estimate(detector._inner, img, intrinsics, debug_telemetry)


class FunnelStatus(IntEnum):
    """Status of a candidate in the fast-path decoding funnel.

    Mirrors the Rust ``locus_core::batch::FunnelStatus`` enum. Values match
    the ``u8`` codes stored in :attr:`DetectionBatch.rejected_funnel_status`,
    where only the ``Rejected*`` variants appear in practice.
    """

    NoneReason = 0
    """Candidate had not been processed by the funnel."""

    PassedContrast = 1
    """Passed the O(1) contrast gate. Never appears in ``rejected_funnel_status``."""

    RejectedContrast = 2
    """Rejected by the O(1) contrast gate — geometry-only failure."""

    RejectedSampling = 3
    """Rejected during homography DDA / SIMD sampling / Hamming check."""


@dataclass(frozen=True)
class DetectionBatch:
    """
    Vectorized detection results.

    This dataclass contains parallel NumPy arrays representing a batch of detections.
    """

    ids: np.ndarray  # Shape: (N,), Dtype: int32
    corners: np.ndarray  # Shape: (N, 4, 2), Dtype: float32
    error_rates: np.ndarray  # Shape: (N,), Dtype: float32
    poses: np.ndarray | None = None  # Shape: (N, 7), Dtype: float32. [tx, ty, tz, qx, qy, qz, qw]
    telemetry: "PipelineTelemetry | None" = None
    rejected_corners: np.ndarray | None = None  # Shape: (M, 4, 2), Dtype: float32
    rejected_error_rates: np.ndarray | None = None  # Shape: (M,), Dtype: float32
    # Shape: (M,), Dtype: uint8. Codes from `FunnelStatus`.
    rejected_funnel_status: np.ndarray | None = None

    @property
    def centers(self) -> np.ndarray:
        """Compute centers from corners: (N, 2)"""
        return np.mean(self.corners, axis=1)

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class PipelineTelemetry:
    """
    Intermediate artifacts captured during the detection pipeline.

    The underlying pixel data is copied out of the Rust arena at result
    construction time, so these arrays remain valid across frames.
    """

    binarized: np.ndarray  # Shape: (H, W), Dtype: uint8
    threshold_map: np.ndarray  # Shape: (H, W), Dtype: uint8
    gwlf_fallback_count: int = 0
    gwlf_avg_delta: float = 0.0
    subpixel_jitter: np.ndarray | None = None  # Shape: (N, 4, 2), Dtype: float32
    reprojection_errors: np.ndarray | None = None  # Shape: (N,), Dtype: float32


class Detector:
    """High-level detector.

    Construction:
        ``Detector(profile="standard")`` — load a shipped JSON profile by name.
        ``Detector(config=my_cfg)`` — use a pre-built :class:`DetectorConfig`.
        ``Detector()`` — equivalent to ``profile="standard"``.

    Per-call orchestration options (``decimation``, ``threads``, ``families``)
    stay outside the profile because they describe *how* the detector is
    invoked, not *what* it looks for.
    """

    def __init__(
        self,
        profile: ProfileName | None = None,
        config: DetectorConfig | None = None,
        *,
        decimation: int | None = None,
        threads: int | None = None,
        families: list[TagFamily] | None = None,
    ) -> None:
        if profile is not None and config is not None:
            raise ValueError("Pass either `profile` or `config`, not both.")

        if config is None:
            config = DetectorConfig.from_profile(profile or "standard")

        if families is None:
            families = [TagFamily.AprilTag36h11]

        self._inner = _create_detector_from_config(
            config_json=config.model_dump_json(),
            decimation=decimation,
            threads=threads,
            families=[int(f) for f in families],
        )

    def config(self) -> DetectorConfig:
        """Returns the current detector configuration as a nested model.

        Rust serializes its live config into the profile-JSON format and Python
        re-parses it, so the readback is total over every field — no per-field
        transcription to drift out of sync.
        """
        return DetectorConfig.model_validate_json(self._inner.config())

    def set_families(self, families: list[TagFamily]):
        """Update the tag families to be detected."""
        family_values = [int(f) for f in families]
        self._inner.set_families(family_values)

    def detect(
        self,
        img: np.ndarray,
        intrinsics: CameraIntrinsics | None = None,
        tag_size: float | None = None,
        debug_telemetry: bool = False,
        **kwargs,
    ) -> DetectionBatch:
        """
        Detect tags in the image.

        Args:
            img: Input grayscale image (np.uint8).
            intrinsics: Optional CameraIntrinsics for 3D pose estimation.
            tag_size: Optional physical tag size (meters).

        Returns:
            A vectorized DetectionBatch object.
        """
        if img.dtype != np.uint8:
            raise ValueError(f"Input image must be uint8, got {img.dtype}")

        raw = self._inner.detect(
            img,
            intrinsics=intrinsics,
            tag_size=tag_size,
            debug_telemetry=debug_telemetry,
            **kwargs,
        )

        telemetry = None
        if raw.telemetry is not None:
            t = raw.telemetry
            telemetry = PipelineTelemetry(
                binarized=t.binarized,
                threshold_map=t.threshold_map,
                gwlf_fallback_count=t.gwlf_fallback_count,
                gwlf_avg_delta=t.gwlf_avg_delta,
                subpixel_jitter=t.subpixel_jitter,
                reprojection_errors=t.reprojection_errors,
            )

        return DetectionBatch(
            ids=raw.ids,
            corners=raw.corners,
            error_rates=raw.error_rates,
            poses=raw.poses,
            rejected_corners=raw.rejected_corners,
            rejected_error_rates=raw.rejected_error_rates,
            rejected_funnel_status=raw.rejected_funnel_status,
            telemetry=telemetry,
        )

    def detect_concurrent(
        self,
        frames: list[np.ndarray],
        intrinsics: CameraIntrinsics | None = None,
        tag_size: float | None = None,
    ) -> list[DetectionBatch]:
        """
        Detect tags in multiple frames concurrently.

        Releases the GIL for the entire parallel section. Telemetry and
        rejected-corner data are not available via this method.

        Args:
            frames: List of grayscale uint8 images.
            intrinsics: Optional CameraIntrinsics for 3D pose estimation.
            tag_size: Optional physical tag size (meters).

        Returns:
            A list of DetectionBatch, one per input frame, in the same order.
        """
        for i, img in enumerate(frames):
            if img.dtype != np.uint8:
                raise ValueError(f"Frame {i} must be uint8, got {img.dtype}")

        raw_results = self._inner.detect_concurrent(
            frames,
            intrinsics=intrinsics,
            tag_size=tag_size,
        )

        return [
            DetectionBatch(
                ids=r.ids,
                corners=r.corners,
                error_rates=r.error_rates,
                poses=r.poses,
                rejected_corners=r.rejected_corners,
                rejected_error_rates=r.rejected_error_rates,
                rejected_funnel_status=r.rejected_funnel_status,
            )
            for r in raw_results
        ]


__all__ = [
    "HAS_NON_RECTIFIED",
    "AdaptivePpbConfig",
    "AprilGrid",
    "BoardEstimateResult",
    "BoardEstimator",
    "CameraIntrinsics",
    "CharucoBoard",
    "CharucoEstimateResult",
    "CharucoRefiner",
    "CharucoTelemetryResult",
    "CornerRefinementMode",
    "DetectOptions",
    "DetectionBatch",
    "DetectionResult",
    "Detector",
    "DetectorBuilder",
    "DetectorConfig",
    "DistortionModel",
    "EdLinesImbalanceGatePolicy",
    "FunnelStatus",
    "LocusFeatureError",
    "PipelineTelemetryResult",
    "Pose",
    "QuadExtractionMode",
    "QuadExtractionPolicy",
    "SegmentationConnectivity",
    "TagFamily",
    "init_tracy",
]
