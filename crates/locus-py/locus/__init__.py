from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from ._config import (
    AdaptivePpbConfig,
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
    EdLinesImbalanceGatePolicy,
    PipelineTelemetryResult,
    QuadExtractionMode,
    SegmentationConnectivity,
    TagFamily,
    ThresholdMode,
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
    the ``u8`` codes stored in :attr:`DetectionBatch.rejected_funnel_status`.
    """

    NoneReason = 0
    """Candidate had not been processed by the funnel."""

    PassedContrast = 1
    """Passed the O(1) contrast gate. In ``rejected_funnel_status`` this means
    the candidate passed the contrast gate but still failed to decode — the
    Rust pipeline (``crates/locus-core/src/funnel.rs``) never overwrites
    ``funnel_status`` after the funnel gate runs, so a rejected candidate
    carrying this value failed at the later decode stage, not the funnel."""

    RejectedContrast = 2
    """Rejected by the O(1) contrast gate — geometry-only failure."""

    RejectedSampling = 3
    """Reserved for a homography-DDA/SIMD-sampling rejection. No code path in
    ``locus-core`` currently sets this value (verified empirically: 0
    occurrences across a 3378-candidate real-data run) — decode failures show
    up as ``PassedContrast`` above instead. Kept distinct in case a future
    change wires this up."""


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
    subpixel_jitter: np.ndarray | None = None  # Shape: (N, 4, 2), Dtype: float32
    reprojection_errors: np.ndarray | None = None  # Shape: (N,), Dtype: float32


class Detector:
    """High-level detector.

    Construction:
        ``Detector(profile="standard")`` — load a shipped JSON profile by name.
        ``Detector(config=my_cfg)`` — use a pre-built :class:`DetectorConfig`.
        ``Detector()`` — equivalent to ``profile="standard"``.

    Per-call orchestration options (``decimation``, ``threads``, ``families``,
    ``max_concurrent_frames``) stay outside the profile because they describe
    *how* the detector is invoked, not *what* it looks for.

    ``threads`` sets the Rayon worker count. ``0`` or ``None`` (the default)
    uses the global Rayon pool (``RAYON_NUM_THREADS`` / core count); ``n > 0``
    builds one scoped ``n``-thread pool at construction and runs every
    ``detect`` / ``detect_concurrent`` call inside it, bounding this detector's
    CPU footprint. Detection output is identical for every value.

    ``max_concurrent_frames`` sizes the pool of frame contexts that
    :meth:`detect_concurrent` leases (default ``1``); frames beyond it get a
    temporary context each.
    """

    def __init__(
        self,
        profile: ProfileName | None = None,
        config: DetectorConfig | None = None,
        *,
        decimation: int | None = None,
        threads: int | None = None,
        families: list[TagFamily] | None = None,
        max_concurrent_frames: int | None = None,
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
            max_concurrent_frames=max_concurrent_frames,
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


class DetectorBuilder:
    """Fluent builder for :class:`Detector`.

    Carries orchestration only: the configuration (a shipped profile or a
    :class:`DetectorConfig`), tag families, decimation, thread count and the
    concurrent-frame pool size. Detection settings live in
    :class:`DetectorConfig`; edit one and hand it to :meth:`with_config`::

        cfg = locus.DetectorConfig.from_profile("standard")
        cfg.quad.min_area = 400
        detector = (
            locus.DetectorBuilder()
            .with_config(cfg)
            .with_family(locus.TagFamily.AprilTag36h11)
            .with_max_concurrent_frames(8)
            .build()
        )

    :meth:`build` re-validates the configuration through the Pydantic model, so
    a field edited after the config was constructed is checked like any other.
    """

    def __init__(self) -> None:
        self._config: DetectorConfig | None = None
        self._families: list[TagFamily] = []
        self._decimation: int | None = None
        self._threads: int | None = None
        self._max_concurrent_frames: int | None = None

    def with_profile(self, profile: ProfileName) -> "DetectorBuilder":
        """Use a shipped profile (``"standard"``, ``"grid"``, ``"high_accuracy"``)."""
        self._config = DetectorConfig.from_profile(profile)
        return self

    def with_config(self, config: DetectorConfig) -> "DetectorBuilder":
        """Use a :class:`DetectorConfig`. Defaults to the ``standard`` profile."""
        self._config = config
        return self

    def with_family(self, family: TagFamily) -> "DetectorBuilder":
        """Add a tag family to decode. Defaults to ``AprilTag36h11`` when none is added."""
        if family not in self._families:
            self._families.append(family)
        return self

    def with_decimation(self, decimation: int) -> "DetectorBuilder":
        """Decimate the input by this factor before segmentation."""
        self._decimation = decimation
        return self

    def with_threads(self, threads: int) -> "DetectorBuilder":
        """Intra-frame Rayon thread count; ``0`` uses the global pool."""
        self._threads = threads
        return self

    def with_max_concurrent_frames(self, n: int) -> "DetectorBuilder":
        """Size of the frame-context pool used by :meth:`Detector.detect_concurrent`."""
        self._max_concurrent_frames = n
        return self

    def build(self) -> Detector:
        """Validate the configuration and construct the :class:`Detector`."""
        config = self._config or DetectorConfig.from_profile("standard")
        config = DetectorConfig.model_validate_json(config.model_dump_json())
        return Detector(
            config=config,
            decimation=self._decimation,
            threads=self._threads,
            families=list(self._families) or None,
            max_concurrent_frames=self._max_concurrent_frames,
        )


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
    "ThresholdMode",
    "init_tracy",
]
