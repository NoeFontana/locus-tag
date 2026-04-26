from dataclasses import dataclass
from typing import Any

import numpy as np

from ._config import (
    DecoderConfig,
    DetectOptions,
    DetectorConfig,
    PoseConfig,
    ProfileName,
    QuadConfig,
    SegmentationConfig,
    ThresholdConfig,
)
from .locus import (
    AprilGrid,
    BoardEstimateResult,
    CameraIntrinsics,
    CharucoBoard,
    CharucoEstimateResult,
    CharucoTelemetryResult,
    CornerRefinementMode,
    DecodeMode,
    DetectionResult,
    DetectorBuilder,
    EdLinesImbalanceGatePolicy,
    PipelineTelemetryResult,
    PoseEstimationMode,
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
    DistortionModel = _RustDistortionModel
else:

    class _LeanDistortionModelMeta(type):
        _STRIPPED = ("BrownConrady", "KannalaBrandt")

        def __getattr__(cls, name: str) -> Any:
            if name in cls._STRIPPED:
                raise LocusFeatureError(
                    f"DistortionModel.{name} is unavailable.\n\n{_DISTORTION_REMEDIATION}"
                )
            raise AttributeError(name)

    class DistortionModel(metaclass=_LeanDistortionModelMeta):  # type: ignore[no-redef]
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
            config=config._to_ffi_config(),
            decimation=decimation,
            threads=threads,
            families=[int(f) for f in families],
        )

    def config(self) -> DetectorConfig:
        """Returns the current detector configuration as a nested model."""
        raw = self._inner.config()
        return DetectorConfig(
            threshold=ThresholdConfig(
                tile_size=raw.threshold_tile_size,
                min_range=raw.threshold_min_range,
                enable_sharpening=raw.enable_sharpening,
                enable_adaptive_window=raw.enable_adaptive_window,
                min_radius=raw.threshold_min_radius,
                max_radius=raw.threshold_max_radius,
                constant=raw.adaptive_threshold_constant,
                gradient_threshold=raw.adaptive_threshold_gradient_threshold,
            ),
            quad=QuadConfig(
                min_area=raw.quad_min_area,
                max_aspect_ratio=raw.quad_max_aspect_ratio,
                min_fill_ratio=raw.quad_min_fill_ratio,
                max_fill_ratio=raw.quad_max_fill_ratio,
                min_edge_length=raw.quad_min_edge_length,
                min_edge_score=raw.quad_min_edge_score,
                subpixel_refinement_sigma=raw.subpixel_refinement_sigma,
                upscale_factor=raw.upscale_factor,
                max_elongation=raw.quad_max_elongation,
                min_density=raw.quad_min_density,
                extraction_mode=raw.quad_extraction_mode,
                edlines_imbalance_gate=raw.edlines_imbalance_gate,
            ),
            decoder=DecoderConfig(
                min_contrast=raw.decoder_min_contrast,
                refinement_mode=raw.refinement_mode,
                decode_mode=raw.decode_mode,
                max_hamming_error=raw.max_hamming_error,
                gwlf_transversal_alpha=raw.gwlf_transversal_alpha,
            ),
            pose=PoseConfig(
                huber_delta_px=raw.huber_delta_px,
                tikhonov_alpha_max=raw.tikhonov_alpha_max,
                sigma_n_sq=raw.sigma_n_sq,
                structure_tensor_radius=raw.structure_tensor_radius,
            ),
            segmentation=SegmentationConfig(
                connectivity=raw.segmentation_connectivity,
                margin=raw.segmentation_margin,
            ),
        )

    def set_families(self, families: list[TagFamily]):
        """Update the tag families to be detected."""
        family_values = [int(f) for f in families]
        self._inner.set_families(family_values)

    def detect(
        self,
        img: np.ndarray,
        intrinsics: CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
        debug_telemetry: bool = False,
        **kwargs,
    ) -> DetectionBatch:
        """
        Detect tags in the image.

        Args:
            img: Input grayscale image (np.uint8).
            intrinsics: Optional CameraIntrinsics for 3D pose estimation.
            tag_size: Optional physical tag size (meters).
            pose_estimation_mode: Fast or Accurate.

        Returns:
            A vectorized DetectionBatch object.
        """
        if img.dtype != np.uint8:
            raise ValueError(f"Input image must be uint8, got {img.dtype}")

        raw = self._inner.detect(
            img,
            intrinsics=intrinsics,
            tag_size=tag_size,
            pose_estimation_mode=pose_estimation_mode,
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
            telemetry=telemetry,
        )

    def detect_concurrent(
        self,
        frames: list[np.ndarray],
        intrinsics: CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ) -> list[DetectionBatch]:
        """
        Detect tags in multiple frames concurrently.

        Releases the GIL for the entire parallel section. Telemetry and
        rejected-corner data are not available via this method.

        Args:
            frames: List of grayscale uint8 images.
            intrinsics: Optional CameraIntrinsics for 3D pose estimation.
            tag_size: Optional physical tag size (meters).
            pose_estimation_mode: Fast or Accurate.

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
            pose_estimation_mode=pose_estimation_mode,
        )

        return [
            DetectionBatch(
                ids=r.ids,
                corners=r.corners,
                error_rates=r.error_rates,
                poses=r.poses,
                rejected_corners=r.rejected_corners,
                rejected_error_rates=r.rejected_error_rates,
            )
            for r in raw_results
        ]


__all__ = [
    "HAS_NON_RECTIFIED",
    "AprilGrid",
    "BoardEstimateResult",
    "BoardEstimator",
    "CameraIntrinsics",
    "CharucoBoard",
    "CharucoEstimateResult",
    "CharucoRefiner",
    "CharucoTelemetryResult",
    "CornerRefinementMode",
    "DecodeMode",
    "DetectOptions",
    "DetectionBatch",
    "DetectionResult",
    "Detector",
    "DetectorBuilder",
    "DetectorConfig",
    "DistortionModel",
    "EdLinesImbalanceGatePolicy",
    "LocusFeatureError",
    "PipelineTelemetryResult",
    "Pose",
    "PoseEstimationMode",
    "QuadExtractionMode",
    "SegmentationConnectivity",
    "TagFamily",
    "init_tracy",
]
