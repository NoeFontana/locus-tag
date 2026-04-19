import enum
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._config import DetectOptions, DetectorConfig
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
    DetectorPreset,
    PipelineTelemetryResult,
    PoseEstimationMode,
    QuadExtractionMode,
    SegmentationConnectivity,
    TagFamily,
    init_tracy,
)
from .locus import BoardEstimator as _BoardEstimator
from .locus import CharucoRefiner as _CharucoRefiner
from .locus import (
    PyPose as Pose,
)
from .locus import (
    create_detector as _create_detector,
)


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
    def __init__(
        self,
        decimation: int | None = None,
        threads: int | None = None,
        families: list[TagFamily] | None = None,
        preset: DetectorPreset | None = None,
        threshold_tile_size: int | None = None,
        threshold_min_range: int | None = None,
        adaptive_threshold_constant: int | None = None,
        quad_min_area: int | None = None,
        quad_min_fill_ratio: float | None = None,
        quad_min_edge_score: float | None = None,
        decoder_min_contrast: float | None = None,
        max_hamming_error: int | None = None,
        **kwargs: Any,
    ):
        # Default family if none provided (legacy behavior)
        if families is None:
            families = [TagFamily.AprilTag36h11]

        family_values = [int(f) for f in families]

        # Collect explicit config arguments
        config_args = {
            "threshold_tile_size": threshold_tile_size,
            "threshold_min_range": threshold_min_range,
            "adaptive_threshold_constant": adaptive_threshold_constant,
            "quad_min_area": quad_min_area,
            "quad_min_fill_ratio": quad_min_fill_ratio,
            "quad_min_edge_score": quad_min_edge_score,
            "decoder_min_contrast": decoder_min_contrast,
            "max_hamming_error": max_hamming_error,
        }

        # Merge explicit config with additional kwargs
        merged_kwargs = {**config_args, **kwargs}

        # Validate preset constraints
        if (
            preset == DetectorPreset.Grid
            and merged_kwargs.get("segmentation_connectivity") == SegmentationConnectivity.Eight
        ):
            warnings.warn(
                "Grid preset relies on 4-connectivity; enforcing 8-connectivity reduces touching-tag separation.",
                stacklevel=2,
            )

        # Prepare kwargs for Rust by converting enums and filtering None
        final_rust_kwargs: dict[str, Any] = {}
        for k, v in merged_kwargs.items():
            if v is None:
                continue
            if isinstance(v, bool):
                final_rust_kwargs[k] = v
            elif hasattr(v, "__int__") and not isinstance(v, (int, float)):
                # Handle PyO3 enums and standard enums
                final_rust_kwargs[k] = int(v)
            elif isinstance(v, enum.Enum):
                final_rust_kwargs[k] = v.value
            else:
                final_rust_kwargs[k] = v

        self._inner = _create_detector(
            decimation=decimation,
            threads=threads,
            families=family_values,
            preset=preset,
            **final_rust_kwargs,
        )

    @staticmethod
    def standard_config() -> "Detector":
        """Create a detector with high-fidelity production defaults."""
        return Detector(preset=DetectorPreset.Standard)

    def config(self) -> DetectorConfig:
        """Returns the current detector configuration."""
        raw = self._inner.config()
        # Create a dictionary of all fields to populate the Pydantic model
        fields = {field: getattr(raw, field) for field in DetectorConfig.model_fields}
        return DetectorConfig(**fields)

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
    "DetectorPreset",
    "PipelineTelemetryResult",
    "PoseEstimationMode",
    "Pose",
    "QuadExtractionMode",
    "SegmentationConnectivity",
    "TagFamily",
    "init_tracy",
]
