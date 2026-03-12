import enum
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._config import DetectOptions, DetectorConfig
from .locus import (
    CameraIntrinsics,
    CornerRefinementMode,
    DecodeMode,
    PoseEstimationMode,
    SegmentationConnectivity,
    TagFamily,
    init_tracy,
)
from .locus import (
    PyPose as Pose,
)
from .locus import (
    create_detector as _create_detector,
)
from .locus import (
    fast_config as _fast_config,
)
from .locus import (
    production_config as _production_config,
)


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

    @property
    def centers(self) -> np.ndarray:
        """Compute centers from corners: (N, 2)"""
        return np.mean(self.corners, axis=1)

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class PipelineTelemetry:
    """
    Zero-copy intermediate artifacts from the detection pipeline.

    WARNING: These arrays are ephemeral. They point directly into the Rust
    arena memory and become invalid the moment another frame is processed
    by the same Detector instance.
    """

    binarized: np.ndarray  # Shape: (H, W), Dtype: uint8
    threshold_map: np.ndarray  # Shape: (H, W), Dtype: uint8


class Detector:
    def __init__(
        self,
        decimation: int = 1,
        threads: int = 0,
        families: list[TagFamily] | None = None,
        threshold_tile_size: int = 8,
        threshold_min_range: int = 10,
        adaptive_threshold_constant: int = 0,
        quad_min_area: int = 16,
        quad_min_fill_ratio: float = 0.10,
        quad_min_edge_score: float = 4.0,
        decoder_min_contrast: float = 20.0,
        max_hamming_error: int = 2,
        **kwargs: Any,
    ):
        # Map enum to int values for Rust. Rust expects a Vec<i32>.
        if families is None:
            families = [TagFamily.AprilTag36h11]

        family_values = [int(f) for f in families]

        # Merge explicit args into kwargs for create_detector
        rust_kwargs: dict[str, Any] = {
            "threshold_tile_size": threshold_tile_size,
            "threshold_min_range": threshold_min_range,
            "adaptive_threshold_constant": adaptive_threshold_constant,
            "quad_min_area": quad_min_area,
            "quad_min_fill_ratio": quad_min_fill_ratio,
            "quad_min_edge_score": quad_min_edge_score,
            "decoder_min_contrast": decoder_min_contrast,
            "max_hamming_error": max_hamming_error,
        }
        rust_kwargs.update(kwargs)

        # Prepare kwargs for Rust by converting enums to ints
        final_rust_kwargs: dict[str, Any] = {}
        for k, v in rust_kwargs.items():
            # Boolean types must be preserved
            if isinstance(v, bool):
                final_rust_kwargs[k] = v
            # PyO3 enums might not inherit from enum.Enum but are int-convertible
            elif hasattr(v, "__int__"):
                final_rust_kwargs[k] = int(v)
            elif isinstance(v, enum.Enum):
                final_rust_kwargs[k] = v.value
            else:
                final_rust_kwargs[k] = v

        self._inner = _create_detector(
            decimation=decimation, threads=threads, families=family_values, **final_rust_kwargs
        )

    @staticmethod
    def production_config() -> "Detector":
        """Create a detector with high-fidelity production defaults."""
        d = Detector.__new__(Detector)
        d._inner = _production_config()
        return d

    @staticmethod
    def fast_config() -> "Detector":
        """Create a detector with low-latency defaults."""
        d = Detector.__new__(Detector)
        d._inner = _fast_config()
        return d

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

        res_dict = self._inner.detect(
            img,
            intrinsics=intrinsics,
            tag_size=tag_size,
            pose_estimation_mode=pose_estimation_mode,
            debug_telemetry=debug_telemetry,
            **kwargs,
        )

        tel_res = res_dict.pop("telemetry", None)
        telemetry = None
        if tel_res is not None:
            telemetry = PipelineTelemetry(**tel_res)

        return DetectionBatch(**res_dict, telemetry=telemetry)


__all__ = [
    "Detector",
    "TagFamily",
    "SegmentationConnectivity",
    "CornerRefinementMode",
    "DecodeMode",
    "PoseEstimationMode",
    "CameraIntrinsics",
    "Pose",
    "DetectorConfig",
    "DetectOptions",
    "DetectionBatch",
    "init_tracy",
]
