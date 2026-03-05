from dataclasses import dataclass
from typing import Optional, List
import enum
import numpy as np

from ._config import DetectOptions, DetectorConfig
from .locus import (
    SegmentationConnectivity,
    CornerRefinementMode,
    DecodeMode,
    PoseEstimationMode,
    TagFamily,
    CameraIntrinsics,
    PyPose as Pose,
    create_detector as _create_detector,
    init_tracy,
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
    poses: Optional[np.ndarray] = None  # Shape: (N, 7), Dtype: float32. [tx, ty, tz, qx, qy, qz, qw]

    @property
    def centers(self) -> np.ndarray:
        """Compute centers from corners: (N, 2)"""
        return np.mean(self.corners, axis=1)

    def __len__(self) -> int:
        return len(self.ids)

class Detector:
    def __init__(self, decimation: int = 1, threads: int = 0, families: list[TagFamily] = None, **kwargs):
        # Map enum to int values for Rust. Rust expects a Vec<i32>.
        if families is None:
            families = [TagFamily.AprilTag36h11]
        
        family_values = [int(f) for f in families]
        
        # Prepare kwargs for Rust by converting enums to ints
        rust_kwargs = {}
        for k, v in kwargs.items():
            # PyO3 enums might not inherit from enum.Enum but are int-convertible
            if hasattr(v, "__int__"):
                rust_kwargs[k] = int(v)
            elif isinstance(v, enum.Enum):
                rust_kwargs[k] = v.value
            else:
                rust_kwargs[k] = v

        self._inner = _create_detector(
            decimation=decimation,
            threads=threads,
            families=family_values,
            **rust_kwargs
        )

    def detect(self, img: np.ndarray, **kwargs) -> DetectionBatch:
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
            
        res_dict = self._inner.detect(img, **kwargs)
        return DetectionBatch(**res_dict)

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
