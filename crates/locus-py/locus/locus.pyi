import enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

class TagFamily(enum.IntEnum):
    AprilTag36h11 = 0
    AprilTag41h12 = 1
    ArUco4x4_50 = 2
    ArUco4x4_100 = 3

class SegmentationConnectivity(enum.IntEnum):
    Four = 0
    Eight = 1

class CornerRefinementMode(enum.IntEnum):
    None_ = 0
    Edge = 1
    GridFit = 2
    Erf = 3

class DecodeMode(enum.IntEnum):
    Hard = 0
    Soft = 1

class PoseEstimationMode(enum.IntEnum):
    Fast = 0
    Accurate = 1

class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None: ...

@dataclass(frozen=True)
class DetectionBatch:
    ids: np.ndarray  # Shape: (N,), Dtype: int32
    corners: np.ndarray  # Shape: (N, 4, 2), Dtype: float32
    error_rates: np.ndarray  # Shape: (N,), Dtype: float32
    poses: Optional[np.ndarray] = None  # Shape: (N, 7), Dtype: float32. [tx, ty, tz, qx, qy, qz, qw]
    
    @property
    def centers(self) -> np.ndarray: ...
    
    def __len__(self) -> int: ...

class Detector:
    def __init__(
        self,
        decimation: int = 1,
        threads: int = 0,
        families: Optional[List[TagFamily]] = None,
        threshold_tile_size: int = 4,
        threshold_min_range: int = 10,
        adaptive_threshold_constant: int = 3,
        quad_min_area: int = 16,
        quad_min_fill_ratio: float = 0.30,
        quad_min_edge_score: float = 0.0,
        decoder_min_contrast: float = 20.0,
        max_hamming_error: int = 2,
        upscale_factor: Optional[int] = None,
        refinement_mode: Optional[Union[CornerRefinementMode, int]] = None,
        decode_mode: Optional[Union[DecodeMode, int]] = None,
        segmentation_connectivity: Optional[Union[SegmentationConnectivity, int]] = None,
        **kwargs: Any
    ) -> None: ...
    
    def detect(
        self, 
        img: np.ndarray, 
        intrinsics: Optional[CameraIntrinsics] = None,
        tag_size: Optional[float] = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast
    ) -> DetectionBatch: ...

def init_tracy() -> None: ...
