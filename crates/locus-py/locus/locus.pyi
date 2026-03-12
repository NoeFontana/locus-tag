import enum
from typing import Any

import numpy as np

class TagFamily(enum.IntEnum):
    AprilTag16h5 = 0
    AprilTag36h11 = 1
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
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None: ...
    @property
    def fx(self) -> float: ...
    @property
    def fy(self) -> float: ...
    @property
    def cx(self) -> float: ...
    @property
    def cy(self) -> float: ...

class PyPose:
    @property
    def quaternion(self) -> list[float]: ...  # [x, y, z, w]
    @property
    def translation(self) -> list[float]: ...  # [x, y, z]

class PyDetectorConfig:
    threshold_tile_size: int
    threshold_min_range: int
    enable_bilateral: bool
    bilateral_sigma_space: float
    bilateral_sigma_color: float
    enable_sharpening: bool
    enable_adaptive_window: bool
    threshold_min_radius: int
    threshold_max_radius: int
    adaptive_threshold_constant: int
    adaptive_threshold_gradient_threshold: int
    quad_min_area: int
    quad_max_aspect_ratio: float
    quad_min_fill_ratio: float
    quad_max_fill_ratio: float
    quad_min_edge_length: float
    quad_min_edge_score: float
    subpixel_refinement_sigma: float
    segmentation_margin: int
    segmentation_connectivity: SegmentationConnectivity
    upscale_factor: int
    decoder_min_contrast: float
    refinement_mode: CornerRefinementMode
    decode_mode: DecodeMode
    max_hamming_error: int

class Detector:
    def detect(
        self,
        img: np.ndarray,
        intrinsics: CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
        debug_telemetry: bool = False,
    ) -> dict[str, Any]: ...
    def config(self) -> PyDetectorConfig: ...
    def set_families(self, families: list[int]) -> None: ...

def create_detector(
    decimation: int = 1,
    threads: int = 0,
    families: list[int] = [],
    **kwargs: Any,
) -> Detector: ...
def production_config() -> Detector: ...
def fast_config() -> Detector: ...
def init_tracy() -> None: ...
