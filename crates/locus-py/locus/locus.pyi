import enum
from typing import Any

import numpy as np

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
    def rotation(self) -> np.ndarray: ...  # 3x3
    @property
    def translation(self) -> np.ndarray: ...  # 3x1

class Detector:
    def detect(
        self,
        img: np.ndarray,
        intrinsics: CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ) -> dict[str, Any]: ...

def create_detector(
    decimation: int = 1,
    threads: int = 0,
    families: list[int] = [],
    **kwargs: Any,
) -> Detector: ...
def init_tracy() -> None: ...
