from ._config import DetectOptions, DetectorConfig
from .locus import (
    SegmentationConnectivity as _SegmentationConnectivity,
    CornerRefinementMode as _CornerRefinementMode,
    DecodeMode as _DecodeMode,
    PoseEstimationMode as _PoseEstimationMode,
    CameraIntrinsics,
    PyPose as Pose,
    create_detector as _create_detector,
    init_tracy,
)
import enum
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

class Result:
    """
    Vectorized detection results.
    """
    def __init__(self, data: dict):
        self._data = data

    @property
    def ids(self) -> np.ndarray:
        return self._data["ids"]

    @property
    def centers(self) -> np.ndarray:
        return self._data["centers"]

    @property
    def corners(self) -> np.ndarray:
        return self._data["corners"]

    @property
    def hamming(self) -> np.ndarray:
        return self._data["hamming"]

    @property
    def decision_margin(self) -> np.ndarray:
        return self._data["decision_margin"]

    def __len__(self):
        return len(self.ids)

    def to_dict(self) -> dict:
        return self._data

    def to_list(self) -> list:
        n = len(self)
        res = []
        for i in range(n):
            d = {
                "id": self.ids[i],
                "center": self.centers[i],
                "corners": self.corners[i],
                "hamming": self.hamming[i],
                "decision_margin": self.decision_margin[i],
            }
            res.append(d)
        return res

class Detector:
    def __init__(self, decimation: int = 1, threads: int = 0, families: list[TagFamily] = None, **kwargs):
        # Map enum to int values for Rust
        family_values = [int(f) for f in families] if families else []
        
        # Prepare kwargs for Rust by converting enums to ints
        rust_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, enum.Enum):
                rust_kwargs[k] = int(v)
            else:
                rust_kwargs[k] = v

        self._inner = _create_detector(
            decimation=decimation,
            threads=threads,
            families=family_values,
            **rust_kwargs
        )

    def detect(self, img: np.ndarray) -> Result:
        res_dict = self._inner.detect(img)
        return Result(res_dict)

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
    "Result",
    "init_tracy",
]
