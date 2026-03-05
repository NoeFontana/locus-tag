import enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class TagFamily(enum.Enum):
    AprilTag36h11 = 0
    AprilTag41h12 = 1
    ArUco4x4_50 = 2
    ArUco4x4_100 = 3

class Pose:
    rotation: List[List[float]]
    translation: List[float]
    def __init__(self) -> None: ...

class Detector:
    def __init__(
        self,
        decimation: int = ...,
        threads: int = ...,
        families: List[TagFamily] = ...,
        **kwargs: Any
    ) -> None: ...
    
    def detect(self, img: np.ndarray) -> Dict[str, np.ndarray]: ...

def init_tracy() -> None: ...
