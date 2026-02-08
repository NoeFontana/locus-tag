import enum
from typing import Any

class CornerRefinementMode(enum.Enum):
    None_ = 0
    Subpixel = 1
    Erf = 2

class SegmentationConnectivity(enum.Enum):
    Four = 4
    Eight = 8

class TagFamily(enum.Enum):
    AprilTag36h11 = 0
    AprilTag16h5 = 1
    ArUco4x4_50 = 2
    ArUco4x4_100 = 3

class DecodeMode(enum.Enum):
    Hard = 0
    Soft = 1

class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None: ...

class PoseEstimationMode(enum.Enum):
    Fast = 0
    Accurate = 1

class Pose:
    rotation: list[list[float]]
    translation: list[float]
    def __init__(self) -> None: ...

class Detection:
    id: int
    center: list[float]
    corners: list[list[float]]
    hamming: int
    decision_margin: float
    bits: int
    pose: Pose | None
    pose_covariance: list[list[float]] | None
    def __init__(self) -> None: ...

class PipelineStats:
    threshold_ms: float
    segmentation_ms: float
    quad_extraction_ms: float
    decoding_ms: float
    total_ms: float
    num_candidates: int
    num_detections: int
    num_rejected_by_contrast: int
    num_rejected_by_hamming: int
    def __init__(self) -> None: ...

class FullDetectionResult:
    detections: list[Detection]
    candidates: list[Any]
    stats: PipelineStats
    threshold_image: Any
    segmentation_image: Any
    def get_binarized(self) -> Any: ...
    def get_labels(self) -> Any: ...
    def __init__(self) -> None: ...

class Detector:
    enable_sharpening: bool
    def __init__(
        self,
        threshold_tile_size: int = ...,
        threshold_min_range: int = ...,
        enable_bilateral: bool = ...,
        bilateral_sigma_space: float = ...,
        bilateral_sigma_color: float = ...,
        enable_sharpening: bool = ...,
        enable_adaptive_window: bool = ...,
        threshold_min_radius: int = ...,
        threshold_max_radius: int = ...,
        adaptive_threshold_constant: int = ...,
        adaptive_threshold_gradient_threshold: int = ...,
        quad_min_area: int = ...,
        quad_max_aspect_ratio: float = ...,
        quad_min_fill_ratio: float = ...,
        quad_max_fill_ratio: float = ...,
        quad_min_edge_length: float = ...,
        quad_min_edge_score: float = ...,
        subpixel_refinement_sigma: float = ...,
        segmentation_margin: int = ...,
        segmentation_connectivity: SegmentationConnectivity = ...,
        upscale_factor: int = ...,
        decoder_min_contrast: float = ...,
        refinement_mode: CornerRefinementMode = ...,
        decode_mode: DecodeMode = ...,
    ) -> None: ...
    def detect(
        self,
        img: Any,
        decimation: int = ...,
        intrinsics: tuple[float, float, float, float] | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ) -> list[Detection]: ...
    def detect_with_options(
        self,
        img: Any,
        families: list[TagFamily],
        decimation: int = ...,
        intrinsics: tuple[float, float, float, float] | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ) -> list[Detection]: ...
    def detect_with_stats(
        self,
        img: Any,
        decimation: int = ...,
        intrinsics: tuple[float, float, float, float] | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ) -> tuple[list[Detection], PipelineStats]: ...
    def set_families(self, families: list[TagFamily]) -> None: ...
    def extract_candidates(self, img: Any, decimation: int = ...) -> list[Any]: ...
    def detect_full(
        self,
        img: Any,
        decimation: int = ...,
        intrinsics: tuple[float, float, float, float] | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ) -> FullDetectionResult: ...

def detect_tags(img: Any) -> list[Detection]: ...
def detect_tags_with_stats(img: Any) -> tuple[list[Detection], PipelineStats]: ...
def dummy_detect(img: Any) -> list[Detection]: ...
def debug_threshold(img: Any) -> Any: ...
def debug_segmentation(img: Any) -> Any: ...
