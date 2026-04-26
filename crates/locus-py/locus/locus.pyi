import enum
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TagFamily(enum.IntEnum):
    AprilTag16h5 = 0
    AprilTag36h11 = 1
    ArUco4x4_50 = 2
    ArUco4x4_100 = 3
    ArUco6x6_250 = 4

class SegmentationConnectivity(enum.IntEnum):
    Four = 0
    Eight = 1

class CornerRefinementMode(enum.IntEnum):
    # The ``None`` variant matches the Rust name but shadows a Python keyword at
    # the attribute level. Access it as ``getattr(CornerRefinementMode, "None")``
    # or via a JSON/string spelling through ``DetectorConfig``.
    None_ = 0  # exposed as ``None`` at runtime — name kept here so mypy can resolve it
    Edge = 1
    Erf = 2
    Gwlf = 3

class DecodeMode(enum.IntEnum):
    Hard = 0
    Soft = 1

class PoseEstimationMode(enum.IntEnum):
    Fast = 0
    Accurate = 1

class QuadExtractionMode(enum.IntEnum):
    ContourRdp = 0
    EdLines = 1

class DistortionModel(enum.IntEnum):
    """Lens distortion model tag for `CameraIntrinsics`.

    `BrownConrady` and `KannalaBrandt` are only available when the `non_rectified`
    Cargo feature is compiled in. Accessing them on a lean wheel raises
    `locus.LocusFeatureError` at runtime — see `locus.HAS_NON_RECTIFIED`.
    """

    Pinhole = 0
    BrownConrady = 1
    KannalaBrandt = 2

# ---------------------------------------------------------------------------
# Config / misc structs
# ---------------------------------------------------------------------------

class CameraIntrinsics:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion_model: DistortionModel = ...,
        dist_coeffs: list[float] | None = ...,
    ) -> None: ...
    distortion_model: DistortionModel
    dist_coeffs: list[float]
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
    max_hamming_error: int | None
    gwlf_transversal_alpha: float
    quad_max_elongation: float
    quad_min_density: float
    quad_extraction_mode: QuadExtractionMode
    huber_delta_px: float
    tikhonov_alpha_max: float
    sigma_n_sq: float
    structure_tensor_radius: int
    edlines_imbalance_gate: bool
    def __init__(
        self,
        *,
        threshold_tile_size: int,
        threshold_min_range: int,
        enable_sharpening: bool,
        enable_adaptive_window: bool,
        threshold_min_radius: int,
        threshold_max_radius: int,
        adaptive_threshold_constant: int,
        adaptive_threshold_gradient_threshold: int,
        quad_min_area: int,
        quad_max_aspect_ratio: float,
        quad_min_fill_ratio: float,
        quad_max_fill_ratio: float,
        quad_min_edge_length: float,
        quad_min_edge_score: float,
        subpixel_refinement_sigma: float,
        segmentation_margin: int,
        segmentation_connectivity: SegmentationConnectivity,
        upscale_factor: int,
        decoder_min_contrast: float,
        refinement_mode: CornerRefinementMode,
        decode_mode: DecodeMode,
        max_hamming_error: int | None,
        gwlf_transversal_alpha: float,
        quad_max_elongation: float,
        quad_min_density: float,
        quad_extraction_mode: QuadExtractionMode,
        huber_delta_px: float,
        tikhonov_alpha_max: float,
        sigma_n_sq: float,
        structure_tensor_radius: int,
        edlines_imbalance_gate: bool,
    ) -> None: ...

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class PipelineTelemetryResult:
    """Intermediate pipeline artifacts emitted when ``debug_telemetry=True``."""
    @property
    def binarized(self) -> npt.NDArray[np.uint8]: ...  # (H, W)
    @property
    def threshold_map(self) -> npt.NDArray[np.uint8]: ...  # (H, W)
    @property
    def subpixel_jitter(self) -> npt.NDArray[np.float32] | None: ...  # (N, 4, 2)
    @property
    def reprojection_errors(self) -> npt.NDArray[np.float32] | None: ...  # (N,)
    @property
    def gwlf_fallback_count(self) -> int: ...
    @property
    def gwlf_avg_delta(self) -> float: ...

class DetectionResult:
    """Typed result from :meth:`Detector.detect`."""
    @property
    def ids(self) -> npt.NDArray[np.int32]: ...  # (N,)
    @property
    def corners(self) -> npt.NDArray[np.float32]: ...  # (N, 4, 2)
    @property
    def error_rates(self) -> npt.NDArray[np.float32]: ...  # (N,)
    @property
    def poses(self) -> npt.NDArray[np.float32] | None: ...  # (N, 7) [tx,ty,tz,qx,qy,qz,qw]
    @property
    def rejected_corners(self) -> npt.NDArray[np.float32]: ...  # (M, 4, 2)
    @property
    def rejected_error_rates(self) -> npt.NDArray[np.float32]: ...  # (M,)
    @property
    def telemetry(self) -> PipelineTelemetryResult | None: ...

class CharucoTelemetryResult:
    """Debug telemetry from :meth:`CharucoRefiner.estimate`."""
    @property
    def rejected_saddles(self) -> npt.NDArray[np.float32]: ...  # (R, 2)
    @property
    def rejected_determinants(self) -> npt.NDArray[np.float32]: ...  # (R,)

class CharucoEstimateResult:
    """Typed result from :meth:`CharucoRefiner.estimate`."""
    @property
    def ids(self) -> npt.NDArray[np.int32]: ...  # (N,)
    @property
    def corners(self) -> npt.NDArray[np.float32]: ...  # (N, 4, 2)
    @property
    def saddle_ids(self) -> npt.NDArray[np.int32]: ...  # (S,)
    @property
    def saddle_pts(self) -> npt.NDArray[np.float32]: ...  # (S, 2)
    @property
    def saddle_obj(self) -> npt.NDArray[np.float64]: ...  # (S, 3)
    @property
    def board_pose(self) -> npt.NDArray[np.float64] | None: ...  # (7,) [tx,ty,tz,qx,qy,qz,qw]
    @property
    def board_cov(self) -> npt.NDArray[np.float64] | None: ...  # (6, 6)
    @property
    def telemetry(self) -> CharucoTelemetryResult | None: ...

class BoardEstimateResult:
    """Typed result from :meth:`BoardEstimator.estimate`."""
    @property
    def ids(self) -> npt.NDArray[np.int32]: ...  # (N,)
    @property
    def corners(self) -> npt.NDArray[np.float32]: ...  # (N, 4, 2)
    @property
    def board_pose(self) -> npt.NDArray[np.float64] | None: ...  # (7,) [tx,ty,tz,qx,qy,qz,qw]
    @property
    def board_cov(self) -> npt.NDArray[np.float64] | None: ...  # (6, 6)

# ---------------------------------------------------------------------------
# Board topology
# ---------------------------------------------------------------------------

class CharucoBoard:
    """Configuration for a ChAruco board."""
    def __init__(
        self,
        rows: int,
        cols: int,
        square_length: float,
        marker_length: float,
        family: TagFamily,
    ) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...

class AprilGrid:
    """Configuration for an AprilGrid board."""
    def __init__(
        self,
        rows: int,
        cols: int,
        spacing: float,
        marker_length: float,
        family: TagFamily,
    ) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...

class BoardEstimator:
    """Estimator for multi-tag board poses (AprilGrid)."""
    def __init__(self, board: AprilGrid) -> None: ...
    @classmethod
    def from_charuco(cls, board: CharucoBoard) -> BoardEstimator: ...
    def estimate(
        self,
        detector: Detector,
        img: npt.NDArray[np.uint8],
        intrinsics: CameraIntrinsics,
    ) -> BoardEstimateResult: ...

class CharucoRefiner:
    """Extracts ChAruco saddle points and estimates board pose."""
    def __init__(self, board: CharucoBoard) -> None: ...
    def estimate(
        self,
        detector: Detector,
        img: npt.NDArray[np.uint8],
        intrinsics: CameraIntrinsics,
        debug_telemetry: bool = False,
    ) -> CharucoEstimateResult: ...

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    def __init__(
        self,
        profile: Literal["standard", "grid", "high_accuracy", "render_tag_hub"] | None = None,
        config: Any | None = None,
        *,
        decimation: int | None = None,
        threads: int | None = None,
        families: list[TagFamily] | None = None,
    ) -> None: ...
    def detect(
        self,
        img: npt.NDArray[np.uint8],
        intrinsics: CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
        debug_telemetry: bool = False,
    ) -> DetectionResult: ...
    def detect_concurrent(
        self,
        frames: list[npt.NDArray[np.uint8]],
        intrinsics: CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ) -> list[DetectionResult]:
        """Detect tags in multiple frames concurrently using Rayon.

        Releases the GIL for the entire parallel section. Pool size is set by
        ``max_concurrent_frames`` at construction time.
        Telemetry and rejected-corner data are not available via this method.
        """
        ...
    def config(self) -> PyDetectorConfig: ...
    def set_families(self, families: list[int]) -> None: ...

class DetectorBuilder:
    """Fluent builder for constructing a :class:`Detector`."""
    def __init__(self) -> None: ...
    def with_decimation(self, decimation: int) -> DetectorBuilder: ...
    def with_threads(self, threads: int) -> DetectorBuilder: ...
    def with_family(self, family: TagFamily) -> DetectorBuilder: ...
    def with_upscale_factor(self, factor: int) -> DetectorBuilder: ...
    def with_corner_refinement(self, mode: CornerRefinementMode) -> DetectorBuilder: ...
    def with_decode_mode(self, mode: DecodeMode) -> DetectorBuilder: ...
    def with_connectivity(self, connectivity: SegmentationConnectivity) -> DetectorBuilder: ...
    def with_threshold_tile_size(self, size: int) -> DetectorBuilder: ...
    def with_threshold_min_range(self, range: int) -> DetectorBuilder: ...
    def with_adaptive_threshold_constant(self, c: int) -> DetectorBuilder: ...
    def with_quad_min_area(self, area: int) -> DetectorBuilder: ...
    def with_quad_min_fill_ratio(self, ratio: float) -> DetectorBuilder: ...
    def with_quad_min_edge_score(self, score: float) -> DetectorBuilder: ...
    def with_max_hamming_error(self, errors: int) -> DetectorBuilder: ...
    def with_decoder_min_contrast(self, contrast: float) -> DetectorBuilder: ...
    def with_gwlf_transversal_alpha(self, alpha: float) -> DetectorBuilder: ...
    def with_quad_max_elongation(self, elongation: float) -> DetectorBuilder: ...
    def with_quad_min_density(self, density: float) -> DetectorBuilder: ...
    def with_quad_extraction_mode(self, mode: QuadExtractionMode) -> DetectorBuilder: ...
    def with_sharpening(self, enable: bool) -> DetectorBuilder: ...
    def with_max_concurrent_frames(self, n: int) -> DetectorBuilder: ...
    def build(self) -> Detector: ...

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def _create_detector_from_config(
    config: PyDetectorConfig,
    decimation: int | None = None,
    threads: int | None = None,
    families: list[int] = [],
) -> Detector: ...
def init_tracy() -> None: ...
def _shipped_profile_json(name: str) -> str:
    """Return the raw JSON text for a shipped profile, read from Rust's
    compile-time-embedded bytes. Raises ``ValueError`` if the name is not
    one of the shipped profiles.
    """
    ...
