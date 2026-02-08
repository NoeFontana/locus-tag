from ._config import DetectOptions, DetectorConfig
from .locus import (
    CameraIntrinsics,
    CornerRefinementMode,
    DecodeMode,
    Detection,
    FullDetectionResult,
    PipelineStats,
    Pose,
    PoseEstimationMode,
    SegmentationConnectivity,
    TagFamily,
    debug_segmentation,
    debug_threshold,
    detect_tags,
    detect_tags_with_stats,
    dummy_detect,
)
from .locus import Detector as _RustDetector


class Detector:
    """
    High-performance tag detector.

    This class wraps the Rust implementation to provide a Pydantic-validated
    configuration interface.
    """

    @classmethod
    def checkerboard(cls, **kwargs) -> "Detector":
        """
        Create a detector optimized for checkerboard patterns.

        This profile enables sharpening, uses 4-way connectivity, and
        disables bilateral filtering to maximize edge recall for small tags.
        """
        config = {
            "enable_sharpening": True,
            "enable_bilateral": False,
            "segmentation_connectivity": SegmentationConnectivity.Four,
            "decoder_min_contrast": 10.0,
            "quad_min_area": 8,
            "quad_min_edge_length": 2.0,
        }
        config.update(kwargs)
        return cls(config=DetectorConfig(**config))

    def __init__(self, config: DetectorConfig | None = None, **kwargs):
        if config is None:
            # Allow passing individual parameters as kwargs
            config = DetectorConfig(**kwargs)

        # Initialize the underlying Rust detector
        self._inner = _RustDetector(
            threshold_tile_size=config.threshold_tile_size,
            threshold_min_range=config.threshold_min_range,
            enable_bilateral=config.enable_bilateral,
            bilateral_sigma_space=config.bilateral_sigma_space,
            bilateral_sigma_color=config.bilateral_sigma_color,
            enable_sharpening=config.enable_sharpening,
            enable_adaptive_window=config.enable_adaptive_window,
            threshold_min_radius=config.threshold_min_radius,
            threshold_max_radius=config.threshold_max_radius,
            adaptive_threshold_constant=config.adaptive_threshold_constant,
            adaptive_threshold_gradient_threshold=config.adaptive_threshold_gradient_threshold,
            quad_min_area=config.quad_min_area,
            quad_max_aspect_ratio=config.quad_max_aspect_ratio,
            quad_min_fill_ratio=config.quad_min_fill_ratio,
            quad_max_fill_ratio=config.quad_max_fill_ratio,
            quad_min_edge_length=config.quad_min_edge_length,
            quad_min_edge_score=config.quad_min_edge_score,
            subpixel_refinement_sigma=config.subpixel_refinement_sigma,
            segmentation_margin=config.segmentation_margin,
            segmentation_connectivity=config.segmentation_connectivity,
            upscale_factor=config.upscale_factor,
            decoder_min_contrast=config.decoder_min_contrast,
            refinement_mode=config.refinement_mode,
            decode_mode=config.decode_mode,
        )

    def detect(
        self,
        img,
        decimation: int = 1,
        intrinsics: tuple[float, float, float, float] | CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ):
        """Detect tags using configured defaults."""
        if isinstance(intrinsics, CameraIntrinsics):
            intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)

        return self._inner.detect(
            img,
            decimation=decimation,
            intrinsics=intrinsics,
            tag_size=tag_size,
            pose_estimation_mode=pose_estimation_mode,
        )

    def detect_with_options(
        self,
        img,
        options: DetectOptions | None = None,
        decimation: int = 1,
        intrinsics: tuple[float, float, float, float] | CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
        **kwargs,
    ):
        """Detect tags with per-call options."""
        if options is None:
            options = DetectOptions(
                decimation=decimation,
                intrinsics=intrinsics,
                tag_size=tag_size,
                pose_estimation_mode=pose_estimation_mode,
                **kwargs,
            )

        # Handle unpacking in options if needed, though DetectOptions usually takes tuple.
        # But if user passed CameraIntrinsics to DetectOptions constructor...
        # DetectOptions is pydantic, verifying types. I should check if I updated DetectOptions to allow CameraIntrinsics.

        if not options.families:
            # We need to unpack options.intrinsics here too if it's not a tuple
            # But wait, DetectOptions is typed as tuple | None in _config.py
            # So if I want to support CameraIntrinsics in DetectOptions, I need to update _config.py too.
            pass

        # ... logic ...

        # Let's keep it simple: support CameraIntrinsics in the method arguments,
        # and ensure it's converted to tuple before passing to Rust.

        passed_intrinsics = options.intrinsics
        if isinstance(passed_intrinsics, CameraIntrinsics):
            passed_intrinsics = (
                passed_intrinsics.fx,
                passed_intrinsics.fy,
                passed_intrinsics.cx,
                passed_intrinsics.cy,
            )

        if not options.families:
            return self._inner.detect(
                img,
                decimation=options.decimation,
                intrinsics=passed_intrinsics,
                tag_size=options.tag_size,
                pose_estimation_mode=options.pose_estimation_mode,
            )

        return self._inner.detect_with_options(
            img,
            families=options.families,
            decimation=options.decimation,
            intrinsics=passed_intrinsics,
            tag_size=options.tag_size,
            pose_estimation_mode=options.pose_estimation_mode,
        )

    def detect_with_stats(
        self,
        img,
        decimation: int = 1,
        intrinsics: tuple[float, float, float, float] | CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ):
        """Detect tags and return performance statistics."""
        if isinstance(intrinsics, CameraIntrinsics):
            intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)

        return self._inner.detect_with_stats(
            img,
            decimation=decimation,
            intrinsics=intrinsics,
            tag_size=tag_size,
            pose_estimation_mode=pose_estimation_mode,
        )

    # ... set_families ...

    # ... extract_candidates ...

    def detect_full(
        self,
        img,
        decimation: int = 1,
        intrinsics: tuple[float, float, float, float] | CameraIntrinsics | None = None,
        tag_size: float | None = None,
        pose_estimation_mode: PoseEstimationMode = PoseEstimationMode.Fast,
    ) -> FullDetectionResult:
        """Perform full detection and return all intermediate debug data."""
        if isinstance(intrinsics, CameraIntrinsics):
            intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)

        return self._inner.detect_full(
            img,
            decimation=decimation,
            intrinsics=intrinsics,
            tag_size=tag_size,
            pose_estimation_mode=pose_estimation_mode,
        )


__all__ = [
    "Detection",
    "Detector",
    "FullDetectionResult",
    "PipelineStats",
    "TagFamily",
    "SegmentationConnectivity",
    "CornerRefinementMode",
    "DecodeMode",
    "PoseEstimationMode",
    "Pose",
    "CameraIntrinsics",
    "DetectorConfig",
    "DetectOptions",
    "detect_tags",
    "detect_tags_with_stats",
    "dummy_detect",
    "debug_threshold",
    "debug_segmentation",
]
