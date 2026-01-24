from ._config import DetectOptions, DetectorConfig
from .locus import (
    Detection,
    PipelineStats,
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
        return cls(**config)

    def __init__(self, config: DetectorConfig = None, **kwargs):
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
        )

    def detect(self, img, decimation: int = 1):
        """Detect tags using configured defaults."""
        return self._inner.detect(img, decimation=decimation)

    def detect_with_options(
        self, img, options: DetectOptions = None, decimation: int = 1, **kwargs
    ):
        """Detect tags with per-call options."""
        if options is None:
            options = DetectOptions(decimation=decimation, **kwargs)

        if not options.families:
            return self._inner.detect(img, decimation=options.decimation)

        return self._inner.detect_with_options(img, options.families)

    def detect_with_stats(self, img, decimation: int = 1):
        """Detect tags and return performance statistics."""
        return self._inner.detect_with_stats(img, decimation=decimation)

    def set_families(self, families: list[TagFamily]):
        """Set the default tag families to decode."""
        self._inner.set_families(families)

    @property
    def enable_sharpening(self) -> bool:
        """Check if sharpening is enabled."""
        return self._inner.enable_sharpening


__all__ = [
    "Detection",
    "Detector",
    "PipelineStats",
    "TagFamily",
    "SegmentationConnectivity",
    "DetectorConfig",
    "DetectOptions",
    "detect_tags",
    "detect_tags_with_stats",
    "dummy_detect",
    "debug_threshold",
    "debug_segmentation",
]
