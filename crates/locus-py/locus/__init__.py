from ._config import DetectOptions, DetectorConfig
from .locus import (
    Detection,
    PipelineStats,
    TagFamily,
    debug_segmentation,
    debug_threshold,
    detect_tags,
    detect_tags_gradient,
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

    def __init__(self, config: DetectorConfig = None, **kwargs):
        if config is None:
            # Allow passing individual parameters as kwargs
            config = DetectorConfig(**kwargs)

        # Initialize the underlying Rust detector
        self._inner = _RustDetector(
            threshold_tile_size=config.threshold_tile_size,
            threshold_min_range=config.threshold_min_range,
            quad_min_area=config.quad_min_area,
            quad_max_aspect_ratio=config.quad_max_aspect_ratio,
            quad_min_fill_ratio=config.quad_min_fill_ratio,
            quad_max_fill_ratio=config.quad_max_fill_ratio,
            quad_min_edge_length=config.quad_min_edge_length,
            quad_min_edge_score=config.quad_min_edge_score,
        )

    def detect(self, img):
        """Detect tags using configured defaults."""
        return self._inner.detect(img)

    def detect_with_options(self, img, options: DetectOptions = None, **kwargs):
        """Detect tags with per-call options."""
        if options is None:
            options = DetectOptions(**kwargs)

        if not options.families:
            return self._inner.detect(img)

        return self._inner.detect_with_options(img, options.families)

    def detect_with_stats(self, img):
        """Detect tags and return performance statistics."""
        return self._inner.detect_with_stats(img)

    def set_families(self, families: list[TagFamily]):
        """Set the default tag families to decode."""
        self._inner.set_families(families)


__all__ = [
    "Detection",
    "Detector",
    "PipelineStats",
    "TagFamily",
    "DetectorConfig",
    "DetectOptions",
    "detect_tags",
    "detect_tags_gradient",
    "detect_tags_with_stats",
    "dummy_detect",
    "debug_threshold",
    "debug_segmentation",
]
