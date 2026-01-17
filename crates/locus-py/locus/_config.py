from pydantic import BaseModel, ConfigDict, Field

from .locus import TagFamily, SegmentationConnectivity


class DetectorConfig(BaseModel):
    """
    Pipeline-level configuration for the Locus detector.
    These settings are typically set once during detector instantiation.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    threshold_tile_size: int = Field(default=4, ge=2, le=64)
    threshold_min_range: int = Field(default=2, ge=0, le=255)

    enable_bilateral: bool = Field(default=True)
    bilateral_sigma_space: float = Field(default=0.8, ge=0.1)
    bilateral_sigma_color: float = Field(default=30.0, ge=0.1)

    enable_sharpening: bool = Field(default=False)

    enable_adaptive_window: bool = Field(default=True)
    threshold_min_radius: int = Field(default=2, ge=1)
    threshold_max_radius: int = Field(default=7, ge=1)

    quad_min_area: int = Field(default=16, ge=1)
    quad_max_aspect_ratio: float = Field(default=3.0, ge=1.0)
    quad_min_fill_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    quad_max_fill_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    quad_min_edge_length: float = Field(default=4.0, ge=0.0)
    quad_min_edge_score: float = Field(default=0.4, ge=0.0)
    subpixel_refinement_sigma: float = Field(default=0.6, ge=0.0)
    segmentation_connectivity: SegmentationConnectivity = Field(default=SegmentationConnectivity.Eight)
    upscale_factor: int = Field(default=1, ge=1)
    decoder_min_contrast: float = Field(default=20.0, ge=0.0)


class DetectOptions(BaseModel):
    """
    Per-call options for tag detection.
    Controls which tag families to decode for a specific frame.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    families: list[TagFamily] = Field(default_factory=list)

    @classmethod
    def all(cls) -> "DetectOptions":
        return cls(
            families=[
                TagFamily.AprilTag36h11,
                TagFamily.AprilTag16h5,
                TagFamily.ArUco4x4_50,
                TagFamily.ArUco4x4_100,
            ]
        )
