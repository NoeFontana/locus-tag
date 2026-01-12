from pydantic import BaseModel, ConfigDict, Field

from .locus import TagFamily


class DetectorConfig(BaseModel):
    """
    Pipeline-level configuration for the Locus detector.
    These settings are typically set once during detector instantiation.
    """

    model_config = ConfigDict(frozen=True)

    threshold_tile_size: int = Field(default=8, ge=2, le=64)
    threshold_min_range: int = Field(default=10, ge=0, le=255)

    quad_min_area: int = Field(default=400, ge=10)
    quad_max_aspect_ratio: float = Field(default=3.0, ge=1.0)
    quad_min_fill_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    quad_max_fill_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    quad_min_edge_length: float = Field(default=4.0, ge=0.0)
    quad_min_edge_score: float = Field(default=10.0, ge=0.0)


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
