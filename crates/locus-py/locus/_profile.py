"""Nested DetectorProfile schema.

Forward-looking model grouping detector configuration into semantic blocks
with an ``extends:`` inheritance slot. Coexists with the flat
:mod:`locus._config` model used by the current :class:`locus.Detector`
kwargs path; the profile→kwargs translator lands in a follow-up phase.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar, cast

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, model_validator

from .locus import (
    CornerRefinementMode,
    DecodeMode,
    PoseEstimationMode,
    QuadExtractionMode,
    SegmentationConnectivity,
    TagFamily,
)

_E = TypeVar("_E")


def _coerce(enum_cls: type[_E]):
    # PyO3 "int-enum" classes are not constructible via ``EnumCls(int)``. The
    # variants are class attributes whose ``int()`` yields the discriminant;
    # build a reverse lookup at definition time.
    lookup: dict[int, _E] = {}
    for attr in dir(enum_cls):
        if attr.startswith("_"):
            continue
        variant = getattr(enum_cls, attr)
        if isinstance(variant, enum_cls):
            lookup[int(cast(Any, variant))] = variant

    def _inner(value: Any) -> _E:
        if isinstance(value, enum_cls):
            return value
        try:
            return lookup[int(value)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{enum_cls.__name__}: value {value!r} is not a valid variant "
                f"(allowed ints: {sorted(lookup)})"
            ) from exc

    return _inner


# PyO3 "int-enum" classes are not standard IntEnum; Pydantic cannot
# round-trip them by default. Each Annotated alias below accepts ints/enum
# instances on load (via BeforeValidator) and emits the int on dump.
# The declarations are expanded inline rather than generated because mypy
# does not accept dynamically-built Annotated values as field annotations.
_SERIALIZE_AS_INT = PlainSerializer(int, return_type=int)

_SegConnField = Annotated[
    SegmentationConnectivity, BeforeValidator(_coerce(SegmentationConnectivity)), _SERIALIZE_AS_INT
]
_QuadExtractionField = Annotated[
    QuadExtractionMode, BeforeValidator(_coerce(QuadExtractionMode)), _SERIALIZE_AS_INT
]
_CornerRefinementField = Annotated[
    CornerRefinementMode, BeforeValidator(_coerce(CornerRefinementMode)), _SERIALIZE_AS_INT
]
_DecodeModeField = Annotated[DecodeMode, BeforeValidator(_coerce(DecodeMode)), _SERIALIZE_AS_INT]
_PoseEstimationField = Annotated[
    PoseEstimationMode, BeforeValidator(_coerce(PoseEstimationMode)), _SERIALIZE_AS_INT
]
_TagFamilyField = Annotated[TagFamily, BeforeValidator(_coerce(TagFamily)), _SERIALIZE_AS_INT]


class ThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tile_size: int = Field(default=8, ge=2, le=64)
    min_range: int = Field(default=10, ge=0, le=255)
    enable_sharpening: bool = False
    enable_adaptive_window: bool = False
    min_radius: int = Field(default=2, ge=1)
    max_radius: int = Field(default=15, ge=1)
    constant: int = 0
    gradient_threshold: int = Field(default=10, ge=0, le=255)

    @model_validator(mode="after")
    def _check_radius_ordering(self) -> ThresholdConfig:
        if self.min_radius > self.max_radius:
            raise ValueError(
                f"min_radius ({self.min_radius}) must be <= max_radius ({self.max_radius})"
            )
        return self


class QuadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    min_area: int = Field(default=16, ge=1)
    max_aspect_ratio: float = Field(default=10.0, ge=1.0)
    min_fill_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    max_fill_ratio: float = Field(default=0.98, ge=0.0, le=1.0)
    min_edge_length: float = Field(default=4.0, gt=0.0)
    min_edge_score: float = Field(default=4.0, ge=0.0)
    subpixel_refinement_sigma: float = Field(default=0.6, ge=0.0)
    segmentation_margin: int = 1
    segmentation_connectivity: _SegConnField = Field(
        default_factory=lambda: SegmentationConnectivity.Eight
    )
    upscale_factor: int = Field(default=1, ge=1)
    max_elongation: float = Field(default=0.0, ge=0.0)
    min_density: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_mode: _QuadExtractionField = Field(
        default_factory=lambda: QuadExtractionMode.ContourRdp
    )

    @model_validator(mode="after")
    def _check_fill_ratio_ordering(self) -> QuadConfig:
        if self.min_fill_ratio >= self.max_fill_ratio:
            raise ValueError(
                f"min_fill_ratio ({self.min_fill_ratio}) must be strictly less than "
                f"max_fill_ratio ({self.max_fill_ratio})"
            )
        return self


class DecoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    min_contrast: float = Field(default=20.0, ge=0.0)
    refinement_mode: _CornerRefinementField = Field(
        default_factory=lambda: CornerRefinementMode.Erf
    )
    decode_mode: _DecodeModeField = Field(default_factory=lambda: DecodeMode.Hard)
    max_hamming_error: int = Field(default=2, ge=0)
    gwlf_transversal_alpha: float = Field(default=0.01, ge=0.0)
    families: list[_TagFamilyField] = Field(default_factory=list)


class PoseConfig(BaseModel):
    """Pose estimation and LM-solver knobs.

    The four LM fields (``huber_delta_px``, ``tikhonov_alpha_max``,
    ``sigma_n_sq``, ``structure_tensor_radius``) exist in the Rust
    ``DetectorConfig`` but are not yet plumbed through
    :func:`locus.create_detector`. The profile schema is forward-looking and
    reserves them now so the future loader can pass them through without a
    schema bump.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    huber_delta_px: float = Field(default=1.5, ge=0.0)
    tikhonov_alpha_max: float = Field(default=0.25, ge=0.0)
    sigma_n_sq: float = Field(default=4.0, ge=0.0)
    structure_tensor_radius: int = Field(default=2, ge=1, le=8)
    estimation_mode: _PoseEstimationField = Field(default_factory=lambda: PoseEstimationMode.Fast)


class DetectorProfile(BaseModel):
    """Nested, named detector configuration with an ``extends:`` slot.

    Phase A0 produces the model only. The ``extends`` field is declared and
    typed, but resolution (walking the parent chain, merging overrides) is a
    follow-up phase.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        arbitrary_types_allowed=True,
    )

    name: str = Field(min_length=1)
    extends: str | None = None
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    quad: QuadConfig = Field(default_factory=QuadConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)

    @model_validator(mode="after")
    def _check_cross_group_compat(self) -> DetectorProfile:
        # Mirrors DetectorConfig::validate() in crates/locus-core/src/config.rs:
        # EdLines quad extraction is incompatible with Erf refinement and Soft decode.
        if self.quad.extraction_mode == QuadExtractionMode.EdLines:
            if self.decoder.refinement_mode == CornerRefinementMode.Erf:
                raise ValueError(
                    "quad.extraction_mode=EdLines is incompatible with decoder.refinement_mode=Erf"
                )
            if self.decoder.decode_mode == DecodeMode.Soft:
                raise ValueError(
                    "quad.extraction_mode=EdLines is incompatible with decoder.decode_mode=Soft"
                )
        return self


__all__ = [
    "DecoderConfig",
    "DetectorProfile",
    "PoseConfig",
    "QuadConfig",
    "ThresholdConfig",
]
