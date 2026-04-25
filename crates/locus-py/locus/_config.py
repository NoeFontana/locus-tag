"""Nested DetectorConfig schema and JSON profile loader.

Single source of truth for the Python side of Locus detector configuration.
The schema mirrors the grouping in the canonical JSON profiles shipped
inside ``locus-core`` (``crates/locus-core/profiles/*.json``); Rust
deserializes those same files (via ``include_str!``) into its flat
``DetectorConfig``, and Python reads the exact embedded bytes through
the ``_shipped_profile_json`` FFI hook. If the two ever disagree, the
JSON is authoritative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, TypeVar, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    model_validator,
)

from .locus import (
    CameraIntrinsics,
    CornerRefinementMode,
    DecodeMode,
    PoseEstimationMode,
    PyDetectorConfig,
    QuadExtractionMode,
    SegmentationConnectivity,
    TagFamily,
    _shipped_profile_json,
)

_E = TypeVar("_E")

ProfileName: TypeAlias = Literal["standard", "grid", "high_accuracy", "render_tag_hub"]
SHIPPED_PROFILES: tuple[ProfileName, ...] = (
    "standard",
    "grid",
    "high_accuracy",
    "render_tag_hub",
)


def _enum_registry(enum_cls: type[_E]) -> tuple[dict[int, _E], dict[str, _E], dict[_E, str]]:
    # PyO3 int-enums are not `enum.IntEnum`, so walk class attributes to build
    # a constructible lookup. Called 3x per enum at import time (once each from
    # `_coerce`, `_serialize_name`, `_enum_field`) — cheap enough that caching
    # is not worth the typeshed friction around `functools.cache`.
    by_int: dict[int, _E] = {}
    by_name: dict[str, _E] = {}
    to_name: dict[_E, str] = {}
    for attr in dir(enum_cls):
        if attr.startswith("_"):
            continue
        variant = getattr(enum_cls, attr)
        if isinstance(variant, enum_cls):
            by_int[int(cast(Any, variant))] = variant
            by_name[attr] = variant
            to_name[variant] = attr
    return by_int, by_name, to_name


def _coerce(enum_cls: type[_E]):
    by_int, by_name, _ = _enum_registry(enum_cls)

    def _inner(value: Any) -> _E:
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            try:
                return by_name[value]
            except KeyError as exc:
                raise ValueError(
                    f"{enum_cls.__name__}: {value!r} is not a valid variant name "
                    f"(allowed: {sorted(by_name)})"
                ) from exc
        try:
            return by_int[int(value)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{enum_cls.__name__}: value {value!r} is not a valid variant "
                f"(allowed ints: {sorted(by_int)}; allowed names: {sorted(by_name)})"
            ) from exc

    return _inner


def _serialize_name(enum_cls: type[_E]):
    _, _, to_name = _enum_registry(enum_cls)

    def _inner(value: _E) -> str:
        try:
            return to_name[value]
        except KeyError as exc:
            raise ValueError(f"Unknown {enum_cls.__name__} variant: {value!r}") from exc

    return _inner


def _enum_field(enum_cls: type[_E]) -> Any:
    # String-in JSON, enum-instance in Python. Publishes the allowed variant
    # names to JSON Schema so editor validation and `schemas/profile.schema.json`
    # remain useful.
    _, by_name, _ = _enum_registry(enum_cls)
    return Annotated[
        enum_cls,
        BeforeValidator(_coerce(enum_cls)),
        PlainSerializer(_serialize_name(enum_cls), when_used="json", return_type=str),
        WithJsonSchema({"type": "string", "enum": sorted(by_name)}),
    ]


if TYPE_CHECKING:
    # mypy needs plain type aliases to accept these in field annotations.
    # At runtime the `else` branch installs the `Annotated[...]` wrapper so
    # Pydantic sees the validators/serializers; the two views describe the
    # same values — see `_enum_field` for the runtime metadata shape.
    _CornerRefinementField: TypeAlias = CornerRefinementMode
    _DecodeModeField: TypeAlias = DecodeMode
    _QuadExtractionField: TypeAlias = QuadExtractionMode
    _SegConnField: TypeAlias = SegmentationConnectivity
else:
    _CornerRefinementField = _enum_field(CornerRefinementMode)
    _DecodeModeField = _enum_field(DecodeMode)
    _QuadExtractionField = _enum_field(QuadExtractionMode)
    _SegConnField = _enum_field(SegmentationConnectivity)


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
                f"threshold.min_radius ({self.min_radius}) must be <= "
                f"max_radius ({self.max_radius})"
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
    upscale_factor: int = Field(default=1, ge=1)
    max_elongation: float = Field(default=0.0, ge=0.0)
    min_density: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_mode: _QuadExtractionField = Field(
        default_factory=lambda: QuadExtractionMode.ContourRdp
    )
    edlines_imbalance_gate: bool = Field(default=False)

    @model_validator(mode="after")
    def _check_fill_ratio_ordering(self) -> QuadConfig:
        if self.min_fill_ratio >= self.max_fill_ratio:
            raise ValueError(
                f"quad.min_fill_ratio ({self.min_fill_ratio}) must be strictly less than "
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


class PoseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    huber_delta_px: float = Field(default=1.5, ge=0.0)
    tikhonov_alpha_max: float = Field(default=0.25, ge=0.0)
    sigma_n_sq: float = Field(default=4.0, ge=0.0)
    structure_tensor_radius: int = Field(default=2, ge=1, le=8)


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    connectivity: _SegConnField = Field(default_factory=lambda: SegmentationConnectivity.Eight)
    margin: int = 1


class DetectorConfig(BaseModel):
    """Nested detector configuration — Python source of truth.

    The three shipped profiles live in ``crates/locus-core/profiles/*.json``
    and are embedded into the Rust crate at compile time; the wheel reads
    the exact same bytes through the FFI. Load via :meth:`from_profile`
    for a shipped profile or :meth:`from_profile_json` for a user-supplied
    JSON string.
    """

    model_config = ConfigDict(extra="forbid", frozen=False, arbitrary_types_allowed=True)

    name: str | None = None
    extends: str | None = None
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    quad: QuadConfig = Field(default_factory=QuadConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)

    @model_validator(mode="after")
    def _check_extends_unresolved(self) -> DetectorConfig:
        if self.extends is not None:
            raise NotImplementedError(
                f"Profile inheritance (extends={self.extends!r}) is declared in the schema "
                "but not yet resolved by the loader. Inline the parent profile's values for "
                "now; resolution will land in a follow-up."
            )
        return self

    @model_validator(mode="after")
    def _check_cross_group_compat(self) -> DetectorConfig:
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

    @classmethod
    def from_profile(cls, name: ProfileName) -> DetectorConfig:
        """Load a shipped profile by name."""
        if name not in SHIPPED_PROFILES:
            raise ValueError(
                f"Unknown shipped profile {name!r}; expected one of {sorted(SHIPPED_PROFILES)}"
            )
        return cls.model_validate_json(_shipped_profile_json(name))

    @classmethod
    def from_profile_json(cls, json_str: str) -> DetectorConfig:
        """Load a user-supplied profile from a JSON string."""
        return cls.model_validate_json(json_str)

    def _to_ffi_config(self) -> PyDetectorConfig:
        # Cross-group validation has already fired in this Pydantic model;
        # Rust's `DetectorConfig::validate()` remains the final gate.
        return PyDetectorConfig(
            threshold_tile_size=self.threshold.tile_size,
            threshold_min_range=self.threshold.min_range,
            enable_sharpening=self.threshold.enable_sharpening,
            enable_adaptive_window=self.threshold.enable_adaptive_window,
            threshold_min_radius=self.threshold.min_radius,
            threshold_max_radius=self.threshold.max_radius,
            adaptive_threshold_constant=self.threshold.constant,
            adaptive_threshold_gradient_threshold=self.threshold.gradient_threshold,
            quad_min_area=self.quad.min_area,
            quad_max_aspect_ratio=self.quad.max_aspect_ratio,
            quad_min_fill_ratio=self.quad.min_fill_ratio,
            quad_max_fill_ratio=self.quad.max_fill_ratio,
            quad_min_edge_length=self.quad.min_edge_length,
            quad_min_edge_score=self.quad.min_edge_score,
            subpixel_refinement_sigma=self.quad.subpixel_refinement_sigma,
            upscale_factor=self.quad.upscale_factor,
            quad_max_elongation=self.quad.max_elongation,
            quad_min_density=self.quad.min_density,
            quad_extraction_mode=self.quad.extraction_mode,
            edlines_imbalance_gate=self.quad.edlines_imbalance_gate,
            decoder_min_contrast=self.decoder.min_contrast,
            refinement_mode=self.decoder.refinement_mode,
            decode_mode=self.decoder.decode_mode,
            max_hamming_error=self.decoder.max_hamming_error,
            gwlf_transversal_alpha=self.decoder.gwlf_transversal_alpha,
            huber_delta_px=self.pose.huber_delta_px,
            tikhonov_alpha_max=self.pose.tikhonov_alpha_max,
            sigma_n_sq=self.pose.sigma_n_sq,
            structure_tensor_radius=self.pose.structure_tensor_radius,
            segmentation_connectivity=self.segmentation.connectivity,
            segmentation_margin=self.segmentation.margin,
        )


class DetectOptions(BaseModel):
    """Per-call options for tag detection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    families: list[TagFamily] = Field(default_factory=list)
    decimation: int = Field(default=1, ge=1)
    intrinsics: tuple[float, float, float, float] | CameraIntrinsics | None = Field(default=None)
    tag_size: float | None = Field(default=None, ge=0.0)
    pose_estimation_mode: PoseEstimationMode = Field(
        default_factory=lambda: PoseEstimationMode.Fast
    )

    @classmethod
    def all(cls) -> DetectOptions:
        return cls(
            families=[
                TagFamily.AprilTag36h11,
                TagFamily.ArUco4x4_50,
                TagFamily.ArUco4x4_100,
            ]
        )


__all__ = [
    "SHIPPED_PROFILES",
    "DecoderConfig",
    "DetectOptions",
    "DetectorConfig",
    "PoseConfig",
    "ProfileName",
    "QuadConfig",
    "SegmentationConfig",
    "ThresholdConfig",
]
