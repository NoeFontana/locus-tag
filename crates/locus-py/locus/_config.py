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

import warnings
from collections.abc import Callable
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
    EdLinesImbalanceGatePolicy,
    PoseEstimationMode,
    PyDetectorConfig,
    QuadExtractionMode,
    SegmentationConnectivity,
    TagFamily,
    _shipped_profile_json,
)

_E = TypeVar("_E")

ProfileName: TypeAlias = Literal[
    "standard",
    "grid",
    "high_accuracy",
    "render_tag_hub",
    "general",
    "max_recall_adaptive",
]
SHIPPED_PROFILES: tuple[ProfileName, ...] = (
    "standard",
    "grid",
    "high_accuracy",
    "render_tag_hub",
    "general",
    "max_recall_adaptive",
)
_ADAPTIVE_PROFILES: frozenset[ProfileName] = frozenset({"high_accuracy", "max_recall_adaptive"})
_STATIC_ONLY_PROFILES: frozenset[ProfileName] = frozenset(SHIPPED_PROFILES) - _ADAPTIVE_PROFILES


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


def _enum_field(enum_cls: type[_E], coercer: Callable[[Any], _E] | None = None) -> Any:
    # String-in JSON, enum-instance in Python. Pass `coercer` to extend the
    # default int/name/instance coercion (e.g. legacy-shape acceptance).
    _, by_name, _ = _enum_registry(enum_cls)
    return Annotated[
        enum_cls,
        BeforeValidator(coercer if coercer is not None else _coerce(enum_cls)),
        PlainSerializer(_serialize_name(enum_cls), when_used="json", return_type=str),
        WithJsonSchema({"type": "string", "enum": sorted(by_name)}),
    ]


def _coerce_imbalance_gate(value: Any) -> EdLinesImbalanceGatePolicy:
    # `bool` is a subclass of `int`, so catch it *before* the int path.
    if isinstance(value, bool):
        warnings.warn(
            "edlines_imbalance_gate: boolean values are deprecated; use "
            'EdLinesImbalanceGatePolicy.Enabled / .Disabled (or "Enabled" / '
            '"Disabled") instead.',
            DeprecationWarning,
            stacklevel=3,
        )
        return EdLinesImbalanceGatePolicy.Enabled if value else EdLinesImbalanceGatePolicy.Disabled
    return _coerce(EdLinesImbalanceGatePolicy)(value)


if TYPE_CHECKING:
    # mypy needs plain type aliases to accept these in field annotations.
    # At runtime the `else` branch installs the `Annotated[...]` wrapper so
    # Pydantic sees the validators/serializers; the two views describe the
    # same values — see `_enum_field` for the runtime metadata shape.
    _CornerRefinementField: TypeAlias = CornerRefinementMode
    _DecodeModeField: TypeAlias = DecodeMode
    _QuadExtractionField: TypeAlias = QuadExtractionMode
    _SegConnField: TypeAlias = SegmentationConnectivity
    _ImbalanceGateField: TypeAlias = EdLinesImbalanceGatePolicy
else:
    _CornerRefinementField = _enum_field(CornerRefinementMode)
    _DecodeModeField = _enum_field(DecodeMode)
    _QuadExtractionField = _enum_field(QuadExtractionMode)
    _SegConnField = _enum_field(SegmentationConnectivity)
    _ImbalanceGateField = _enum_field(EdLinesImbalanceGatePolicy, _coerce_imbalance_gate)


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


class AdaptivePpbConfig(BaseModel):
    """Per-candidate extraction routing based on pixels-per-bit (PPB).

    Candidates with `ppb < threshold` run `low_extraction` + `low_refinement`;
    candidates with `ppb >= threshold` run `high_extraction` + `high_refinement`.
    At `ppb == threshold` exactly the low route wins (deterministic tie-break
    for snapshot stability).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    threshold: float = Field(default=2.5, gt=1.0, lt=5.0)
    low_extraction: _QuadExtractionField = Field(
        default_factory=lambda: QuadExtractionMode.ContourRdp
    )
    high_extraction: _QuadExtractionField = Field(
        default_factory=lambda: QuadExtractionMode.EdLines
    )
    low_refinement: _CornerRefinementField = Field(default_factory=lambda: CornerRefinementMode.Erf)
    # Python reserves the bare keyword `None`, but PyO3 exposes the enum
    # variant by its Rust name, so we reach it through `getattr`.
    high_refinement: _CornerRefinementField = Field(
        default_factory=lambda: getattr(CornerRefinementMode, "None")
    )


class _AdaptivePpbPolicy(BaseModel):
    """Wrapper mirroring serde's externally-tagged `{"AdaptivePpb": {...}}` form."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    AdaptivePpb: AdaptivePpbConfig


# Serde's default externally-tagged enum format yields either the unit string
# "Static" or a wrapped object `{"AdaptivePpb": {...}}`. Pydantic accepts both
# via this union; Rust's `QuadExtractionPolicy` deserializes the same shapes.
QuadExtractionPolicy: TypeAlias = Literal["Static"] | _AdaptivePpbPolicy


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
    edlines_imbalance_gate: _ImbalanceGateField = Field(
        default_factory=lambda: EdLinesImbalanceGatePolicy.Disabled
    )
    extraction_policy: QuadExtractionPolicy = "Static"

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
    max_hamming_error: int | None = Field(default=None, ge=0)
    """Maximum Hamming errors accepted during decoding.

    ``None`` (the default) defers to each registered family's tightest
    safe budget (16h5 = 0, 4x4_* = 1, 36h11 = 2, 6x6_250 = 2). An
    explicit integer overrides every family uniformly.
    """
    gwlf_transversal_alpha: float = Field(default=0.01, ge=0.0)
    post_decode_refinement: bool = Field(default=False)
    """Run an edge-fit corner re-refit after decoding (Phase C.5).

    ``False`` (the default) preserves byte-identical behaviour. When
    ``True``, each Valid candidate's four outer edges are independently
    fitted with the shared ERF step model and intersected pairwise to
    recover four sub-pixel corners; the homography is then re-solved.
    Phase A's GWLF / structure-tensor covariance is preserved — the line
    fit's Cramér-Rao bound is too optimistic for synthetic-PSF imagery
    and would over-trust the refined corners in the weighted pose solver.
    """


class PoseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    huber_delta_px: float = Field(default=1.5, ge=0.0)
    tikhonov_alpha_max: float = Field(default=0.25, ge=0.0)
    sigma_n_sq: float = Field(default=4.0, ge=0.0)
    structure_tensor_radius: int = Field(default=2, ge=1, le=8)
    pose_consistency_fpr: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Target false-positive rate for the reprojection-consistency gate.

    ``0.0`` (the default) disables the gate, preserving legacy behaviour.
    Positive values are interpreted as a per-tag FPR from which a
    chi-squared critical value is derived; rejected poses are surfaced as
    ``Detection.pose = None``. Values >= 1.0 are rejected as degenerate.
    """
    pose_consistency_gate_sigma_px: float = Field(default=1.0, gt=0.0)
    """Pixel σ assumed by the χ² consistency gate's null distribution.

    Independent of ``sigma_n_sq`` so the gate's calibration stays valid
    even when the LM weights residuals with anisotropic structure-tensor
    or GWLF info matrices. Default ``1.0 px`` — tighter than ``sigma_n_sq``
    (``≈ 4 px²``) so the gate catches sub-2-px false-positive residuals
    that the looser LM noise model would let through.
    """


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
        # Mirrors the four AdaptivePpb rules enforced in
        # `crates/locus-core/src/config.rs::DetectorConfig::validate`.
        if isinstance(self.quad.extraction_policy, _AdaptivePpbPolicy):
            p = self.quad.extraction_policy.AdaptivePpb
            if p.low_extraction == p.high_extraction:
                raise ValueError(
                    "quad.extraction_policy.AdaptivePpb has identical low/high extraction "
                    'modes; use "Static" for single-mode operation'
                )
            # `threshold` bounds are already enforced by the Pydantic Field
            # constraints (gt=1.0, lt=5.0) on AdaptivePpbConfig; re-checking
            # here would duplicate the message surface.
            for ext, refine in (
                (p.low_extraction, p.low_refinement),
                (p.high_extraction, p.high_refinement),
            ):
                if ext == QuadExtractionMode.EdLines:
                    if refine == CornerRefinementMode.Erf:
                        raise ValueError(
                            "quad.extraction_policy.AdaptivePpb route uses EdLines with "
                            "refinement_mode=Erf, which is incompatible"
                        )
                    if self.decoder.decode_mode == DecodeMode.Soft:
                        raise ValueError(
                            "quad.extraction_policy.AdaptivePpb route uses EdLines with "
                            "decoder.decode_mode=Soft, which is incompatible"
                        )
        return self

    @classmethod
    def from_profile(cls, name: ProfileName) -> DetectorConfig:
        """Load a shipped profile by name.

        General-purpose profiles must stay on ``QuadExtractionPolicy::Static``.
        Adaptive routing lives in explicit opt-in profiles
        (currently only ``max_recall_adaptive``).
        """
        if name not in SHIPPED_PROFILES:
            raise ValueError(
                f"Unknown shipped profile {name!r}; expected one of {sorted(SHIPPED_PROFILES)}"
            )
        cfg = cls.model_validate_json(_shipped_profile_json(name))
        if name in _STATIC_ONLY_PROFILES and cfg.quad.extraction_policy != "Static":
            raise ValueError(
                f'Shipped profile {name!r} must carry quad.extraction_policy="Static"; '
                "adaptive routing lives in opt-in profiles only"
            )
        return cfg

    @classmethod
    def from_profile_json(cls, json_str: str) -> DetectorConfig:
        """Load a user-supplied profile from a JSON string."""
        return cls.model_validate_json(json_str)

    def _to_ffi_config(self) -> PyDetectorConfig:
        # Cross-group validation has already fired in this Pydantic model;
        # Rust's `DetectorConfig::validate()` remains the final gate.
        #
        # `PyDetectorConfig` stays `Copy` on the Rust side, so the enum-shaped
        # `quad.extraction_policy` is flattened across a boolean discriminator
        # + five scalar fields here. Rust's `From<PyDetectorConfig>` rebuilds
        # the enum. See `crates/locus-py/src/lib.rs` for the round-trip.
        is_adaptive = isinstance(self.quad.extraction_policy, _AdaptivePpbPolicy)
        if is_adaptive:
            assert isinstance(self.quad.extraction_policy, _AdaptivePpbPolicy)
            p = self.quad.extraction_policy.AdaptivePpb
            ppb_threshold = p.threshold
            ppb_low_ext = p.low_extraction
            ppb_high_ext = p.high_extraction
            ppb_low_ref = p.low_refinement
            ppb_high_ref = p.high_refinement
        else:
            # Placeholder scalars — Rust's `From<PyDetectorConfig>` ignores
            # them when `quad_extraction_policy_is_adaptive == False`.
            ppb_threshold = 0.0
            ppb_low_ext = QuadExtractionMode.ContourRdp
            ppb_high_ext = QuadExtractionMode.EdLines
            ppb_low_ref = CornerRefinementMode.Erf
            ppb_high_ref = getattr(CornerRefinementMode, "None")

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
            quad_extraction_policy_is_adaptive=is_adaptive,
            adaptive_ppb_threshold=ppb_threshold,
            adaptive_ppb_low_extraction=ppb_low_ext,
            adaptive_ppb_high_extraction=ppb_high_ext,
            adaptive_ppb_low_refinement=ppb_low_ref,
            adaptive_ppb_high_refinement=ppb_high_ref,
            decoder_min_contrast=self.decoder.min_contrast,
            refinement_mode=self.decoder.refinement_mode,
            decode_mode=self.decoder.decode_mode,
            max_hamming_error=self.decoder.max_hamming_error,
            gwlf_transversal_alpha=self.decoder.gwlf_transversal_alpha,
            post_decode_refinement=self.decoder.post_decode_refinement,
            huber_delta_px=self.pose.huber_delta_px,
            tikhonov_alpha_max=self.pose.tikhonov_alpha_max,
            sigma_n_sq=self.pose.sigma_n_sq,
            structure_tensor_radius=self.pose.structure_tensor_radius,
            pose_consistency_fpr=self.pose.pose_consistency_fpr,
            pose_consistency_gate_sigma_px=self.pose.pose_consistency_gate_sigma_px,
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
    "AdaptivePpbConfig",
    "DecoderConfig",
    "DetectOptions",
    "DetectorConfig",
    "PoseConfig",
    "ProfileName",
    "QuadConfig",
    "QuadExtractionPolicy",
    "SegmentationConfig",
    "ThresholdConfig",
]
