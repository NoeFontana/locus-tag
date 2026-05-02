"""Pydantic schemas for the Phase 0 rotation-tail diagnostic harness."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SceneRecord(BaseModel):
    """One row per scene in `scenes.json`."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    tag_id: int

    # Hub metadata
    distance_m: float
    angle_of_incidence_deg: float
    pixel_area: float
    occlusion_ratio: float

    # GT pose: scalar-last quaternion + translation
    gt_quaternion_xyzw: tuple[float, float, float, float]
    gt_translation_xyz: tuple[float, float, float]

    # Detected pose (production-equivalent path). None if production missed.
    detected: bool
    detected_quaternion_xyzw: Optional[tuple[float, float, float, float]] = None
    detected_translation_xyz: Optional[tuple[float, float, float]] = None

    # Alternate IPPE branch (LM-refined).
    alternate_quaternion_xyzw: Optional[tuple[float, float, float, float]] = None
    alternate_translation_xyz: Optional[tuple[float, float, float]] = None

    # Errors against GT — degrees and millimetres.
    rotation_error_chosen_deg: Optional[float] = None
    rotation_error_alternate_deg: Optional[float] = None
    translation_error_chosen_mm: Optional[float] = None
    translation_error_alternate_mm: Optional[float] = None

    # Diagnostic d² values.
    branch_chosen_idx: int  # 0/1; 255 sentinel = no branch selected.
    aggregate_d2_chosen: float = float("nan")
    aggregate_d2_alternate: float = float("nan")
    branch_d2_ratio: float = float("nan")
    max_corner_d2: float = float("nan")

    # LM telemetry (from per-iteration retention; None if call skipped).
    lm_iterations: Optional[int] = None
    lm_convergence: Optional[int] = None  # 0=grad, 1=step, 2=max-iter, 3=chol-fail

    # Per-image noise floor σ.
    image_noise_sigma: float

    # Production-equivalent latency (no diagnostic overhead). Microseconds.
    latency_us: float

    # Estimated PPM = sqrt(quad_area_px) / tag_size_m.
    ppm_estimated: float


class CornerRecord(BaseModel):
    """One row per (scene × corner) in `corners.parquet` — 4 rows per scene."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    corner_idx: int  # 0..3

    gt_corner_x: float
    gt_corner_y: float
    detected_corner_x: Optional[float] = None
    detected_corner_y: Optional[float] = None
    residual_norm_px: Optional[float] = None

    # Mahalanobis d² and Huber IRLS weight at the LM-converged pose.
    final_mahalanobis_d2: Optional[float] = None
    final_irls_weight: Optional[float] = None

    # Structure-tensor anisotropy.
    structure_tensor_lambda_max: Optional[float] = None
    structure_tensor_lambda_min: Optional[float] = None
    structure_tensor_R: Optional[float] = None  # λ_min / λ_max ∈ [0, 1]

    # Leave-one-out: fraction of rotation error reduction when this corner
    # is dropped from LM. Positive ⇒ this corner is hurting the fit.
    leave_one_out_rotation_err_drop_pct: Optional[float] = None


class ScenesFile(BaseModel):
    """Top-level container for `scenes.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="rotation_tail_diag/v1")
    config_name: str
    profile: str
    pose_estimation_mode: str
    sigma_n_sq_configured: float
    n_scenes: int
    scenes: list[SceneRecord]


class FailureMode(BaseModel):
    """Per-scene classification result."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    mode: str  # branch_flip / corner_outlier / grazing_angle / sigma_miscalibration / other
    evidence: dict[str, float | int | str | None | list[float]]


class FailureModesFile(BaseModel):
    """Top-level container for `failure_modes.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="rotation_tail_diag/v1")
    classifier_version: str
    sigma_n_sq_configured: float
    population: dict[str, int]  # mode → count
    classifications: list[FailureMode]
