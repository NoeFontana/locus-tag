"""Pose-covariance calibration audit.

Compares the 6×6 covariance returned by ``refine_pose_lm_weighted`` (via the
bench-internals ``refine_pose_lm_weighted_with_telemetry`` entry-point) against
empirical 6-DOF se(3) residuals on the hub 1080p corpus.

Key statistic: the squared 6-DOF Mahalanobis distance

    d² = δᵀ Σ⁻¹ δ        with    δ = log_SE3( T_det · T_gt⁻¹ )

is asymptotically χ²(6)-distributed (mean 6, variance 12) when the covariance
is well calibrated. We aggregate the empirical distribution across the hub
1080p corpus, compute a KL divergence between the empirical histogram and
χ²(6), and break out the calibration per-axis (each diagonal block in
isolation) to finger-point the formula bug if any.

Outputs a JSON report and Q-Q plot under
``diagnostics/pose_cov_audit_<ISO-date>/``.

Run::

    uv run --group bench tools/bench/pose_cov_audit.py \\
        --hub-config locus_v1_tag36h11_1920x1080
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import locus
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from locus import bench as lb
from locus._config import DetectorConfig
from tqdm import tqdm

from tools.bench.utils import HubDatasetLoader

# χ²(6) reference moments — Σ truly equal to (JᵀWJ)⁻¹ would give these.
CHI2_K = 6
CHI2_MEAN = float(CHI2_K)
CHI2_VAR = float(2 * CHI2_K)

# Each per-axis 1-DOF block is asymptotically χ²(1) when calibrated.
CHI2_1DOF_MEAN = 1.0
CHI2_1DOF_VAR = 2.0

# Histogram resolution for the KL estimate. The bin edges are picked relative
# to χ²(6) tail behaviour (P(χ²₆ > 30) ≈ 4×10⁻⁵).
KL_NUM_BINS = 60
KL_MAX_X = 60.0


# ---------------------------------------------------------------------------
# se(3) tangent-space helpers
# ---------------------------------------------------------------------------


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert a scalar-last unit quaternion to a 3×3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_log(rot: np.ndarray) -> np.ndarray:
    """Map SO(3) → so(3) (axis-angle, 3-vector). Numerically robust near 0/π."""
    cos_theta = (np.trace(rot) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        # First-order: log(R) ≈ (R − Rᵀ)/2 unhatted.
        skew = 0.5 * (rot - rot.T)
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)
    if abs(math.pi - theta) < 1e-6:
        # Near-π: numerically stable axis from the largest diagonal.
        diag = np.diag(rot)
        i = int(np.argmax(diag))
        col = (rot[:, i] + np.eye(3)[:, i]).astype(np.float64)
        col_norm = float(np.linalg.norm(col))
        if col_norm < 1e-9:
            return np.array([math.pi, 0.0, 0.0], dtype=np.float64)
        return (math.pi / col_norm) * col
    skew = (theta / (2.0 * math.sin(theta))) * (rot - rot.T)
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)


def _quat_xyzw_to_rotation(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3×3 rotation matrix."""
    qx, qy, qz, qw = q_xyzw
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _project_canonical_tag_corners(
    intrinsics: Any,  # locus.CameraIntrinsics
    gt_trans: np.ndarray,
    gt_quat_xyzw: np.ndarray,
    tag_size: float,
) -> np.ndarray:
    """Project the canonical centred tag corners through (R, t, K).

    Canonical corners (matches `centered_tag_corners` in pose_weighted.rs):
        (-s/2, -s/2, 0), (s/2, -s/2, 0), (s/2, s/2, 0), (-s/2, s/2, 0)
    Returns a (4, 2) array of pixel coordinates.

    Pinhole only: distortion is intentionally not applied here. Hub
    `locus_v1_tag36h11_*` configs are rectified; this is the same projection
    used by the LM's reprojection step.
    """
    s = float(tag_size) / 2.0
    obj_pts = np.array(
        [[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]],
        dtype=np.float64,
    )
    rot = _quat_xyzw_to_rotation(gt_quat_xyzw)
    world = obj_pts @ rot.T + gt_trans  # (4, 3)
    z = world[:, 2]
    proj = world[:, :2] / z[:, None]  # (4, 2) normalised image coords
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)
    image_pts = np.empty((4, 2), dtype=np.float64)
    image_pts[:, 0] = proj[:, 0] * fx + cx
    image_pts[:, 1] = proj[:, 1] * fy + cy
    return image_pts


def _se3_residual(
    det_trans: np.ndarray,
    det_quat_xyzw: np.ndarray,
    gt_trans: np.ndarray,
    gt_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Compute a 6-vector residual δ = [Δt; Δθ] in the GT tangent space.

    ``Δt = R_gtᵀ (t_det − t_gt)`` and ``Δθ = log(R_gtᵀ R_det)``. This matches
    the parameter ordering returned by the LM solver: indices 0..2 are the
    translation block, 3..5 the rotation block.
    """
    r_gt = _quat_xyzw_to_rot(gt_quat_xyzw)
    r_det = _quat_xyzw_to_rot(det_quat_xyzw)
    delta_t = r_gt.T @ (det_trans - gt_trans)
    delta_r = _rot_log(r_gt.T @ r_det)
    return np.concatenate([delta_t, delta_r]).astype(np.float64)


# ---------------------------------------------------------------------------
# χ² helpers (avoid a scipy dependency — small, self-contained)
# ---------------------------------------------------------------------------


def _gammainc_lower(s: float, x: float) -> float:
    """Regularised lower incomplete gamma P(s, x) = γ(s, x) / Γ(s).

    Implements Numerical Recipes' standard branching: a power-series for
    ``x < s + 1`` and a continued fraction for the upper tail. Accurate to
    ``< 1e-12`` over the range we care about (``x ∈ [0, 60]``, ``s ∈ {1/2, 3}``).
    """
    if x < 0.0 or s <= 0.0:
        raise ValueError(f"_gammainc_lower out of range: s={s}, x={x}")
    if x == 0.0:
        return 0.0

    log_pref = -x + s * math.log(x) - math.lgamma(s)

    if x < s + 1.0:
        # Power series.
        term = 1.0 / s
        total = term
        for n in range(1, 200):
            term *= x / (s + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return math.exp(log_pref) * total

    # Lentz's continued fraction for the upper incomplete: Γ(s,x)/Γ(s).
    fpmin = 1e-300
    b = x + 1.0 - s
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    upper = math.exp(log_pref) * h
    return 1.0 - upper


def _chi2_cdf(x: float | np.ndarray, k: int) -> np.ndarray:
    """χ²(k) cumulative distribution function."""
    arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(arr)
    s = 0.5 * k
    for idx, v in np.ndenumerate(arr):
        if v <= 0.0:
            out[idx] = 0.0
        else:
            out[idx] = _gammainc_lower(s, 0.5 * float(v))
    return out


def _chi2_ppf(p: np.ndarray, k: int) -> np.ndarray:
    """χ²(k) inverse CDF via Newton iteration on top of ``_chi2_cdf``.

    Mean ``k`` makes a good seed; the χ² density is unimodal, so a couple of
    Newton refinements (with bisection fallback) give 12-digit precision.
    """
    out = np.empty_like(p, dtype=np.float64)
    for i, prob in enumerate(np.asarray(p, dtype=np.float64)):
        if prob <= 0.0:
            out[i] = 0.0
            continue
        if prob >= 1.0:
            out[i] = float("inf")
            continue
        # Seed: rough Wilson-Hilferty approximation.
        z = math.sqrt(2.0) * _erfinv_approx(2.0 * prob - 1.0)
        x = max(k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3, 1e-6)
        for _ in range(50):
            cdf_x = _gammainc_lower(0.5 * k, 0.5 * x)
            # Density of χ²(k): pdf = x^(k/2-1) e^(-x/2) / (2^(k/2) Γ(k/2)).
            log_pdf = (
                (0.5 * k - 1.0) * math.log(x)
                - 0.5 * x
                - 0.5 * k * math.log(2.0)
                - math.lgamma(0.5 * k)
            )
            pdf = math.exp(log_pdf)
            if pdf < 1e-300:
                break
            step = (cdf_x - prob) / pdf
            x_new = x - step
            if x_new <= 0.0:
                x_new = 0.5 * x
            if abs(x_new - x) < 1e-10 * max(1.0, x):
                x = x_new
                break
            x = x_new
        out[i] = x
    return out


def _erfinv_approx(y: float) -> float:
    """Winitzki's elementary inverse-erf approximation (~5e-3 accurate)."""
    a = 0.147
    sgn = 1.0 if y >= 0 else -1.0
    yc = max(min(abs(y), 1.0 - 1e-15), -1.0 + 1e-15)
    ln = math.log(1.0 - yc * yc)
    term = 2.0 / (math.pi * a) + 0.5 * ln
    return sgn * math.sqrt(math.sqrt(term * term - ln / a) - term)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _empirical_histogram_pmf(
    samples: np.ndarray, max_x: float, num_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_edges, pmf) on ``[0, max_x]`` with ``num_bins`` bins.

    Right-tail samples (``> max_x``) are clipped into the last bin so the pmf
    sums to 1; KL is biased low for those, which is acceptable since the
    χ²(6) tail past ``max_x`` carries < 5×10⁻⁹ probability.
    """
    edges = np.linspace(0.0, max_x, num_bins + 1)
    counts, _ = np.histogram(np.clip(samples, 0.0, max_x), bins=edges)
    n = max(int(counts.sum()), 1)
    pmf = counts.astype(np.float64) / float(n)
    return edges, pmf


def _chi2_bin_pmf(edges: np.ndarray, k: int) -> np.ndarray:
    """χ²(k) probability mass per histogram bin."""
    cdf = _chi2_cdf(edges, k)
    return np.diff(cdf)


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p ‖ q) over discrete bins. Empty bins are floored at ``eps``."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


# ---------------------------------------------------------------------------
# Hardware metadata (per constraints.md §6)
# ---------------------------------------------------------------------------


def _capture_hardware_metadata(rayon_threads: str | None) -> dict[str, Any]:
    """Capture verified hardware metadata from system tools."""
    meta: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "build_profile": "--release",
        "rayon_num_threads": rayon_threads or "unset",
    }

    lscpu = shutil.which("lscpu")
    if lscpu:
        try:
            out = subprocess.check_output([lscpu], text=True, timeout=5)
            wanted = {
                "Architecture",
                "CPU(s)",
                "Vendor ID",
                "Model name",
                "Thread(s) per core",
                "L1d cache",
                "L2 cache",
                "L3 cache",
            }
            cpu: dict[str, str] = {}
            for line in out.splitlines():
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                if key in wanted:
                    cpu[key] = val.strip()
            meta["cpu"] = cpu
        except (OSError, subprocess.SubprocessError):
            meta["cpu"] = {"error": "lscpu invocation failed"}
    else:
        meta["cpu"] = {"error": "lscpu not found"}

    return meta


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------


@dataclass
class SceneSample:
    scene_id: str
    tag_id: int
    delta_se3: list[float]  # 6-vector
    cov_flat: list[float]  # 36 row-major
    d2_total: float
    d2_per_axis: list[float]  # 6 floats
    convergence: int
    iterations: int
    final_per_corner_d2: list[float]  # 4 floats (per-corner Mahalanobis at convergence)
    final_per_corner_irls_weight: list[float]  # 4 floats (Huber weights at convergence)
    # Path B follow-up — GT-corner residual analysis (added 2026-05-03).
    gt_corners_px: list[list[float]]  # 4 × 2 (GT 3D corners projected via intrinsics)
    corner_residuals_px: list[list[float]]  # 4 × 2 (det − gt, per-corner)
    corner_residual_norms_px: list[float]  # 4 floats (‖det − gt‖ per corner)
    # Counterfactual: re-run weighted LM with GT corners as inputs. If `d²_gt`
    # collapses while `d²_total` is large, the miscalibration is in the corner
    # fitter (or upstream of the LM); otherwise it sits in the LM / model.
    d2_gt_corners: float


def _select_detection(batch: locus.DetectionBatch, gt_tag_id: int) -> int | None:
    if len(batch) == 0:
        return None
    ids = np.asarray(batch.ids)
    matches = np.where(ids == gt_tag_id)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def run_audit(
    config_name: str,
    profile: str,
    output_dir: Path,
    rayon_threads: str | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = HubDatasetLoader()
    ds = loader.load_dataset(config_name)
    if ds.intrinsics is None or ds.tag_size is None:
        raise RuntimeError(f"Hub dataset {config_name} missing intrinsics or tag_size")

    intrinsics = ds.intrinsics
    tag_size = float(ds.tag_size)

    cfg = DetectorConfig.from_profile(profile)  # pyright: ignore[reportArgumentType]
    detector = locus.Detector(config=cfg, families=[locus.TagFamily.AprilTag36h11])
    sigma_n_sq = float(cfg.pose.sigma_n_sq)
    tikhonov_alpha_max = float(cfg.pose.tikhonov_alpha_max)
    structure_tensor_radius = int(cfg.pose.structure_tensor_radius)

    samples: list[SceneSample] = []
    skipped = {"no_image": 0, "no_detection": 0, "lm_singular": 0, "no_gt_pose": 0}

    image_keys = sorted(ds.gt_map.keys())
    for img_name in tqdm(image_keys, desc="audit"):
        gt_for_img = ds.gt_map[img_name]
        tags = gt_for_img["tags"]
        if not tags:
            continue
        gt_tag_id, gt_tag_data = next(iter(tags.items()))
        gt_pose = gt_tag_data.get("pose")
        if gt_pose is None:
            skipped["no_gt_pose"] += 1
            continue

        img = cv2.imread(str(ds.images_dir / img_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped["no_image"] += 1
            continue

        batch = detector.detect(
            img,
            intrinsics=intrinsics,
            tag_size=tag_size,
            pose_estimation_mode=locus.PoseEstimationMode.Accurate,
        )
        det_idx = _select_detection(batch, int(gt_tag_id))
        if det_idx is None:
            skipped["no_detection"] += 1
            continue

        det_corners = np.asarray(batch.corners[det_idx], dtype=np.float64)
        if batch.poses is None:
            skipped["no_detection"] += 1
            continue
        det_pose = np.asarray(batch.poses[det_idx], dtype=np.float64)

        covs = []
        for c in det_corners:
            cov = lb.compute_corner_covariance(
                img,
                float(c[0]),
                float(c[1]),
                tikhonov_alpha_max,
                sigma_n_sq,
                structure_tensor_radius,
            )
            covs.append(list(cov))

        det_quat = list(det_pose[3:7].astype(float))
        det_trans = list(det_pose[0:3].astype(float))

        telem = lb.refine_pose_lm_weighted_with_telemetry(
            intrinsics,
            det_corners.tolist(),
            tag_size,
            det_quat,
            det_trans,
            covs,
        )

        cov_flat = np.asarray(telem["covariance"], dtype=np.float64)
        cov6 = cov_flat.reshape(6, 6)

        # Refined pose returned by the bench LM (should match production).
        refined = telem["pose"]
        ref_trans = np.asarray(refined["translation"], dtype=np.float64)
        ref_quat = np.asarray(refined["quaternion"], dtype=np.float64)

        delta = _se3_residual(
            ref_trans,
            ref_quat,
            np.asarray(gt_pose[0:3], dtype=np.float64),
            np.asarray(gt_pose[3:7], dtype=np.float64),
        )

        try:
            cov_inv = np.linalg.inv(cov6)
        except np.linalg.LinAlgError:
            skipped["lm_singular"] += 1
            continue

        d2_total = float(delta @ cov_inv @ delta)

        # Per-axis 1-DOF Mahalanobis: δᵢ² / Σᵢᵢ.
        diag = np.diag(cov6)
        d2_axis = (delta * delta) / np.where(diag > 1e-30, diag, 1e-30)

        # GT-corner residual analysis (Path B follow-up).
        gt_trans = np.asarray(gt_pose[0:3], dtype=np.float64)
        gt_quat = np.asarray(gt_pose[3:7], dtype=np.float64)
        gt_corners = _project_canonical_tag_corners(intrinsics, gt_trans, gt_quat, tag_size)
        corner_residuals = det_corners - gt_corners  # (4, 2)
        corner_residual_norms = np.linalg.norm(corner_residuals, axis=1)  # (4,)

        # Counterfactual LM with GT corners as inputs. Σ_corner is recomputed at
        # GT pixel locations (they are sub-pixel so the structure tensor still
        # samples the same neighbourhood, modulo bilinear smoothing). Initial
        # pose seeded from GT to remove init-bias from the comparison.
        gt_covs = []
        for c in gt_corners:
            cov_gt = lb.compute_corner_covariance(
                img,
                float(c[0]),
                float(c[1]),
                tikhonov_alpha_max,
                sigma_n_sq,
                structure_tensor_radius,
            )
            gt_covs.append(list(cov_gt))
        telem_gt = lb.refine_pose_lm_weighted_with_telemetry(
            intrinsics,
            gt_corners.tolist(),
            tag_size,
            list(gt_quat.astype(float)),
            list(gt_trans.astype(float)),
            gt_covs,
        )
        cov_gt_flat = np.asarray(telem_gt["covariance"], dtype=np.float64)
        cov_gt6 = cov_gt_flat.reshape(6, 6)
        ref_gt = telem_gt["pose"]
        ref_gt_trans = np.asarray(ref_gt["translation"], dtype=np.float64)
        ref_gt_quat = np.asarray(ref_gt["quaternion"], dtype=np.float64)
        delta_gt = _se3_residual(ref_gt_trans, ref_gt_quat, gt_trans, gt_quat)
        try:
            cov_gt_inv = np.linalg.inv(cov_gt6)
            d2_gt = float(delta_gt @ cov_gt_inv @ delta_gt)
        except np.linalg.LinAlgError:
            d2_gt = float("nan")

        samples.append(
            SceneSample(
                scene_id=img_name.removesuffix(".png"),
                tag_id=int(gt_tag_id),
                delta_se3=delta.tolist(),
                cov_flat=cov_flat.tolist(),
                d2_total=d2_total,
                d2_per_axis=d2_axis.tolist(),
                convergence=int(telem["convergence"]),
                iterations=int(telem["iterations"]),
                final_per_corner_d2=list(map(float, telem["final_per_corner_d2"])),
                final_per_corner_irls_weight=list(
                    map(float, telem["final_per_corner_irls_weight"])
                ),
                gt_corners_px=gt_corners.tolist(),
                corner_residuals_px=corner_residuals.tolist(),
                corner_residual_norms_px=corner_residual_norms.tolist(),
                d2_gt_corners=d2_gt,
            )
        )

    if not samples:
        raise RuntimeError("No samples collected — every scene was skipped.")

    d2 = np.asarray([s.d2_total for s in samples], dtype=np.float64)
    edges, pmf_emp = _empirical_histogram_pmf(d2, KL_MAX_X, KL_NUM_BINS)
    pmf_chi2 = _chi2_bin_pmf(edges, k=CHI2_K)
    kl = _kl_divergence(pmf_emp, pmf_chi2)

    # Per-axis stats (each in 1-DOF reference).
    axis_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    per_axis_stats: dict[str, dict[str, float | str]] = {}
    for i, name in enumerate(axis_names):
        d2_i = np.asarray([s.d2_per_axis[i] for s in samples], dtype=np.float64)
        emp_mean = float(np.mean(d2_i))
        emp_var = float(np.var(d2_i, ddof=1)) if d2_i.size > 1 else float("nan")
        # Calibration ratio: empirical / theoretical mean.
        ratio = emp_mean / CHI2_1DOF_MEAN
        if ratio < 0.5:
            verdict = "over-estimating uncertainty (Σ too large)"
        elif ratio > 2.0:
            verdict = "under-estimating uncertainty (Σ too small)"
        else:
            verdict = "near-calibrated"
        per_axis_stats[name] = {
            "empirical_mean_d2": emp_mean,
            "empirical_var_d2": emp_var,
            "ideal_mean_d2": CHI2_1DOF_MEAN,
            "ideal_var_d2": CHI2_1DOF_VAR,
            "calibration_ratio": ratio,
            "verdict": verdict,
        }

    # Q-Q plot: empirical d² quantiles vs χ²(6) quantiles.
    d2_sorted = np.sort(d2)
    n = len(d2_sorted)
    plot_pos = (np.arange(1, n + 1) - 0.5) / n
    theo_q = _chi2_ppf(plot_pos, k=CHI2_K)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(theo_q, d2_sorted, "o", markersize=3, alpha=0.7, label="empirical")
    lim = max(float(np.max(theo_q)), float(np.max(d2_sorted)), 1.0)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Theoretical χ²(6) quantile")
    ax.set_ylabel("Empirical d² quantile")
    ax.set_title(
        f"Pose-cov Q-Q plot — {config_name}\n"
        f"n={n}, mean={float(d2.mean()):.2f} (ideal 6.00), KL={kl:.4f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    qq_path = output_dir / "qq_plot.png"
    fig.tight_layout()
    fig.savefig(qq_path, dpi=120)
    plt.close(fig)

    # Histogram comparison plot.
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    ax2.bar(centers, pmf_emp, width=width, alpha=0.5, label="empirical")
    ax2.plot(centers, pmf_chi2, "r-", linewidth=2, label="χ²(6)")
    ax2.set_xlabel("Mahalanobis d²")
    ax2.set_ylabel("PMF")
    ax2.set_title(f"d² histogram vs χ²(6) — {config_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    hist_path = output_dir / "histogram.png"
    fig2.tight_layout()
    fig2.savefig(hist_path, dpi=120)
    plt.close(fig2)

    # Aggregate report.
    if kl <= 0.1:
        kl_verdict = "well-calibrated (KL ≤ 0.1)"
    elif kl <= 0.5:
        kl_verdict = "drift (0.1 < KL ≤ 0.5)"
    else:
        kl_verdict = "miscalibrated (KL > 0.5)"

    report = {
        "config_name": config_name,
        "profile": profile,
        "iso_date": dt.date.today().isoformat(),
        "hardware": _capture_hardware_metadata(rayon_threads),
        "n_scenes_total": len(image_keys),
        "n_samples": len(samples),
        "skipped": skipped,
        "global_d2": {
            "empirical_mean": float(d2.mean()),
            "empirical_var": float(d2.var(ddof=1)) if len(d2) > 1 else float("nan"),
            "ideal_mean": CHI2_MEAN,
            "ideal_var": CHI2_VAR,
            "kl_divergence_to_chi2_6": kl,
            "kl_verdict": kl_verdict,
            "p10": float(np.percentile(d2, 10)),
            "p50": float(np.percentile(d2, 50)),
            "p90": float(np.percentile(d2, 90)),
            "p99": float(np.percentile(d2, 99)),
        },
        "per_axis": per_axis_stats,
        "plots": {
            "qq_plot": str(qq_path.relative_to(output_dir)),
            "histogram": str(hist_path.relative_to(output_dir)),
        },
        "pose_config": {
            "sigma_n_sq": sigma_n_sq,
            "tikhonov_alpha_max": tikhonov_alpha_max,
            "structure_tensor_radius": structure_tensor_radius,
        },
    }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    samples_payload = [
        {
            "scene_id": s.scene_id,
            "tag_id": s.tag_id,
            "delta_se3": s.delta_se3,
            "cov_flat": s.cov_flat,
            "d2_total": s.d2_total,
            "d2_per_axis": s.d2_per_axis,
            "convergence": s.convergence,
            "iterations": s.iterations,
            "final_per_corner_d2": s.final_per_corner_d2,
            "final_per_corner_irls_weight": s.final_per_corner_irls_weight,
            "gt_corners_px": s.gt_corners_px,
            "corner_residuals_px": s.corner_residuals_px,
            "corner_residual_norms_px": s.corner_residual_norms_px,
            "d2_gt_corners": s.d2_gt_corners,
        }
        for s in samples
    ]
    (output_dir / "samples.json").write_text(json.dumps(samples_payload, indent=2))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hub-config",
        type=str,
        default="locus_v1_tag36h11_1920x1080",
        help="Hub dataset config name to audit.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="high_accuracy",
        help="Locus detector profile (default: high_accuracy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: diagnostics/pose_cov_audit_<ISO-date>/).",
    )
    parser.add_argument(
        "--rayon-threads",
        type=str,
        default=None,
        help="Value of RAYON_NUM_THREADS during the run (logged into the report).",
    )
    args = parser.parse_args()

    today = dt.date.today().isoformat()
    output_dir = args.output_dir or Path("diagnostics") / f"pose_cov_audit_{today}"

    report = run_audit(
        config_name=args.hub_config,
        profile=args.profile,
        output_dir=output_dir,
        rayon_threads=args.rayon_threads,
    )

    g = report["global_d2"]
    print()
    print(f"  samples:           {report['n_samples']} / {report['n_scenes_total']}")
    print(f"  global d² mean:    {g['empirical_mean']:.4f} (ideal {g['ideal_mean']:.1f})")
    print(f"  global d² var:     {g['empirical_var']:.4f} (ideal {g['ideal_var']:.1f})")
    print(f"  KL(emp ‖ χ²₆):    {g['kl_divergence_to_chi2_6']:.4f} → {g['kl_verdict']}")
    print(f"  output:            {output_dir}/report.json")
    print()
    print("  per-axis calibration ratios (1.0 = ideal):")
    for name, stats_axis in report["per_axis"].items():
        print(
            f"    {name:>2}: ratio={stats_axis['calibration_ratio']:.3f}  ({stats_axis['verdict']})"
        )


if __name__ == "__main__":
    main()
