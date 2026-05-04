"""Validate the S3 Rust port end-to-end on scene_0008.

Runs `Detector.detect()` twice — once with `edlines_use_anchor_walk = False`
(baseline), once with `True` (S3 path active) — and compares the corner
residuals of the canonical c1 (the failing corner per
``scene_0008_root_cause_2026-05-03.md §1``).

Acceptance per design memo §4.3:
  - max corner ‖Δ‖ < 1.0 px → ship-clean
  - 1.0 ≤ ‖Δ‖ < 2.0 px → conditional ship
  - ‖Δ‖ ≥ 2.0 px → falsified

Usage::

    PYTHONPATH=. uv run --group bench tools/bench/edlines_s3_validate_scene_0008.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import locus
import numpy as np
from locus._config import DetectorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.bench.utils import HubDatasetLoader  # noqa: E402

HUB_CONFIG = "locus_v1_tag36h11_1920x1080"
SCENE = "scene_0008_cam_0000"


def project_canonical_corners(
    intrinsics: locus.CameraIntrinsics, t: np.ndarray, q_xyzw: np.ndarray, tag_size: float
) -> np.ndarray:
    """Project the 4 canonical centred tag corners through (R,t,K)."""
    s = tag_size / 2.0
    obj = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float64)
    qx, qy, qz, qw = q_xyzw
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    world = obj @ rot.T + t
    z = world[:, 2]
    proj = world[:, :2] / z[:, None]
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx_v, cy_v = float(intrinsics.cx), float(intrinsics.cy)
    return np.column_stack([proj[:, 0] * fx + cx_v, proj[:, 1] * fy + cy_v])


def run_one(cfg: DetectorConfig, ds, label: str) -> None:
    img = cv2.imread(str(ds.images_dir / f"{SCENE}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"missing scene image for {SCENE}")
    gt_for_img = ds.gt_map[f"{SCENE}.png"]
    tag_id, tag_data = next(iter(gt_for_img["tags"].items()))
    pose_d = np.asarray(tag_data["pose"], dtype=np.float64)
    # Flat 7-vector: [tx, ty, tz, qx, qy, qz, qw].
    gt_t = pose_d[:3]
    gt_q = pose_d[3:7]
    tag_size = float(ds.tag_size)
    intrinsics = ds.intrinsics

    detector = locus.Detector(config=cfg, families=[locus.TagFamily.AprilTag36h11])
    batch = detector.detect(
        img,
        intrinsics=intrinsics,
        tag_size=tag_size,
        pose_estimation_mode=locus.PoseEstimationMode.Accurate,
    )
    # DetectionBatch exposes corners via attributes, not detections().
    n = int(batch.ids.shape[0])
    if n == 0:
        print(f"[{label}] NO DETECTION")
        return
    # Filter to the matching tag id.
    matches = np.where(batch.ids == int(tag_id))[0]
    if matches.size == 0:
        print(f"[{label}] NO MATCHING TAG (detected ids: {list(batch.ids)})")
        return
    idx = int(matches[0])
    det_corners = np.asarray(batch.corners[idx], dtype=np.float64)
    gt_corners = project_canonical_corners(intrinsics, gt_t, gt_q, tag_size)
    deltas = np.linalg.norm(det_corners - gt_corners, axis=1)
    print(f"[{label}] corner residuals (vs GT, canonical order):")
    label_names = ["c0 (BL)", "c1 (BR/img-TL)", "c2 (TR)", "c3 (TL/img-BR)"]
    for nm, d in zip(label_names, deltas, strict=True):
        print(f"  {nm:20s}: {d:.3f} px")
    print(f"  max ‖Δ‖: {deltas.max():.3f} px, mean: {deltas.mean():.3f} px")


def main() -> None:
    loader = HubDatasetLoader()
    ds = loader.load_dataset(HUB_CONFIG)
    print(f"Image: {SCENE}.png\n")

    print("=== Baseline (use_anchor_walk = False, current EdLines) ===")
    cfg_off = DetectorConfig.from_profile("high_accuracy")
    cfg_off.quad.edlines_use_anchor_walk = False
    run_one(cfg_off, ds, "OFF")

    print("\n=== S3 (use_anchor_walk = True, gradient-anchor walk) ===")
    cfg_on = DetectorConfig.from_profile("high_accuracy")
    cfg_on.quad.edlines_use_anchor_walk = True
    run_one(cfg_on, ds, "S3")


if __name__ == "__main__":
    main()
