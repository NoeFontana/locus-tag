"""Corner-RMSE and reprojection-RMSE eval for the model-edge pose stage.

Complements ``model_edge_eval.py`` (which reports pose rot/trans percentiles) with
the two image-space accuracy metrics, baseline vs ``pose_edge_refinement_enabled``:

- **2D corner RMSE** ``||detected_corner - GT_corner||`` (px), order-preserving.
  Expected **identical** — the stage never writes ``batch.corners``; it emits only
  the pose. The ``corners`` column asserts byte-identity vs baseline.
- **Reprojection RMSE** ``||project(est_pose, model_corner) - GT_corner||`` (px):
  pose accuracy in image space, the same quantity as the Rust ``hub.rs``
  ``mean_reprojection_rmse``. Changes with the refined pose.

render-tag is pinhole (``DistortionModel.Pinhole``, no ``dist_coeffs``), so a plain
pinhole projection is exact; the model-corner order ``[BL, BR, TR, TL]`` is validated
to reproject the GT pose to 0.000 px.

Usage:
    PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache RAYON_NUM_THREADS=1 \\
      uv run --group bench tools/bench/model_edge_rmse.py
"""

from __future__ import annotations

import os
from typing import Any, cast

import cv2
import locus
import numpy as np

from tools.bench.metrics import corner_rmse_px, percentiles
from tools.bench.utils import HUB_CACHE_DIR, HubDatasetLoader, _quat_to_rot

CONFIGS = (
    (os.environ["RENDER_TAG_SOTA_CONFIG"],)
    if "RENDER_TAG_SOTA_CONFIG" in os.environ
    else (
        "locus_v1_tag36h11_640x480",
        "locus_v1_tag36h11_1280x720",
        "locus_v1_tag36h11_1920x1080",
        "locus_v1_tag36h11_3840x2160",
    )
)

# Base profile whose corner baseline the edge stage is measured against (see
# model_edge_eval.py). Defaults to shipped `high_accuracy`.
BASE_PROFILE = os.environ.get("MODEL_EDGE_BASE_PROFILE", "high_accuracy")


def _project(model: np.ndarray, pose: np.ndarray, intr: object) -> np.ndarray:
    """Pinhole-project body-frame `model` points (N×3) through `pose`
    `[tx,ty,tz,qx,qy,qz,qw]`. Reuses the shared `utils._quat_to_rot` so the
    quaternion convention cannot drift from the pose-error harness. **Pinhole
    only** — callers must assert the intrinsics carry no distortion.
    """
    cam = (_quat_to_rot(*pose[3:7]) @ model.T).T + pose[:3]
    u = intr.fx * cam[:, 0] / cam[:, 2] + intr.cx  # type: ignore[attr-defined]
    v = intr.fy * cam[:, 1] / cam[:, 2] + intr.cy  # type: ignore[attr-defined]
    return np.stack([u, v], 1)


def _run(
    loader: HubDatasetLoader, cfg_name: str, enabled: bool
) -> tuple[list[float], list[float], dict[tuple[str, int], np.ndarray]]:
    ds = loader.load_dataset(cfg_name)
    intr = ds.intrinsics
    assert intr is not None and ds.tag_size is not None
    # The reprojection here is pinhole-only; refuse to silently report wrong
    # numbers on a distorted dataset (Locus's own Pose::project is distortion-aware).
    assert not list(intr.dist_coeffs), (
        f"{cfg_name}: model_edge_rmse reprojection is pinhole-only but the dataset "
        f"carries distortion ({intr.distortion_model}, {list(intr.dist_coeffs)})"
    )
    s = ds.tag_size / 2.0
    model = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])  # BL,BR,TR,TL
    # Self-check the model-corner order + projection convention against the GT:
    # reprojecting a GT pose through `model` must land on the GT corners (~0 px).
    # Catches a corner-order / quaternion-convention drift at runtime instead of
    # silently inflating both baseline and edge equally.
    k0 = sorted(ds.gt_map.keys())[0]
    g0 = ds.gt_map[k0]["tags"]
    t0 = g0[sorted(g0.keys())[0]]
    gt_reproj = corner_rmse_px(
        _project(model, np.asarray(t0["pose"], dtype=np.float64), intr),
        np.asarray(t0["corners"], dtype=np.float64),
    )
    assert gt_reproj < 1e-3, (
        f"{cfg_name}: GT-pose reprojection is {gt_reproj:.4f} px (expected ~0); "
        f"model-corner order or quaternion convention mismatch"
    )
    cfg = locus.DetectorConfig.from_profile(cast(Any, BASE_PROFILE))
    cfg.pose.pose_edge_refinement_enabled = enabled
    det = locus.Detector(config=cfg, families=[locus.TagFamily.AprilTag36h11], threads=1)

    corner_rmse: list[float] = []
    reproj_rmse: list[float] = []
    corners: dict[tuple[str, int], np.ndarray] = {}
    for img_name in sorted(ds.gt_map.keys()):
        img_path = ds.images_dir / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        batch = det.detect(img, intrinsics=intr, tag_size=ds.tag_size)
        gt_tags = ds.gt_map[img_name]["tags"]
        poses = getattr(batch, "poses", None)
        for i in range(len(batch.ids)):
            tid = int(batch.ids[i])
            if tid not in gt_tags:
                continue
            gtc = np.asarray(gt_tags[tid]["corners"], dtype=np.float64)
            det_c = np.asarray(batch.corners[i], dtype=np.float64)
            corner_rmse.append(corner_rmse_px(det_c, gtc))
            corners[(img_name, tid)] = det_c
            if poses is not None:
                p = np.asarray(poses[i], dtype=np.float64)
                reproj_rmse.append(corner_rmse_px(_project(model, p, intr), gtc))
    return corner_rmse, reproj_rmse, corners


def main() -> None:
    loader = HubDatasetLoader(root=HUB_CACHE_DIR)
    print(f"base profile: {BASE_PROFILE}  |  pose mode: Accurate (intrinsics + tag_size)")
    print(
        f"{'config':<12} {'variant':<9} | {'corner_rmse 2D (px)':<24} | "
        f"{'reproj_rmse (px)':<24} | corners"
    )
    print(
        f"{'':12} {'':9} | {'mean':>7} {'p95':>7} {'p99':>6} | {'mean':>7} {'p95':>7} {'p99':>6} |"
    )
    print("-" * 90)
    for cfg_name in CONFIGS:
        short = cfg_name.replace("locus_v1_tag36h11_", "")
        base_corners: dict[tuple[str, int], np.ndarray] = {}
        for enabled, lbl in [(False, "baseline"), (True, "edge")]:
            crmse, rrmse, cmap = _run(loader, cfg_name, enabled)
            cm, (cp95, cp99) = float(np.mean(crmse)), percentiles(crmse, [95, 99])
            rm, (rp95, rp99) = float(np.mean(rrmse)), percentiles(rrmse, [95, 99])
            if lbl == "baseline":
                base_corners = cmap
                ident = "n/a"
            else:
                same = len(base_corners) == len(cmap) and all(
                    k in base_corners and np.array_equal(base_corners[k], v)
                    for k, v in cmap.items()
                )
                ident = "IDENTICAL" if same else "DIFFER"
            print(
                f"{short:<12} {lbl:<9} | {cm:>7.4f} {cp95:>7.4f} {cp99:>6.4f} | "
                f"{rm:>7.4f} {rp95:>7.4f} {rp99:>6.4f} | {ident}"
            )


if __name__ == "__main__":
    main()
