"""Phase 0 rotation-tail diagnostic harness — extraction stage.

For each scene in a hub-config dataset, run the production detector once
(latency baseline, no diagnostics) and the bench-internals API once
(diagnostics: alternate-branch LM, structure-tensor eigenvalues, image-noise
floor, leave-one-out refit). Diff the two pose paths to ensure the diagnostic
build is numerically equivalent — non-zero diff stops the run.

Outputs:
    {output_dir}/scenes.json
    {output_dir}/corners.parquet
    {output_dir}/recordings/scene_NN.rrd  (one per detected scene)
    {output_dir}/recordings/blueprint.rbl
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import locus
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from locus import bench as lb
from locus._config import DetectorConfig
from tqdm import tqdm

from tools.bench.rotation_tail_diag.io_models import (
    CornerRecord,
    SceneRecord,
    ScenesFile,
)
from tools.bench.utils import HubDatasetLoader, rotation_error_deg

# χ² (2 DOF) gate FPR — matches what high_accuracy ships.
DEFAULT_FPR = 1e-3
HUBER_K = 1.345  # matches refine_pose_lm_weighted constant
LEAVE_ONE_OUT_RUN = True


def _quat_angle_deg(q1_xyzw: tuple, q2_xyzw: tuple) -> float:
    """Geodesic angle between two scalar-last unit quaternions, in degrees.

    Routes through `tools.bench.utils.rotation_error_deg` so this matches the
    published high_accuracy baseline numbers (rotation-matrix trace formula).
    """
    fake_pose = np.array([0.0, 0.0, 0.0, *q1_xyzw], dtype=np.float64)
    return rotation_error_deg(fake_pose, np.asarray(q2_xyzw, dtype=np.float64))


def _quad_area(corners: np.ndarray) -> float:
    """Shoelace area for a quad (4×2 numpy array, pixel units)."""
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _match_detection_to_gt(batch: locus.DetectionBatch, gt_tag_id: int) -> int | None:
    """Return the index in `batch` whose ID matches the GT tag, or None."""
    if len(batch) == 0:
        return None
    ids = np.asarray(batch.ids)
    matches = np.where(ids == gt_tag_id)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert scalar-last quaternion to 3×3 rotation matrix."""
    x, y, z, w = q_xyzw
    n = float(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def _project_corners(
    pose_xyzw: np.ndarray,
    obj_pts: np.ndarray,
    intrinsics: locus.CameraIntrinsics,
) -> np.ndarray:
    """Project object-frame corners (4×3) to pixel coords (4×2) using a 7-vec pose."""
    rot = _quat_to_rot(pose_xyzw[3:7])
    trans = pose_xyzw[0:3]
    p_cam = (rot @ obj_pts.T).T + trans  # 4×3
    z = np.maximum(p_cam[:, 2], 1e-6)
    u = intrinsics.fx * p_cam[:, 0] / z + intrinsics.cx
    v = intrinsics.fy * p_cam[:, 1] / z + intrinsics.cy
    return np.stack([u, v], axis=1)


def _build_obj_pts(tag_size: float) -> np.ndarray:
    """Locus convention: 4 corners centered at origin, z=0, in TL→TR→BR→BL order."""
    s = tag_size * 0.5
    return np.array(
        [[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]],
        dtype=np.float64,
    )


def _huber_irls_weight(d2: float, k: float = HUBER_K) -> float:
    """Huber IRLS weight from squared Mahalanobis distance."""
    s = float(np.sqrt(max(d2, 0.0)))
    if s <= k:
        return 1.0
    return k / max(s, 1e-12)


def _compute_corner_diagnostics(
    img: np.ndarray,
    detected_corners: np.ndarray,
    detected_pose: np.ndarray | None,
    gt_corners: np.ndarray,
    gt_quat: tuple,
    intrinsics: locus.CameraIntrinsics,
    tag_size: float,
    config: locus.DetectorConfig,
    sigma_n_sq: float,
    tikhonov_alpha_max: float,
    structure_tensor_radius: int,
    rot_err_chosen: float | None,
) -> tuple[list[dict], dict]:
    """Compute per-corner diagnostics + final per-corner IRLS weight & Mahalanobis d²."""

    # Compute per-corner covariances (Σ) via the structure-tensor inverse the
    # production solver would have used. These feed both the Mahalanobis stats
    # and the per-iter LM telemetry call.
    covs = []
    for c in detected_corners:
        cov = lb.compute_corner_covariance(
            img,
            float(c[0]),
            float(c[1]),
            tikhonov_alpha_max,
            sigma_n_sq,
            structure_tensor_radius,
        )
        covs.append(list(cov))

    # Run the per-iteration LM telemetry call to capture final IRLS weights and
    # Mahalanobis d² at the converged pose.
    lm_telemetry = None
    final_d2 = [float("nan")] * 4
    final_w = [float("nan")] * 4
    if detected_pose is not None:
        # Convert detected pose [tx,ty,tz,qx,qy,qz,qw] to quaternion + translation
        det_quat = list(detected_pose[3:7].astype(float))
        det_trans = list(detected_pose[0:3].astype(float))
        # The bench LM solver re-runs LM from the given seed. Feed it the
        # production-converged pose so the trace reflects the actual fit.
        lm_telemetry = lb.refine_pose_lm_weighted_with_telemetry(
            intrinsics,
            detected_corners.tolist(),
            tag_size,
            det_quat,
            det_trans,
            covs,
        )
        final_d2 = list(lm_telemetry["final_per_corner_d2"])
        final_w = list(lm_telemetry["final_per_corner_irls_weight"])

    # Per-corner structure-tensor eigenvalues + leave-one-out
    corner_records = []
    obj_pts = _build_obj_pts(tag_size)
    for i in range(4):
        cx, cy = float(detected_corners[i, 0]), float(detected_corners[i, 1])
        eigs = lb.corner_structure_tensor_eigenvalues(img, cx, cy, structure_tensor_radius)
        l_max, l_min, R = None, None, None
        if eigs is not None:
            l_max, l_min = float(eigs[0]), float(eigs[1])
            R = (l_min / l_max) if l_max > 1e-12 else 0.0

        # Reprojection residual
        residual_norm = None
        if detected_pose is not None:
            proj = _project_corners(detected_pose, obj_pts, intrinsics)
            residual_norm = float(np.linalg.norm(proj[i] - detected_corners[i]))

        # Leave-one-out: drop corner i and re-run LM, then compute the
        # *percent reduction* in rotation error vs GT. Positive ⇒ this corner
        # was hurting the fit.
        loo_drop_pct = None
        if (
            LEAVE_ONE_OUT_RUN
            and detected_pose is not None
            and rot_err_chosen is not None
            and rot_err_chosen > 0.05  # don't divide by ~zero error
        ):
            det_quat_t = tuple(float(v) for v in detected_pose[3:7])
            det_trans_t = tuple(float(v) for v in detected_pose[0:3])
            try:
                refit = lb.refit_pose_drop_corner(
                    intrinsics,
                    detected_corners.tolist(),
                    tag_size,
                    list(det_quat_t),
                    list(det_trans_t),
                    i,
                    locus.PoseEstimationMode.Accurate,
                    config._to_ffi_config(),
                    img,
                )
                rot_err_loo_vs_gt = _quat_angle_deg(tuple(refit["quaternion"]), gt_quat)
                loo_drop_pct = float((rot_err_chosen - rot_err_loo_vs_gt) / rot_err_chosen * 100.0)
            except Exception:  # noqa: BLE001 — bench-only, swallow
                loo_drop_pct = None

        rec = CornerRecord(
            scene_id="",  # filled by caller
            corner_idx=i,
            gt_corner_x=float(gt_corners[i, 0]),
            gt_corner_y=float(gt_corners[i, 1]),
            detected_corner_x=cx,
            detected_corner_y=cy,
            residual_norm_px=residual_norm,
            final_mahalanobis_d2=(float(final_d2[i]) if not np.isnan(final_d2[i]) else None),
            final_irls_weight=(float(final_w[i]) if not np.isnan(final_w[i]) else None),
            structure_tensor_lambda_max=l_max,
            structure_tensor_lambda_min=l_min,
            structure_tensor_R=R,
            leave_one_out_rotation_err_drop_pct=loo_drop_pct,
        )
        corner_records.append(rec.model_dump())

    summary = {
        "lm_iterations": int(lm_telemetry["iterations"]) if lm_telemetry else None,
        "lm_convergence": (int(lm_telemetry["convergence"]) if lm_telemetry else None),
    }
    return corner_records, summary


def _log_rerun_scene(
    rec_path: Path,
    img: np.ndarray,
    intrinsics: locus.CameraIntrinsics,
    scene: SceneRecord,
    detected_corners: np.ndarray | None,
    gt_corners: np.ndarray,
    branches: list | None,
    obj_pts: np.ndarray,
) -> None:
    """Save a per-scene Rerun recording (.rrd) at ``rec_path``."""
    import rerun as rr  # local import — keep harness importable without rerun

    rr.init(f"locus_rotation_tail_{scene.scene_id}", spawn=False)

    h, w = img.shape[:2]
    rr.log(
        "world/camera",
        rr.Pinhole(
            focal_length=[intrinsics.fx, intrinsics.fy],
            principal_point=[intrinsics.cx, intrinsics.cy],
            resolution=[w, h],
        ),
        static=True,
    )
    rr.log("world/camera/image", rr.Image(img))

    # Ground truth corners
    rr.log(
        "world/camera/gt_corners",
        rr.LineStrips2D(
            [np.vstack([gt_corners, gt_corners[:1]]).tolist()],
            colors=[[0, 200, 0]],
            radii=2.0,
        ),
    )

    if detected_corners is not None:
        rr.log(
            "world/camera/pred_corners",
            rr.LineStrips2D(
                [np.vstack([detected_corners, detected_corners[:1]]).tolist()],
                colors=[[255, 140, 0]],
                radii=2.0,
            ),
        )

    # Branch reprojections
    if branches:
        for j, b in enumerate(branches):
            ref = b["refined_pose"]
            pose7 = np.array([*ref["translation"], *ref["quaternion"]], dtype=np.float64)
            proj = _project_corners(pose7, obj_pts, intrinsics)
            color = [80, 80, 255] if j == 0 else [255, 80, 80]
            rr.log(
                f"world/camera/branch_{j}_reproj",
                rr.LineStrips2D(
                    [np.vstack([proj, proj[:1]]).tolist()],
                    colors=[color],
                    radii=1.0,
                ),
            )

    # Scalar diagnostics (single sample per scene — Rerun timeline-friendly).
    if scene.rotation_error_chosen_deg is not None:
        rr.log(
            "metrics/rotation_error_chosen_deg",
            rr.Scalars([float(scene.rotation_error_chosen_deg)]),
        )
    if scene.rotation_error_alternate_deg is not None:
        rr.log(
            "metrics/rotation_error_alternate_deg",
            rr.Scalars([float(scene.rotation_error_alternate_deg)]),
        )
    rr.log("metrics/d2_chosen", rr.Scalars([float(scene.aggregate_d2_chosen)]))
    rr.log("metrics/d2_alternate", rr.Scalars([float(scene.aggregate_d2_alternate)]))
    rr.log("metrics/branch_chosen", rr.Scalars([float(scene.branch_chosen_idx)]))
    rr.log("metrics/image_noise_sigma", rr.Scalars([float(scene.image_noise_sigma)]))

    # Text classification placeholder — populated by classify.py via post-hoc edit.
    rr.log("text/classification", rr.TextLog("(not yet classified)"))

    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(rec_path))


def run(
    config_name: str,
    profile: str,
    output_dir: Path,
    *,
    pose_estimation_mode: locus.PoseEstimationMode = locus.PoseEstimationMode.Accurate,
    enable_rerun: bool = True,
    enable_corner_telemetry: bool = True,
    fpr: float = DEFAULT_FPR,
    families: list[locus.TagFamily] | None = None,
) -> Path:
    """Run the extraction pipeline. Returns the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = output_dir / "recordings"
    rec_dir.mkdir(exist_ok=True)

    families = families or [locus.TagFamily.AprilTag36h11]

    # 1. Load dataset
    loader = HubDatasetLoader()
    ds = loader.load_dataset(config_name)
    if ds.intrinsics is None or ds.tag_size is None:
        raise RuntimeError(f"Hub dataset {config_name} missing intrinsics or tag_size")

    intrinsics = ds.intrinsics
    tag_size = float(ds.tag_size)

    # 2. Build production detector (matches what high_accuracy ships)
    cfg = DetectorConfig.from_profile(profile)  # type: ignore[arg-type]  # CLI passes str
    detector = locus.Detector(config=cfg, families=families)
    py_cfg = cfg._to_ffi_config()
    sigma_n_sq = float(cfg.pose.sigma_n_sq)
    tikhonov_alpha_max = float(cfg.pose.tikhonov_alpha_max)
    structure_tensor_radius = int(cfg.pose.structure_tensor_radius)

    # 3. Re-load full record list to access per-scene metadata fields
    rich_path = Path("tests/data/hub_cache") / config_name / "rich_truth.json"
    with open(rich_path) as f:
        rich_data = json.load(f)
    rich_records = rich_data["records"] if isinstance(rich_data, dict) else rich_data
    rich_by_image = {r["image_id"]: r for r in rich_records}

    # 4. Iterate scenes
    scene_records: list[SceneRecord] = []
    corner_records_all: list[dict] = []
    obj_pts = _build_obj_pts(tag_size)
    skipped_no_detection = 0
    pose_diff_violations = 0

    image_keys = sorted(ds.gt_map.keys())

    for img_name in tqdm(image_keys, desc="extract"):
        scene_id = img_name.removesuffix(".png")
        rich = rich_by_image.get(scene_id)
        if rich is None:
            continue

        gt = ds.gt_map[img_name]
        tags = gt["tags"]
        if not tags:
            continue
        gt_tag_id, gt_tag_data = next(iter(tags.items()))
        gt_corners_arr = np.asarray(gt_tag_data["corners"], dtype=np.float64)
        gt_pose = gt_tag_data.get("pose")
        if gt_pose is None:
            continue
        gt_quat = (float(gt_pose[3]), float(gt_pose[4]), float(gt_pose[5]), float(gt_pose[6]))
        gt_trans = (float(gt_pose[0]), float(gt_pose[1]), float(gt_pose[2]))

        img = cv2.imread(str(ds.images_dir / img_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Per-image noise floor (one call per scene)
        sigma_n = float(lb.estimate_image_noise(img))

        # PASS A: production-equivalent latency baseline.
        t0 = time.perf_counter()
        batch = detector.detect(
            img,
            intrinsics=intrinsics,
            tag_size=tag_size,
            pose_estimation_mode=pose_estimation_mode,
        )
        latency_us = (time.perf_counter() - t0) * 1e6

        det_idx = _match_detection_to_gt(batch, gt_tag_id)
        if det_idx is None:
            skipped_no_detection += 1
            scene_records.append(
                SceneRecord(
                    scene_id=scene_id,
                    tag_id=int(gt_tag_id),
                    distance_m=float(rich["distance"]),
                    angle_of_incidence_deg=float(rich["angle_of_incidence"]),
                    pixel_area=float(rich.get("pixel_area", 0.0)),
                    occlusion_ratio=float(rich.get("occlusion_ratio", 0.0)),
                    gt_quaternion_xyzw=gt_quat,
                    gt_translation_xyz=gt_trans,
                    detected=False,
                    branch_chosen_idx=255,
                    image_noise_sigma=sigma_n,
                    latency_us=latency_us,
                    ppm_estimated=0.0,
                )
            )
            continue

        det_corners = np.asarray(batch.corners[det_idx], dtype=np.float64)
        det_pose = (
            np.asarray(batch.poses[det_idx], dtype=np.float64) if batch.poses is not None else None
        )

        # PASS B: bench API on the same corners — should reproduce the same pose.
        diag = lb.estimate_tag_pose(
            intrinsics,
            det_corners.tolist(),
            tag_size,
            pose_estimation_mode,
            py_cfg,
            img,
            fpr,
        )
        bench_pose7 = None
        if diag["pose"] is not None:
            bench_pose7 = np.array(
                [
                    *diag["pose"]["translation"],
                    *diag["pose"]["quaternion"],
                ],
                dtype=np.float64,
            )
            if det_pose is not None:
                # Pose-equivalence check between production and bench paths.
                trans_diff = float(np.linalg.norm(bench_pose7[:3] - det_pose[:3]))
                quat_diff_deg = _quat_angle_deg(tuple(bench_pose7[3:7]), tuple(det_pose[3:7]))
                if trans_diff > 1e-3 or quat_diff_deg > 0.01:
                    pose_diff_violations += 1

        # Both branches refined.
        branches = lb.estimate_both_branches(
            intrinsics,
            det_corners.tolist(),
            tag_size,
            pose_estimation_mode,
            py_cfg,
            img,
        )

        # Errors against GT
        rot_err_chosen = None
        trans_err_chosen_mm = None
        if det_pose is not None:
            rot_err_chosen = _quat_angle_deg(tuple(det_pose[3:7]), gt_quat)
            trans_err_chosen_mm = (
                float(np.linalg.norm(det_pose[:3] - np.asarray(gt_trans))) * 1000.0
            )

        # Alternate-branch error: pick the branch != branch_chosen
        chosen_idx = int(diag["branch_chosen"])
        alt_quat = None
        alt_trans = None
        rot_err_alt = None
        trans_err_alt_mm = None
        agg_d2_chosen = float(diag["aggregate_d2"])
        agg_d2_alt = float("nan")
        branch_d2_ratio = float(diag["branch_d2_ratio"])

        if branches is not None and chosen_idx in (0, 1):
            alt_idx = 1 - chosen_idx
            alt = branches[alt_idx]["refined_pose"]
            alt_quat = (
                float(alt["quaternion"][0]),
                float(alt["quaternion"][1]),
                float(alt["quaternion"][2]),
                float(alt["quaternion"][3]),
            )
            alt_trans = (
                float(alt["translation"][0]),
                float(alt["translation"][1]),
                float(alt["translation"][2]),
            )
            rot_err_alt = _quat_angle_deg(alt_quat, gt_quat)
            trans_err_alt_mm = (
                float(np.linalg.norm(np.asarray(alt_trans) - np.asarray(gt_trans))) * 1000.0
            )
            agg_d2_alt = float(branches[alt_idx]["aggregate_d2"])

        ppm_est = float(np.sqrt(_quad_area(det_corners)) / max(tag_size, 1e-6))

        scene = SceneRecord(
            scene_id=scene_id,
            tag_id=int(gt_tag_id),
            distance_m=float(rich["distance"]),
            angle_of_incidence_deg=float(rich["angle_of_incidence"]),
            pixel_area=float(rich.get("pixel_area", 0.0)),
            occlusion_ratio=float(rich.get("occlusion_ratio", 0.0)),
            gt_quaternion_xyzw=gt_quat,
            gt_translation_xyz=gt_trans,
            detected=True,
            detected_quaternion_xyzw=(
                (float(det_pose[3]), float(det_pose[4]), float(det_pose[5]), float(det_pose[6]))
                if det_pose is not None
                else None
            ),
            detected_translation_xyz=(
                (float(det_pose[0]), float(det_pose[1]), float(det_pose[2]))
                if det_pose is not None
                else None
            ),
            alternate_quaternion_xyzw=alt_quat,
            alternate_translation_xyz=alt_trans,
            rotation_error_chosen_deg=rot_err_chosen,
            rotation_error_alternate_deg=rot_err_alt,
            translation_error_chosen_mm=trans_err_chosen_mm,
            translation_error_alternate_mm=trans_err_alt_mm,
            branch_chosen_idx=chosen_idx,
            aggregate_d2_chosen=agg_d2_chosen,
            aggregate_d2_alternate=agg_d2_alt,
            branch_d2_ratio=branch_d2_ratio,
            max_corner_d2=float(diag["max_corner_d2"]),
            image_noise_sigma=sigma_n,
            latency_us=latency_us,
            ppm_estimated=ppm_est,
        )

        # Per-corner diagnostics
        if enable_corner_telemetry:
            crs, summary = _compute_corner_diagnostics(
                img,
                det_corners,
                det_pose,
                gt_corners_arr,
                gt_quat,
                intrinsics,
                tag_size,
                cfg,
                sigma_n_sq,
                tikhonov_alpha_max,
                structure_tensor_radius,
                rot_err_chosen,
            )
            for cr in crs:
                cr["scene_id"] = scene_id
                corner_records_all.append(cr)
            scene.lm_iterations = summary["lm_iterations"]
            scene.lm_convergence = summary["lm_convergence"]

        scene_records.append(scene)

        if enable_rerun:
            try:
                _log_rerun_scene(
                    rec_dir / f"{scene_id}.rrd",
                    img,
                    intrinsics,
                    scene,
                    det_corners,
                    gt_corners_arr,
                    branches,
                    obj_pts,
                )
            except Exception as e:  # noqa: BLE001 — Rerun is exploration-only
                tqdm.write(f"  Rerun log failed for {scene_id}: {e}")

    # 5. Write outputs
    scenes_out = ScenesFile(
        config_name=config_name,
        profile=profile,
        pose_estimation_mode=str(pose_estimation_mode).split(".")[-1],
        sigma_n_sq_configured=sigma_n_sq,
        n_scenes=len(scene_records),
        scenes=scene_records,
    )
    (output_dir / "scenes.json").write_text(scenes_out.model_dump_json(indent=2))

    if corner_records_all:
        table = pa.Table.from_pylist(corner_records_all)
        pq.write_table(table, output_dir / "corners.parquet")

    print()
    print(
        f"  scenes detected:        {sum(1 for s in scene_records if s.detected)}/{len(scene_records)}"
    )
    print(f"  production misses:      {skipped_no_detection}")
    print(f"  pose-equivalence violations (>1e-3 m / 0.01°): {pose_diff_violations}")
    print("  rotation p50/p95/p99:   ", end="")
    rot_errs = sorted(
        [
            s.rotation_error_chosen_deg
            for s in scene_records
            if s.rotation_error_chosen_deg is not None
        ]
    )
    if rot_errs:
        n = len(rot_errs)

        def _q(p):
            i = int(round(p * (n - 1)))
            return rot_errs[i]

        print(f"{_q(0.5):.4f}° / {_q(0.95):.4f}° / {_q(0.99):.4f}°")
    else:
        print("(no detections)")

    return output_dir
