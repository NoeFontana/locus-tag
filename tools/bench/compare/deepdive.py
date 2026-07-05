"""Rerun per-instance deep-dive: visually inspect the worst-Locus examples.

Corners aren't stored in the parquet, so this re-runs detection (with the same
tuned wrappers) on the selected worst-Locus (+ a few best) instances and logs ONE
``.rrd`` with a ``frame_idx`` timeline over instances. Each library's corners go
on a **separate entity path** so overlays are individually toggleable in the
viewer; ground truth is overlaid in green. Mirrors the vocabulary of
``rotation_tail_diag.extract._log_rerun_scene`` / ``tools.viz_rerun_hub``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from tools.bench.compare.generate import ChosenConfig
from tools.bench.matching import MATCH_DISTANCE_THRESHOLD_PX
from tools.bench.metrics import corner_rmse_px
from tools.bench.plots._compare_style import series_color_map
from tools.bench.utils import (
    WRAPPER_BY_LIBRARY,
    HubDatasetLoader,
    LibraryWrapper,
    rotation_error_deg,
)

# GT overlay color (0–255). Per-series overlay colors are derived from the shared
# report palette (series_color_map) so a library reads the same color in the SVG
# report and the rerun deep-dive.
GT_COLOR = [0, 200, 0]


def _series_rgb(series: list[str]) -> dict[str, list[int]]:
    return {
        s: [int(round(c * 255)) for c in rgba[:3]] for s, rgba in series_color_map(series).items()
    }


def _closed(corners: np.ndarray) -> list[list[float]]:
    return np.vstack([corners, corners[:1]]).tolist()


def _select(
    worst_df: pl.DataFrame, best_df: pl.DataFrame | None, max_deepdive: int
) -> list[dict[str, Any]]:
    """Pick the instances to visualize: worst-Locus first, then a few best."""
    n_best = min(3, max_deepdive // 4)
    rows: list[dict[str, Any]] = []
    for r in worst_df.head(max_deepdive - n_best).iter_rows(named=True):
        rows.append({**r, "group": "worst"})
    if best_df is not None and n_best:
        for r in best_df.head(n_best).iter_rows(named=True):
            rows.append({**r, "group": "best"})
    return rows[:max_deepdive]


def _build_wrappers(configs: list[ChosenConfig], family: int) -> dict[str, LibraryWrapper]:
    wrappers: dict[str, LibraryWrapper] = {}
    for c in configs:
        series = f"{c.library}:{c.profile_label}"
        wrappers[series] = WRAPPER_BY_LIBRARY[c.library].from_params(
            family=family, params=c.param_values, space=c.space
        )
    return wrappers


def _nearest_detection(dets: list[dict], tag_id: int, gt_center: np.ndarray) -> dict | None:
    """The same-id detection matched to this GT, or None if none within threshold.

    Uses the same ``MATCH_DISTANCE_THRESHOLD_PX`` as the Collector that produced the
    parquet, so the deep-dive's focus metric agrees with the metric row that selected
    the instance (a threshold-less nearest pick could log a "matched" error for a
    detection the parquet counted as missed).
    """
    same_id = [d for d in dets if int(d["id"]) == tag_id]
    if not same_id:
        return None
    best = min(same_id, key=lambda d: float(np.linalg.norm(np.asarray(d["center"]) - gt_center)))
    if float(np.linalg.norm(np.asarray(best["center"]) - gt_center)) >= MATCH_DISTANCE_THRESHOLD_PX:
        return None
    return best


def emit_deepdive(
    *,
    worst_df: pl.DataFrame,
    best_df: pl.DataFrame | None,
    configs: list[ChosenConfig],
    data_dir: Path,
    out_dir: Path,
    family: int,
    max_deepdive: int = 24,
) -> Path | None:
    """Emit one combined ``.rrd`` over the selected instances. Returns its path.

    Returns ``None`` if there are no instances to visualize.
    """
    import rerun as rr

    selected = _select(worst_df, best_df, max_deepdive)
    if not selected:
        return None

    wrappers = _build_wrappers(configs, family)
    series_rgb = _series_rgb(list(wrappers))
    loaders: dict[str, Any] = {}
    img_cache: dict[tuple[str, str], np.ndarray | None] = {}
    det_cache: dict[tuple[str, str, str], list[dict]] = {}

    def dataset(name: str) -> Any:
        if name not in loaders:
            loaders[name] = HubDatasetLoader(root=data_dir).load_dataset(name)
        return loaders[name]

    def image(ds_name: str, image_id: str, ds: Any) -> np.ndarray | None:
        key = (ds_name, image_id)
        if key not in img_cache:
            img_cache[key] = cv2.imread(str(ds.images_dir / image_id), cv2.IMREAD_GRAYSCALE)
        return img_cache[key]

    def detections(
        series: str, ds_name: str, image_id: str, img: np.ndarray, ds: Any
    ) -> list[dict]:
        key = (series, ds_name, image_id)
        if key not in det_cache:
            dets, _ = wrappers[series].detect(img, intrinsics=ds.intrinsics, tag_size=ds.tag_size)
            det_cache[key] = dets
        return det_cache[key]

    rr.init("locus_compare_deepdive", spawn=False)
    for i, inst in enumerate(selected):
        ds = dataset(inst["dataset"])
        img = image(inst["dataset"], inst["image_id"], ds)
        if img is None:
            continue
        rr.set_time(timeline="frame_idx", sequence=i)

        if ds.intrinsics is not None:
            rr.log(
                "world/camera",
                rr.Pinhole(
                    focal_length=[ds.intrinsics.fx, ds.intrinsics.fy],
                    principal_point=[ds.intrinsics.cx, ds.intrinsics.cy],
                    resolution=[img.shape[1], img.shape[0]],
                ),
            )
        rr.log("world/camera/image", rr.Image(img))

        tags = ds.gt_map[inst["image_id"]]["tags"]
        gt_strips = [_closed(np.asarray(t["corners"], dtype=np.float64)) for t in tags.values()]
        rr.log("world/camera/gt_corners", rr.LineStrips2D(gt_strips, colors=[GT_COLOR], radii=1.5))

        tag_id = int(inst["tag_id"])
        gt = tags.get(tag_id)
        gt_corners = np.asarray(gt["corners"], dtype=np.float64) if gt is not None else None
        gt_center = gt_corners.mean(axis=0) if gt_corners is not None else np.zeros(2)
        gt_pose = (
            np.asarray(gt["pose"], dtype=np.float64) if gt and gt.get("pose") is not None else None
        )

        for series in wrappers:
            dets = detections(series, inst["dataset"], inst["image_id"], img, ds)
            color = series_rgb.get(series, [200, 200, 200])
            strips = [_closed(np.asarray(d["corners"], dtype=np.float64)) for d in dets]
            path = f"world/camera/{series.replace(':', '_')}_corners"
            rr.log(path, rr.LineStrips2D(strips, colors=[color], radii=1.0))

            focus = _nearest_detection(dets, tag_id, gt_center)
            if focus is not None and gt_corners is not None:
                rmse = corner_rmse_px(focus["corners"], gt_corners)
                rr.log(f"metrics/{series}/corner_rmse_px", rr.Scalars([rmse]))
                dp = focus.get("pose")
                if dp is not None and gt_pose is not None:
                    dp = np.asarray(dp, dtype=np.float64)
                    rr.log(
                        f"metrics/{series}/trans_err_m",
                        rr.Scalars([float(np.linalg.norm(dp[:3] - gt_pose[:3]))]),
                    )
                    rr.log(
                        f"metrics/{series}/rot_err_deg",
                        rr.Scalars([rotation_error_deg(dp, gt_pose[3:7])]),
                    )

        delta = inst.get("delta")
        rr.log(
            "text/instance",
            rr.TextLog(
                f"[{inst['group']}] {inst['image_id']} tag={tag_id} "
                f"stratum={inst.get('stratum_id')} delta={delta} kind={inst.get('failure_kind')}"
            ),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = out_dir / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rrd = rec_dir / "compare_deepdive.rrd"
    rr.save(str(rrd))
    return rrd
