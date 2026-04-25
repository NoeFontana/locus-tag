"""Capture the SOTA pose-precision table for the 1080p render-tag subset.

Runs OpenCV `cv2.aruco`, AprilTag-C (`pupil_apriltags`), and Locus across the
three profiles relevant to render-tag evaluation (`standard`, `high_accuracy`,
`render_tag_hub`) on `locus_v1_tag36h11_1920x1080`. For each detector reports
recall, precision, translation error mean/p50/p95/p99, rotation error
mean/p50/p95/p99, and mean latency.

Usage:
    PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \\
      uv run --group bench tools/bench/render_tag_sota_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import locus
from locus._config import ProfileName

from tools.bench.utils import (
    HUB_CACHE_DIR,
    AprilTagWrapper,
    HubDatasetLoader,
    LocusWrapper,
    OpenCVWrapper,
    aggregate_pose_stats,
    evaluate_tag_pose,
)

HUB_CONFIG = "locus_v1_tag36h11_1920x1080"
LOCUS_PROFILES: tuple[ProfileName, ...] = ("standard", "high_accuracy", "render_tag_hub")


def main() -> None:
    ds = HubDatasetLoader(root=HUB_CACHE_DIR).load_dataset(HUB_CONFIG)
    eval_tag_size = ds.tag_size if ds.tag_size is not None else 1.0
    family = locus.TagFamily.AprilTag36h11

    results: dict[str, dict[str, Any]] = {}
    for profile in LOCUS_PROFILES:
        cfg = locus.DetectorConfig.from_profile(profile)
        detector = locus.Detector(config=cfg, families=[family])
        wrapper = LocusWrapper(name=f"Locus ({profile})", detector=detector, family=int(family))
        stats = evaluate_tag_pose(
            wrapper, ds, eval_tag_size, pose_estimation_mode=locus.PoseEstimationMode.Accurate
        )
        results[wrapper.name] = aggregate_pose_stats(stats)

    cv_wrap = OpenCVWrapper(family=int(family))
    cv_wrap.name = "OpenCV cv2.aruco"
    at_wrap = AprilTagWrapper(nthreads=8, family=int(family))
    at_wrap.name = "AprilTag-C (pupil)"
    for wrap in (cv_wrap, at_wrap):
        results[wrap.name] = aggregate_pose_stats(evaluate_tag_pose(wrap, ds, eval_tag_size))

    out_path = Path("/tmp/render_tag_sota_full.json")
    out_path.write_text(json.dumps({HUB_CONFIG: results}, indent=2))
    print(f"\nSaved results to {out_path}\n")

    print(
        f"{'Detector':<24} | {'Rec %':>6} | {'Prec %':>6} | "
        f"{'t mean':>7} | {'t p50':>6} | {'t p95':>6} | {'t p99':>6} | "
        f"{'r mean':>7} | {'r p50':>6} | {'r p95':>6} | {'r p99':>7} | {'lat ms':>6}"
    )
    print("-" * 140)
    for name, r in results.items():
        print(
            f"{name:<24} | {r['recall']:>6.2f} | {r['precision']:>6.2f} | "
            f"{r['trans_mean_m']:>7.4f} | {r['trans_p50_m']:>6.4f} | "
            f"{r['trans_p95_m']:>6.4f} | {r['trans_p99_m']:>6.4f} | "
            f"{r['rot_mean_deg']:>7.4f} | {r['rot_p50_deg']:>6.4f} | "
            f"{r['rot_p95_deg']:>6.4f} | {r['rot_p99_deg']:>7.4f} | {r['latency_ms']:>6.2f}"
        )


if __name__ == "__main__":
    main()
