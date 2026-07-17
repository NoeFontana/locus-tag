"""Model-edge pose-refinement eval: shipped `high_accuracy` vs the opt-in
`pose.pose_edge_refinement_enabled` stage, across render-tag resolutions.

The stage aligns a decoded tag's internal bit-grid edges to the image and refines
the 6-DoF pose against them (rotation from ~40 distributed edges; translation
re-anchored to the 4 corners). See
docs/engineering/benchmarking/model_edge_refinement_20260715.md.

Usage:
    PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache RAYON_NUM_THREADS=1 \\
      uv run --group bench tools/bench/model_edge_eval.py
"""

from __future__ import annotations

import os
from typing import Any

import locus

from tools.bench.utils import (
    HUB_CACHE_DIR,
    HubDatasetLoader,
    LocusWrapper,
    aggregate_pose_stats,
    evaluate_tag_pose,
)

# Resolutions to sweep (override with a single config via the env var).
_DEFAULT = (
    "locus_v1_tag36h11_640x480",
    "locus_v1_tag36h11_1280x720",
    "locus_v1_tag36h11_1920x1080",
    "locus_v1_tag36h11_3840x2160",
)
CONFIGS = (
    (os.environ["RENDER_TAG_SOTA_CONFIG"],) if "RENDER_TAG_SOTA_CONFIG" in os.environ else _DEFAULT
)


def _run(loader: HubDatasetLoader, cfg_name: str, enabled: bool) -> dict[str, Any]:
    ds = loader.load_dataset(cfg_name)
    fam = locus.TagFamily.AprilTag36h11
    cfg = locus.DetectorConfig.from_profile("high_accuracy")
    cfg.pose.pose_edge_refinement_enabled = enabled
    detector = locus.Detector(config=cfg, families=[fam], threads=1)
    wrapper = LocusWrapper(name="locus", detector=detector, family=int(fam))
    tag_size = ds.tag_size if ds.tag_size is not None else 1.0
    return aggregate_pose_stats(evaluate_tag_pose(wrapper, ds, tag_size))


def main() -> None:
    loader = HubDatasetLoader(root=HUB_CACHE_DIR)
    print(
        f"{'config':<30} {'variant':<10} | {'rot p95':>7} {'rot p99':>7} | "
        f"{'t p99 mm':>8} {'t mean':>7} | {'lat ms':>6}"
    )
    print("-" * 88)
    for cfg_name in CONFIGS:
        short = cfg_name.replace("locus_v1_tag36h11_", "")
        for enabled, lbl in [(False, "baseline"), (True, "edge")]:
            r = _run(loader, cfg_name, enabled)
            print(
                f"{short:<30} {lbl:<10} | {r['rot_p95_deg']:>7.3f} {r['rot_p99_deg']:>7.3f} | "
                f"{r['trans_p99_m'] * 1000:>8.2f} {r['trans_mean_m'] * 1000:>7.2f} | "
                f"{r['latency_ms']:>6.2f}"
            )


if __name__ == "__main__":
    main()
