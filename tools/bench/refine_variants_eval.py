"""Head-to-head render-tag pose eval across Locus corner-refinement variants.

Isolates the *sole surviving lever* on the rotation-p99 gap (corner-refinement
robustness — see the 2026-07-14 rotation-tail diagnostic) by running the same
`high_accuracy` detector with only its large-marker refinement route swapped:

    V0  EdLines + None        — shipped baseline (rot p99 ~0.600 deg)
    V-GWLF  EdLines + Gwlf     — apriltag-style gradient-weighted edge-line refit
                                 + intersection + covariance-weighted PnP

The 2026-07-14 finding (docs/engineering/benchmarking/refine_variants_20260714.md):
GWLF reaches rot p99 ~0.398 deg (near OpenCV-apriltag 0.376) but regresses
translation p99 18.6->63 mm — it swaps Locus's error profile for apriltag's, so it
is NOT a Pareto win and stays unshipped. Everything here is config-only; the three
shipped profiles are untouched, so detection snapshots stay byte-identical.

Usage:
    PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache RAYON_NUM_THREADS=1 \\
      uv run --group bench tools/bench/refine_variants_eval.py

    # other resolutions:
    RENDER_TAG_SOTA_CONFIG=locus_v1_tag36h11_3840x2160 uv run ... refine_variants_eval.py
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import locus
from locus._config import CornerRefinementMode

from tools.bench.utils import (
    HUB_CACHE_DIR,
    HubDatasetLoader,
    LocusWrapper,
    aggregate_pose_stats,
    evaluate_tag_pose,
)

HUB_CONFIG = os.environ.get("RENDER_TAG_SOTA_CONFIG", "locus_v1_tag36h11_1920x1080")
BASE_PROFILE = "high_accuracy"

# `None` is a Python keyword; PyO3 exposes the variant under its Rust name.
_NONE = getattr(CornerRefinementMode, "None")
_GWLF = CornerRefinementMode.Gwlf


def _set_refinement(cfg: Any, low: Any, high: Any) -> Any:
    """Swap the AdaptivePpb route refinements on a fresh `cfg`, in place.

    `high_accuracy` routes large markers (ppb >= threshold) through the *high*
    route and small ones through the *low* route; we swap both so the whole
    corpus exercises the variant under test. Mutates and returns `cfg`.
    """
    ppb = getattr(cfg.quad.extraction_policy, "AdaptivePpb", None)
    if ppb is None:  # Static policy — nothing routed; leave untouched.
        return cfg
    ppb.low_refinement = low
    ppb.high_refinement = high
    return cfg


def _variants() -> dict[str, Callable[[Any], Any]]:
    """Name -> config transform applied to a *fresh* `from_profile` config.

    A fresh config per variant avoids deep-copying PyO3 enum members (which are
    not picklable).
    """
    return {
        # V0: shipped high_accuracy (EdLines large-marker route -> None).
        "V0 EdLines+None (baseline)": lambda c: c,
        # V-GWLF: apriltag-style edge-line refit + covariance-weighted PnP.
        # Fixes rotation p99 but regresses translation ~3x (see doc); NOT shipped.
        "V-GWLF EdLines+Gwlf": lambda c: _set_refinement(c, _GWLF, _GWLF),
    }


def main() -> None:
    ds = HubDatasetLoader(root=HUB_CACHE_DIR).load_dataset(HUB_CONFIG)
    eval_tag_size = ds.tag_size if ds.tag_size is not None else 1.0
    family = locus.TagFamily.AprilTag36h11

    results: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, list[float]]] = {}
    for name, transform in _variants().items():
        cfg = transform(locus.DetectorConfig.from_profile(BASE_PROFILE))
        detector = locus.Detector(config=cfg, families=[family], threads=1)
        wrapper = LocusWrapper(name=name, detector=detector, family=int(family))
        stats = evaluate_tag_pose(wrapper, ds, eval_tag_size)
        results[name] = aggregate_pose_stats(stats)
        # Keep the raw per-detection error arrays for tail (p99) analysis.
        raw[name] = {
            "rot_errs_deg": [float(v) for v in stats["rot_errs"]],
            "trans_errs_m": [float(v) for v in stats["trans_errs"]],
        }

    out_path = Path("/tmp/refine_variants.json")
    out_path.write_text(
        json.dumps(
            {
                "hub_config": HUB_CONFIG,
                "base_profile": BASE_PROFILE,
                "pose_mode": "Accurate",
                "threads": 1,
                "results": results,
                "raw": raw,
            },
            indent=2,
        )
    )
    print(f"\nSaved results to {out_path}  (config {HUB_CONFIG})\n")

    print(
        f"{'Variant':<34} | {'Rec %':>6} | {'Prec %':>6} | "
        f"{'t mean':>7} | {'t p95':>6} | {'t p99':>6} | "
        f"{'r mean':>7} | {'r p50':>6} | {'r p95':>6} | {'r p99':>7} | {'lat ms':>6}"
    )
    print("-" * 132)
    for name, r in results.items():
        print(
            f"{name:<34} | {r['recall']:>6.2f} | {r['precision']:>6.2f} | "
            f"{r['trans_mean_m']:>7.4f} | {r['trans_p95_m']:>6.4f} | {r['trans_p99_m']:>6.4f} | "
            f"{r['rot_mean_deg']:>7.4f} | {r['rot_p50_deg']:>6.4f} | "
            f"{r['rot_p95_deg']:>6.4f} | {r['rot_p99_deg']:>7.4f} | {r['latency_ms']:>6.2f}"
        )
    print("\nTarget: rot p99 < 0.376 deg (OpenCV-apriltag), guardrails: recall,")
    print("trans p99, mean corner RMSE must not regress vs V0.")


if __name__ == "__main__":
    main()
