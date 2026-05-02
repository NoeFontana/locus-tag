"""Histogram pass over hub-cache corpora to propose §5 bucket boundaries.

Runs a quantile-based scan of the four `single_tag_locus_v1_tag36h11_*`
corpora under `tests/data/hub_cache/` and prints proposed cut points for
each stratification axis. Two cut sets are emitted:

  - **Raw**: 33/66 percentiles of the pooled distribution.
  - **Snapped**: rounded to a human-readable grid (dist → 0.1 m, aoi → 5°,
    ppm → 100 px/m). The snapped values are what land in §5 and in
    `tools/bench/strata.py`. The redistribution check at the bottom asserts
    the snap moves ≤ ``MAX_SNAP_DELTA`` records relative to the raw cuts.

Re-run this whenever a patch bump per ``stratification.md`` §6 changes which
corpora are in scope or when a new physical regime lands.

Usage:
    PYTHONPATH=. uv run python tools/bench/strata_histogram.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.bench.utils import max_edge_px

HUB_ROOT = Path("tests/data/hub_cache")
TARGET_CONFIGS = (
    "single_tag_locus_v1_tag36h11_640x480",
    "single_tag_locus_v1_tag36h11_1280x720",
    "single_tag_locus_v1_tag36h11_1920x1080",
    "single_tag_locus_v1_tag36h11_3840x2160",
)

# Snap grids — chosen for human-readable cuts that round-trip cleanly through
# diff reports. Equal across re-runs so corpus changes don't silently shift
# `dist=1.49 → 1.5` style precision.
SNAP_DIST_M = 0.1
SNAP_AOI_DEG = 5.0
SNAP_PPM = 100.0

# Records that may land in a different bucket after snapping. The snap is
# meant to gain human readability at the cost of *some* fidelity — but if a
# coarse grid merges adjacent buckets entirely, the script aborts so the
# engineer revisits the grid. 15% of pool size is the soft ceiling: anything
# tighter punishes natural rounding (e.g. dist 0.67 → 0.7 reshuffles a few
# records by design); anything looser permits a grid that flattens a bucket.
MAX_SNAP_DELTA_FRACTION = 0.15


@dataclass(frozen=True)
class TagRecord:
    config: str
    image_id: str
    tag_id: int
    distance_m: float
    aoi_deg: float
    ppm: float  # derived
    resolution_h: int
    velocity: float | None
    occlusion_ratio: float
    tag_size_mm: float


def load_records(root: Path = HUB_ROOT) -> list[TagRecord]:
    out: list[TagRecord] = []
    for config in TARGET_CONFIGS:
        rich_path = root / config / "rich_truth.json"
        if not rich_path.exists():
            print(f"  [skip] {rich_path} not found")
            continue
        with rich_path.open() as f:
            data = json.load(f)
        entries = data["records"] if isinstance(data, dict) and "records" in data else data
        for e in entries:
            if e.get("record_type") != "TAG":
                continue
            corners = np.asarray(e["corners"], dtype=np.float64)
            tag_size_mm = float(e["tag_size_mm"])
            ppm = max_edge_px(corners) / (tag_size_mm / 1000.0)
            out.append(
                TagRecord(
                    config=config,
                    image_id=e.get("image_filename") or e.get("image_id") or "",
                    tag_id=int(e["tag_id"]),
                    distance_m=float(e["distance"]),
                    aoi_deg=float(e["angle_of_incidence"]),
                    ppm=ppm,
                    resolution_h=int(e["resolution"][1]),
                    velocity=e.get("velocity"),
                    occlusion_ratio=float(e.get("occlusion_ratio", 0.0)),
                    tag_size_mm=tag_size_mm,
                )
            )
    return out


def _snap(value: float, grid: float) -> float:
    """Snap to the nearest grid multiple (banker's-rounding-free)."""
    return round(value / grid) * grid


def _bucket_counts(values: np.ndarray, lo: float, hi: float) -> tuple[int, int, int]:
    return (
        int((values <= lo).sum()),
        int(((values > lo) & (values <= hi)).sum()),
        int((values > hi).sum()),
    )


def _bucket_assignment(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return per-record bucket index 0/1/2."""
    out = np.full(values.shape, 1, dtype=np.int8)
    out[values <= lo] = 0
    out[values > hi] = 2
    return out


def _summary(label: str, values: np.ndarray) -> None:
    p = np.percentile(values, [0, 33, 50, 66, 100])
    print(
        f"  {label:>10}  n={len(values):>5}  "
        f"min={p[0]:>8.2f}  p33={p[1]:>8.2f}  p50={p[2]:>8.2f}  "
        f"p66={p[3]:>8.2f}  max={p[4]:>8.2f}  "
        f"mean={values.mean():>8.2f}  std={values.std():>7.2f}"
    )


def main() -> int:
    records = load_records()
    print(f"\nLoaded {len(records)} TAG records from {len(TARGET_CONFIGS)} corpora\n")

    for config in TARGET_CONFIGS:
        n = sum(1 for r in records if r.config == config)
        if n:
            print(f"  {config:<50}  n={n}")
    print()

    distance = np.array([r.distance_m for r in records])
    aoi = np.array([r.aoi_deg for r in records])
    ppm = np.array([r.ppm for r in records])
    res_h = np.array([r.resolution_h for r in records])

    print("Per-axis distribution (pooled across all four corpora):\n")
    _summary("distance_m", distance)
    _summary("aoi_deg", aoi)
    _summary("ppm", ppm)
    print()

    print("Resolution histogram:")
    for h in sorted(set(res_h.tolist())):
        print(f"  H={h}: n={int((res_h == h).sum())}")
    print()

    n_velocity_null = sum(1 for r in records if r.velocity is None)
    print(f"Motion: velocity is null for {n_velocity_null}/{len(records)} records")
    print()

    p33_d, p66_d = np.percentile(distance, [33, 66])
    p33_a, p66_a = np.percentile(aoi, [33, 66])
    p33_p, p66_p = np.percentile(ppm, [33, 66])

    snap_d_lo, snap_d_hi = _snap(p33_d, SNAP_DIST_M), _snap(p66_d, SNAP_DIST_M)
    snap_a_lo, snap_a_hi = _snap(p33_a, SNAP_AOI_DEG), _snap(p66_a, SNAP_AOI_DEG)
    snap_p_lo, snap_p_hi = _snap(p33_p, SNAP_PPM), _snap(p66_p, SNAP_PPM)

    print("=" * 78)
    print("§5 bucket boundary table — SNAPPED values (use these in strata.py):\n")
    n_d = _bucket_counts(distance, snap_d_lo, snap_d_hi)
    n_a = _bucket_counts(aoi, snap_a_lo, snap_a_hi)
    n_p = _bucket_counts(ppm, snap_p_lo, snap_p_hi)
    print("| Axis | Slug    | Range                                | n   |")
    print("| ---- | ------- | ------------------------------------ | --- |")
    print(f"| res  | sd      | `H ≤ 480`                            | {int((res_h <= 480).sum()):>3} |")
    print(f"| res  | hd      | `480 < H ≤ 720`                      | {int(((res_h > 480) & (res_h <= 720)).sum()):>3} |")
    print(f"| res  | fhd     | `720 < H ≤ 1080`                     | {int(((res_h > 720) & (res_h <= 1080)).sum()):>3} |")
    print(f"| res  | uhd     | `1080 < H`                           | {int((res_h > 1080).sum()):>3} |")
    print(f"| ppm  | lo      | `ppm ≤ {snap_p_lo:>5.0f}`                       | {n_p[0]:>3} |")
    print(f"| ppm  | mid     | `{snap_p_lo:>5.0f} < ppm ≤ {snap_p_hi:>5.0f}`              | {n_p[1]:>3} |")
    print(f"| ppm  | hi      | `{snap_p_hi:>5.0f} < ppm`                        | {n_p[2]:>3} |")
    print(f"| aoi  | frontal | `angle ≤ {snap_a_lo:>4.0f}°`                     | {n_a[0]:>3} |")
    print(f"| aoi  | oblique | `{snap_a_lo:>4.0f}° < angle ≤ {snap_a_hi:>4.0f}°`           | {n_a[1]:>3} |")
    print(f"| aoi  | grazing | `{snap_a_hi:>4.0f}° < angle`                      | {n_a[2]:>3} |")
    print(f"| dist | near    | `d ≤ {snap_d_lo:>4.1f}`                          | {n_d[0]:>3} |")
    print(f"| dist | mid     | `{snap_d_lo:>4.1f} < d ≤ {snap_d_hi:>4.1f}`                 | {n_d[1]:>3} |")
    print(f"| dist | far     | `{snap_d_hi:>4.1f} < d`                           | {n_d[2]:>3} |")
    print(f"| mot  | static  | `‖velocity‖ ≤ 0.0` (or null)         | {n_velocity_null:>3} |")
    print(f"| mot  | motion  | `0.0 < ‖velocity‖`                   | {len(records) - n_velocity_null:>3} |")
    print()

    # Snap audit: how many records change bucket vs the raw 33/66 cuts.
    raw_d = _bucket_assignment(distance, p33_d, p66_d)
    raw_a = _bucket_assignment(aoi, p33_a, p66_a)
    raw_p = _bucket_assignment(ppm, p33_p, p66_p)
    snp_d = _bucket_assignment(distance, snap_d_lo, snap_d_hi)
    snp_a = _bucket_assignment(aoi, snap_a_lo, snap_a_hi)
    snp_p = _bucket_assignment(ppm, snap_p_lo, snap_p_hi)
    delta_d = int((raw_d != snp_d).sum())
    delta_a = int((raw_a != snp_a).sum())
    delta_p = int((raw_p != snp_p).sum())
    total = delta_d + delta_a + delta_p

    threshold = int(MAX_SNAP_DELTA_FRACTION * len(records))
    print("Snap audit (records that changed bucket vs raw 33/66 percentiles):")
    print(f"  distance: {delta_d}   aoi: {delta_a}   ppm: {delta_p}   total: {total}")
    print(f"  threshold: {threshold} ({MAX_SNAP_DELTA_FRACTION:.0%} of {len(records)})")
    if total > threshold:
        print(f"\n  FAIL: snap moves {total} records — grid is too coarse, revisit constants.")
        return 1
    print("  PASS: snap is within tolerance.")
    print()
    print("Stratum cardinality (excluding `unk`): 4 × 3 × 3 × 3 × 2 = 216 strata")
    print("(Most are empty; only populated combinations matter for the diff gate.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
