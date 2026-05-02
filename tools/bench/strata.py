"""Stratification axis bucketing — raw physical values → bucket slugs → ``stratum_id``.

The boundary table below is the single source of truth for the §5 cuts in
``docs/engineering/stratification.md``. Cuts were chosen by
``tools/bench/strata_histogram.py`` running 33/66-percentile equal-frequency
splits over the four ``single_tag_locus_v1_tag36h11_*`` corpora and snapping
to a human-readable grid (dist→0.1m, aoi→5°, ppm→100 px/m). The snap audit
in the histogram script enforces ≤15% bucket reshuffle versus raw
percentiles; re-run that script before adjusting any constant here.

Per stratification.md §6 a re-bucketing is a v1 patch bump and invalidates
existing baselines — re-run ``bench aggregate`` after edits.

The grammar of the produced ``stratum_id`` matches ``tools/bench/schema.py``
``_STRATUM_KEYS`` and ``_validate_stratum_id`` exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Bucket boundaries (see strata_histogram.py for derivation)
# ---------------------------------------------------------------------------

# Resolution: H pixels — corpora are deterministic, no rounding
RES_SD_MAX = 480
RES_HD_MAX = 720
RES_FHD_MAX = 1080
# > RES_FHD_MAX → uhd

# Distance: metres
DIST_NEAR_MAX = 0.7
DIST_MID_MAX = 1.5
# > DIST_MID_MAX → far

# Angle of incidence: degrees (0° = frontal)
AOI_FRONTAL_MAX = 35.0
AOI_OBLIQUE_MAX = 50.0
# > AOI_OBLIQUE_MAX → grazing

# Pixels-per-metre of tag (derived: max_edge_px(corners) / (tag_size_mm/1000))
PPM_LO_MAX = 800.0
PPM_MID_MAX = 1300.0
# > PPM_MID_MAX → hi

# Motion: ‖velocity‖ in m/s (`null` velocity → static; degenerate in v1 corpora)
MOT_STATIC_MAX = 0.0


@dataclass(frozen=True, slots=True)
class AxisValues:
    """Raw continuous axis values for a single record. NaN/None means `unk`."""

    resolution_h: int | None
    distance_m: float
    aoi_deg: float
    ppm: float
    velocity: float | None  # `None` means static


def _is_unk(value: float) -> bool:
    return math.isnan(value)


def _bucket_res(h: int | None) -> str:
    if h is None:
        return "unk"
    if h <= RES_SD_MAX:
        return "sd"
    if h <= RES_HD_MAX:
        return "hd"
    if h <= RES_FHD_MAX:
        return "fhd"
    return "uhd"


def _bucket_dist(d: float) -> str:
    if _is_unk(d):
        return "unk"
    if d <= DIST_NEAR_MAX:
        return "near"
    if d <= DIST_MID_MAX:
        return "mid"
    return "far"


def _bucket_aoi(a: float) -> str:
    if _is_unk(a):
        return "unk"
    if a <= AOI_FRONTAL_MAX:
        return "frontal"
    if a <= AOI_OBLIQUE_MAX:
        return "oblique"
    return "grazing"


def _bucket_ppm(p: float) -> str:
    if _is_unk(p):
        return "unk"
    if p <= PPM_LO_MAX:
        return "lo"
    if p <= PPM_MID_MAX:
        return "mid"
    return "hi"


def _bucket_mot(velocity: float | None) -> str:
    """Static unless velocity is finite and > 0; `unk` if velocity is NaN."""
    if velocity is None:
        return "static"
    if _is_unk(velocity):
        return "unk"
    if velocity <= MOT_STATIC_MAX:
        return "static"
    return "motion"


def compute_stratum_id(axes: AxisValues) -> str:
    """Map raw axis values to the canonical pipe-separated ``stratum_id``.

    Output passes ``tools/bench/schema.py::_validate_stratum_id``.
    """
    return (
        f"res={_bucket_res(axes.resolution_h)}|"
        f"ppm={_bucket_ppm(axes.ppm)}|"
        f"aoi={_bucket_aoi(axes.aoi_deg)}|"
        f"dist={_bucket_dist(axes.distance_m)}|"
        f"mot={_bucket_mot(axes.velocity)}"
    )
