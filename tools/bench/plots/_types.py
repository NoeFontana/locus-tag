"""Shared Literal types for the plot modules.

Centralising these means a single place to widen / rename axis or metric
identifiers; consumers (sweep, report) import from here so a typo at one
call site is caught against this contract.
"""

from __future__ import annotations

from typing import Literal

ContinuousAxis = Literal["distance_m", "aoi_deg", "ppm"]

Metric = Literal[
    "recall",
    "trans_err_p50_m",
    "rot_err_p50_deg",
    "repro_err_p50_px",
    "latency_p50_ms",
]

GroupBy = Literal["stratum_id", "resolution_h"]
