"""Per-instance comparative benchmarking: compare locus vs OpenCV ArUco vs
pupil_apriltags image-by-image and by stratum, surface the images where Locus
underperforms (improvement levers), and emit a structured report + embeddable
vector illustrations + a rerun deep-dive.

Built on the existing bench substrate (Collector/ObservationRecord, the tuning
wrappers' ``from_params``, ``pareto/<lib>.json``, stratification). The analysis
layer is polars-native; plots are matplotlib→SVG.
"""

from __future__ import annotations
