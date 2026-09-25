# Render-tag 1080p SOTA pursuit (2026-04-25)

> **Superseded**: see [`render_tag_sota_20260713.md`](render_tag_sota_20260713.md)
> for current numbers. Kept as the historical record of how `render_tag_hub`
> (later folded into `high_accuracy`) reached SOTA.

Goal: beat AprilTag-C and OpenCV `cv2.aruco` on `locus_v1_tag36h11_1920x1080`
(50 scenes, single tag each). Result achieved via two changes:

1. **`enable_sharpening` + minor threshold/geometry tuning**: 94% → 96%
   recall, mean RMSE 0.2162 → 0.1734 px. Every JSON-exposed knob beyond
   this was a no-op for the remaining 2 misses (scenes 0002, 0033) — they
   rejected inside EdLines's own hard-coded gates, not the JSON-tunable
   ones. `ContourRdp` recovers those 2 scenes but blows mean RMSE to 1.33px
   (2.3× the SOTA gate) — EdLines's sub-pixel fit is doing real work that a
   coarser extractor can't replace.
2. **EdLines axis-aligned imbalance gate** (opt-in
   `edlines_imbalance_gate`, shipped in `render_tag_hub.json`, later
   folded into `high_accuracy`): fixes a TRBL boundary-segmentation
   degeneracy on near-axis-aligned tags. Full rationale in
   [`lessons.md` §4.3](lessons.md). Opt-in because it regresses
   brown_conrady distortion recall (0.929 → 0.869) if left global — many
   legitimate distorted aprilgrid sub-tags sit in the 8-15% min-arc band
   the gate would misfire on.

Net result (1920×1080, vs. toughest external competitor per metric):
recall tied at 100% (OpenCV/AprilTag-C), translation p50/p95/p99 2-7×
tighter than AprilTag-C, rotation p50 tied, rotation p95/p99 lost to
OpenCV by 1.4-1.5× (EdLines grazing-angle pose ambiguity — same effect
noted in the 2160p report), latency tied with `high_accuracy` (11.67ms)
and 2.2-3.8× faster than the externals. First Locus profile to clear
AprilTag-C on every translation percentile at 100% recall.

2160p stayed at 94% in this pass (pre- the `MAX_CANDIDATES` truncation
fix — see `render_tag_2160p_20260425.md`). No regression on any other
regression suite; the gate defaults `false` everywhere except
`render_tag_hub.json`.

**External pose-convention gotchas** (still relevant, see
`tools/bench/utils.py`): AprilTag-C's tag frame is 180° rotated about z
vs. Locus/GT (`R @ diag(-1,-1,1)` fix); OpenCV must use
`solvePnP(SOLVEPNP_ITERATIVE)` with explicit y-down center-origin object
points — `SOLVEPNP_IPPE_SQUARE` picks an inconsistent branch on ~half of
symmetric synthetic tags. Without these fixes rotation error reads
~180° even when translation is correct.
