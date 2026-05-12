# EdLines S1 — corner-region exclusion sweep (2026-05-04)

Empirical follow-up to `edlines_segmentation_design_2026-05-03.md` §3. Implements
the corner-pixel exclusion filter and sweeps Δ ∈ {0, 1, 2, 3, 5} px on the full
50-scene `locus_v1_tag36h11_1920x1080` corpus.

**TL;DR:** S1 confirms the corner-pixel-misassignment hypothesis (monotonic
improvement on scene_0008 across Δ; Phase-2 telemetry shows the top edge fit
collapses to a degenerate `y = 456.5` integer-row boundary). However it
**does not meet the §3.5 acceptance bar of -2 px** on scene_0008 — best is
-1.19 px at Δ=5, which also drops one scene to no-detection. The corpus-wide
mean ‖r‖ improvement (-14 % at Δ=5, ‑11 % at Δ=2) was **unexpected and is a
separable finding**: corner-region pixel contamination is mildly biasing
slopes across the whole corpus, not just on near-axis-aligned tags.

Per the design memo's mechanical decision rule (§6 step 2: ship if §3.5
passes, otherwise stop and read telemetry), **S1 does not pass acceptance and
we proceed to S3** — but a follow-up could ship S1 at Δ ∈ {2, 3} as a
corpus-level bias reducer if the d² regression is acceptable for downstream
consumers (it is the same trade documented for the §3 1 % exclusion-zone
experiment).

## §1 Hardware

AMD EPYC-Milan KVM, 8 logical CPUs (`Architecture: x86_64`). `--release`
build with `bench-internals` feature. CPython 3.14.3, rustc 1.92.0,
`RAYON_NUM_THREADS=8`.

## §2 Implementation (during the experiment)

For the sweep, `EdLinesConfig::corner_exclusion_px: f64` (default 0.0,
byte-identical off-path) was added with a `OnceLock`-cached read of
`LOCUS_EDLINES_CORNER_EXCLUSION_PX`. `extract_boundary_segments` was
modified to track the four selected extremals (TRBL or Diagonal) and,
after angular-arc bin assignment, drop boundary points whose squared
Euclidean distance to either of the arc's two endpoint extremals fell
within `Δ²`. ~25 LoC change.

**This filter is reverted in the same change as this memo** (see §6 for
the decision). The diagnostic dump survives:

```
EDLINES_PHASE1 bbox=<min_x,min_y,max_x,max_y> mode=<axis|diag>
    arc_pts=<n0,n1,n2,n3>
    line0=<nx,ny,d,cx,cy> line1=… line2=… line3=…
```

Emitted from `run_pipeline_with_mode` only when
`LOCUS_DUMP_EDLINES_PHASE1=1`. Surfaces the full Phase 1 + Phase 2 state
per component for any future EdLines investigation.

## §3 Sweep result

```
Δ_px | n  | scene_0008 c1 ‖r‖ | corpus mean ‖r‖ | corpus p99 ‖r‖ | corpus max ‖r‖ | corpus mean d² | corpus p99 d²
-----+----+-------------------+------------------+-----------------+-----------------+-----------------+----------------
0.0  | 50 | 3.833             | 0.1908           | 0.6972          | 3.833           | 714.7           |  9 435
1.0  | 50 | 3.797 (-1 %)      | 0.1741 (-9 %)    | 0.6899 (-1 %)   | 3.797           | 857.5           | 11 380
2.0  | 50 | 3.632 (-5 %)      | 0.1692 (-11 %)   | 0.6924 (-1 %)   | 3.632           | 803.3           | 10 026
3.0  | 50 | 3.378 (-12 %)     | 0.1660 (-13 %)   | 0.6925 (-1 %)   | 3.378           | 846.3           | 12 295
5.0  | 49 | 2.647 (-31 %)     | 0.1640 (-14 %)   | 0.7093 (+2 %)   | 2.647           | 882.9           | 12 685
```

(Full per-Δ summary at `diagnostics/edlines_s1_sweep/summary.json`.)

### §3.1 What Δ=5 dropped

At Δ=5, `n=49` (one scene falls to `no_detection: 1`). The component bbox
comes too close to the boundary-points-after-trim threshold and Phase 2
fails on at least one edge. This is a recall regression: at Δ=3 all 50
scenes still produce detections.

The dropped scene is most likely a small-bbox tag where 5-px trimming both
ends of every arc removes almost all data. We don't identify it by ID
because the audit only records aggregate skipped counts, but the
per-Δ-3 vs per-Δ-5 row count delta is unambiguous.

## §4 Phase 1 telemetry confirms the hypothesis

Dump for scene_0008's tag (bbox 881,456 → 1048,606, axis mode):

```
arc_pts     = 117, 169, 155, 197         (1.7× imbalance)
line0 (T→R) = nx=0.000000, ny=1.000000, d=-456.500000
                → y = 456.5 (perfectly horizontal, at integer boundary)
line3 (L→T) = nx=-0.985736, ny=-0.168300, d=969.633036
                → x = 983.66 − 0.171·y
```

The true top edge connects GT corners (903.44, 456.41) and (1049.71,
455.25): slope ≈ −0.007, never crossing y = 456.5 over its 146 px run.
The fact that Phase 2 IRLS converges to **`y = 456.5` exactly** — not
`456.27` (true) and not `456.41` (corner-1 GT) — is the integer-pixel
rounding signature: when nearly all monotone-scan boundary points sit on
either row 456 or row 457 due to a near-axis-aligned edge, the L1/L2
fit has no slope information and minimises by clamping to the
midpoint. This is exactly the pixel-rounding floor S3 (gradient-anchor
walk) targets.

The detected Phase-2 corner from intersecting `line3` with `line0` is at
(905.74, 456.5) — 2.3 px from GT in (x, y). Phase 5 (joint GN) inflates
this to (907.25, 456.84) — total ~3.83 px. So the partition does have
~2.3 px of Phase-2 baseline error, of which:

- Δ=5 recovers 1.19 px (52 %) by stripping pixels in 5-px disks at each
  arc endpoint. The remaining 1.1 px is **everything-along-the-edge**
  rounding bias that no point-removal scheme will fix — it requires a
  gradient-driven (sub-pixel-from-the-start) line extractor, i.e. S3.

## §5 Verdict against design-memo criteria

### §3.4 (Falsification)

> "S1 is falsified iff: no Δ ∈ {1,2,3,5} reduces scene_0008 corner 1 ‖r‖
>  by ≥ 1 px, AND the telemetry shows < 5 boundary points removed per arc."

**Not falsified.** Δ=5 reduces by 1.19 px > 1.0 px. Telemetry shows
substantial point removal (Δ=2 alone removes ~5 % of total boundary
points across the four arcs of scene_0008's tag).

The hypothesis "corner-region pixels are biasing the IRLS slope" is
**confirmed**.

### §3.4 (Non-shippable)

> "S1 is non-shippable iff: best Δ regresses corpus mean ‖r‖ by ≥ 0.03 px,
>  OR regresses corpus p99 ‖r‖ by ≥ 0.05 px, OR any baseline-passing
>  scene starts failing snapshot validation."

- Corpus mean ‖r‖: **improves** by 0.027 px at Δ=5 (better than the
  ±0.03 band, which was set to allow up to 0.03 px regression). Not
  triggered.
- Corpus p99 ‖r‖: at Δ=5 increases by 0.012 px (+1.7 %). Below the 0.05
  px threshold. Not triggered for Δ ≤ 5.
- Any scene falls to no-detection: **YES at Δ=5**. Triggered for Δ=5
  only. **Δ=3 stays at full detection count.**

### §3.5 (Acceptance)

> "Ship if best Δ:
>   - Reduces scene_0008 corner 1 ‖r‖ by ≥ 2 px
>   - Corpus mean ‖r‖ within ±0.005 px of baseline (0.191)
>   - Corpus p99 ‖r‖ within ±0.05 px of baseline (0.697)"

- scene_0008 -2 px: **NOT MET**. Best is -1.19 px (Δ=5), which also
  trips the no-detection regression. Δ=3 is -0.46 px.
- Corpus mean within ±0.005: NOT MET — improves by 0.027 px (better
  than the symmetric band; sign is favourable).
- Corpus p99 within ±0.05: MET at all Δ.

**Acceptance not met on scene_0008 alone.**

### §4.6 (d² regression — observed but not gated)

`mean d²` rises 715 → 803 (Δ=2) → 882 (Δ=5), ~12-23 %. p99 d² similar.
Mechanism is the same as the §3 1 % exclusion-zone experiment: the GN's
σ²_k = Σr²_i / (n − 2) shrinks faster with reduced n than ‖r‖ shrinks,
so d² grows even though ‖r‖ shrinks. This is a **calibration regression**
on Σ_pose. It was observed in the F5 experiment too and is the
same root cause: removing points without recalibrating the variance
model. Not falsifying for S1 acceptance, but a downstream concern.

## §6 Decision (principal-engineer call, 2026-05-04)

**Don't ship S1. Don't proceed to S3. Defer architectural work; ship the
runtime gate.**

Three reasons.

### §6.1 The d² regression is in the wrong direction for the top-line metric

This entire investigation chain originated from
`rotation_tail_diagnostic_phase0_20260502.md`: the rotation-tail p99 is
0.771° and we want to lower it.  d² is the closest analytical proxy to
that tail.

S1 makes mean d² **worse** by +12 % (Δ=2) → +23 % (Δ=5) and p99 d² worse
by +6 → +34 %.  The mechanism is mechanical: σ²_k = Σr²_i / (n − 2)
shrinks faster than ‖r‖ shrinks when n drops.  The LM solver becomes
*more confidently wrong* — same bias direction, tighter Σ.  Downstream
consumers (Kalman, factor graph) weight the biased measurement *more*,
producing systematic rather than noisy error.

**We never measured `rotation_error_chosen_deg` under S1 — only ‖r‖ and
d².**  Shipping on the ‖r‖ improvement alone is metric-pleasing on a
proxy, not quality-improving on the metric of record.  This is exactly
the F5 pattern (`edlines_sota_experiments_2026-05-03.md` §3) which we
already chose not to ship.

### §6.2 S3 inherits the Phase C.5 PSF-floor risk and is multi-day on synthetic data alone

S3 (gradient-anchor walk) is the only architectural intervention that
crosses the rounding floor S1's telemetry surfaced (scene_0008's
`y = 456.5` exact).  But:

- `post_decode_refinement_20260426.md` documents a ~0.6 px corner-RMSE
  floor on Blender PSF for ERF-style sub-pixel methods.  S3's
  gradient-driven anchor walk on synthetic PSF inherits this risk
  category: synthetic PSF lacks the gradient sharpness needed for true
  sub-pixel resolution beyond ~0.6 px, and S3 may not actually clear
  scene_0008's 4 px gap on synthetic data alone.
- `edlines_sota_experiments_2026-05-03.md §5` already recommends
  deferring improvements #2 / #3 (≡ S3) until either real-camera
  evidence emerges or the runtime gate proves insufficient.
- `edlines_phase5_decoupling_2026-05-03.md §6` separately recommends
  the runtime gate as the practical fix.

Three converging negative-result memos all point to the same answer:
the data we need is real-camera, not more synthetic experiments.
Continuing on synthetic data alone is exactly the trap
`feedback_check_negative_results_first.md` warned about.

### §6.3 The runtime gate already detects scene_0008 and costs ~4 h

Track A's `corner_geometry_outlier` already fires on scene_0008
(`min_irls = 0.219`, `max_corner_d² = 37.7` vs χ²(1) at α=10⁻⁴ = 15.137,
per `scene_0008_root_cause_2026-05-03.md` §5).  A ~4 h follow-up gates
the failing corner / inflates Σ_pose / rejects the detection — the same
counterfactual p99 rotation drop (0.171° per Track A) without any
EdLines architectural risk.

## §7 Action items

1. **Revert** the `EdLinesConfig::corner_exclusion_px` field, the env-var
   read, and the per-arc filter in `extract_boundary_segments` (~25
   LoC).  Done in this same change.
2. **Keep** the `dump_phase1` / `LOCUS_DUMP_EDLINES_PHASE1` telemetry
   hook.  It is dead code with no env-var (zero overhead off-path) and
   surfaces the rounding-floor signature for any future investigator —
   the most durable artefact of this experiment.
3. **Keep** this memo, `edlines_segmentation_design_2026-05-03.md`, and
   `tools/bench/edlines_s1_sweep.py` — they document the falsification
   chain and are reproducible.
4. **Do not** start S3 implementation.  The design sketch in
   `edlines_segmentation_design_2026-05-03.md §4` is preserved for
   future revival.
5. **Pursue** the runtime gate (Track A's `corner_geometry_outlier`) as
   a separate small track (~4 h).  This is the practical fix on
   synthetic data alone.

## §8 What this PR closes

- The S1 implementation + sweep + falsification chain.
- The decision on S3 *on synthetic data alone* — deferred per
  `edlines_sota_experiments_2026-05-03.md §5`.

What this PR does NOT close:

- The validity of S3 (or any Phase 1-2 architectural rework) *if
  real-camera evidence emerges* of scene_0008-class failures at
  non-trivial rates.
- The runtime-gate work — separate track, owned independently.

## §9 Reproducing

The S1 filter is reverted in the same change as this memo, so the sweep
harness `tools/bench/edlines_s1_sweep.py` is preserved for future
revival but does **not** run as-is — re-implementing the
`corner_exclusion_px` filter in `extract_boundary_segments` (~25 LoC,
see git history of this branch) is required before the sweep produces
non-baseline rows.  The Phase-1 telemetry hook IS preserved and runs
without re-implementation.

For the Phase 1 telemetry dump (single scene; demonstrates the
`y = 456.5` integer-row floor on scene_0008):

```bash
uv run maturin develop --release \
    --manifest-path crates/locus-py/Cargo.toml --features bench-internals

LOCUS_DUMP_EDLINES_PHASE1=1 \
    uv run --group bench python -c "
import cv2, locus
from locus._config import DetectorConfig
img = cv2.imread('tests/data/hub_cache/locus_v1_tag36h11_1920x1080/images/scene_0008_cam_0000.png',
                 cv2.IMREAD_GRAYSCALE)
locus.Detector(config=DetectorConfig.from_profile('high_accuracy'),
               families=[locus.TagFamily.AprilTag36h11]).detect(img)
" 2>&1 | grep '^EDLINES_PHASE1' | grep 'bbox=881,456'
```

Expected output: `line0=0.000000,1.000000,-456.500000,...` — i.e. the
Phase-2 IRLS top-edge line is exactly `y = 456.5`, the integer-pixel
midpoint between rows 456 and 457.  This is the durable diagnostic
artefact of the S1 chain.
