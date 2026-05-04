# EdLines S3 — gradient-anchor walk Phase 1-2 replacement (2026-05-04)

Design memo for the S3 architectural rework recommended in
`edlines_segmentation_design_2026-05-03.md §4`. The decision to commit was
made after the §1 falsification experiment confirmed the core mechanism
works on Blender PSF.

## §1 What the falsification proved

Today's experiment (1 hour) directly tested whether sub-pixel gradient
methods can recover scene_0008's tag corner from raw gray-image data,
bypassing Phase 1-2's binary boundary tracer entirely. Both the top
and left edges were fit independently using gradient methods (no
Phase-1 boundary partition), then intersected.

| Method | Corner ‖Δ‖ vs GT |
|---|---:|
| Phase-2 IRLS, both edges (binary tracer) | **2.29 px** |
| 50%-transition regression, both edges | 0.32 px |
| **gradient-peak regression, both edges** | **0.32 px** |
| image-fit reference (top: 50%; left: memo §2) | 0.32 px |

Gradient-only methods hit the **same 0.32 px floor as the
known-good 50%-transition reference from `scene_0008_root_cause_2026-05-03.md §2`.**
This is the synthetic Blender PSF floor — better than Phase C.5's
documented ~0.6 px ERF floor because the tag-border edges in this
scene are sharp; corpus-wide expected ≤ 0.6 px.

**Recovery: 86 % of Phase 1-2's 2.29 px error budget.**

Per-edge breakdown:

| Edge | Phase-2 IRLS Δ intercept | gradient-peak Δ intercept |
|---|---:|---:|
| top | -7.07 px (forced slope=0) | +0.10 px |
| left | +9.33 px (slope error 7 %) | -0.34 px |

scene_0008's full 3.83 px error decomposes as:

```
Phase 1-2 contribution:        2.30 px  ← S3 targets this
Phase 4-5 amplification:       1.53 px  ← unchanged by S3
                              -------
                               3.83 px  observed
```

If S3 lands at the 0.3 px floor on Phase 1-2 → Phase 4-5 amplification
of an already-good starting point should be smaller (current 0.66×
attenuation factor of binary-tracer error → on a clean line, plausibly
1× to 3× of the 0.3 px floor → final corner error 0.3 to 1.0 px).

This is a 75-92 % improvement over the current 3.83 px on scene_0008.

## §2 Algorithm

A focused subset of Akinlar & Topal's ED-Lines (2010), trimmed to what
the falsification proved sufficient. Helmholtz a-contrario validation
(NFA) is **not** in v1 — added in v2 only if v1 produces too many false
segments.

### §2.1 Inputs and outputs

```
Inputs:
  - Gray image (full resolution)
  - Component bbox + label mask (from existing Phase 0 segmentation)

Output:
  - 4 sub-pixel lines (nx, ny, d) defining the four tag edges, CW-ordered
  - 4 sub-pixel corners as the intersections of adjacent line pairs
```

### §2.2 Stage A — anchor pixel extraction

For each pixel `(x, y)` inside the component bbox + a 2-px padding:

1. Compute Sobel gradient `(gx, gy)`. Cache `|∇|² = gx² + gy²` and
   `θ = atan2(gy, gx)` quantised to 4 octants (vertical / horizontal /
   diag-NE / diag-NW).
2. **Anchor criterion**: `(x, y)` is an anchor iff
   - `|∇|² > τ²_anchor`
   - `|∇(x, y)| > |∇(x ± δx, y ± δy)|` along the gradient direction
     (non-maximum suppression in the gradient orientation, similar to
     Canny's NMS step but cheaper because we use the 4-octant
     quantisation).

Default `τ_anchor = 16` (from the falsification — distinguishes tag
edge gradients ~60 from interior-bit boundary gradients ~50 from
background noise gradients ~5).

### §2.3 Stage B — gradient-direction chain walking

For each anchor pixel, walk along the **edge tangent direction**
(perpendicular to gradient) building a chain of connected anchors.

Smart-routing algorithm (from ED-Lines):

```
def walk(anchor):
    chain = [anchor]
    for direction in [+tangent, -tangent]:
        cursor = anchor
        while True:
            # Three candidate next pixels: ahead, ahead-left, ahead-right
            candidates = [cursor + direction, cursor + direction.rot(+45°),
                          cursor + direction.rot(-45°)]
            # Pick highest-gradient candidate that is an anchor
            next_p = max(c in candidates if is_anchor(c), key=|∇|)
            if next_p is None or len(chain) > MAX_CHAIN: break
            chain.append(next_p)
            cursor = next_p
    return chain  # ordered list of pixels along the edge
```

Each anchor belongs to exactly one chain (mark consumed).

### §2.4 Stage C — chain → linear segments

Split each chain at curvature changes. Algorithm: sliding-window line
fit; if the fit error exceeds 1 px RMS, split at the worst-fitting point.

Each segment is a candidate edge.

### §2.5 Stage D — top-4 segment selection

Pick the 4 longest segments such that:

- All 4 segments are at least `MIN_LEN_PX = max(20, bbox_short / 4)` px long
- Pairwise angle constraint: each pair of adjacent (CW-ordered) segments
  has an angle of `90° ± 30°` between them
- Pair-wise non-parallel: no two of the 4 segments are within 15° of
  parallel (rejects the case where two parallel "best" segments are picked
  and there's no top/bottom)

If no valid 4-segment set exists, return failure (caller falls back to
the current Phase 1-2 path or rejects the candidate).

### §2.6 Stage E — sub-pixel line fit

For each of the 4 chosen segments, fit a TLS line through the chain's
anchor positions (with sub-pixel adjustment using the 3-point parabolic
vertex of `|∇|` along the gradient direction at each anchor — exactly the
falsification method).

### §2.7 Stage F — corner extraction

Order the 4 lines CW (by angle of their tangent). Adjacent line pairs
intersect to give the 4 corners.

The result mirrors the existing EdLines pipeline output type
(`[Point; 4]` corners + `CornerCovariances`).

## §3 Integration

### §3.1 Config surface

```rust
pub struct EdLinesConfig {
    // ... existing fields ...
    /// Use gradient-anchor walk for Phase 1-2 (S3 architecture).
    /// Default: false (current binary-tracer path, byte-identical).
    pub use_anchor_walk: bool,
    /// Anchor gradient-magnitude threshold τ_anchor (squared internally).
    /// Default: 16.0 (intensity units, matching falsification).
    pub anchor_threshold: f64,
}
```

Plumbed through `DetectorConfig::edlines_anchor_walk` (analogous to
`edlines_imbalance_gate`) → JSON shim with `#[serde(default)]` →
Pydantic `QuadConfig.edlines_anchor_walk: bool` → `.pyi` stub.

**Off-path byte-identical**: when `use_anchor_walk = false`, the old
`run_pipeline_with_mode` path runs unchanged. No snapshot rebless on
disabled-state.

### §3.2 Pipeline routing

In `extract_quad_edlines` (`crates/locus-core/src/edlines.rs`):

```rust
if cfg.use_anchor_walk {
    let result = run_anchor_walk(arena, gray, labels, comp_label, stat, cfg);
    // result already contains sub-pixel lines + corners; skip Phase 4-5.
} else {
    // existing AXIS → DIAG fallback chain
    run_pipeline_with_mode(...)
}
```

**Phase 4-5 are NOT run when S3 is on**. The anchor walk produces
sub-pixel lines directly; Phase 4 (sub-pixel re-fit) and Phase 5 (joint
GN) were tuned for the binary-tracer baseline and have known
amplification properties (see EdLines #1 falsification —
`edlines_sota_experiments_2026-05-03.md §2`). On already-clean lines,
they could regress (small perturbations amplified). v1 skips them.

This is a **complete Phase 1-5 replacement**, not an addition. Total
code surface ~250-300 LoC.

## §4 Falsification + acceptance criteria

### §4.1 Day-2 falsification (anchor + chain walking)

**Cheap test**: dump anchor + chain visualisation for scene_0008's tag.
Expected: 4 chains, one per edge, each ≥ 100 anchors. If the chains
fragment (>4 chains for a single edge) or merge (single chain spans
multiple edges), the chain-walking algorithm needs revision before
proceeding.

**Falsification**: if scene_0008 produces fewer than 4 valid chains
matching the tag edges, S3 v1 is dead in the water (need NFA
validation or different chain-splitting heuristic).

### §4.2 Day-4 falsification (full S3 on scene_0008)

Run S3 on scene_0008 alone; measure corner residuals.

| Outcome | Action |
|---|---|
| corner ‖Δ‖ < 1.0 px | proceed to corpus sweep (day 5) |
| 1.0 ≤ ‖Δ‖ < 2.0 px | document as partial fix; corpus sweep with caveat |
| ‖Δ‖ ≥ 2.0 px | falsified — synthetic PSF floor higher than expected; defer |

### §4.3 Day-5 acceptance (corpus)

Run pose_cov_audit on the full 50-scene `locus_v1_tag36h11_1920x1080`
corpus with `use_anchor_walk = true`.

**Ship if all of**:
- scene_0008 corner 1 ‖r‖ < 1.0 px
- Corpus mean ‖r‖ within ±0.01 px of baseline (0.191 px)
  - tighter than S1's ±0.005 because S3 has more degrees of freedom
- Corpus p99 ‖r‖ within ±0.05 px of baseline (0.697 px)
- No scene drops to no-detection
- `cargo nextest --release --all-features` passes
- `cargo test regression_render_tag` passes (off-path; with use_anchor_walk = true, snapshots may need rebless on the gate-firing scenes)

**Non-shippable if any of**:
- scene_0008 corner 1 ‖r‖ ≥ 2.0 px (S3 didn't help materially)
- Corpus mean ‖r‖ regresses by ≥ 0.05 px (S3 hurts typical scenes)
- Any scene drops to no-detection (anchor walk too aggressive)

If between "ship" and "non-shippable" — partial improvement —
**document as conditional ship**: opt-in on `high_accuracy` only, with
the trade documented in the memo. Same call-pattern as the runtime gate
(PR #239).

## §5 Out of scope (v1)

- **NFA / Helmholtz a-contrario validation**: classical ED-Lines uses
  this to filter false segments. We rely on the top-4-longest +
  pairwise-angle gate instead. Add NFA in v2 only if v1 produces too
  many spurious quads.
- **Multi-scale / image pyramid**: anchor extraction at decimated
  resolutions. Track G separately.
- **Distortion handling**: S3 operates in image-pixel space like the
  current EdLines path. Distorted intrinsics still gated upstream
  (`detector.rs:194-208`).
- **Phase 4-5 integration**: skipped per §3.2. If S3 corner output
  needs further refinement, that's a separate change after we measure
  the actual quality.

## §6 Schedule

| Day | Deliverable |
|---|---|
| 1 (today) | This memo + branch `s3-anchor-walk` off main |
| 2 | Stage A (anchors) + Stage B (chains) + day-2 falsification dump on scene_0008 |
| 3 | Stages C-E (segment split, top-4, line fit) — independent unit tests |
| 4 | Stage F (corner extraction) + integration behind config flag + day-4 falsification on scene_0008 |
| 5 | Corpus sweep + decision memo + (if shipping) snapshot rebless |

Total: 5 days. Each day has a checkpoint that can stop the work cleanly
if results don't justify continuing.

## §7 Reproducing today's falsification

```bash
PYTHONPATH=. uv run --group bench python tools/bench/edlines_s3_falsification.py \
    --scene-id scene_0008_cam_0000 \
    --hub-config locus_v1_tag36h11_1920x1080
```

(Script exists in this PR's tools/bench/ directory.) The numbers in
§1's table are reproducible to 4 dp with the script + the hub corpus's
scene_0008 image.

## §8 Reference

Akinlar & Topal, *ED-Lines: A real-time line segment detector with a
false detection control*, Pattern Recognition Letters 32, 2011. The
canonical algorithm we're implementing.
