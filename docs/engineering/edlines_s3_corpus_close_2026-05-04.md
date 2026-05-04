# EdLines S3 — corpus sweep close-out (2026-05-04)

Day-5 deliverable for the S3 (gradient-anchor walk) Phase 1-5 replacement
investigation per `edlines_s3_anchor_walk_design_2026-05-04.md`.

**Verdict: non-shippable per memo §4.3 mechanical criteria.** S3 v1 fixes
scene_0008's targeted failure brilliantly (-90 % on c1) but destroys most of
the rest of the corpus.

## §1 Hardware

AMD EPYC-Milan KVM, 8 logical CPUs (`Architecture: x86_64`). `--release`
build with `bench-internals`. CPython 3.14.3, rustc 1.92.0,
`RAYON_NUM_THREADS=8`.

## §2 Result

50-scene `locus_v1_tag36h11_1920x1080` corpus, baseline (current EdLines)
vs S3 (`edlines_use_anchor_walk = true`):

| metric | baseline | S3 | Δ |
|---|---:|---:|---|
| n detected | 50 | **30** | **-20 scenes lost (-40 %)** |
| corpus mean ‖r‖ | 0.191 | 0.887 | **+370 %** |
| corpus p99 ‖r‖ | 0.697 | 1.936 | **+176 %** |
| corpus max ‖r‖ | 3.833 | 2.016 | -47 % |
| corpus mean rot° | 0.134 | **3.508** | **+2 526 %** |
| corpus p99 rot° | 0.771 | **70.54** | **+9 050 %** |
| corpus max rot° | 0.874 | **98.82** | catastrophic |
| global d² mean | 714.7 | 5 328 983 | wildly miscal |
| scene_0008 c1 | 3.833 | **0.385** | **-90 %** ✓ |
| scene_0008 rot° | 0.874 | **0.075** | **-91 %** ✓ |

## §3 Verdict against memo §4.3 acceptance

| Criterion | Threshold | Observed | Pass/Fail |
|---|---|---|---|
| scene_0008 c1 | < 1.0 px | 0.385 px | ✓ pass |
| Corpus mean ‖r‖ within ±0.01 of baseline (0.191) | ±0.01 | +0.696 | ❌ FAIL (70× over) |
| Corpus p99 ‖r‖ within ±0.05 of baseline (0.697) | ±0.05 | +1.239 | ❌ FAIL (25× over) |
| No scene drops to no-detection | 0 drops | 20 drops | ❌ FAIL |

**Memo §4.3 non-shippable rules**:

- "Corpus mean ‖r‖ regresses by ≥ 0.05 px" → **+0.696 px regression** → triggered
- "Any scene drops to no-detection" → **20 scenes dropped** → triggered

**Conclusion: S3 v1 is non-shippable.** Do not enable on any production profile.

## §4 What went right

The mechanism works on scene_0008. The Python prototype on day 3 predicted
0.87 px on c1; the Rust port lands 0.385 px (better than expected, likely
because Phase 4-5 amplification in the prototype wasn't bypassed cleanly).
Falsification was sound; this is not a `the algorithm doesn't work` story.

## §5 What went wrong

The 20 scene-drops + catastrophic rotations on the surviving 30 indicate
S3 v1 has implementation issues that prevent it from generalising to the
typical corpus. The most plausible causes (in order of likelihood):

1. **Anchor threshold too high for real tag images.** `tau_anchor = 16.0`
   was calibrated on scene_0008's image where the tag has high contrast.
   Other scenes in the corpus may have tags at greater distance / lower
   contrast where `|∇| < 16` over much of the boundary. Loss-of-detection
   on these scenes is consistent with anchors below the gradient-magnitude
   floor.
2. **Top-4 segment selection too strict.** `min_segment_span_floor_px = 20`
   + `bbox_short / 4` rejects small tags or tags with fragmented chains.
   Combined with the gradient-orientation gate breaking long edges into
   segments, the corpus's smaller / more rotated tags may not produce
   four valid segments.
3. **Corner ordering / canonical rotation.** S3's `order_lines_cw` sorts
   by angle from centroid; the resulting CW order may not match the
   canonical (BL, BR, TR, TL) order the decoder expects. The decoder
   tries all 4 rotations against codeword tables, but if the corner
   geometry itself is mirrored (det < 0), the decoder may pick the
   chiral-mirror rotation, producing a 90/180° pose error. This is
   consistent with 4 / 30 surviving scenes having rot° ≥ 65°.
4. **Sub-pixel adjustment producing NaN/Inf on edge probes.** No empirical
   evidence yet, but plausible if `subpixel_adjust_anchor` returns
   non-finite values that propagate into TLS line fits.

None of these are fundamental algorithm bugs — they are implementation
calibration issues. Each is fixable with focused work, but doing so on
synthetic data alone has the same critique as S1 / Phase C.5: we cannot
distinguish whether fixes generalise to real cameras until we have
real-camera data.

## §6 Decision

Per memo §4.3 mechanical criteria + the shipping principle from PR #239 / S1
close-out: **do not ship S3 v1**. The runtime gate (PR #239) addresses
scene_0008's d² miscalibration honestly without changing the pose; S3 v1
would have *added* geometric accuracy to scene_0008 specifically, but the
20-scene loss + catastrophic rotation tail makes this unshippable.

What we keep:

1. **Code stays in tree behind the off-by-default flag.** S3 path runs only
   when `edlines_use_anchor_walk = true`. All shipped JSON profiles have
   it as `false` (or omit it — defaults to false via `#[serde(default)]`).
   No production user is exposed to S3 v1 without explicit opt-in.
2. **The 3 unit tests on synthetic squares stay green.** The synthetic-edge
   pipeline works.
3. **The Python prototypes (day-2 / day-3) and validation harness
   (`tools/bench/edlines_s3_validate_scene_0008.py`) stay**. They are the
   shortest path to re-run S3 on any future input.
4. **The design memo + this close-out memo document the path** for any
   future S3 v2 attempt:
   - Adaptive `tau_anchor` (per-component, scaled by local gradient
     statistics)
   - Looser top-4 selection (smaller `min_segment_span_floor_px`,
     re-merge fragmented chains)
   - Explicit chirality check on extracted corner quad (det of the
     2×2 matrix formed by adjacent corner displacements)
   - Sub-pixel adjustment NaN guard

## §7 What this matches in the broader investigation

This is the **fourth converging negative-result** on architectural EdLines
work targeting scene_0008 / synthetic-data residuals:

1. F4 falsification (`edlines_phase5_decoupling_2026-05-03.md`) — Phase 5
   chord-lock decoupling regresses corpus.
2. F5 / EdLines #1 (`edlines_sota_experiments_2026-05-03.md`) — Phase 3/4
   iteration regresses corpus.
3. S1 (`edlines_s1_corner_exclusion_2026-05-04.md`) — corner-pixel
   exclusion: scene_0008 partial fix, d² regression, 1 scene drop.
4. **S3 v1 (this memo)** — gradient-anchor walk: scene_0008 -90 % fix, 20
   scene drops + catastrophic rotation tail.

Each architectural intervention exhibits the same pattern: **scene_0008
specifically is fixable**, but generalising the fix to a synthetic corpus
without regressing the typical scenes has not been achievable. The unifying
explanation is that scene_0008 is a single point in a high-dimensional
failure manifold of synthetic data, and any algorithm tuned to that point
shifts other points off the working manifold.

The principal-engineer call from the runtime-gate decision applies again:
**defer architectural EdLines work until real-camera evidence**. The
runtime gate (PR #239) is the practical fix on synthetic data alone.

## §8 Reproducing

```bash
uv run maturin develop --release \
    --manifest-path crates/locus-py/Cargo.toml --features bench-internals

# Baseline:
PYTHONPATH=. uv run --group bench tools/bench/pose_cov_audit.py \
    --hub-config locus_v1_tag36h11_1920x1080 \
    --output-dir diagnostics/edlines_s3_corpus/baseline

# S3:
PYTHONPATH=. uv run --group bench tools/bench/pose_cov_audit.py \
    --hub-config locus_v1_tag36h11_1920x1080 \
    --output-dir diagnostics/edlines_s3_corpus/anchor_walk \
    --use-anchor-walk

# Compare:
PYTHONPATH=. uv run --group bench python tools/bench/edlines_s3_validate_scene_0008.py
```

Single-scene scene_0008 verification (where S3 *does* succeed):

```
=== Baseline ===
  c1 (BR/img-TL)      : 3.833 px
  max ‖Δ‖: 3.833 px

=== S3 ===
  c1 (BR/img-TL)      : 0.385 px
  max ‖Δ‖: 1.141 px
```
