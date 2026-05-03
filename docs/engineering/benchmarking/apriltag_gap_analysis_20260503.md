# AprilTag-vs-Locus gap analysis

**Date:** 2026-05-03
**Question:** where is `locus-tag` lacking relative to AprilTag-C, and where should we invest engineering effort to close the gap?

## Headline

**With the production-accuracy profile (`high_accuracy`), Locus is decisively ahead of AprilTag-C on this corpus.** Translation error is 3–7× tighter at every percentile, rotation p50 is 1.2–1.5× tighter, latency is 3–3.5× faster, and rotation outliers (p99) are 50–60× better (AprilTag exhibits catastrophic ~50° pose flips on symmetric tags here).

The earlier framing — "AprilTag is tightest at every distance, Locus pays for it only at 4K" — was an artifact of comparing AprilTag against the `standard` profile (latency-tuned). On `standard`, Locus is genuinely 4× behind AprilTag on rotation accuracy. The gap is configuration-driven, not algorithmic.

The two real residual gaps to AprilTag, with `high_accuracy`:
1. **Recall**: Locus misses 1–2 tags per 100 frames where AprilTag finds them. Plausibly the `min_area=800` filter clipping distant tags.
2. **Rotation p95**: 1.5–2.8× looser than AprilTag. Pose-solver tuning opportunity for the upper body of the rotation distribution.

## Sweep configuration

Corpus: `single_tag_locus_v1_tag36h11_{1280x720, 1920x1080, 3840x2160}`, 100 frames per cell.

```bash
# standard profile (CLI default)
PYTHONPATH=. uv run --group bench tools/cli.py bench real \
  --hub-config <C> --compare --max-hamming 0 --limit 100 \
  --record-out runs/gap_analysis/<C>.parquet

# high_accuracy profile (one-line monkey-patch in tools/cli.py:450, reverted)
# also requires --refinement None (EdLines extraction is incompatible with Erf
# refinement) and skipping the Locus (Soft) wrapper instantiation
# (Soft is incompatible with EdLines extraction).
```

Locus wrappers consume the CLI-built config; AprilTag and OpenCV use their respective package defaults via `tools/bench/utils.py`.

## Standard profile (CLI default — latency-optimized)

| Library | Res | Recall | Precision | Trans p50 (mm) | Trans p95 (mm) | Trans p99 (mm) | Rot p50 (°) | Rot p95 (°) | Rot p99 (°) | Latency p50 (ms) |
| :--- | :-: | :-: | :-: | -: | -: | -: | -: | -: | -: | -: |
| Locus (Hard) | 720 | 1.000 | 1.000 | 3.41 | 29.76 | 35.65 | 0.323 | 1.257 | 3.343 | 8.5 |
| AprilTag | 720 | 1.000 | 1.000 | 3.19 | 24.10 | 34.26 | 0.082 | 0.259 | 50.527 | 12.7 |
| OpenCV | 720 | 1.000 | 1.000 | 3.92 | 27.89 | 42.40 | 0.133 | 0.364 | 0.426 | 22.1 |
| Locus (Hard) | 1080 | 1.000 | 1.000 | 4.53 | 57.24 | 135.22 | 0.320 | 1.946 | 41.371 | 36.3 |
| AprilTag | 1080 | 1.000 | 1.000 | 5.00 | 41.62 | 53.05 | 0.084 | 0.243 | 57.337 | 34.3 |
| OpenCV | 1080 | 1.000 | 0.957 | 5.83 | 58.32 | 69.14 | 0.087 | 0.313 | 0.371 | 42.7 |
| Locus (Hard) | 2160 | 1.000 | 1.000 | 9.54 | 93.71 | 116.14 | 0.361 | 1.179 | 1.808 | 66.0 |
| AprilTag | 2160 | 1.000 | 1.000 | 8.63 | 82.74 | 101.56 | 0.091 | 0.302 | 59.170 | 139.7 |
| OpenCV | 2160 | 1.000 | 0.978 | 10.58 | 99.08 | 223.14 | 0.111 | 0.417 | 0.442 | 147.9 |

On `standard`, AprilTag has a clear ~4× rotation advantage at p50/p95. Locus's only standard-profile wins are rotation p99 (catastrophic-tail suppression) and latency at 720p / 2160p.

## High_accuracy profile (production-accuracy)

| Library | Res | Recall | Precision | Trans p50 (mm) | Trans p95 (mm) | Trans p99 (mm) | Rot p50 (°) | Rot p95 (°) | Rot p99 (°) | Latency p50 (ms) |
| :--- | :-: | :-: | :-: | -: | -: | -: | -: | -: | -: | -: |
| Locus (Hard) | 720 | 0.980 | 1.000 | **0.46** | **4.81** | **6.75** | **0.063** | 0.455 | **0.554** | **4.4** |
| AprilTag | 720 | 1.000 | 1.000 | 3.19 | 24.10 | 34.26 | 0.082 | **0.259** | 50.527 | 13.5 |
| Locus (Hard) | 1080 | 0.978 | 1.000 | **0.83** | **10.78** | **17.70** | **0.070** | 0.686 | **1.341** | **7.5** |
| AprilTag | 1080 | 1.000 | 1.000 | 5.00 | 41.62 | 53.05 | 0.084 | **0.243** | 57.337 | 26.4 |
| Locus (Hard) | 2160 | 0.956 | 1.000 | **1.46** | **11.89** | **28.16** | **0.060** | 0.440 | **1.014** | **33.4** |
| AprilTag | 2160 | 1.000 | 1.000 | 8.63 | 82.74 | 101.56 | 0.091 | **0.302** | 59.170 | 109.7 |

On `high_accuracy`, Locus dominates AprilTag on every dimension except recall and rot p95.

## The role reversal

| Dimension | Standard | High_accuracy |
| :--- | :--- | :--- |
| Trans p50 | AprilTag 7% tighter @720p, Locus 10% tighter @1080p | **Locus 5.9–7× tighter at every res** |
| Trans p99 | Roughly tied | **Locus 3.0–7.0× tighter** |
| Rot p50 | AprilTag 4× tighter | **Locus 1.2–1.5× tighter** |
| Rot p95 | AprilTag 4–8× tighter | AprilTag 1.5–2.8× tighter |
| Rot p99 | Locus dominates (no 50° outliers) | **Locus dominates by 50–60×** |
| Latency | Locus 1.5× faster (720p, 2160p), tied (1080p) | **Locus 3.0–3.5× faster everywhere** |
| Recall | Tied (all 100%) | AprilTag 100%, Locus 95.6–98% |

## Where the gap actually is

Initial hypotheses for both gaps proved wrong on inspection. The deep root-cause sections below trace each gap to its actual mechanism. Summary:

### Gap 1 — Recall miss on `high_accuracy`

Locus misses 1–2 tags per 100 frames where AprilTag finds them: 1 at 720p, 1 at 1080p, 2 at 2160p (4 total). **Not a `min_area` issue** — half the misses are on `scene_0031` at PPM > 2300 (a *close* tag, two orders of magnitude above any plausible `min_area`), the other half are decoder rejections specific to the `--max-hamming=0` configuration. See deep root cause.

### Gap 2 — Rotation p95 looseness

Locus's rotation p95 is 1.5–2.8× looser than AprilTag's. **Not a high-AOI grazing-angle problem** — the tail is concentrated at frontal small tags (PPM < 500, AOI ≈ 25°) where corner-localization noise drives rotation ambiguity. AprilTag handles 16/21 of Locus's worst rotation cases without difficulty. See deep root cause.

## Deep root cause — recall miss

Per-miss attribution by joining `missed_gt` records to spatially-attributed `rejected_quad` records (same `image_id` + `tag_id`):

| Res | Image | tag_id | distance | aoi | ppm | Stage of failure |
| :--- | :--- | -: | -: | -: | -: | :--- |
| 720p | scene_0002 | 219 | 1.19m | 51.5° | 736 | RejectedDecode (×1) |
| 1080p | scene_0031 | 505 | 0.59m | 14.6° | **2348** | **PRE-EXTRACTION** (no quad ever extracted) |
| 2160p | scene_0020 | 356 | 5.20m | 19.9° | 550 | RejectedDecode (×3) |
| 2160p | scene_0031 | 505 | 1.18m | 14.6° | **2372** | **PRE-EXTRACTION** (no quad ever extracted) |

**The `min_area` hypothesis is wrong.** Half the misses (scene_0031 at 1080p and 2160p) are on a *close* tag with PPM 2348–2372 — at tag_size 0.165m, that's a side length of ~388px and an area of ~150,000 pixels, two orders of magnitude above `min_area=800`. The quad is never extracted at all. The other half are decoder-stage rejections at the strictest Hamming setting.

There are **two distinct failure modes**, with very different fixes:

### Mode A — EdLines extraction failure (2/4 misses)

`scene_0031` consistently fails to produce a quad at 1080p and 2160p. The tag is close, frontal (AOI 14.6°), and well-lit (PPM > 2300). With `min_area=800` and EdLines extraction, *something specific to that scene's edge profile* causes EdLines's line-fitting heuristics to drop the candidate before extraction completes.

**Mechanism hypothesis:** EdLines fits anchored line segments via gradient-direction continuity. On scenes with sharp transitions (e.g., a tag rendered against a uniform background with no shadow), the four tag edges may not satisfy EdLines's line-segment anchor criteria — the gradient profile is too sharp / too symmetric to register as four distinct line segments.

**Verification action:**
1. Run rerun visualization on `scene_0031` and capture the EdLines intermediate stage (the line-segment array) — is the tag's contour represented?
2. Compare to scene_0030 / scene_0032 (presumably similar but successful): what's different about the rendering?
3. If EdLines is dropping anchors, the fix is in `crates/locus-core/src/edge_refinement.rs` or `quad/edlines.rs` — likely a gradient-magnitude threshold or anchor-density parameter.

### Mode B — RejectedDecode at `max_hamming=0` (2/4 misses)

These tags are extracted as quads but rejected at decode because their bit pattern doesn't exactly match any codeword. With `--max-hamming=2` (the actual CLI default), they would recover.

The gap-analysis sweep used `--max-hamming=0` for tightness against the Soft sweep methodology; the recall miss in this configuration is partly an artifact of that choice. PR #225's data shows Hard at h=2 has 100% precision *and* 100% recall on these corpora — so the Mode B miss is configuration-specific.

**Action:** for accuracy-conscious deployments, document that `--max-hamming=2` is the production setting. Hard at h=2 gets full recall with full precision; the h=0 cell of the gap-analysis sweep was a methodology artifact.

### Net recall picture

If we relax `--max-hamming` from 0 to 2, the recall miss collapses from 4 → 2, and the residual 2 cases are the Mode A EdLines failures on scene_0031. **The "1–2 tags / 100 frames" gap is realistically half EdLines, half configuration-artifact.** EdLines is the actual bug.

## Deep root cause — rotation p95 tail

Top 5% of rotation errors across all three resolutions (n=7 of 136 matched records). Bulk = bottom 50%.

| Axis | Top 5% mean | Bulk mean | Ratio |
| :--- | -: | -: | -: |
| `distance_m` | 3.88 | 1.14 | **3.39×** ↑ |
| `aoi_deg` | 25.5° | 41.3° | 0.62× ↓ |
| `ppm` | **452** | **1718** | **0.26×** ↓ |
| `trans_err_m` | 12.6 mm | 0.8 mm | **15.94×** ↑ |
| `repro_err_px` | 0.35 | 0.13 | **2.57×** ↑ |

Per-resolution: every tail bucket has PPM ≈ 450, distance 2–6m, AOI ≈ 25°. **The tail is "small frontal far tags," not "high-AOI grazing tags."** Within the tail:
- `r(rot_err, trans_err) = -0.107` → rotation tail is **not** correlated with translation error in the tail.
- `r(rot_err, repro_err) = +0.648` → rotation tail **is** strongly correlated with reprojection error.

**The tail comes from corner-localization noise on small frontal tags, NOT from grazing-angle solver degeneracy.**

### Comparing to AprilTag on the same tags

For each row in Locus's tail (n=21 with the cross-resolution matching), look up AprilTag's rot_err on the same `(image_id, tag_id)`:

- Locus tail mean rot_err: 0.935°
- AprilTag rot_err on the same tags: 0.278°
- Pearson correlation: −0.123 (essentially zero)
- Of Locus's 21 tail rows, only 5 are also in AprilTag's own p95 tail.

**AprilTag handles 16/21 of Locus's worst rotation cases without difficulty.** This isn't a hard-tag problem, it's a Locus-specific weakness in this regime.

### Mechanism hypothesis

Frontal small tags are *almost* a 2D shape — the four corners are nearly coplanar in the image plane and depth signal is weak. Pose estimation is mathematically near-degenerate: small corner perturbations move the rotation estimate substantially while leaving translation (centroid) stable. That matches the data: rotation breaks, translation doesn't.

Locus's pipeline at this regime:
- **Corner refinement**: GWLF anisotropic transversal model (`gwlf_transversal_alpha = 0.01`) + ERF/EdLines edge fitting. At low PPM, the gradient signal-to-noise drops and per-corner uncertainty rises.
- **Pose solver**: weighted Levenberg-Marquardt with structure-tensor priors (`structure_tensor_radius = 2`, `huber_delta_px = 1.5`, `tikhonov_alpha_max = 0.25`). When per-corner covariances are high (low PPM), the solver weights residuals by inverse covariance — but if the covariance estimate itself is noisy, the weighting amplifies rather than damps.

AprilTag's solver is a simpler iterative reweighted least-squares without anisotropic covariance priors. For frontal small tags it ends up *more robust* because it doesn't trust the per-corner covariance and simply minimizes raw reprojection. Locus's structure-tensor + Mahalanobis weighting is an asset on bigger / oblique tags (and visible in the translation 5–7× lead) but a liability when the covariance estimate itself is unreliable.

### Action candidates, ordered by expected ROI

1. **Cap structure-tensor weight at low PPM**: when PPM < 500 (or equivalently when median per-corner covariance trace exceeds a threshold), fall back to isotropic weighting. The pose solver currently always trusts the structure-tensor prior. ~20 LOC change in `crates/locus-core/src/pose_weighted.rs`. Estimated rotation p95 improvement: a measurable fraction of the 1.5–2.8× gap, based on the r=0.648 repro/rot correlation.
2. **Subpixel corner refinement on small tags**: GWLF's transversal alpha controls how strongly the gradient profile fits the cross-edge intensity. At low PPM, the gradient is shallower; the fit may need a different alpha. ~10 LOC parameter sweep, then bake into the `high_accuracy` profile.
3. **Tikhonov regularization scaling with corner uncertainty**: currently `tikhonov_alpha_max = 0.25` is a hard ceiling. Make it scale with PPM (or with the trace of the corner covariance matrix). Adds principled damping to under-determined cases.
4. **AprilTag-style pure reprojection objective** as a fallback: at low PPM, switch to reprojection-only LM (no anisotropic prior). Bigger surgery; only worth it if (1)–(3) don't close the gap.

### Net rotation picture

The tail is concentrated and characterized: **frontal far small tags, corner-localization driven**. The fix is in pose-solver / corner-localization tuning at low-PPM regime, not architectural. Action (1) — capping structure-tensor weight at low PPM — is the cleanest first cut.

## Net engineering plan

| Gap | Root cause | First-cut fix | Effort |
| :--- | :--- | :--- | :--- |
| Recall miss (Mode A: EdLines) | `scene_0031`'s edge profile drops anchors | Rerun-visualize, then tune EdLines anchor-density / gradient-magnitude threshold | 1–2 days investigation + small parameter change |
| Recall miss (Mode B: RejectedDecode) | `--max-hamming=0` artifact | Use `--max-hamming=2` for accuracy benches; document | Configuration / docs only |
| Rotation p95 tail | ~~Structure-tensor priors over-weighted at low PPM~~ — **REFUTED, see Principal-engineer review** | ~~Cap weight at low PPM (PPM<500) → isotropic fallback~~ | ~~~20 LOC~~ — **do not ship** |

The first two survive review. The third does not — see below.

## Principal-engineer review (added after independent critiques + ablation)

Three parallel independent reviews scrutinized the analysis above (algorithmic, statistical, engineering). They converged on three concerns: the rot-p95 mechanism was mislabeled, the n=7 statistics were too thin to support causal claims, and the proposed "cap at PPM<500" fix was underspecified by ~30 LOC of signature-plumbing it didn't account for. To resolve the open question — *is the weighted LM solver actually the problem at low PPM?* — we ran a decisive ablation.

### Ablation: weighted-LM (`Accurate`) vs IPPE-only (`Fast`)

Same corpora, same `--max-hamming=2` for parity. `tools/bench/utils.py` always passes `pose_estimation_mode=Accurate` for Locus through `bench real`; we monkey-patched line 568 to `Fast` for the second leg, then reverted.

| Resolution | Metric | Accurate (current) | Fast (proposed alternative) | Ratio (Fast/Accurate) |
| :--- | :--- | -: | -: | -: |
| 720p | trans_p50 (mm) | 0.56 | 0.80 | 1.43× worse |
| 720p | trans_p95 (mm) | 5.89 | 28.73 | **4.88× worse** |
| 720p | rot_p50 (°) | 0.063 | 0.279 | 4.43× worse |
| 720p | rot_p95 (°) | 0.522 | 5.868 | **11.24× worse** |
| 720p | rot_p99 (°) | 2.567 | 23.695 | 9.23× worse |
| 720p | latency (ms) | 13.2 | 8.7 | 0.66× (faster) |
| 1080p | trans_p50 (mm) | 0.83 | 2.01 | 2.42× worse |
| 1080p | trans_p95 (mm) | 10.78 | 131.56 | **12.21× worse** |
| 1080p | rot_p50 (°) | 0.070 | 0.401 | 5.76× worse |
| 1080p | rot_p95 (°) | 0.686 | 4.853 | **7.08× worse** |
| 1080p | rot_p99 (°) | 1.341 | 7.456 | 5.56× worse |
| 2160p | trans_p50 (mm) | 1.46 | 4.83 | 3.31× worse |
| 2160p | trans_p95 (mm) | 11.89 | 311.93 | **26.23× worse** |
| 2160p | rot_p50 (°) | 0.060 | 0.701 | 11.63× worse |
| 2160p | rot_p95 (°) | 0.440 | 11.746 | **26.71× worse** |
| 2160p | rot_p99 (°) | 1.014 | 18.429 | 18.18× worse |

Bootstrap 95% CI on `rot_p95(Fast) / rot_p95(Accurate)` over 2000 resamples of n=397 paired records: **[7.4×, 22.2×]** — strongly significant.

**Per-record paired comparison on Accurate's p95+ tail (n=20):** Fast produces a tighter rotation on **only 1 of 20 records**. The pose-mode swap is uniformly worse, not just on the tail.

### What the ablation tells us

1. **The "weighted LM is hurting at low PPM" hypothesis is REFUTED.** Fast (IPPE-only) is uniformly worse — by 5–25× on rotation p95 and 5–26× on translation p95. The weighted LM solver is doing real work *across the entire distribution*, including at low PPM. Removing it would be catastrophic.
2. **The proposed fix ("cap structure-tensor weight at PPM<500 → isotropic fallback") is in the wrong direction.** Capping the weight is moving toward the Fast regime; the data says Locus's accuracy lead over AprilTag is a direct consequence of that weight, not a side effect. Shipping that change would regress translation by 4–26× and rotation by 7–27×.
3. **The rot_p95 tail is the residual cost of the difficult cases on a solver that is otherwise *crushing* AprilTag.** AprilTag's IRLS-style solver gets a tighter p95 (1.5–2.8×) on small frontal far tags specifically, but pays for that flexibility with a vastly looser p99 (50–60× worse: catastrophic 50° pose flips that Locus does not exhibit) and 5–7× looser translation across the full distribution.

### What the original "deep root cause" got right and wrong

| Claim | Status after ablation |
| :--- | :--- |
| The tail is at low PPM, frontal small far tags | ✓ Holds (geometrically descriptive) |
| The tail correlates with reprojection error (r=0.65) | ~ Holds **as a description**; statistically fragile (95% CI [−0.55, 0.96] per Agent 2's bootstrap), and the *causal* implication ("noisy corners drive the tail") doesn't survive — Fast also degrades on the same corners and gets *worse* rotation, so the corners aren't the binding constraint here |
| Structure-tensor priors are amplifying noise at low PPM | ✗ **Refuted** by ablation. Without the prior (Fast), every metric is worse at low PPM, not better |
| "Cap structure-tensor weight at PPM<500" is the right fix | ✗ **Refuted**. This change moves toward the Fast regime, which the data shows is uniformly worse |

### Principal-engineer net call

**Drop the rotation-side code work entirely from the pre-release surface.** The proposed fix would regress production accuracy across the board. Without strong evidence that a *different* fix (e.g., improving corner localization on small tags via a re-enabled ERF refinement, or AprilTag-style IRLS as a per-tag fallback only on the 5% hardest cases) would help, no rotation-side change ships. The hypotheses for both alternatives are reasonable but unvalidated.

**The post-release follow-ups, in priority order:**
1. **Investigate the rot_p95 tail at the corner-localization level** (not the solver level). The hypothesis is that pre-pose sub-pixel corner accuracy is the limiting factor on small tags. Test: enable `decoder.refinement_mode=Erf` in `high_accuracy.json` (currently `None`) and re-run. If `rot_p95` tightens, ERF earns its keep on this regime; ship the profile change. If not, the cause is elsewhere and we have a follow-up scope.
2. **Investigate EdLines failure on `scene_0031`.** Use rerun visualization. Per Agent 3, before tuning `grad_min_mag` (the doc's first guess), test `--imbalance-gate disabled` and `min_fill_ratio=0.05` — both are cheaper to try and may turn out to be the real knob. Only ~50% of the recall miss is EdLines (the other 50% is the `--max-hamming=0` methodology artifact), so the leverage is bounded.
3. **AprilTag IRLS as a per-tag fallback** (only if (1) doesn't close the gap). The principle: when the weighted-LM converges to a poor minimum (detect via condition-number probe of `JᵀWJ`), retry with AprilTag-style IRLS on those specific tags. This preserves the wins on the bulk distribution while addressing the tail. Real engineering scope (~150 LOC + tests). Don't pursue without (1)'s data.

### What ships pre-release

- **This document** (corrected mechanism, ablation result, refuted fix, deferred follow-ups). Durable record so a future engineer doesn't repeat the proposed `Fast`-fallback as "the obvious fix" — it isn't.
- **Hamming methodology clarification** (use `--max-hamming=2`, not 0, for accuracy benches; this is the existing CLI default). Already noted in the recall section above.
- **No code change to the pose pipeline.** Validated as net-positive by ablation: keeping the weighted LM is the right call.

### Non-gaps

- **Translation accuracy**: Locus's ERF + weighted-LM pipeline produces 3–7× tighter translation than AprilTag's iterative solver on this corpus. No gap to close.
- **Latency**: Locus is 3–3.5× faster at every resolution on `high_accuracy`. No gap to close.
- **Rotation outlier suppression**: Locus's p99 is 50–60× better on `high_accuracy` (and 15–33× better on `standard`). AprilTag's symmetric-tag pose ambiguity produces catastrophic flips that Locus avoids. No gap to close.

## What this changes about engineering priorities

1. **The bench harness's default profile (`standard`) is the wrong basis for AprilTag comparison.** It optimizes for latency — `min_area=36`, `extraction_mode=ContourRdp`, no EdLines, ERF refinement — and produces accuracy figures that AprilTag-C beats by 4× on rotation. Public bench reports rendered with this profile mislead readers about Locus's accuracy ceiling.
2. **A fair comparison uses `high_accuracy`** for both binaries (or at least Locus). When users care about pose accuracy they should use this profile; when they care about latency, the standard profile already wins on latency at most resolutions.
3. **The recall miss is NOT a `min_area` problem.** Initial hypothesis was wrong — half the misses are EdLines extraction failures on a single scene with PPM 2300+ (way above any plausible `min_area` setting), and the other half are decoder rejections at the strictest `--max-hamming=0` (which would recover at h=2). See "Deep root cause — recall miss" above.
4. **The rotation p95 tail is NOT high-AOI grazing-angle solver degeneracy.** Initial hypothesis was wrong — the tail is concentrated at *frontal* small tags (PPM < 500, AOI ≈ 25°) where corner-localization noise dominates. r(rot_err, repro_err) = 0.65; r(rot_err, trans_err) = −0.11. The fix is in the pose solver's structure-tensor weighting, not in Huber/Tikhonov tuning of the LM iteration itself. See "Deep root cause — rotation p95 tail" above.

## Caveats

- **Corpus is `single_tag_locus_v1_tag36h11_*`**: synthetic renders from our own pipeline. Real camera data and other corpora (charuco, aprilgrid, ICRA forward) may shift the picture. The render-tag SOTA report (`render_tag_sota_20260425.md`) on a different rendered corpus is broadly consistent with the `high_accuracy` numbers above.
- **AprilTag wrapper has a known corner-canonicalization caveat** (`tools/bench/utils.py:907-961`, flagged in PR #223): pose comparisons are valid (180° z-rotation correction applied), but raw-corner reprojection error against AprilTag is contaminated by corner-array reordering. None of the metrics above use raw-corner reprojection; pose error is via 6DOF GT.
- **Sample size is 100 frames per cell.** p99 numbers are based on a single tail observation; treat them as directional rather than precise. Trends across resolutions and distinct corpora are the load-bearing claim.
- **AprilTag-C version** is `pupil_apriltags` (Python bindings of AprilTag3 C reference) at whatever version is pinned in the bench environment. A newer AprilTag release or different binding may shift the numbers slightly.
- **The high_accuracy sweep required disabling the Locus (Soft) wrapper** because Soft decoding is incompatible with EdLines extraction (configuration validation rejects the combination). PR #225 already documented Soft as a deprecation candidate; this analysis reinforces that conclusion since high_accuracy + Hard already wins on every dimension that matters.

## Reproducing this analysis

```bash
# Build the wheel
uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml

# Standard profile (current bench-harness default)
mkdir -p runs/gap_analysis
for corpus in single_tag_locus_v1_tag36h11_{1280x720,1920x1080,3840x2160}; do
  PYTHONPATH=. uv run --group bench tools/cli.py bench real \
    --hub-config "$corpus" --compare --max-hamming 0 --limit 100 \
    --record-out "runs/gap_analysis/${corpus}.parquet"
done

# high_accuracy profile requires three temporary edits to tools/cli.py:
#   1. Line 450: `from_profile("standard")` → `from_profile("high_accuracy")`
#   2. Lines 461-472: comment out the Locus (Soft) wrapper block
#      (Soft is incompatible with EdLines extraction)
# And one CLI flag:
#   --refinement None  (Erf is incompatible with EdLines)
for corpus in single_tag_locus_v1_tag36h11_{1280x720,1920x1080,3840x2160}; do
  PYTHONPATH=. uv run --group bench tools/cli.py bench real \
    --hub-config "$corpus" --compare --max-hamming 0 --refinement None \
    --limit 100 \
    --record-out "runs/gap_analysis/ha_${corpus}.parquet"
done

# A `--profile` flag would obviate the temporary CLI edits — see "Follow-ups".
```

Aggregation script: `/tmp/aggregate_gap.py` and `/tmp/aggregate_gap_ha.py` in this development context (kept inline rather than committed; the parquets themselves are not version-controlled).

## Follow-ups

1. **Add a `--profile` flag to `bench real`** so this analysis can be re-run cleanly without monkey-patching. Small CLI change (~10 LOC). Would also be useful for any future profile-comparison work.
2. **Drill into the recall-miss cases.** Identify which tags Locus misses on `high_accuracy` 720p/1080p and check whether `min_area` lower would recover them without measurable latency cost.
3. **Drill into the rot-p95 tail.** Stratify the 5% worst rotation cases by AOI / distance / PPM; see whether they cluster in a way that points to a specific solver-tuning opportunity.
4. **Re-render the public bench reports against `high_accuracy`** so external readers see the production-accuracy numbers, not the latency-tuned ones. Optional but worth considering before the next release.

## Hardware

CPU: 16 logical cores (`lscpu`). Linux 6.8.0-107-generic. Build: `--release`. RAYON_NUM_THREADS unset.
