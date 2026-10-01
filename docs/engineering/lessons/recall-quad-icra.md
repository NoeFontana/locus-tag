# Recall, quad extraction & ICRA — lessons

**Status:** ACTIVE — the 2026-07 issues are CLOSED (fixes shipped; `max_recall_adaptive` removed); the [2026-10-01 real-image recall](#2026-10-01-real-image-recall-liu4k-euroc-the-segmentation-model) findings are open.
**Last updated:** 2026-10-01
**Owning code:** quad extraction (`extract_quads_soa` / `extract_quads_soa_with_camera` and `pixel_count_descending_order` in the quad module), the AdaptivePpb extraction router (`quad.extraction_policy`), the decoder Hamming/bit-sample path, and the shipped profiles under `crates/locus-core/profiles/` (`standard.json`, `high_accuracy.json`).

## TL;DR
Extraction-mode choice, not funnel/threshold/refinement knobs, governs recall vs. pose-tail trade-offs: EdLines wins the clean-render pose tail but culls ~50% of dense small-tag candidates; ContourRdp wins crude-render recall but destroys the render-tag rotation tail. No single static `(extraction, refinement)` pair wins both regimes — the durable fix is PPB-based adaptive routing (`AdaptivePpb`, threshold 2.5), which lets `high_accuracy` recover ICRA recall (+14.2 pt) while staying byte-identical on render-tag. A separate quad-truncation bug (truncating candidates by pixel_count *before* geometric filtering) silently dropped tag-sized candidates behind large background blobs; the fix moved truncation caller-side, post-filter. The `max_recall_adaptive` profile validated this routing but was later removed in the config consolidation — its behavior lives on inside `high_accuracy`'s AdaptivePpb block.

## Lessons

**Quad truncation fix.** `pixel_count_descending_order` bundled two things: (1) sort component indices by `pixel_count` descending (load-bearing — lifts ICRA `standard` recall ~2.8 pp and stabilizes order-sensitive dedup on crude renders), and (2) truncate `component_stats` to `MAX_CANDIDATES = 1024` **before** per-component geometric filtering (the bug). Tag candidates have small pixel_count relative to background blobs (texture/shadow/noise), so pre-filter truncation kept giant blobs (which gates would reject anyway) and discarded real tags. Fix: drop *only* the pre-filter truncation; keep the desc ordering verbatim; truncate caller-side after `extract_single_quad` filters geometrically (Rayon `collect()` preserves input order, so truncation drops the smallest *survivors* — the intended 4K-recall behavior). Impact: distortion Brown–Conrady recall `0.8701→0.9354` (+6.5 pp), Kannala–Brandt `0.8088→0.8130`; render-tag byte-identical. Same fix on the `_with_camera` path.

**ICRA-forward high_accuracy diagnostic.** `high_accuracy` (EdLines + `refinement_mode=None`) collapsed on ICRA forward frames — mean recall `0.4631` vs `standard`'s `0.7236`, with 5 of 6 dense frames detecting zero tags. The `icra_forward_diagnostic.rs` harness (per-frame funnel / rejected-size / decode-Hamming attribution) root-caused it: the funnel was *not* the bottleneck (both profiles pass 100% through contrast), so threshold/sharpening flips could not help. EdLines under-produced by ~50% (`~110` candidates/frame vs `~206`), and its imprecise corners pushed decode Hamming into the 6–10 bucket (97/102 rejected on one frame). Cause was attributable to `extraction_mode=EdLines` on dense small-tag scenes. Resolution: adopt the `AdaptivePpb` router (threshold 2.5; low = ContourRdp+Erf, high = EdLines+None) — a Pareto win: ICRA forward recall `0.4631→0.6053` (+14.2 pt), RMSE `0.7535→0.5572`, render-tag / distortion / board snapshots byte-identical (well-resolved tags have PPB≫2.5 and stay on EdLines+None; far-field PPB<2.5 tags route to ContourRdp+Erf).

**max_recall_adaptive calibration (REMOVED profile).** A now-removed profile that first shipped the `AdaptivePpb` routing (threshold 2.5, sharpening on): ICRA forward recall `0.7380` (+27 pp vs high_accuracy, matching `standard`), every render-tag p99 rotation under budget. Its calibration sweep is the durable evidence: disabling sharpening cost `-19 pp` ICRA recall *and* worsened the render-tag p99 tail (3°+ outliers), so sharpening does real work on both extraction paths; lowering the threshold `2.5→1.5` had near-zero effect (candidates already sit above 2.5) with a small ICRA cost — the shipped config was a local Pareto optimum. Its residual render-tag RMSE gap (~0.6 px vs high_accuracy's ~0.2 px) is intrinsic to ContourRdp+Erf, which yields robust corners but no per-corner Fisher covariance prior. **The profile was dropped in the config consolidation** — its routing was absorbed into `high_accuracy`, so nothing was lost, but do not resurrect the standalone profile expecting new behavior.

**Hub regression snapshot.** A point-in-time perf/accuracy table (AMD EPYC-Milan, `--release`, single-threaded), now historical. Durable methodology notes: (a) `high_accuracy` is the accuracy *baseline* because its 4–6× tighter pose bounds catch pose-solver regressions that `standard`'s wide thresholds mask; small-tag recall is instead carried by the `standard`-profile robustness suite. (b) Latency scales ~linearly with pixel count; `high_accuracy` is ~2× faster than `standard` (EdLines cheaper than ContourRdp). (c) The robustness subsets are KPI watchlists, not regressions: `tag16h5` precision is codebook-bound (halving allowed Hamming is the dominant lever, remainder is dense-codebook ambiguity); `low_key` has a hard config recall ceiling (KPI for future contrast-robust threshold work); `raw_pipeline` is largely config-bound.

## 2026-10-01 — Real-image recall (Liu4K, EuRoC): the segmentation model

**Status:** ACTIVE — root causes established; fixes proposed, none shipped.
Numbers: [`benchmarking/liu4k_euroc_sota_20261001.md`](../benchmarking/liu4k_euroc_sota_20261001.md);
reproduce with `cargo xtask sota` (`xtask/README.md`).

On Liu4K (4K photos, markers blended into textured backgrounds) shipped `standard` finds
29 % of markers against aruco_nano's 66 % and OpenCV's 58 %; on EuRoC `cam_april`
(Kalibr AprilGrid) it decodes 20 % of the in-view tags. Four causes, all upstream of decode,
explain the recall gap. Together they took Locus to **67.2 % recall / F1 80.3 on Liu4K
(aruco_nano: 66.3 % / 79.7) and 90 % on EuRoC**.

**RC1 — The CCL threshold is a min/max midpoint, with no validity gate and no guard band.**
Segmentation does not read the binarized image: it marks `pixel < threshold_map`, and that
map is `(min + max) / 2` over a fixed 3×3-tile (24 px) neighbourhood. `threshold.min_range`
reaches only the `binarized` buffer, which nothing downstream consumes. Measured on the
detector's own `threshold_map` (debug telemetry), ≥ 95 % of Liu4K misses fail before a quad
exists. Small and mid-size markers **merge** with texture (81–89 % of misses): near an
edge the threshold is midpoint(black, brightest texture), so darker texture joins the
marker. Large markers **shatter** (88 % of ≥ 250 px misses): inside a border thicker than
about two tiles, min ≈ max and the threshold falls to the noise midpoint. A local-mean
rule (`I < μ − C`, as OpenCV and aruco_nano use) leaves a guard band on the bright side of
every edge, which is the mechanism PR #383 (`LocalMean`) targets. Do **not** re-try the
validity gate alone: PR #383 measured it at −0.18 pp, because merging happens in
*valid* tiles.

**RC2 — The quad pre-gates assume a filled blob.** `quad.min_fill_ratio`, `min_density` and
`max_elongation` were calibrated on filled dark components. A scale-free local-mean
threshold produces a hollow edge band (fill ≈ 4w/L: 0.09 for a 7 px band on a 300 px
marker). With r=7 / C=3 the gates cut ≥ 250 px recall to 0.2 %; with the gates off it is
56.6 %. PR #383's r=24 partly works because a wide window thickens the band.
8-connectivity also bridges edge bands to clutter through single diagonal contacts:
`Four` is worth +2.0 pp (65.2 → 67.2 %). Turning the gates off is not shippable as is,
though. On render-tag `tag16h5` precision falls from 83 % to 50 %, because the gates were
also doing false-positive control for a weak dictionary. That job belongs to a
dictionary-aware criterion (expected false accepts ∝ candidates × P(random code within
the Hamming budget)), not to geometric shape statistics.

**RC3 — The threshold offset C is in grey levels, not in units of noise.** Liu4K prefers
C ≈ 3 and EuRoC C ≈ 8; their median noise (`compute_image_noise_floor` estimator) is
0.87 vs 1.72, and both optima sit near 3.5–4.7 σ̂ₙ. `C = clamp(round(k·σ̂ₙ), 2, 20)` is flat
for k = 3…5 on Liu4K (67.2–67.3 %) and selects C ≈ 6–7 on EuRoC by itself (88.7 % at
k = 5). "Darker than the local mean by k noise standard deviations" is the statistically
meaningful form of the foreground test, and it replaces a per-dataset constant with one
physical parameter.

**RC4 — Tag border width is not part of the layout model.** Kalibr AprilGrids (EuRoC)
print tag36h11 with a **2-bit** black border (10 cells across). Locus samples the 8-cell
lattice; on a correctly localized EuRoC tag that reads 8 bit errors from any code, while
the 10-cell lattice reads the true id with 0 errors. Locus decodes these grids at all only
because the decoder retries homography scales 0.9 / 1.1: 0.9 approximates the exact 0.8,
with samples about 0.2 cells from cell boundaries. Recall therefore degrades with blur and
obliquity for a reason unrelated to image quality. The wrong lattice plus a 2-error
Hamming budget also yields **false ids**: `tests/test_euroc.py::test_euroc_no_false_positives`
(id 549 on a 36-tag grid) and `::test_euroc_detection_recall` (10.3 %) fail on `main`
whenever the dataset is present; CI skips them. OpenCV decodes the grid with
`markerBorderBits = 2`; aruco_nano hard-codes `markerSize + 2` and AprilTag 3 a 1-bit
border, so `cargo xtask sota` reports both as unsupported (references are never patched).
Fix direction: generate the sampling lattice from (data bits, border bits) and add a Kalibr
layout, rather than relying on the scale retry.

**Latency follows from the same pipeline order (L1–L3).** Serial, 1 thread, 4K: aruco_nano
52 ms/image; the Locus RC1–RC3 candidate 314 ms; `standard` 194 ms. Stage spans show the
candidate spending 240 of 327 ms in `quad_extraction`, **78 % of it sub-pixel refinement**:
about 776 quads are refined per image to decode 6, and the O(1) contrast funnel rejects
about 1 of them on textured photos. With refinement disabled it runs at 131 ms for −1.24 pp
recall. The fix direction is **decode-first ordering**: coarse-decode unrefined quads, refine
only decoded tags (emitted corners unchanged), and refine-and-retry Hamming near-misses.
Separately, the local-mean rule halves CCL cost (79 → 35 ms) by removing flat-region
speckle.

**Revisions to earlier statements on this page.** The "`low_key` hard config recall
ceiling" was the threshold model: the local-mean candidate takes `low_key` from 14 % to 98 %
(detection level). "The render-tag RMSE gap on ContourRdp+Erf routes is intrinsic" is
refuted: it is a photometric bias plus the 1-DOF ERF's seed-direction scatter (see
[rotation-tail 2026-10-01](rotation-tail-and-edge-refinement.md#2026-10-01-corner-bias-is-photometric-not-a-psf-floor)).

**Ship gates still open for the candidate** (r=7, C = k·σ̂ₙ, sharpening off, 4-conn, gates
off): ICRA forward −0.36 pp recall and +0.019 px corner RMSE (0.280 → 0.298); render-tag
`tag16h5` precision (RC2); serial latency (L1). Sharpening, and PR #384's shoot-limited
variant, are a crutch for RC1 on the ContourRdp route; on Liu4K, sharpening off is best
once the threshold model is fixed.

## Re-attempt / watch-outs
- Do **not** reintroduce a standalone `max_recall_adaptive` profile — its `AdaptivePpb` block already lives in `high_accuracy`. Any revival must justify why the shared router isn't sufficient.
- Do **not** try to fix the ICRA/high_accuracy gap by knob-flipping (sharpening, static ContourRdp, static Erf): every static extraction/refinement pair either destroys the render-tag pose tail (`0.56°→102°`) or blows mean RMSE 4× (`0.20→0.86`). "No static pair wins both regimes" still holds — PPB routing sidesteps it, it does not refute it.
- Keep the quad-truncation edge case covered: truncation must stay **caller-side, after geometric filtering** (small tags hide behind large-pixel-count background blobs). Never truncate `component_stats` pre-filter. Preserve the pixel_count-descending order (with deterministic index tie-break) — load-bearing for order-sensitive dedup on crude renders.
- ~~The render-tag RMSE gap on ContourRdp+Erf routes is intrinsic (no Fisher covariance prior).~~ Superseded 2026-10-01: the gap is a photometric (sRGB gamma) bias plus the 1-DOF ERF's seed-direction scatter. See the rotation-tail page.
- Test-infra trap: a relative `LOCUS_ICRA_DATASET_DIR` once resolved against the crate root and silently fell through to a 1-frame stub, producing a false "byte-identical" result. Relative paths now resolve against the workspace root and missing dataset dirs panic rather than skip. Verify ICRA snapshots against the real 50-frame dataset.

## Provenance
Distilled 2026-07-19 from (removed; see git history): `icra_forward_high_accuracy_diagnostic_20260426`, `quad_truncation_fix_20260426`, `max_recall_adaptive_calibration_20260426`, `hub_regression_20260423`.
