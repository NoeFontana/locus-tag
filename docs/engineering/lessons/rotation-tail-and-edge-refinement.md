# Pose rotation-error tail — lessons

**Status:** RESOLVED for single-frame — model-edge pose refinement shipped in v0.7.0 (`high_accuracy`); corner-level levers remain trade-bound. The [2026-10-01 photometric finding](#2026-10-01-corner-bias-is-photometric-not-a-psf-floor) is ACTIVE and qualifies the "~0.6 px edge-line floor" below.
**Last updated:** 2026-10-04
**Owning code:** `crates/locus-core/src/model_edge.rs` and the pose LM stack (`pose.rs` / `pose_weighted.rs`).

## TL;DR
The residual rotation-p99 tail on `high_accuracy` Accurate-mode pose (≈0.60° at 1080p) was first read as a *physical law* — the IPPE planar-pose ambiguity that any 4-corner PnP inherits. That belief was refuted: the tail is not physics but **corner localization**, and it concentrates in a handful of scenes (removing the worst 2 tags already hits the target). We then chased it at the corner level — GWLF/apriltag-edge refit, per-corner repair, per-tag switching (even an oracle), and internal fusion — and every one proved **trade-bound**: rotation improved only by adopting a common-mode corner profile that structurally regressed translation (e.g. GWLF: rot p99 0.600°→0.398° but trans p99 18.6→63.0 mm). The reframe that finally won was to stop reshaping the same 4 corners and **add independent information**: refine the 6-DoF pose against the decoded tag's ~40 internal bit-grid edges. That over-constrains rotation without disturbing corner-anchored translation, cutting rot p99 47–76% across resolutions (1080p 0.600°→0.249°, under OpenCV apriltag's 0.376°) at Locus's best-in-class translation. It shipped **on by default in `high_accuracy` in v0.7.0**.

## The saga (what was hypothesized, tried, and concluded)

| # | Hypothesis | Test | Verdict | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| 1 | The rot-p99 tail is a **physical law** — IPPE planar-pose ambiguity intrinsic to any 4-point PnP; not fixable single-frame. | Phase-0 diagnostic forensics on `locus_v1_tag36h11_1920x1080` (50 scenes), `high_accuracy` + Accurate. | **Refuted.** The tail is a small set of outlier scenes, not a distribution-wide physics floor; substituting good corners collapses the error. | Phase-0: rot p50 0.057° / p95 0.473° / p99 0.771°; tail carried by ~2 scenes (`scene_0008` 0.87°, `scene_0005` 0.66°). Corpus-relative reclassification found `corner_geometry_outlier` / `ppm_starved` / `sigma_miscalibration` modes — corner-localization and calibration, not an irreducible ambiguity. |
| 2 | Reframe: the tail is **corner localization** — a few EdLines Phase-1 arc-partition gross failures, correctable by better sub-pixel corners. | Sort shipped rot errors, drop worst-k tags; corner-error decomposition vs GT. | **Confirmed as diagnosis.** Removing the worst **2** tags lands at the apriltag target (p99 0.600°→0.384°); p95 (0.385°) is already competitive. The gap is a handful of gross corner failures (tag 563 / scene_0008: one corner 3.83 px off), not the other 47 tags. | `refine_variants_20260714.md` Result 2. |
| 3 | Corner-level refit (**GWLF** = apriltag-style gradient-weighted edge-line fit → intersect) closes rotation. | Route `high_accuracy` large markers through GWLF; measure rot/trans vs shipped EdLines. | **Trade-bound.** Rotation nearly reaches target but translation regresses 3.4× (49/50 tags worse). GWLF corners are ~3× worse in absolute position (mean RMSE 0.63 vs 0.21 px — the ~0.6 px edge-line floor on Blender PSF) but more *consistent*; consistency pulls rotation in, absolute error inflates translation. | rot p99 0.600°→0.398°, trans p99 18.6→63.0 mm. `refine_variants_20260714.md` Result 1. Not shipped. |
| 4 | Selectively **repair only failing corners** (EdgeLineGated: snap to GWLF only when they disagree past a gate). | Prototype `CornerRefinementMode::EdgeLineGated`, gate τ=1.5. | **Falsified.** Gating (0.70°) beats no-handling (0.77°) but *loses* to the shipped `outlier_drop` (0.60°) — dropping a catastrophic corner beats correcting it to the 0.6 px floor — and neutralises `outlier_drop`. Reverted; no dead knob shipped. | `refine_variants_20260714.md` Result 3. |
| 5 | **Per-corner repair / per-tag switch / internal fusion** can be steered to a Pareto win. | Swap-worst-only; greedy per-tag GWLF switch (oracle); per-corner median of {EdLines, GWLF, ContourRdp+Erf}. | **Falsified — trade is fundamental at the per-tag level.** One-corner swap is *worse* than all-four (you can't mix corner systems). Oracle per-tag switch reaches rot p99 0.377° only at trans p99 26.4 mm (+42%), and switch-worthy vs not overlaps fully (undetectable). Fusion drags to the offset cluster: trans p99 19.9→34.3 mm. | `refine_variants_20260714.md` Result 4. Corner-refinement level **exhausted**; only remaining win = *add independent information*. |
| 6 | **Phase C.5 post-decode re-refit** (re-fit the 4 outer edges, intersect, re-solve homography) improves accuracy. | Optional stage behind `decoder.post_decode_refinement`. | **Superseded.** Real but marginal: means improve uniformly (ICRA RMSE −5.1%, render-tag −0.3…−0.7%) but **p99 rotation stays within sub-1% noise** — it does not close the tail. Also learned: writing the optimistic CRB covariance regressed render-tag p99 rotation 3–5%, so the covariance column is deliberately preserved. Same edge-intersection floor as GWLF; only touches the 4 outer edges. | `post_decode_refinement_20260426`. |
| 7 | **Add information: model-edge pose refinement** — refine 6-DoF pose against the decoded tag's ~40 *internal* bit-grid edges, not just 4 corners. | Opt-in Accurate-mode stage (`model_edge.rs`): measure→fit Nielsen-LM against 50%-intensity edge crossings under Huber δ=0.5 px + 7 samples/boundary; re-anchor translation to the 4 trusted corners; no-worse + χ² gates. | **WON — shipped v0.7.0.** Over-constrains rotation from the interior without disturbing corner-anchored translation. Rot p99 −47…−76% at every resolution; reprojection RMSE falls everywhere; recall/precision 100%/100%; corner RMSE unchanged by construction. | `model_edge_refinement_20260715.md`. 1080p rot p99 0.600°→**0.249°** (p95 0.385°→0.180°), under OpenCV apriltag's 0.376°, at trans p99 ~20 mm vs apriltag ~55 mm. Larger win on degraded imagery (p99 −56…−84%). On by default in `high_accuracy`; left off in `standard` (no χ² gate there → tail would drift). |

## 2026-10-01 — Corner bias is photometric, not a PSF floor

**Status:** ACTIVE — mechanism established on controlled renders; no fix shipped.
Reproduce: `PYTHONPATH=. uv run --group bench python tools/bench/photometric_corner_bias.py`.

On clean render-tag images `standard` (ContourRdp + ERF) corners sit **0.59 px inward**
(radial, mean) of the GT corners, and ERF moves the integer contour seed *further* inward
(−0.50 → −0.59 px). The image itself explains it. Across 1800 GT-edge profiles the 50 %
intensity crossing lies **0.40 px inside** the GT edge, and the normalized intensity *at*
the GT edge is **0.74 = sRGB(0.5)**. Blender integrates coverage in linear light and writes
sRGB-encoded PNGs, so a blurred edge's midpoint *in encoded intensity* sits on the dark
side. Every refiner that fits a symmetric step or gradient peak to encoded intensities
inherits this.

A controlled experiment with analytically exact GT (area-coverage render, blur in linear
light, optional sRGB encoding, `photometric_corner_bias.py`) separates three defects
(radial bias / RMSE in px; 40 and 120 px tags agree):

| Photometry, blur σ | `standard` (ContourRdp + ERF) | `high_accuracy` (EdLines) |
| :--- | :--- | :--- |
| linear, 0.5 | −0.02 / 0.46–0.51 | **+0.52…+0.54** / 0.54–0.70 |
| linear, 1.0 | −0.03 / 0.46–0.52 | **+0.58…+0.61** / 0.62–0.79 |
| sRGB, 0.5 | −0.30 / 0.50–0.67 | +0.18…+0.25 / 0.23–0.38 |
| sRGB, 1.0 | −0.66 / 0.82–0.91 | −0.11…−0.16 / 0.21–0.31 |
| sRGB, 1.5 | −0.97…−1.02 / 1.18–1.22 | −0.34…−0.51 / 0.46–0.54 |

1. **Gamma bias.** Both refiners assume a linear photometric response. On gamma-encoded
   input (sRGB renders, phones, webcams, JPEG) they move edges toward the dark side, in
   proportion to blur. The prediction is quantitative: for this render's 8–92 % intensity
   range the encoded midpoint corresponds to 34 % linear coverage, i.e. an offset of
   Φ⁻¹(0.34)·σ_eff per edge, giving −0.33 / −0.60 / −0.88 px radial at σ = 0.5 / 1.0 / 1.5
   against −0.31 / −0.65 / −0.95 measured.
2. **EdLines carries an intrinsic ~+0.5 px outward bias in linear light**, which the sRGB
   bias cancels near σ ≈ 1 — the render-tag regime. See
   [EdLines 2026-10-01](edlines-segmentation.md#2026-10-01-intrinsic-outward-bias-masked-by-srgb).
3. **ERF is unbiased in linear light but scatters 0.46–0.80 px RMSE on clean renders.**
   The quad-path fit is 1-DOF: it re-fits each edge's offset and keeps the edge *direction*
   of the integer RDP vertices, so seed quantization passes straight into the corner.

**Real data (EuRoC, Kalibr AprilGrid).** Against the grid's X-junctions, which are unbiased
by symmetry, ERF corners sit −0.39 px inward and EdLines corners +0.17 px outward. That is
the signature the sRGB rows predict at σ ≈ 0.5, so the sensor output is likely
gamma-like (an earlier "printed-target shrink" reading was retracted after the synthetic
control). Replacing only the refiner on Locus quads with `cv2.cornerSubPix` cuts
leave-one-tag-out corner error 0.695 → 0.248 px. That gain is junction symmetry, not a
general win: on plain L-corners `cornerSubPix` is itself biased inward by 0.14–0.41 px as
blur grows.

**What this changes in the saga above (hypotheses, not conclusions).** Row 3 attributes
GWLF's 0.63 px absolute error to "the ~0.6 px edge-line floor on Blender PSF". GWLF is a
gradient-weighted line fit on encoded intensities, so the gamma bias above is a
competing explanation of that floor. If it holds, the corner-level frontier ("consistency
vs. absolute error") was partly measured against a biased reference. Likewise the
AprilGrid saddle-refinement negative (saddles ~1 px "away from the tag interior") is
consistent with line-fit corners being biased inward, rather than with saddle leakage.
**Re-attempt condition now met for GWLF-style corners:** evaluate them through a
linearized sampler (photometric response applied as an f32 LUT inside the sampler, never by
re-quantizing the 8-bit image, which cost 10–12 pp recall in a probe), with EdLines'
outward bias fixed in the same change. Otherwise render-tag regresses when the
cancellation breaks.

## 2026-10-04 — The marker calibrates its own photometric inset

**Status:** SHIPPED in `decoder.corner_subpix` (`standard`, `grid`). Owning code:
`crates/locus-core/src/marker_inset.rs`. Supersedes the "no fix shipped" status of the
2026-10-01 section above for every decoded marker with cells ≥ 3.3 px on a rectified image.

**Trigger.** Fusing the junction corner with whole-edge line corners (same PR) halved the debiased
corner scatter but *raised* single-tag rotation p99 on render-tag (high_iso 0.44° → 0.62°,
tag16h5 0.57° → 0.80°). The tail was a few small (39–78 px), oblique (52–58° AoI) tags.

**Root cause, step by step.**
1. A pure 0.6 px uniform inset added to the ground-truth corners reproduces the tail on the
   same scenes (0.55–0.77°), plus 80–170 mm of translation error. The fused corners sat almost
   exactly on that "pure inset" prediction; `main`'s noisier corners partly cancelled it by
   chance.
2. Along an oblique edge the inset is not even constant: blur, and hence the tone-curve shift,
   changes with depth (0.71 px at one end of an edge, 0.35 px at the other). A whole-edge line
   therefore tilts, and its scatter-based covariance cannot see a systematic tilt.
3. The inset is shared by every gradient detector (OpenCV and aruco_nano: −0.6 to −0.8 px radial
   on render-tag and ChArUco), and it depends on the dataset's tone curve. A fixed sRGB
   linearisation fixes render-tag (−0.59 → −0.04) but breaks ICRA (0.00 → +0.26) and low_key
   (+0.29 → +1.14).

**Falsified: a pose-level inset nuisance.** Adding `δ·d_k` (exact edge-offset direction) to the
single-tag pose LM, marginalised by variable projection with a Gaussian prior, cannot work. The
Cramér–Rao bound on `δ` is 4–23 px on the tail tags at σ = 0.15 px, because an inset is
first-order depth plus pose. Measured: σ_δ ≤ 1 px is a no-op and σ_δ = 10 px is catastrophic.
Boards can identify it (#432) because their layout fixes the marker centres.

**What works: the decoded marker measures its own inset.** Every bit boundary of the decoded
pattern shifts toward its dark side by the same `δ`, whatever the tone curve and blur, and the
layout says where each boundary is. Per marker, a robust fit of all boundary offsets separates
`δ` (by polarity), an interior layout scale `s`, and the detected corners' inset `ε` per side
(pinned by the outer boundaries); the corners move outward by `ε`. On the Blender renders the
fit gives δ ≈ 0.41 px, `s − 1` = 0.0000, and an ε equal to the ground-truth inset of each
corner estimator.

**Two traps, and how the model handles them.**
- *ICRA's artwork.* Its interior dark features are about 4 % of a cell thin while the outer
  edge is exact (border ring 0.960 cells thick, measured with ground-truth corners). Within one
  marker that is indistinguishable from a photometric shift, and it cannot happen on a printed
  marker, where ink spread or erosion moves the outer edge too. Accepted as a dataset artefact
  (decision 2026-10-04): ICRA corners move outward by ≈ 0.2 px.
- *Undeclared lens distortion.* The default build cannot represent distortion, so distorted
  images run the pinhole path. A homography maps lines to lines, so distortion shows as a bow
  along each boundary. The model carries four bow parameters, kept only when an F-test (99 %)
  shows they are significant. On rectified images they then cost nothing; with them, a
  barrel-distorted synthetic marker is recovered to < 0.15 px (1.01 px without). Declared
  distortion skips the calibration and the edge-line fusion entirely.

**Measured** (regression snapshots, render-tag 1080p, the suites' built-in pose path; with the
edge-line band capped at 4 px, which on large markers had reached the bit edges):

| Set | Corner mean RMSE (px) | Rotation p99 (°) | Translation p99 (mm) |
| :-- | :-- | :-- | :-- |
| high_iso | 0.711 → 0.062 | 0.438 → 0.201 | 66 → 21 |
| tag16h5 | 0.704 → 0.055 | 0.563 → 0.379 | 128 → 17 |
| low_key (n = 6) | 0.022 → 0.005 | 0.564 → 0.153 | 4.9 → 1.0 |
| raw_pipeline | 0.497 → 0.303 | unchanged (one 2.6° frame) | 41 → 21 |

Boards: ChArUco rotation p50 0.0094° → 0.0022°; ChArUco refiner rotation mean 0.32° → 0.10°.
The board-level inset nuisance (#432) is still needed for corners the calibration skips:
without it, board translation p95 rises 0.65 → 11 mm (AprilGrid).

**Re-attempt / revisit only if:** a real-camera dataset shows interior dark features scaling
differently from the outline (the ICRA pattern). That would need a frame-level split of `δ`
(constant in px) from artwork erosion (proportional to the cell), regressing across the tags of
a frame.

## Durable conclusion
Every "reshape the same 4 corners" lever is trade-bound because a corner-refinement method has a fixed error *profile*: you can trade Locus's low-absolute-error/high-variance EdLines corners for apriltag/GWLF's high-absolute-error/low-variance corners, buying rotation consistency at the cost of translation bias — but you cannot escape the frontier, because rotation and translation are read off the *same four observations*. The only way to improve both at once is to add observations the corners don't carry. Model-edge refinement does exactly that: the decoded interior pattern supplies ~40 independent, interior-distributed edge constraints that pin orientation an order of magnitude better than 4 corners, while translation stays anchored to the trusted corners. Adding information beat reshaping information.

## Still open / re-attempt only if
- **Corner-level refinement remains exhausted / trade-bound** — do not re-attempt GWLF replacement, per-corner gated repair, per-tag switching, or internal corner fusion without new evidence; all are empirically falsified (see `MEMORY` anti-patterns and `refine_variants_20260714.md`).
- **Translation carries a bounded, gated trade** in the model-edge stage — concentrated in t p95 at high resolution (1080p +1.1 mm, 2160p +4.9 mm) where a large rotation correction pulls the corner-anchored translation. Gated and opt-in; revisit only if a real-camera regression appears.
- **Multi-tag and temporal levers are not yet pursued** — board joint-solve and multi-frame/SE(3) fusion are the next "add-info" tier above single-tag corner refinement; open for the residual t-p95 trade and for `standard`'s gross-outlier tail.
- **`standard` tail is unaddressed** — enabling model-edge there needs the full package (edge refinement + χ² consistency gate + outlier-drop) plus its own recall/precision benchmark (ICRA included); deferred past v0.7.0. `standard`'s real gap is the absent pose gates, not corner refinement.
- **Absolute-unit constants** (Huber δ, `NielsenConfig::POSE` grad_tol/damping_floor) are tuned on synthetic Blender data — make them scale-relative first if real-camera data ever shows a regression.

## Live references (not superseded)
- Corner-level variant study: [`benchmarking/refine_variants_20260714.md`](../benchmarking/refine_variants_20260714.md)
- Shipped model-edge refinement: [`benchmarking/model_edge_refinement_20260715.md`](../benchmarking/model_edge_refinement_20260715.md)

## Provenance
Distilled 2026-07-19 from (removed; see git history): `rotation_tail_diagnostic_phase0_20260502`, `rotation_tail_diagnostic_phase0_20260503`, `post_decode_refinement_20260426`. Related: `MEMORY` anti-patterns on rotation-tail levers.
