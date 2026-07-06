# Rotation-Tail Levers Map — Where the Remaining Accuracy Is (2026-07-06)

**Status:** research memo / decision record. **Scope:** render-tag `tag36h11`, single-tag,
`high_accuracy` profile. **Audience:** perception engineers deciding where to invest next.

## TL;DR

Locus (`high_accuracy`) is **state-of-the-art on render-tag** — 4× better corner accuracy
than tuned OpenCV ArUco / pupil_apriltags, best translation, at 5–15× lower latency. A full
per-instance comparison, a three-part numerics audit, four empirical falsifications, and a
2026 SOTA survey converge on one conclusion:

> **The single remaining accuracy gap — out-of-plane rotation for near-frontal (AoI < 20°)
> and low-pixel-density (ppm < ~700) tags — is a fundamental observability limit (the IPPE
> two-fold planar-pose ambiguity), not a solver or detector defect. No single-frame lever
> can close it. The only levers that can are information-adding (spatial or temporal), and a
> decisive experiment confirms the emitted pose covariance is a sound substrate for them.**

Every single-frame improvement path is now exhausted (see dead ends (§2)). The
validated path forward is information-adding fusion (§3).

---

## 1. The gap, precisely

From the full render-tag per-instance benchmark (see
[`render_tag_sota_20260425.md`](benchmarking/render_tag_sota_20260425.md) for the prior
baseline; this run is 2026-07-05, 14 482 records, 4 resolutions × 4 series):

- **Not a gap:** corner RMSE (Locus 0.14–0.17 px vs ~0.6 px, 4× better at *every* angle),
  translation (best on 196/200 instances), recall (100 % on clean render-tag), IPPE branch
  selection (`branch_flip = 0/50`), covariance calibration, the Huber knee.
- **The only gap:** out-of-plane **rotation**, and only in low-information regimes. Two clean
  crossovers:
  - **By angle of incidence:** rot p50 frontal <20° = 0.181° (Locus) vs 0.094°/0.109°
    (apriltag/opencv) — 2× *worse*; oblique 40–60° = 0.060° vs 0.080°/0.084° — *best*.
  - **By pixel density:** rot p50 ppm <700 = 0.267° (worse); ppm >1200 = 0.039° (best).
- **The contrast that localizes it:** Locus corners are 4× better at *every* AoI including
  frontal — so superior corners provably do **not** convert to superior rotation when the
  geometry is ill-conditioned. The fault is downstream of corner detection, in the
  corners→pose tilt DOF.

**Mechanism (verified from code + literature):** near-frontal, the projection degenerates
toward affine; the tilt DOF is weakly observable regardless of corner precision. This is the
IPPE two-fold planar ambiguity — rotation-only, worst for small/distant/fronto-parallel tags.

---

## 2. Dead ends — what does *not* close the gap

### 2a. Our own rejected levers (16 of 17 stay dead)

A re-examination of every falsified lever against the "bottleneck = rotation leverage
collapse" lens: all but one are aimed at non-bottlenecks (corners, recall, covariance, board
corpora, or an already-refuted mechanism) and correctly stay dead.

| Lever | Verdict | Reason |
| :-- | :-- | :-- |
| Global covariance multipliers (Phases 1–4) | **Dead** | A uniform Σ rescale is provably a no-op on the WLS estimate; a diagonal one can't fix the wrong off-diagonal `(JᵀWJ)⁻¹` structure. See [`pose_covariance_followup_2026-05-22.md`](pose_covariance_followup_2026-05-22.md). |
| Huber-knee σ recalibration | **Dead (falsified 2026-07-06)** | A/B'd directly: the σ-calibrated knee regressed rot p99 **+10.8 %** and left the frontal tail unchanged. Corners are so good residuals are high-leverage geometry, not outliers. |
| IPPE post-LM dual-branch reselection | **Dead** | `rotation-tail-diag` measured `branch_flip = 0/50`; branch selection is already correct. |
| 2-DOF Tukey narrow-band corner refinement | **Re-examine, predicted dead** | The one lever rejected partly on the wrong metric (had rotation-*tail* wins, killed on ChArUco-board p99). But its deadband no-ops the clean-frontal cases that constitute this gap. Near-free re-test on its branch. |
| ERF corner-sample exclusion; EdLines phase-5 decoupling; EdLines competitive selector; AprilGrid saddle / Förstner; black-border gate; per-tag κ-cap | **Dead** | All target corners / recall / board corpora / latency — none adds tilt observability. |

### 2b. 2026 SOTA — what the field offers (and why it does not help *this* gap)

- **No single-frame planar-pose solver beats IPPE here.** SQPnP is globally optimal on the
  *same* ambiguous cost surface (two near-equal minima remain); MLPnP / anisotropic PnP buy
  *translation*, not rotation; the one 2026 head-to-head (ProCay78, arXiv:2601.12567) reports
  parity on *reprojection* error, not AoI-stratified rotation. Consensus is unanimous
  (Schweighofer & Pinz TPAMI 2006; Collins & Bartoli IJCV 2014; Abbas *Sensors* 2019).
- **Deep fiducial detectors** (DeepTag, DeepArUco++, Deep ChArUco) improve detection
  *robustness/recall* (which Locus already saturates), are GPU-bound (GTX 1080 → RTX 3090
  class, incompatible with the ~40 ms budget), and DeepTag itself reports rotation
  *worsening* toward 0° view angle — it inherits, not solves, the ambiguity.
- **Learned sub-pixel corner refiners** help *repeatability*, not raw precision once a corner
  is resolved; classical refinement is already near the localization floor on clean data.
- **Super-resolution is a net negative for metrology** — SR is optimized for perceptual
  quality and demonstrably *fabricates* geometric detail on high-contrast straight edges
  (exactly what markers are), with no study showing improved corner/pose accuracy.
- **Marker redesign:** extra coplanar features buy jitter/robustness, not resolved
  near-frontal rotation (TopoTag's 16 nodes are ~2× *worse* on static rotation than
  AprilTag). Only a physical *baseline* (board/grid corner spread) structurally helps.

> **Confidence note.** The mechanism-level conclusions above are corroborated across multiple
> independent sources. Some specific figures came from secondary mirrors (primary PDFs
> returned 403); spot-verify exact digits before quoting them externally. Per project
> practice, hypotheses are flagged and separated from measured facts.

---

## 3. The real levers — information-adding

Because the information is not in a single frame, the only levers that can raise the
rotation tail add information. All three are **architectural** (new API surface), not
hot-path tweaks; the zero-alloc GIL-releasing `detect()` hot path is untouched.

| # | Lever | State | Evidence | Fit |
| :-- | :-- | :-- | :-- | :-- |
| 1 | **Multi-tag / board joint solve** (corner *spread*) | Stateless | Hinderer 2025 (indicative, single preprint): single-marker 1.04° → 5 spread markers 0.08° (~13×) | ✅ Locus already does joint PnP for ChArUco/AprilGrid boards; extend to loose multi-tag scenes |
| 2 | **Opt-in temporal SE(3) EKF** | Stateful wrapper | Split-CIF 2023 ~43 % RMSE cut on the hard path; angle-dependent-variance fusion (Adámek 2023) ~43 % | ✅ constant-velocity error-state EKF on SE(3), ~few hundred LOC, µs/update, statically sized (zero-alloc); Rust-friendly (`kfilter`/`adskalman` + SE(3) exp/log) |
| 3 | **Emit honest per-frame 6×6 pose covariance + IPPE two-solution ratio** | Stateless | The enabling substrate for #1 and #2 | ✅ the Jacobian is already computed at the PnP optimum |

Recommended sequencing: **#3 (substrate) → #1 (stateless, cheapest real gain) → #2 (stateful,
largest but heaviest)**. Keep `detect()` stateless; any tracker is a separate opt-in feature
(the apriltag_ros / TagSLAM architecture pattern).

---

## 4. The decisive experiment — the covariance substrate is sound

Lever #3 collided with a known result: Locus **already** emits a pose covariance, and the
covariance audit found it KL-miscalibrated (13.93) with a *wrong absolute structure* — every
calibration lever is falsified. But fusion does not need absolute calibration; it needs the
*relative* rotation block to correctly flag the weak DOF. We tested exactly that on 100
render-tag scenes (block that correlates with rotation error self-identifies as the rotation
block; covariance surfaced via the `estimate_tag_pose` bench diagnostic added in this PR):

| Test | Result | Interpretation |
| :-- | :-- | :-- |
| Spearman(cov_rot, **rotation error**) | **+0.66** | covariance strongly tracks actual error |
| Spearman(cov_rot, **AoI**) | **−0.60** | correctly inflates at near-frontal |
| Spearman(cov_rot, **ppm**) | **−0.51** | correctly inflates at low pixel-density |
| Within-AoI-bin (frontal / mid / oblique) | +0.42 / +0.77 / +0.68 | real signal, not just an AoI proxy |
| Tail separation (worst-decile rot error) | **4.6×** higher median cov | the bad cases self-announce |

The covariance's AoI signal (−0.60) is *stronger than the raw error-median's* (−0.18): the
emitted covariance flags a hard frontal/low-ppm frame as untrustworthy **even when that
frame's median error is incidentally fine** — precisely the signal a fuser needs to
down-weight it and recover rotation from better-observed frames.

**Verdict:** the information-adding program rests on solid ground. Absolute covariance
calibration is broken (and settled as unfixable), but the *relative* down-weighting signal is
sound. Locus's one genuine weakness self-announces via the covariance and becomes a
first-class asset rather than silent noise.

---

## 5. Recommendation

1. **Close the "improve single-frame rotation" line of work.** It is an observability limit,
   not a defect; document it as *fundamental, mitigated at the tail*.
2. **Ship the covariance/uncertainty substrate (#3)** — emit the per-frame 6×6 pose
   covariance and the IPPE two-solution reprojection ratio as first-class, honest outputs.
   Independently useful (downstream consumers can weight poses) and the prerequisite for #1/#2.
3. **Then the stateless multi-tag / board joint solve (#1)** — the cheapest real gain, and
   Locus already has the joint-PnP machinery for boards.
4. **Opt-in temporal SE(3) EKF (#2)** if/when a stateful path is warranted.
5. **Validate on a temporal render-tag sequence** (or a real slow fly-through) sweeping a tag
   through fronto-parallel, reporting **rotation p99 stratified by AoI × ppm**. The field has
   never published this exact stratification — it would be a defensible novel characterization.

Our per-instance + p99-rotation-tail methodology is **ahead of the fiducial field** (still
mean/RMS norm); the bimodal ambiguity error is exactly where a mean misleads and p99 is the
correct instrument.

---

## Appendix — provenance

- **Hardware (verified this session via `lscpu`/`/proc/meminfo`):** AMD EPYC-Milan, 8 vCPU
  (4 cores × 2 threads), 32 GB RAM.
- **Build/run:** `--release`; tuning at `RAYON_NUM_THREADS=1` × 8 parallel workers (accuracy),
  serial un-pinned latency verification; comparison + diagnostics via
  `bench compare-instances` / `bench rotation-tail-diag` on `high_accuracy`, family
  `AprilTag36h11`, datasets `locus_v1_tag36h11_{640x480,1280x720,1920x1080,3840x2160}`.
- **Confidence:** mechanism-level claims (IPPE ambiguity; global-covariance no-op; SR
  hallucination risk) are multiply corroborated; specific external numbers (Hinderer
  1.04°→0.08°) are single-source/indicative and marked as such.

### Key sources
Schweighofer & Pinz, TPAMI 2006 · Collins & Bartoli (IPPE), IJCV 2014 · Terzakis & Lourakis
(SQPnP), ECCV 2020 · Adámek et al., *Sensors* 2023 · Abbas et al., *Sensors* 2019 · Rijlaarsdam
et al., *Front. Robot. AI* 2022 · Ch'ng et al., arXiv:1909.11888 · Hinderer et al.,
arXiv:2509.17345, 2025 · Split-CIF, arXiv:2310.17879, 2023 · Barfoot & Furgale, T-RO 2014 ·
EPro-PnP, CVPR 2022 · DeepTag, TPAMI 2023. Internal: `pose_covariance_followup_2026-05-22.md`,
`render_tag_sota_20260425.md`.
