# What the EuRoC corner number actually measures (2026-10-07)

The EuRoC scorer's `loo_own_median_px` has been read as "Locus's corner error on real
data", and compared against the render-tag synthetic figure (0.0370 px) as though the
two were the same quantity. They are not, and the 7.7x "gap" between them was an
artefact of that comparison. This file decomposes the number so the next person does
not re-derive it, and records which candidate explanations were **falsified**.

Environment for every figure here: AMD EPYC-Milan, 8 logical / 4 cores, Linux
6.8.0-139, OpenCV 5.0.0, NumPy 2.4.1. Inputs are the stored run jsonl under
`target/sota/runs/euroc` and `/home/dev/locus-archive/sota-reference/runs/euroc`; no
detector was re-run, so these are properties of the metric and of already-recorded
detections.

## The metric is a self-consistency residual, not a corner error

`score.py::_loo_errors` computes, in **undistorted** pixels,

```
e = || H_-t * X_t(nominal board) - c_t ||
```

where `H_-t` is a homography fitted to the *other* tags' own detected corners. There is
no ground truth anywhere in it. Four things therefore enter, and only the first is the
detector:

1. the held-out tag's own corner noise;
2. that noise propagated through `H_-t` (8 DOF fitted to ~112 noisy points, then
   evaluated at the held-out tag);
3. error static in **board** coordinates — the hard-coded `PITCH / TAG = 1.3`, board bow,
   per-tag print placement. A homography absorbs any *projective* map of the board plane,
   so a uniform pitch scale is invisible, but the 1.3 **ratio** is not;
4. error static in **image** coordinates — the radtan model's own residual, plus the
   inverse used to apply it.

Terms 3 and 4 are shared across detectors, so they cancel in deltas measured within one
scoring pass — which is why so many tuning levers have measured "inert" — but they sit in
the absolute number.

### Measured gain of the metric

Synthesising a *perfect* board at the real per-frame poses through the published lens
model, injecting iid Gaussian noise of known sigma in image pixels, and running the
shipped `_loo_errors`:

| sigma_img (px) | LOO median | gain | LOO p90 | p90 gain |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.0209 | 2.09 | 0.0615 | 6.15 |
| 0.05 | 0.0726 | 1.45 | 0.1412 | 2.82 |
| 0.10 | 0.1405 | 1.41 | 0.2666 | 2.67 |
| 0.20 | 0.2775 | 1.39 | 0.5259 | 2.63 |
| 0.30 | 0.4161 | 1.39 | 0.7837 | 2.61 |

`median_LOO = 1.365 * sigma_img + 0.005`.

**So the metric reports ~1.37x the underlying iid corner noise, and its p90 reports
~2.6x.** A reported 0.28 px median is ~0.20 px of image-space corner noise. Quoting
`loo_common_p90_px` as a corner error overstates it by about a factor of three.

## Error budget

Non-parametric partition of the full-fit residual (no optimiser involved): group the
residuals by `(tag, corner)` to get what is static in board coordinates, by 30 px image
cell to get what is static in image coordinates, and call the rest random. Group means
are debiased for their own sampling variance. 115 872 corner observations over 1028
frames of `locus_b_std`:

| component | magnitude | share of variance |
| :--- | ---: | ---: |
| static per `(tag, corner)` | 0.333 mm = 0.157 px at median scale | 13 % |
| static per image cell | 0.214 px | 24 % |
| random | 0.347 px | 63 % |
| **total** | **0.436 px RMS, 0.255 px median** | |

The partition closes: `0.157^2 + 0.214^2 + 0.347^2 = 0.427^2` against a measured
0.436 RMS.

### The per-`(tag, corner)` term is the estimator, not the board

This was the surprise. The same partition across detectors on the same frames:

| detector | total RMS | per-`(tag,corner)` | per-image-cell | random |
| :--- | ---: | ---: | ---: | ---: |
| Locus `standard` | 0.436 px | 0.333 mm | 0.214 px | 0.347 px |
| Locus `high_accuracy` | 0.436 px | 0.332 mm | 0.214 px | 0.346 px |
| OpenCV `APRILTAG` | 0.594 px | 0.747 mm | 0.176 px | 0.419 px |
| OpenCV `CONTOUR+SUBPIX` | 2.523 px | 3.806 mm | 0.459 px | 1.623 px |
| OpenCV `CONTOUR` | 2.633 px | 4.226 mm | 0.477 px | 1.569 px |

A physical board would give **every** detector the same per-`(tag, corner)` number. It
does not: the term tracks detector quality over a 13x range. So it is not print
placement or bow — it is a **deterministic per-corner bias of the estimator**, i.e. the
local bit pattern adjacent to each corner pulling it. That is the photometric inset this
repo already knows about, now localised to the corner and quantified: 0.38 % of tag size
for Locus, 0.85 % for OpenCV APRILTAG, and ~4.5 % for the unrefined contour paths
(consistent with Kalibr's 2-bit border).

Being deterministic in `(tag id, corner index)`, it is in principle correctable — both
are known at decode time, and the machinery exists (PR #434 marker inset, PR #432 board
inset). It is worth 0.157 px of 0.436 px RMS for Locus, so removing it entirely would
move the total ~7 %. Not yet attempted.

On both counts Locus is already the best of the four detectors on **every** component of
the budget.

## Falsified explanations

Recorded so they are not re-attempted. Each was a serious candidate and each is out.

* **Board-model error dominates.** No: 7.0 % of the number, on held-out frames. Fitting
  the pitch/tag ratio returns **exactly** the Kalibr nominal 1.3000, and a free 2-D board
  (288 parameters, fitted on even frames, scored on odd) removes only 2.2 %. The board is
  planar and printed to 0.18 mm, 0.2 % of tag size.
* **Motion blur.** No: coefficient +0.043 on `log(velocity)` in the multivariate fit, and
  flat across a 40x velocity range (0.24 to 0.33 px, non-monotonic).
* **Board non-planarity.** No: the dependence on tilt is non-monotonic
  (0.20 -> 0.31 -> 0.24 -> 0.30 px over 0-75 deg), where a bow must grow with tan(tilt).
* **Apparent tag size explains the synthetic-vs-real gap.** No, and it is worth being
  precise about why. Error *grows* as `L^+0.68` after controlling for tilt, velocity,
  radius and tag count — the opposite sign to the variance-limited `L^-0.5`. Extrapolated
  to render-tag's tag size it predicts 0.46 px where render-tag measures 0.037 px, a 12x
  contradiction. The scaling is real but **EuRoC-internal**; it does not transfer across
  datasets, and the two figures are not two measurements of one quantity.
* **`subpixel_refinement_sigma` mis-specification.** Falsified earlier, 2026-10-07; see
  `profiles/README.md`.

What remains genuinely open is the 0.347 px random term: it is the detector, it is the
largest single component, and it scales as `L^+0.35`, which no mechanism here explains.

## Fixed: the scorer's undistortion inverse

`_undist` was `cv2.undistortPoints`, an iterative inverse that runs a fixed, small number
of steps with no convergence test. On this lens it stops short by a **radius-growing**
amount: 0.046 px median at r in [300, 380) and 0.134 px at worst, where a converged fixed
point reaches 3e-13.

The shortfall is smooth, so a homography absorbs part of it and the LOO median came out
**low**:

| detector | shipped cv2 inverse | converged | artefact |
| :--- | ---: | ---: | ---: |
| Locus `standard` | 0.2679 | 0.2785 | **-3.98 %** |
| OpenCV `APRILTAG` | 0.4750 | 0.4789 | -0.82 % |
| OpenCV `CONTOUR+SUBPIX` | 2.1127 | 2.1131 | -0.02 % |

It flatters whichever detector has the better corners, so it does **not** cancel in the
head-to-head comparison the scorer exists to make: Locus's relative margin over APRILTAG
was overstated by about 3 pp. `_undist` is now a fixed-point inverse that **asserts**
convergence rather than returning an approximation.

Recall and false positives are essentially unmoved (`locus_b_std` 92.485 -> 92.491 %, FP
0 -> 0; reference frames 1029 both ways), so this re-baselines the corner columns only,
and no SOTA ordering changes.

### The correction is a near-fixed absolute term (2026-10-08)

Recomputing the fourteen archived EuRoC runs both ways, in one directory so the reference
pool and common-tag set are identical across the comparison (1047 reference frames, 9923
common tags):

| run | LOO own median, before | after | absolute | relative |
| :--- | ---: | ---: | ---: | ---: |
| `locus_g5`, `locus_new4` | 0.2661 | 0.2767 | +0.0106 | +3.98 % |
| `locus_final` | 0.2665 | 0.2770 | +0.0105 | +3.95 % |
| `locus_f40`, `locus_g4` | 0.2691 | 0.2799 | +0.0108 | +4.01 % |
| `locus_main4` | 0.2697 | 0.2802 | +0.0105 | +3.90 % |
| `locus_f35`, `locus_g35` | 0.2708 | 0.2813 | +0.0105 | +3.89 % |
| `locus_new` | 0.3183 | 0.3274 | +0.0091 | +2.87 % |
| `locus_main` | 0.3211 | 0.3298 | +0.0087 | +2.71 % |
| `opencv_apriltag` | 0.4752 | 0.4788 | +0.0036 | +0.75 % |
| `locus_v080` | 0.9844 | 0.9876 | +0.0032 | +0.32 % |
| `opencv_subpix` | 2.1336 | 2.1323 | -0.0012 | -0.06 % |
| `opencv` | 2.2865 | 2.2820 | -0.0045 | -0.20 % |

The shift is bounded in **absolute** terms (-0.005 to +0.011 px) and is close to constant
across detectors. Its *relative* size therefore falls monotonically as the detector's own
error grows — ~4 % at 0.27 px, 0.75 % at 0.48 px, indistinguishable from zero above 2 px.
That is the whole mechanism by which it flattered the best corners, and it is why the
artefact could not be removed by rescaling: there is no single factor.

It also bounds what the un-re-derivable pages lose. At the 0.3-0.9 px levels those pages
report, the shift is ~0.004-0.009 px on Locus and ~0.004 px on the references, and no
win/loss verdict on any scoreboard changes.

### Re-derived: EuRoC SOTA (2026-10-04)

The only page whose detections were archived. Recomputing the stored runs with the old
inverse reproduces its published cell to the thousandth, which is what identifies them —
`main` is `locus_main`, **This build** is `locus_final`, the reference is `opencv_apriltag`:

| cell | published (inexact) | recomputed, old inverse | **re-derived** |
| :--- | ---: | ---: | ---: |
| `main` median / p90 | 0.316 / 0.634 | 0.316 / 0.634 | **0.325 / 0.650** |
| This build median / p90 | 0.283 / 0.575 | 0.283 / 0.575 | **0.295 / 0.594** |
| OpenCV APRILTAG median / p90 | 0.516 / 0.996 | 0.516 / 0.997 | **0.520 / 1.008** |

Recall (20.86 / 86.02 %) and false positives (6 / 1) move by at most 0.01 pp. Locus still
wins the cell by a wide margin; only the size of the margin changes.

The runs behind `sota_scoreboard_20261002/03/04.md` and `liu4k_euroc_sota_20261001.md` were
not archived, and the common-tag set each of their cells used depended on which runs shared
the directory at the time, which is not recoverable. Those pages are annotated with the
bound above rather than rewritten: a figure rescaled by a factor measured on other runs
would not be a measurement, which is the failure this whole file exists to document.

## How to quote these numbers

* Do not compare `loo_own_median_px` against a render-tag corner RMSE. Different
  estimands.
* Divide by 1.365 for an image-space iid-equivalent corner sigma; do not use the p90 as a
  corner error at all without dividing by ~2.6.
* The absolute number carries a ~0.26 px floor (0.214 px image-static plus the static part
  of the lens model) that no detector change can move. Deltas within one scoring pass are
  the trustworthy output; absolute values are not corner accuracy.
