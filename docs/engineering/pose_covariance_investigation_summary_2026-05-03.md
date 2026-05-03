# Pose-covariance miscalibration — investigation summary (2026-05-03)

V1's audit (`pose_covariance_calibration_audit_2026-05-03.md`) reported
`KL(empirical ‖ χ²(6)) = 13.93`, mean Mahalanobis `d² = 714.7` vs. ideal
`6.0` on `locus_v1_tag36h11_1920x1080`. This memo records the four
follow-up experiments that closed the question.

**Bottom line:** the d² distribution observed on this synthetic corpus is
the correct CRB output for biased corner inputs combined with the
intrinsic anisotropy of the planar-PnP Cramér-Rao bound. There is no
defensible fix from synthetic data alone. Re-open the question against
a real-camera audit corpus (Track C, task #13).

## §1 Verified hardware (per `constraints.md` §6)

AMD EPYC-Milan KVM, 8 logical CPUs, AVX2/FMA/F16C (no AVX-512).
`--release` build with `bench-internals`, `RAYON_NUM_THREADS=8`,
CPython 3.14.3, rustc 1.92.0. Same box as V1.

## §2 Four experiments — what was tried, what was found

### CF1 — α_max sweep (V1's recommended fix #2)

Sweep `pose.tikhonov_alpha_max ∈ {0.5, 1.0}` against V1 audit + render-tag
regression. Mean RMSE preserved on every value, but 1080p p99 rotation
regresses linearly:

| `α_max` | V1 KL | 1080p mean RMSE | 1080p p99 rot |
|---:|---:|---:|---:|
| 0.25 (baseline) | 13.93 | 0.214 | 0.8613° |
| 0.5 | 11.44 | 0.214 (flat) | 1.2357° (**+43.5 %**) |
| 1.0 |  8.99 | 0.214 (flat) | 1.7465° (**+103 %**) |

Slope: ~150 °/unit `α_max` on 1080p p99 rotation, ~5 nat KL drop per unit.
Linear extrapolation: `α_max ≥ 3.0` for `KL ≤ 0.5`, predicted to push
1080p p99 rotation to ≈ 5° — 6.3× regression. Mechanism: 1080p p99 is
dominated by the corner-outlier scene_0008; raising `α_max` evens out
per-corner weights, so the outlier corner gets *more* relative weight.
**CF1 closed: gate-2 (p99 rot) closes before KL approaches spec.**

### Path B — runtime-proxy correlation

Extended `pose_cov_audit.py` to capture `final_per_corner_d2` and
`final_per_corner_irls_weight` (already exposed by the bench PyO3 hook;
audit just wasn't reading them). Spearman correlations against `d²_pose`
on 50 scenes, 8 candidate proxies:

| Proxy | Spearman ρ | Notes |
|---|---:|---|
| `s²` (post-fit MSE) | 0.46 | Doesn't see the bias |
| `min_irls` (inverted) | −0.87 | **Degenerate** — 49/50 scenes are 1.0 |
| `max_corner_d²` | 0.37 | Only fires for scene_0008 |
| `iters` | −0.87 | **Degenerate** — 49/50 are 1 |
| `log\|Σ_pose\|` | 0.69 | Continuous, but measures *scene difficulty* |
| `cond(Σ_pose)` | 0.66 | Same — geometric leverage |

**49 of 50 scenes converge with `min_irls = 1.0`, `max_corner_d² < 1`,
in 1 LM iteration.** The Huber kernel never fires; per-corner
statistics show no anomaly; yet `d²_pose ∈ [0.33, 12 405]`. The LM
absorbs correlated 4-corner bias into a wrong pose with small post-fit
residuals. **No LM-side proxy distinguishes calibrated from
miscalibrated typical scenes.**

### Path B-2 — counterfactual with GT corners

Project canonical 3D tag corners through GT pose + intrinsics; feed
those GT 2D corners back into the same weighted LM (Σ_corner recomputed
at GT pixel locations). Result:

|   | mean | median | p99 |
|---|---:|---:|---:|
| `d²_det` (Locus corners) | 714.71 | 62.76 | 9 435.32 |
| **`d²_gt` (GT corners)** | **0.0000** | **0.0000** | **0.0000** |
| Ideal χ²(6) | 6.00 | 5.35 | 16.81 |

LM converges back to GT pose to floating-point precision on every scene.
**100 % of the V1 miscalibration is upstream of the LM** — not in
intrinsics, not in the model spec, not in numerical conditioning. The
regularization-based fixes (CF1, CF2, CF3, Path β) operate on the
already-correct Hessian; they cannot move the calibration.

### Path α — per-corner residual direction analysis

Decompose 200 per-corner residuals into image-axis, tag-radial, and
tag-edge-perpendicular bases. Initial finding "82 % variance along
image-x" was an artefact of `scene_0008` alone (its 4 corners
contribute residuals up to 3.83 px).

**Excluding scene_0008 (49 scenes, 196 corners):**

|   | Δx (px) | Δy (px) |
|---|---:|---:|
| mean | −0.005 | +0.007 |
| std | **0.147** | **0.136** |

σ_x / σ_y = 1.08 — **isotropic**. Per-corner-index breakdown shows all
4 canonical corners with σ ∈ [0.12, 0.18] in both axes, no structural
pattern. The typical corner residual is noise-floor-like.

## §3 Order-of-magnitude tie-out

V1 §3: principal-axis 1σ of `Σ_pose` ≈ 0.017 px-equivalent.
A 0.14 px corner residual projects onto the principal pose direction
with leverage ~0.07 (planar-PnP geometry):

    d²_principal ≈ (0.07 / 0.017)² ≈ 17

times scene-difficulty multiplier (small / grazing tags amplify) gives
the observed **median d² = 62** and **p99 d² = 9 435**. The numbers
tie out without invoking *any* miscalibration in the noise model.
The d² distribution is the correct CRB output given an irreducible
~0.14 px corner-localisation floor and the intrinsic anisotropy of
`(JᵀWJ)⁻¹` for planar 4-corner PnP.

## §4 Why every "obvious" fix fails on this dataset

| Approach | Why it doesn't work |
|---|---|
| **Inflate `Σ_corner` (CF1 / σ_n bump)** | Diagonal of `Σ_pose` is already too loose (V1 per-axis ratios 0.026–0.27). Inflating `Σ_corner` makes it looser; mean d² drops because principal axes are bounded but per-axis miscal worsens. Plus slows LM convergence ⇒ p99 rot regresses (CF1 evidence). |
| **Tikhonov on returned Hessian (CF2)** | The Hessian is correctly computed for the corners it received (Path B-2 `d²_gt = 0` proves this). Adding a regulariser is a Bayesian prior with no physical motivation. |
| **Anisotropic `Σ_pose` floor (CF3)** | A fixed inflation matrix calibrated to *this* synthetic corpus. Won't transfer — Blender PSF / anti-aliasing don't model real cameras. |
| **Population `B = E[r rᵀ]` (Path β)** | Per-corpus calibration. Honest *form* (KL → 0 by construction) but dishonest *spirit*: gives downstream consumers false confidence on real cameras with different bias signatures. **Defer to Track C.** |
| **Fix the corner fitter (Path α typical scenes)** | Residuals already at noise floor (~0.14 px isotropic). Sub-floor improvement requires research-level work (sub-pixel-grid debias, joint multi-corner refit, Lanczos resampling). Worth pursuing for *render-tag SOTA*, but does not directly buy honest covariance. |

## §5 What's actually fixable

1. **Outlier scenes** (`scene_0008` style). Track A's `corner_geometry_outlier`
   gate (`min_irls < 0.3 OR max_corner_d² > χ²(1)|α=10⁻⁴ = 15.137`)
   already detects them at runtime. A small follow-up could (a) downweight
   the failing corner in the LM step, (b) inflate `Σ_pose` *only when the
   gate fires*, or (c) reject the detection. ~4 h. Doesn't help the 49
   typical scenes but moves the p99 tail.
2. **Real-camera covariance audit (Track C, task #13).** When real-camera
   data exists, the empirical `B = E[r rᵀ]` becomes a meaningful prior
   and Path β ships responsibly. Until then it is a synthetic-data trap.

## §6 What's not worth pursuing on this corpus

- Any further `α_max` sweep (CF1 closed).
- Any λ_tik formula (CF2 closed; mathematically redundant per §4).
- Any per-corpus calibration constant (Path β; defer to real data).
- Any rustdoc warning. CRB anisotropy on biased corners affects every
  pinhole-PnP detector (OpenCV, AprilTag-C, ...). Locus is not unique.

DF Phase 1's plumbing (`tikhonov_stabilize_returned_covariance`,
`adaptive_sigma_n` flags) is dead code given the §4 conclusions —
that branch should be dropped rather than merged.

## §7 Reproducing

```bash
uv run maturin develop --release \
    --manifest-path crates/locus-py/Cargo.toml --features bench-internals

# Audit (V1) + Path B-2 extension (per-corner residuals + counterfactual).
RAYON_NUM_THREADS=8 PYTHONPATH=. \
uv run --group bench tools/bench/pose_cov_audit.py \
    --hub-config locus_v1_tag36h11_1920x1080 \
    --output-dir diagnostics/pathb_gt_corners

# Path B correlation analysis + Path α direction analysis are throwaway
# scripts in /tmp/ during this investigation. Promote to tools/bench/
# when the question reopens against a real-camera corpus.
```

Outputs:
- `diagnostics/pose_cov_audit_2026-05-03/{report,samples}.json` — V1 baseline.
- `diagnostics/pathb_gt_corners/{report,samples}.json` — same audit
  extended with `gt_corners_px`, `corner_residuals_px`,
  `corner_residual_norms_px`, `d2_gt_corners`, `final_per_corner_d2`,
  `final_per_corner_irls_weight`.
