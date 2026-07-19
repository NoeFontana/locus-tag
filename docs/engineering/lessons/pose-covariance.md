# Pose covariance calibration — lessons

**Status:** CLOSED — all single-frame covariance-scaling levers falsified; open frontier is model-edge Fisher covariance (unmerged).
**Last updated:** 2026-07-19
**Owning code:** `crates/locus-core/src/pose_weighted.rs` (`refine_pose_lm_weighted`, the production "Accurate" solver; `compute_corner_covariance`, `finalize_corner_covariance`; 6×6 return path at the `(JᵀWJ)⁻¹` inverse). Audit harness: `tools/bench/pose_cov_audit.py`.

## TL;DR
The 6×6 SE(3) pose covariance emitted by the Accurate solver is anisotropically miscalibrated: on the hub `locus_v1_tag36h11_1920x1080` corpus the empirical squared 6-DoF Mahalanobis distance has mean d² = 714.7 vs χ²(6) ideal 6.0, and KL(empirical‖χ²(6)) = 13.93 nats (gate: ≤0.1 well-calibrated, 0.1–0.5 drift, >0.5 miscalibrated). Four scaling levers (Phases 1–4) were built and measured: pre-LM uniform inflation, post-LM diagonal congruence, GWLF empirical inflation, and per-corner ERF residual-MSE. Every one was falsified — no global or per-corner *diagonal/scalar* rescale moves KL below ~3, because the defect is in the **off-diagonal correlation structure** of `(JᵀWJ)⁻¹`, not its scale. The counterfactual proof: feeding GT corners back through the same LM gives d² = 0.0 to floating-point precision, so 100% of the miscalibration is upstream corner bias interacting with planar-PnP CRB anisotropy, not the noise model or numerics.

## What was tried and what happened
| Lever | Approach | Outcome | Key evidence/number |
| :-- | :-- | :-- | :-- |
| Phase 1 — GWLF empirical inflation | Inflate per-side `cov_l` by `max(1, MSE_emp/σ²_th)` in `refine_quad_gwlf_with_cov` | Unreachable | No shipped profile routes through `CornerRefinementMode::Gwlf`; audit path is structure-tensor, snapshots byte-identical |
| Phase 2 — pre-LM uniform `Σ_c` multiplier | Scale whole structure-tensor `Σ_c` by scalar `m` before weighted LM | Harmful (falsified) | `m=250`: mean d² 714.7→6.45, KL 13.9→5.08, BUT render-tag p99 rot 0.86°→1.90° (+120%); IPPE branch-ambiguity outliers leak past consistency gate |
| Phase 3 — post-LM diagonal 6×6 congruence | `Σ_new[i,j]=√(mᵢmⱼ)·Σ[i,j]` on LM output (PSD/corr-preserving, cannot regress pose) | Falsified (bottoms out) | audit-ratio `m`: mean d² 714.7→83.4, KL 13.9→6.3 — still ≫0.5; forcing per-axis ratios to 1.0 *worsens* joint metric to KL 27.5 |
| Phase 4 — per-corner ERF residual-MSE | `σ²_eff=max(σ_n², min(ε,16σ_n²))` from ERF edge-fit MSE, plumbed through SoA/SIMD (PR #290) | Falsified | KL stays 13.93 ±0.02 across all ε, including run E (true GT residual² upper bound); mean d² falls monotonically (655) but KL never moves; also dormant — `corner_empirical_noise` all-zero in production (every hub tag PPB>5.0, ERF never fires under `AdaptivePpb`) |

Also ruled out (investigation summary): α_max sweep (CF1 — p99-rot gate closes at α_max≈0.5–1.0 long before KL nears spec; extrapolated α_max≥3.0 → ~5° rot); Tikhonov-ε on returned Hessian (CF2 — redundant, Hessian is correct per GT-corner d²=0); per-corpus population `B=E[rrᵀ]` (Path β — honest form, dishonest spirit on synthetic data; defer to real cameras). Typical-scene corner residuals are already at a ~0.14 px isotropic noise floor (σx/σy=1.08).

## Why it's hard
A diagonal/scalar congruence `D·Σ·D` preserves the correlation matrix and only rescales per-axis variances. But the audit's eigen-decomposition shows 99% of the d² mass concentrates in the two *stiffest* principal directions (Σ understates error there by ~17×) while the four soft cardinal axes are simultaneously 3–10× too loose (per-axis diagonal ratios 0.026–0.27). The Hessian has ~7-orders-of-magnitude eigenvalue spread and its principal axes live in *mixed* translation–rotation directions that don't align with cardinals — so the correlation structure of `(JᵀWJ)⁻¹` is wrong, not merely its scale. No monotonic rescaling can simultaneously tighten the stiff mixed axes and loosen the soft cardinals; making the diagonal ratios exactly 1.0 blows the joint metric up (KL 27.5). Any real fix must alter the off-diagonal structure directly.

## Re-attempt only if
- A **fundamentally different covariance source** is used — one that constructs the information matrix from richer geometry rather than rescaling `(JᵀWJ)⁻¹`. This is the current open frontier: a **model-edge body-frame Fisher covariance** (edge + corner residuals) exists as an unmerged draft (PR #343, 2026-07-19, bench-internals only) that cuts KL substantially but still misses the strict ship gate — start there, not from any scaling lever.
- A **real-camera audit corpus** exists: only then does population `B=E[rrᵀ]` / Path β become a defensible prior rather than a synthetic-data trap.
- Phase 4 specifically: reopen **only if both** (1) a corpus/profile where ERF actually fires (non-zero `corner_empirical_noise` in production) **and** (2) a non-diagonal correlation-structure fix land together — condition (2) is load-bearing and unsolved.
- Note the diagonal per-axis / per-corner σ_n² family is provably exhausted (runs C/D/E) — do not re-attempt any diagonal or global scalar rescale.

## Provenance
Distilled 2026-07-19 from (removed; see git history): `pose_covariance_calibration_audit_2026-05-03`, `pose_covariance_investigation_summary_2026-05-03`, `pose_covariance_followup_2026-05-22`, `pose_covariance_phase4_postmortem_2026-05-31`. Related: [`MEMORY` anti-pattern "Global covariance multipliers"], and the [model-edge refinement](../benchmarking/model_edge_refinement_20260715.md) work whose Fisher-covariance offshoot is the open frontier.
