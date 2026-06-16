# Pose-Covariance Phase 4 (per-corner ERF empirical noise) — Postmortem — 2026-05-31

Negative-result postmortem for **Phase 4**, the per-corner ERF residual-MSE
covariance-inflation lever sketched in
`pose_covariance_followup_2026-05-22.md` §4. Phase 4 was the *only* lever the
2026-05-21 sweep had not ruled out. It has now been measured end-to-end and
**falsified on the hub corpus**. No production code ships from this work; this
memo records what was built, what was measured, and the conditions under which
re-attempting could make sense.

This continues the pattern of `pose_covariance_followup_2026-05-22.md` §6 and
the EdLines postmortem (`edlines_sota_followup_postmortem_2026-05-25.md`):
falsified mechanism is recorded in docs, the code stays recoverable in git
history, and `main` stays clean.

## TL;DR

- Phase 4 inflates each corner's structure-tensor covariance with the
  empirical ERF edge-fit residual MSE:
  `σ²_eff = max(σ_n², min(ε_empirical, 16·σ_n²))`.
- The plumbing was implemented and the audit harness extended to read it
  (branch `feat/activate-per-corner-empirical-noise`, PR #290).
- **The §4 acceptance gate was KL < 0.5. Phase 4 leaves KL at 13.93** — the
  baseline value, unchanged to ±0.02 even at the theoretical upper bound where
  `ε` is set to the actual per-corner ground-truth residual². It is falsified.
- Two independent reasons it cannot work as specified:
  1. **Structural** — per-corner *diagonal* inflation cannot repair the
     *off-diagonal* correlation structure of `(JᵀWJ)⁻¹`. Same wall that
     bottomed Phase 3 at KL ≈ 3.
  2. **Reachability** — on every shipped profile the production column
     `corner_empirical_noise` is **all zeros**, because every hub tag has
     PPB > 5.0 and the ERF route never fires under `AdaptivePpb`. The lever is
     dormant by construction.
- **Disposition:** do not ship the ~700 lines of SIMD/FFI/SoA/config plumbing.
  Render-tag protected snapshots are byte-identical with the flag on, which is
  itself proof the mechanism changes nothing observable. The audit-harness
  knobs and the empirical bound below are the salvageable artifacts.

## 1. What was built

Per `pose_covariance_followup_2026-05-22.md` §4, on branch
`feat/activate-per-corner-empirical-noise` (PR #290, re-authored from the
earlier PR #276 / `feat/per-corner-empirical-noise`):

- **SIMD residual² accumulator** in `refine_accumulate_optimized`
  (`edge_refinement.rs`) — `Σ r²` + in-window `sample_count` alongside the
  existing `sum_jtj` / `sum_jt_res`. SIMD-vs-scalar parity test included.
- `ErfEdgeFitter::last_residual_mse()` surfaces the converged per-sample MSE;
  `refine_corner` combines the two adjacent edges' MSE via `combine_edge_mses`
  (arithmetic mean, NaN-zero sentinel fallback).
- **New SoA column** `DetectionBatch::corner_empirical_noise: [[f32; 4];
  MAX_CANDIDATES]`, written by Phase A on the ERF route (`0.0` sentinel
  otherwise), read by Phase D.
- `finalize_corner_covariance` replaces the constant `σ_n²` with
  `max(σ_n², min(ε, 16·σ_n²))`. The 16× ceiling clamp bounds the Phase-2
  tail-cliff failure mode.
- Config flag `pose.use_empirical_corner_noise` (default `false`); only
  `high_accuracy` opted in.
- **Audit-harness extension** (`tools/bench/pose_cov_audit.py`): reads
  `batch.corner_empirical_noise` and passes it as `empirical_n_sq=` to
  `lb.compute_corner_covariance`, plus four knobs to exercise the rule on any
  corpus — `--no-empirical-inflation`, `--synthetic-empirical <px²>`,
  `--gt-residual-empirical`, `--force-ppb-threshold <t>`.

## 2. The measurement that falsifies it

Audit re-run on `locus_v1_tag36h11_1920x1080` (high_accuracy, AMD EPYC-Milan,
8 vCPU, `RAYON_NUM_THREADS` unset). Sweep
(`diagnostics/pose_cov_audit_2026-05-31/report.json` on the PR branch):

| Run | `empirical_n_sq` source | KL(emp ‖ χ²₆) | mean d² | p99 d² |
| :-- | :-- | --: | --: | --: |
| **A** (production today) | 0.0 (SoA column, all-zero) | **13.93** | 714.71 | 9435 |
| B  | flat ε = σ_n² (4.0) | 13.93 | 714.71 | 9435 |
| C  | flat ε = 4·σ_n² (16.0) | 13.95 | 651.95 | 7840 |
| D  | flat ε = 16·σ_n² (ceiling) | 13.93 | 625.51 | 7186 |
| **E** | per-corner GT-residual² (upper bound) | 13.93 | 655.34 | 7921 |

Read this as:

1. **The mechanism is wired correctly.** Mean d² falls monotonically as ε
   grows, and the 16× ceiling engages in run D as designed.
2. **KL does not move.** It stays 13.93 ± 0.02 across every setting including
   run E, the theoretical ceiling where `ε` is the true per-corner residual².
   The §4 gate (KL < 0.5) is missed by ~28×. `report.json` records
   `"kl_verdict": "miscalibrated (KL > 0.5)"`.
3. **The production column is all zeros.** Every hub tag has PPB > 5.0 (the
   `AdaptivePpb.threshold` Pydantic/Rust max), so all candidates take the
   high-PPB EdLines + None route. ERF never fires → no empirical MSE → `0.0`
   sentinel. Run A ≡ baseline.

## 3. Why it cannot work as specified

- **Diagonal inflation, off-diagonal problem.** This is the exact structural
  finding of `pose_covariance_followup_2026-05-22.md` §3: *"no global
  multiplicative scheme can bring the pose-covariance KL below ~3 — the
  correlation structure of (JᵀWJ)⁻¹ is wrong, not just its scale."* Phase 4
  is a per-corner diagonal `σ_n²` rescale; it modulates the *magnitude* of
  individual corner weights but leaves the off-diagonal structure of the
  inverted normal matrix untouched. Run E proves even a perfect per-corner
  residual estimator does not escape this wall.
- **Unreachable on shipped profiles.** Even if the diagonal lever helped, it
  fires on nothing in production. ERF refinement is gated to low-PPB
  candidates under `AdaptivePpb`, and the hub corpus has none. The 14/14
  byte-identical `regression_render_tag` snapshots with the flag on confirm
  zero production effect.

## 4. Re-attempt conditions

Phase 4 should not be reopened until **both** of these change, together:

1. **A corpus where ERF actually fires** — low-PPB tags, or a profile that
   routes more candidates through ERF (lifting the `EdLines + None` high-PPB
   route, currently empirically justified by scene_0008), so
   `corner_empirical_noise` is non-zero in production.
2. **A non-diagonal fix** — something that changes the correlation structure
   of the pose information matrix, not just per-corner weight magnitudes. A
   diagonal lever provably cannot (runs C/D/E above; §3 of the followup).

Condition 1 alone makes the column observable but, per run E, does not move KL.
Condition 2 is the load-bearing one and is unsolved.

## 5. What to salvage

- **The empirical bound in §2** is the durable result: Phase 4's KL ceiling is
  the baseline, full stop.
- **The audit-harness knobs** (`--synthetic-empirical`, `--gt-residual-empirical`,
  `--force-ppb-threshold`, `--no-empirical-inflation`) let a future agent probe
  any per-corner inflation rule *without* shipping production plumbing — they
  compute `ε` Python-side and pass it through `lb.compute_corner_covariance`.
  They live on branch `feat/activate-per-corner-empirical-noise`; recover via
  `git show` exactly as §6 prescribes for the GWLF template.
- **The ~700 lines of SIMD/FFI/SoA/config plumbing** stay on that branch and do
  **not** ship to `main`. They change nothing observable and would only bit-rot.

## 6. Status of the covariance investigation

With Phase 4 falsified, **all four phases are now closed**:

| Phase | Lever | Status |
| :-- | :-- | :-- |
| 1 | GWLF empirical inflation | Unreachable — no profile routes through GWLF |
| 2 | Pre-LM uniform `Σ_c` multiplier | Harmful — render-tag p99 rot 0.86° → 1.90° |
| 3 | Post-LM diagonal `Σ` congruence | Bottoms at KL ≈ 3 — off-diagonal structure wrong |
| 4 | Per-corner ERF residual MSE | Falsified — KL unchanged at 13.93; dormant in production |

There is no known surviving global or per-corner covariance-calibration lever.
Any future work must attack the off-diagonal correlation structure of
`(JᵀWJ)⁻¹` directly.

## 7. Reproduction

On branch `feat/activate-per-corner-empirical-noise`, after
`maturin develop --release --features bench-internals`:

```bash
# Run A (production, all-zero column)
PYTHONPATH=. uv run --group bench tools/bench/pose_cov_audit.py \
  --hub-config locus_v1_tag36h11_1920x1080 --profile high_accuracy

# Run E (theoretical upper bound)
PYTHONPATH=. uv run --group bench tools/bench/pose_cov_audit.py \
  --hub-config locus_v1_tag36h11_1920x1080 --profile high_accuracy \
  --gt-residual-empirical
```

Both report `kl_divergence_to_chi2_6 ≈ 13.93`.

## References

- `pose_covariance_followup_2026-05-22.md` §3 (structural finding), §4 (Phase 4
  sketch + KL < 0.5 gate), §6 (don't-ship-dormant-mechanism precedent).
- `edlines_sota_followup_postmortem_2026-05-25.md` (the docs-only negative-result
  pattern this follows).
- PR #290 `feat/activate-per-corner-empirical-noise` (plumbing + harness, not
  shipped); PR #276 (superseded, closed).
- `diagnostics/pose_cov_audit_2026-05-31/report.json` (the falsifying numbers).
