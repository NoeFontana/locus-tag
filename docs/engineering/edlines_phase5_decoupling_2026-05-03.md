# EdLines #2 — line-parameterised Phase 5 (negative result, 2026-05-03)

Implemented design memo §5.2: replace the chord-locked 8-DoF corner GN with
a per-edge separable line-parameterised GN (`(φ_k, d_k)` × 4 = 8 DoF, but
block-diagonal). Corners are derived as closed-form line intersections;
covariances are propagated by the analytical Jacobian of the intersection
w.r.t. the four line parameters of the two adjacent edges.

**Result: hypothesis falsified.** F4 (chord-locking) is *not* the dominant
cause of scene_0008's failure. The error originates upstream of Phase 5,
in Phases 1–4. Decoupling Phase 5 actually *worsens* the corpus while
producing only marginal improvement on scene_0008.

## §1 Hardware

Same EPYC-Milan KVM, `--release` build with `bench-internals`,
`RAYON_NUM_THREADS=8`. CPython 3.14.3, rustc 1.92.0.

## §2 Implementation

`refine_lines_gauss_newton` (~210 LoC) added alongside
`refine_corners_gauss_newton`, gated behind `const USE_LINE_PHASE5`.

Per-edge state vector `(φ_k, d_k)` parameterises line
`x cos φ_k + y sin φ_k + d_k = 0`. Cost is `Σ_i r_{k,i}²` where
`r = q_x cos φ + q_y sin φ + d` — independent across edges. Each 2-DoF
problem solves in 1 iteration via 2×2 Cholesky-equivalent.

Line covariances `Σ_line_k = σ²_k · H_k⁻¹` where
`σ²_k = Σ_i r_{k,i}² / (n − 2)`. Corner covariance for corner k =
intersect(line_{k−1}, line_k):

    Σ_corner_k = J · diag(Σ_{k−1}, Σ_k) · Jᵀ

with J the 2×4 Jacobian of intersection w.r.t. (φ_{k−1}, d_{k−1}, φ_k, d_k)
derived analytically via the implicit-function theorem on
`F1(x, y) = x cos φ_a + y sin φ_a + d_a = 0` and same for line b.

## §3 Three-way comparison

| Variant | scene_0008 corner 1 ‖r‖ | Corpus mean ‖r‖ | Corpus p99 ‖r‖ | Corpus KL |
|---|---:|---:|---:|---:|
| **Baseline** (chord-locked GN) | 3.833 px | 0.191 | 0.697 | 13.93 |
| **No Phase 5** (just intersect Phase 4 lines) | 3.670 (−4 %) | 0.217 (**+14 %**) | 0.880 | — |
| **Line-GN Phase 5** (3 iters) | 3.389 (−12 %) | 0.256 (**+34 %**) | 0.929 | 17.49 |

## §4 What this proves

### F4 is not dominant

The "no Phase 5" variant intersects Phase 4 lines directly, with **no GN
refinement at all**. If F4 (chord-locked GN producing wrong corners) were
the dominant mechanism, removing Phase 5 should produce a dramatic
improvement on scene_0008. It produces a **4 %** improvement.

The remaining 3.67 px error must originate in Phases 1–4. Specifically,
**Phase 4's lines themselves intersect at (≈ 907.0, 457.0) — not at the
true gray-edge intersection (903.16, 456.27).**

### The chord-locked GN is empirically beneficial on typical scenes

Phase 5's chord-locked GN reduces corpus mean ‖r‖ from 0.217 → 0.191
(−12 % vs no-Phase-5). The joint optimisation across all 4 corners does
provide regularisation when Phase 4 corners are well-conditioned. Removing
Phase 5 makes 49 of 50 scenes *worse*.

### Decoupled line-GN regresses both directions

Decoupled per-edge IRLS-without-Huber-reweighting gave:
- 0.4 px improvement on scene_0008
- 0.065 px regression on corpus mean (~30× the typical noise floor)

The Huber re-weighting in current Phase 4 IRLS suppresses corner-bleed
contamination on typical scenes; the per-edge GN's pure least-squares
inherits it. Adding Huber to the per-edge GN would partially recover but
not change the scene_0008 conclusion.

## §5 Where the bug actually lives

The 4 px error in Phase 4's intersected corners must come from one of:

| Source | Mechanism | Verifiability |
|---|---|---|
| **Phase 1 boundary partition** | TRBL/Diagonal extremals near corner-1 cause arc-boundary placement to put a few boundary points in the wrong arc, biasing Phase 2's line slope. | High — instrument extremal selection |
| **Phase 2 binary IRLS** | Integer-pixel boundary points at corner-1 region are sparse along x; Huber-fit slope is sensitive to which sub-pixel column gets marked "leftmost foreground". | High — log Phase 2 line slopes |
| **Phase 3 sub-pixel ray walk** | Walking along Phase 2's binary line (which already has a slope error) causes probe centres to drift away from the true edge near corner 1, producing systematically biased sub-pixel points. | Medium — would need to dump Phase 3 points |
| **Phase 4 sub-pixel IRLS** | Re-fits Phase 3 points with tighter Huber δ; if Phase 3 points are biased, Phase 4 inherits the bias. | Verified empirically — `fl[3]` and `fl[0]` intersect at ~3.7 px from true corner |

The most likely culprit is **Phase 1 → Phase 2 cascade**. For
scene_0008, the tag is rotated near 90° (canonical bottom edge → image
vertical). The diagonal extremals (NW=c1, NE=c2, SE=c3, SW=c0) place arc
boundaries exactly at the corners — but the *actual integer-pixel* corner
positions in the boundary are off by ~1 px from the geometric corners
(due to rounded boundary tracing). This 1-px offset gets amplified by
the IRLS line fit's lever arm (148 px edge length × 1 px / 148 = 1 px end
displacement) — too small to be the full 4 px error, but a contributor.

The remaining 3 px is likely **Phase 3 walk**: with a slightly-wrong
Phase 2 line, sub-pixel probes near corner 1 walk perpendicular to the
wrong direction. The parabolic fit clamp (±1.5 px) caps how far each
probe can travel, but systematic bias in probe placement compounds.

## §6 Conclusion

**The simple architectural fixes (#1, #2) do not address scene_0008.**
The bug is in the upstream Phase 1–4 cascade for tag geometries near
the diagonal-extremal degenerate case. Fixing it requires:

1. A different boundary-segmentation scheme that places arc boundaries
   away from corners (e.g., dominant-orientation-aware partitioning).
2. **Or** an ERF-style Phase 3 that fits a 2D edge model independent of
   binary line input (improvement #3 in the design memo). This bypasses
   the upstream cascade entirely.

Both are multi-day investments. Per the original recommendation, the
**runtime gate via Track A's `corner_geometry_outlier`** remains the
practical fix on synthetic data alone.

## §7 What to commit

This memo + the empirical comparison data. The implementation
(`refine_lines_gauss_newton`) is reverted from `edlines.rs` — it adds
~210 LoC of dead code that is correct but provides no benefit. The
analytical-Jacobian derivation is preserved here for any future #2
revival.

## §8 Reproducing

```bash
# Apply patch (kept in this branch's history before revert):
git show <commit>:crates/locus-core/src/edlines.rs > /tmp/edlines_with_lineGN.rs

# In the patched file, set USE_LINE_PHASE5 = true (or with bypass: max_iters=0).
# Build + audit:
uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml --features bench-internals
RAYON_NUM_THREADS=8 PYTHONPATH=. uv run --group bench tools/bench/pose_cov_audit.py \
    --hub-config locus_v1_tag36h11_1920x1080 \
    --output-dir diagnostics/edlines_<variant>
```

The two diagnostic dirs from this experiment
(`diagnostics/edlines_lines_phase5/` and `diagnostics/edlines_no_phase5/`)
are deleted post-investigation; results are summarised in §3.
