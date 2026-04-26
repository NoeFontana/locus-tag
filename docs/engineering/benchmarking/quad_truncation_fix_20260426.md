# Quad-extraction truncation fix (2026-04-26)

Recovers the ICRA-forward recall regression introduced by commit
[`5a2f438`](https://github.com/locus-tag/locus-tag/commit/5a2f438) ("fix(quad):
order SoA extraction by pixel-count to lift 2160p recall") and improves the
distortion suite as a side-effect. Branch: `feat-icra-recovery-border-gate`.

The originally-planned `verify_black_border` decode gate and `high_accuracy`
profile fix in `/home/dev/.claude/plans/soft-beaming-pancake.md` are deferred —
that scope produced no working knob-flip without regressing render-tag
rotation p99 by 30–180×, and the diagnostic surfaced this regression first.

## Environment

| Component | Value |
|---|---|
| CPU | AMD EPYC-Milan Processor (4 cores / 8 threads) |
| Arch | x86_64 |
| Build profile | `--release --features bench-internals` |
| Threads | `cargo test -- --test-threads=1` (sequential) for ICRA |

## §1 The bug

`5a2f438` added a `pixel_count_descending_order` helper that capped
`label_result.component_stats` at `MAX_CANDIDATES = 1024` *before* per-component
geometric filtering ran inside `extract_single_quad`. The 2160p win was real
(noise-dominated frames now keep the largest blobs first), but the ordering
inverts on dense small-tag scenes:

- ICRA forward backgrounds produce > 1024 raw CCL components per frame
  (texture, shadows, large structural blobs).
- Tag-sized candidates have *small* `pixel_count` relative to those blobs.
- The pre-filter truncation kept the giant non-tag blobs (which would have
  been geometrically rejected anyway) and discarded the tag candidates.

Net effect on ICRA forward `standard`: `mean_recall` collapsed from `0.7236`
(parent commit `df4e9ce`) to `0.6149` on HEAD before this fix. The snapshot
re-bless at `4998607` masked the regression at `0.7517` with a different
metric path; the underlying detection drop persisted.

## §2 The fix

Filter geometrically first, then partition by `pixel_count` only if the
survivor pool still exceeds the SoA ceiling.

```rust
let mut detections: Vec<(.., u32)> = stats
    .par_iter()
    .enumerate()
    .filter_map(|(label_idx, stat)| {
        // extract_single_quad applies all geometric gates
        ...map(|(c, u, cv)| (c, u, cv, stat.pixel_count))
    })
    .collect();

if detections.len() > MAX_CANDIDATES {
    detections.select_nth_unstable_by(MAX_CANDIDATES - 1, |a, b| b.3.cmp(&a.3));
    detections.truncate(MAX_CANDIDATES);
}
```

Rayon's `par_iter().enumerate().filter_map().collect()` preserves input order
(determinism), and `select_nth_unstable_by` keeps the partition O(n) instead
of O(n log n). Same pattern applied to the `_with_camera` variant.

The 2160p win from `5a2f438` is preserved: when total survivors really do
exceed 1024 (4K + heavy noise), partitioning by `pixel_count` still drops the
smallest survivors first.

## §3 Before / after

### ICRA forward (`tests/data/icra2020`, 50 frames, AprilTag36h11)

Snapshot deltas (this PR rebases all 8 affected ICRA snapshots; everything
not listed was already correct).

| Snapshot | Buggy HEAD | This PR | df4e9ce baseline |
|---|---:|---:|---:|
| `pure_default_standard` | 0.6149 | **0.7236** | 0.7236 |
| `pure_tags_images_soft` | — | **0.9403** | — |
| `pure_default_edlines` | — | **0.7113** | — |
| `pure_default_edlines_moments` | — | **0.7113** | — |
| `pure_default_moments_culling` | — | **0.7381** | — |
| `pure_tags_images` | — | **0.7381** | — |

`standard` recovers exactly to the parent-commit baseline (`df4e9ce`) — the
`0.7517` figure in the prior bless was upstream drift unrelated to quad
ordering, not in scope for this PR.

The `high_accuracy` profile (`mean_recall = 0.4631`) is unchanged — none of
the candidate knob-flips evaluated (sharpening, ContourRdp, Erf refinement)
lifted it without blowing render-tag rotation p99 from `1.897°` to
`26.7°–102°`. Deferred.

### Distortion (`hub_aprilgrid_distortion_*_v1_1920x1080`, 50 scenes each)

Side-effect of the fix — distortion paths use `extract_quads_soa_with_camera`,
which had the same bug.

| Subset | Recall (was) | Recall (now) | RMSE px (was) | RMSE px (now) |
|---|---:|---:|---:|---:|
| Brown–Conrady | 0.8701 | **0.9354** (+6.5 pp) | 1.3886 | **1.0947** |
| Kannala–Brandt | 0.8088 | **0.8130** (+0.4 pp) | 1.5517 | **1.5204** |

### Render-tag SOTA (`tag36h11_1920x1080`, all profiles)

Byte-identical. `render_tag_hub` rotation p99 stays at `1.897°` — the
high-precision suite's results were not gated by truncation order.

### Other gates

- `regression_board_hub` (AprilGrid + Charuco): unchanged.
- `contract_detection_batch` (8 cases): pass.
- `negative_detection`: unchanged false-positive count.
- Cross-compile `aarch64-unknown-linux-gnu`: clean.

## §4 Cleanup (`/simplify`)

Two no-op refactors landed alongside the fix:

1. Removed unused `frame_arena: &Bump` parameter from both `extract_quads_soa`
   and `extract_quads_soa_with_camera`. Per-component allocations route
   through `WORKSPACE_ARENA.with(...)` per Rayon worker — the outer
   per-frame arena was never read.
2. Eliminated the intermediate `Vec<(.., u32)> → Vec<(..)>` strip step. The
   trailing `pixel_count` is now discarded inline in the SoA write loop's
   destructure (`for (i, (corners, unrefined_pts, covs, _)) in ...`).
   Saves one full traversal + one heap allocation per frame
   (~10 MB/frame at 4K).

8 call sites updated: `detector.rs` (4), `hub_bench.rs` (1),
`contract_detection_batch.rs` (2), `test_quad_soa.rs` (1).

## §5 Diagnostic harness

`crates/locus-core/tests/icra_forward_diagnostic.rs` (new, `--ignored`)
attributes per-frame rejections across both `standard` and `high_accuracy`
on ICRA frames 0–5. Output shape per frame:

```
0000.png: valid=N rejected=M  funnel=<histogram>
  rejected_size_hist=<6-bin>  decode_hamming=<5-bin + no_sample>
```

Used to confirm that frames 0–5 in `high_accuracy` collapse pre-decode
(funnel-stage, not Hamming-stage) — i.e., the loss is upstream of the
codebook lookup, ruling out a `verify_black_border` fix as the right tool
for that gap. Run on demand:

```bash
LOCUS_ICRA_DATASET_DIR=tests/data/icra2020 \
  cargo test --release --features bench-internals \
  --test icra_forward_diagnostic -- --ignored --nocapture
```

## §6 Reproduction

```bash
# Truncation regression — must match df4e9ce baseline
TRACY_NO_INVARIANT_CHECK=1 LOCUS_ICRA_DATASET_DIR=tests/data/icra2020 \
  cargo test --release --features bench-internals \
  --test regression_icra2020 -- --test-threads=1

# Side-effect — distortion suite
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
  cargo test --release --features bench-internals \
  --test regression_distortion_hub --test regression_board_hub

# No-regression on render-tag SOTA
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
  cargo test --release --features bench-internals \
  --test regression_render_tag

# Phase-isolation contract + SoA helper
cargo nextest run --release --features bench-internals \
  --test contract_detection_batch --test test_quad_soa
```
