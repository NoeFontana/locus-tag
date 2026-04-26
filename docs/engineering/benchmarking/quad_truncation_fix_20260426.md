# Quad-extraction truncation fix (2026-04-26)

Fixes the dense-scene tag-candidate truncation bug introduced by commit
[`5a2f438`](https://github.com/locus-tag/locus-tag/commit/5a2f438) ("fix(quad):
order SoA extraction by pixel-count to lift 2160p recall") while
preserving the pixel-count-desc iteration order that commit established.
Branch: `feat-icra-recovery-border-gate`.

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

`5a2f438` added a `pixel_count_descending_order` helper that did **two**
things bundled in one function:

1. **Sort component indices by `pixel_count` descending** — load-bearing for
   snapshot stability on noisy renders. Funnel/decoder dedup is
   processing-order-sensitive, and large-blob-first ordering lifts ICRA
   `standard` recall by ~2.8 pp vs natural-label order. ICRA-class datasets
   are crude synthetic renders with hard pixel edges where ordering
   matters; render-tag's PSF/Blender pipeline produces clean candidates
   whose dedup is order-insensitive (its metrics are byte-identical
   regardless of order).
2. **Truncate `component_stats` to `MAX_CANDIDATES = 1024` *before* per-
   component geometric filtering** — this is the bug. Tag-sized candidates
   have small `pixel_count` relative to large background blobs (texture,
   shadows, structural noise). Pre-filter truncation kept the giant blobs
   (which the gates would have rejected anyway) and discarded the tag
   candidates. Visible on the distortion suite: Brown–Conrady recall held
   at `0.870` instead of its true ceiling.

The 5a2f438 commit message focused on the 2160p win — at 4K the
truncation effect is largely benign because most ≥ 1024 survivors really
*are* noise. On 1080p distortion scenes the bug dominated.

## §2 The fix

Drop **only** the pre-filter truncation block from `pixel_count_descending_order`.
The desc-by-pixel-count ordering is preserved verbatim — that's the
load-bearing bit. Survivor truncation now happens caller-side, after
`extract_single_quad` has filtered geometrically. Rayon's
`ParallelIterator::collect()` preserves input order, so survivors come out
in pixel-count-desc order; truncating drops the smallest survivors —
which is the desired 4K-recall behaviour the original commit aimed at.

```rust
fn pixel_count_descending_order(stats: &[ComponentStats]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..stats.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        stats[b as usize].pixel_count
            .cmp(&stats[a as usize].pixel_count)
            .then(a.cmp(&b))    // tie-break for determinism under unstable sort
    });
    order
}

// extract_quads_soa
let order = pixel_count_descending_order(stats);
let mut detections = order
    .par_iter()
    .filter_map(|&label_idx| { extract_single_quad(...) })
    .collect::<Vec<_>>();
detections.truncate(MAX_CANDIDATES);
```

Same pattern in `extract_quads_soa_with_camera`.

## §3 Before / after

### ICRA forward (`tests/data/icra2020`, 50 frames, AprilTag36h11)

All 8 affected snapshots are byte-identical to the bless commit
`4998607` — the desc ordering preserves every result that depended on it.

| Snapshot | Bless `4998607` | This PR |
|---|---:|---:|
| `pure_default_standard` | 0.7517 | **0.7517** ✓ |
| `pure_default_soft` (tags_images_soft) | 0.9623 | **0.9623** ✓ |
| `pure_default_edlines` | 0.7113 | **0.7113** ✓ |
| `pure_default_edlines_moments` | 0.7113 | **0.7113** ✓ |
| `pure_default_moments_culling` | 0.7381 | **0.7381** ✓ |
| `pure_tags_images` | 0.7381 | **0.7381** ✓ |

The `high_accuracy` profile (`mean_recall = 0.4631`) is unchanged — none of
the candidate knob-flips evaluated (sharpening, ContourRdp, Erf refinement)
lifted it without blowing render-tag rotation p99 from `1.897°` to
`26.7°–102°`. Deferred.

### Distortion (`hub_aprilgrid_distortion_*_v1_1920x1080`, 50 scenes each)

The `_with_camera` path had the same bug. With pre-filter truncation
removed, distortion recall lifts to its true ceiling.

| Subset | Recall (was) | Recall (now) | RMSE px (was) | RMSE px (now) |
|---|---:|---:|---:|---:|
| Brown–Conrady | 0.8701 | **0.9354** (+6.5 pp) | 1.3886 | **1.0947** |
| Kannala–Brandt | 0.8088 | **0.8130** (+0.4 pp) | 1.5517 | **1.5204** |

### Render-tag SOTA (`tag36h11_1920x1080`, all profiles)

Byte-identical. `render_tag_hub` rotation p99 stays at `1.897°`. PSF/Blender
rendering produces clean candidates whose downstream dedup is
order-insensitive — so even the order-changing parts of this PR have no
metric impact on render-tag.

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
   per-frame arena was never read. `pixel_count_descending_order` now
   returns a plain `Vec<u32>` of length `stats.len()` (a few KB per
   frame); no bumpalo dependency.
2. Eliminated the intermediate `Vec<(.., u32)> → Vec<(..)>` strip step from
   the original 5a2f438 SoA write loop.

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
# ICRA — must match every committed snapshot byte-for-byte
TRACY_NO_INVARIANT_CHECK=1 LOCUS_ICRA_DATASET_DIR=tests/data/icra2020 \
  cargo test --release --features bench-internals \
  --test regression_icra2020 -- --test-threads=1

# Distortion suite (re-blessed; +6.5 pp Brown-Conrady recall)
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
  cargo test --release --features bench-internals \
  --test regression_distortion_hub --test regression_board_hub

# Render-tag SOTA — byte-identical
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
  cargo test --release --features bench-internals \
  --test regression_render_tag

# Phase-isolation contract + SoA helper
cargo nextest run --release --features bench-internals \
  --test contract_detection_batch --test test_quad_soa
```
