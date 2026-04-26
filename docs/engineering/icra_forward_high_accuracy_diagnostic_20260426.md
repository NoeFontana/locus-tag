# `high_accuracy` profile collapse on ICRA forward frames 0–5

Diagnostic-driven attribution of the recall delta between the `standard`
and `high_accuracy` shipped profiles on the densest crude-render scenes
of the ICRA 2020 dataset.

## §1 The gap

| Profile | ICRA forward mean recall | Frames 0–5 recall |
|---|---:|---:|
| `standard` | `0.7236` | `~0.05` (≈8/154 per frame) |
| `high_accuracy` | `0.4631` | **`0.00`** (5 of 6 frames detect zero) |

`high_accuracy` is the metrology preset (EdLines extraction + Mahalanobis
χ² pose gate, `refinement_mode = None`). It is the right profile for
clean PSF/Blender renders — it lifts render-tag pose tail to
sub-degree p99 — but on dense crude-render small-tag scenes it
collapses.

## §2 The diagnostic

`crates/locus-core/tests/icra_forward_diagnostic.rs` (`#[ignore]`,
`--ignored` to run) loads frames `0000.png`–`0005.png` and prints
per-frame, per-profile attribution:

- `valid=N` / `rejected=M` — accepted vs SoA-rejected candidates.
- `funnel=<hist>` — `RejectedContrast` / `RejectedSampling` /
  `PassedContrast`.
- `rejected_size_hist=<6-bin>` — bounding-box side-length distribution
  of rejected candidates.
- `decode_hamming=<5-bin + no_sample>` — best Hamming distance to any
  dictionary entry across all rotations, for funnel-passers.

```bash
LOCUS_ICRA_DATASET_DIR=/path/to/icra2020 \
  cargo test --release --features bench-internals \
  --test icra_forward_diagnostic -- --ignored --nocapture
```

## §3 What the numbers say

Sample frame-0000 output:

```
standard       valid=9   rejected=205  funnel=PassContrast=205  decode_hamming=h<3=0 3-5=49  6-10=155
high_accuracy  valid=0   rejected=102  funnel=PassContrast=102  decode_hamming=h<3=0 3-5=5   6-10=97
```

Three observations:

1. **The funnel is not the bottleneck.** Both profiles pass 100 % of
   their candidates through the contrast funnel (`RejContrast=0`,
   `RejSampling=0`). Contrast / sharpening / threshold knob flips
   cannot move this dial.
2. **EdLines under-produces by ~50 %.** `high_accuracy` extracts
   `~110 ± 12` candidates per frame versus `standard`'s `~206 ± 1`.
   EdLines (designed for clean photometric edges) culls roughly half
   the dense-scene small-tag candidates that ContourRdp would keep.
3. **Decode-stage corner quality is poor on what survives.** The
   Hamming histogram skews heavily into the `6–10` bucket
   (`97 / 102` rejected candidates on frame 0000 vs `155 / 205` for
   `standard`). With imprecise EdLines corners the bit-sample grid
   lands off-cell, so Hamming overshoots even on candidates that *are*
   tags.

The collapse is therefore **end-to-end attributable to the
`extraction_mode = EdLines` choice on dense small-tag scenes** —
not to funnel, threshold, or refinement.

## §4 Knob-flip experiment matrix

Tested on a fresh experiment branch (`experiment/high-accuracy-extraction-mode`).
ICRA columns are full-dataset `mean_recall`; render-tag (RT) columns
are 1080p p99 rotation error / mean RMSE under `accuracy_baseline`.

| `extraction_mode` | `refinement_mode` | ICRA highacc recall | RT 1080p p99 rot | RT 1080p mean_rmse |
|---|---|---:|---:|---:|
| `EdLines` | `None` (current) | **0.4631** | **0.562°** | **0.2029** |
| `ContourRdp` | `None` | ~0.72 | 102.49° | 1.74 |
| `ContourRdp` | `Erf` | 0.7236 | 103.83° | 1.38 |
| `ContourRdp` | `Gwlf` | 0.6361 | 0.4134° | 0.86 |

Conclusions:

- **No static `(extraction_mode, refinement_mode)` pair simultaneously
  wins both regimes.** EdLines optimises the clean-render pose tail.
  ContourRdp optimises crude-render recall.
- Even `ContourRdp + Gwlf`, which recovers a sub-degree render-tag
  rotation tail, **degrades render-tag mean RMSE 4×** (`0.20 → 0.86`).
  The clean-render regression budget is exhausted.
- ContourRdp alone (`+ None` or `+ Erf`) **destroys** the render-tag
  pose tail (`0.56° → 102°`). The high_accuracy profile's identity
  *is* its sub-degree pose tail; trading that for ICRA recall is not
  acceptable.

`high_accuracy` is left at `EdLines + None` for this PR. The static
profile cannot be fixed by knob-flipping without trading away its own
purpose.

## §5 SOTA path forward

The next-step candidates, in implementation cost order:

1. **Adaptive extraction by candidate size.** Route per-component:
   `ContourRdp` for `pixel_count < threshold`, `EdLines` otherwise.
   Component stats are produced by CCL before extraction runs, so the
   per-component branch is free. Hypothesis: ICRA frames 0–5 have
   small-pixel-count candidates that ContourRdp handles; render-tag's
   large clean tags stay on EdLines, pose tail preserved.
2. **Image pyramid coarse-to-fine.** Detect at `½×` / `¼×`, lift
   corners back to full resolution and re-refine. Standard fix for
   dense small-tag recall (AprilTag3 `decimation`, OpenCV `aruco`
   pyramid). Best long-term unlock — also addresses 2160p far-field
   — but a real engineering project.
3. **Per-bit gradient weighting in decode.** Down-weights bits whose
   sample positions land on weak gradients. Partially compensates for
   imprecise corners without changing extraction. Strict
   precision-additive; doesn't help recall directly.

Path (1) is the next concrete piece of work. (2) and (3) are deferred
to a separate planning round.

## §6 Notes on test infrastructure

While running this diagnostic, the silent-skip bug in
`tests/common::resolve_dataset_root` was discovered: a relative
`LOCUS_ICRA_DATASET_DIR` was resolved against the test CWD (the crate
root, *not* the workspace root) and fell through to the in-tree
1-frame stub fixture; `IcraProvider::new` then returned `None` and
the macro's `if let Some(provider)` short-circuited the snapshot
assertion. PR #205's pre-merge "16 passed" was a false positive.
The fix:

- Relative `LOCUS_ICRA_DATASET_DIR` now resolves against the workspace
  root (mirroring `resolve_hub_root`).
- A non-directory `LOCUS_ICRA_DATASET_DIR` panics rather than falling
  back to the fixture.
- `IcraProvider::new` panics rather than returning `None` when the
  env var is set but the GT or image directory is missing for the
  requested subfolder.

This is a one-time correction. New ICRA snapshots reflect the real
dataset and the previous "byte-identical" claim in
`docs/engineering/benchmarking/quad_truncation_fix_20260426.md` has
been corrected in place.
