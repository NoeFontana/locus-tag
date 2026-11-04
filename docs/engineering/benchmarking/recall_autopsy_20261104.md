# Recall autopsy: where AprilGrid tags are lost (2026-11-04)

A recall number says how many tags were missed, not why. "The threshold never saw the border",
"the border fused with its neighbours" and "the bits read wrong" need unrelated fixes, so each
missed tag is attributed to the first stage that lost it.

## The tool

`crates/locus-core/examples/recall_autopsy.rs` (`bench-internals`) replays each frame with
telemetry, relabels the frame's own binarized image with the profile's connectivity, and samples
the middle of each truth tag's black border ring through the truth homography:

| Stage | Meaning |
|---|---|
| `not_foreground` | under half the border ring is below the threshold |
| `too_small` | the border's component is under `quad.min_area` |
| `fused_tags` | the border's component also holds another truth tag's border |
| `fused_structure` | the component's box is over twice the tag's (connectors, clutter) |
| `no_quad` | a tag-sized component that produced no candidate (quad gates, funnel) |
| `decode_failed` | a candidate over the tag that failed to decode |
| `wrong_id` | decoded, but to another id |
| `detected` | decoded to the right id |

```text
cargo run --release --features bench-internals --example recall_autopsy -- \
    --dataset tests/data/hub_cache/aprilgrid_golden_v1_1920x1080 --profile grid
cargo run --release --features bench-internals --example recall_autopsy -- \
    --truth frames.jsonl --profile grid --border-bits 2 --out autopsy.jsonl
```

`--truth` takes one JSON object per frame, `{"image": ..., "tags": [{"id", "corners"}]}`, with
corners in the detector's convention and order. `--border-bits` places the ring sample in the
middle of the printed border (2 for Kalibr's AprilGrid). `--out` writes one line per truth tag:
its stage, ring foreground fraction, component size and box ratio, the number of tags sharing
its component, and the best Hamming error of a failed candidate over it.

## Kalibr's printed AprilGrid, rendered

Truth from a physically based renderer of Kalibr's printed AprilGrid: 36h11 inside a 2-bit
border, connector squares at `tagSpacing = 0.3`, pixel-footprint integration, Gaussian PSF,
dot gain (ink spread), and shot/read noise through a 12-bit ADC, with blur, ink, levels,
vignetting and noise measured from TUM-VI's frames. Baseline: σ_b 0.5 px, 0.2 mm ink, 20 px tags,
30° incidence, image centre. One knob moves per case; 24 frames, 864 truth tags each. `grid`
profile, `fix/junction-pass-aprilgrid` (#478). The `detected` share reproduces the external
scorecard's recall in every case.

| Case | detected | decode_failed | fused_tags | other |
|---|---|---|---|---|
| baseline | 53.6 % | 32.1 % | 14.4 % | 0 |
| blur 0.3 / 0.8 px | 64.2 / 16.1 % | 19.0 / 67.4 % | 16.8 / 16.6 % | 0 |
| ink 0 / 0.1 / 0.4 mm | 68.5 / 63.4 / 16.8 % | 29.1 / 31.1 / 28.4 % | 2.4 / 5.4 / 54.9 % | 0 |
| tag 12 / 32 px | 0.6 / 39.1 % | 94.8 / 10.9 % | 4.6 / 50.0 % | 0 |
| incidence 0° / 50° / 70° | 74.2 / 23.0 / 2.3 % | 13.7 / 37.7 / 24.8 % | 12.2 / 39.2 / 67.4 % | 0 / 0 / 5.5 % |
| field 40° / 70° | 45.9 / 31.1 % | 28.2 / 24.3 % | 25.8 / 44.1 % | 0 / 0.4 % |
| noiseless | 55.4 % | 32.5 % | 12.0 % | 0 |

Two mechanisms account for every lost tag; thresholding, the quad gates and the funnel lose none.

- **Decoding (13–95 %).** It is present without ink (29 %) and without noise (33 %). Most
  failed candidates miss by 3–9 bits (baseline: 235 of 277), not by one or two. That is the
  signature of sampling the wrong lattice: the 10-cell tag is read as 8 cells at quad scale 0.9.
- **Fusion (2–67 %).** It grows with ink spread (2.4 → 5.4 → 14.4 → 54.9 % from 0 to 0.4 mm),
  with tag size at fixed ink (50 % at 32 px, where the spread covers more pixels) and with
  incidence. At baseline it fuses pairs; at 32 px, components of up to 12 tags — the same
  whole-lattice fusion that defeated pairwise neck-splitting on EuRoC (#449).

The hub's one-cell-border `aprilgrid_golden_v1` (150 frames, 5 169 tags, `grid`) loses 2.6 %:
113 `no_quad` and 11 `decode_failed` on 16–24 px tags, 5 `too_small`.

## Setup

AMD EPYC-Milan, 8 vCPU (KVM, AVX2), Linux 6.8.0-147, rustc 1.92.0, `--release`. Not a latency
measurement.
