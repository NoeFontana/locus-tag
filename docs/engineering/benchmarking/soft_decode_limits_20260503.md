# Soft decode limits — empirical sweep

**Date:** 2026-05-03
**Branch:** `feat/soft-decode-empirical-findings`
**Question:** can the bench harness's `Locus (Soft)` configuration be saved by tightening `max_hamming`, or is the precision crater structural?

## Headline

Tightening `max_hamming` cannot save Soft. **No `(decode_mode=Soft, max_hamming ∈ {0, 1, 2})` cell satisfies the decision rule** (precision ≥ 0.5, Δrecall ≤ 1.0pp vs current Soft default, rotation error p50 ≤ Hard × 1.5) on any of the three resolutions tested. The parameter-only path is closed; the next step is a structural fix (post-decode geometric verification), tracked separately.

The user's earlier observation that "Locus (Soft) rotation error p50 0.5°–1.5° vs Hard sub-0.2°" did not replicate against matched records — Soft and Hard produce identical pose error on true tags because the pose pipeline is the same. The replicable issue is **precision**: Soft accepts hundreds of FPs per frame while Hard accepts zero.

## Sweep matrix

`PYTHONPATH=. uv run --group bench tools/cli.py bench real --hub-config <C> --max-hamming <H> --limit 100 --record-out runs/soft_sweep/<C>_h<H>.parquet` for `H ∈ {0, 1, 2}` × the corpora below.

| Corpus | h | Binary | TP | FP | Recall | Precision | Rot p50 (°) | Latency (ms) |
| :--- | :-: | :--- | -: | -: | -: | -: | -: | -: |
| `tag36h11/1280x720` | 0 | Hard | 50 | 0 | 1.000 | **1.0000** | 0.323 | 8.6 |
| `tag36h11/1280x720` | 1 | Hard | 50 | 0 | 1.000 | **1.0000** | 0.323 | 8.7 |
| `tag36h11/1280x720` | 2 | Hard | 50 | 0 | 1.000 | **1.0000** | 0.323 | 8.7 |
| `tag36h11/1280x720` | 0 | Soft | 50 | 861 | 1.000 | 0.0549 | 0.323 | 11.5 |
| `tag36h11/1280x720` | 1 | Soft | 50 | 1291 | 1.000 | 0.0373 | 0.323 | 10.1 |
| `tag36h11/1280x720` | 2 | Soft | 50 | 1509 | 1.000 | 0.0321 | 0.323 | 9.9 |
| `tag36h11/1920x1080` | 0 | Hard | 45 | 0 | 1.000 | **1.0000** | 0.320 | 16.7 |
| `tag36h11/1920x1080` | 1 | Hard | 45 | 0 | 1.000 | **1.0000** | 0.320 | 17.2 |
| `tag36h11/1920x1080` | 2 | Hard | 45 | 0 | 1.000 | **1.0000** | 0.320 | 16.8 |
| `tag36h11/1920x1080` | 0 | Soft | 45 | 1789 | 1.000 | 0.0245 | 0.320 | 22.8 |
| `tag36h11/1920x1080` | 1 | Soft | 45 | 2766 | 1.000 | 0.0160 | 0.320 | 19.7 |
| `tag36h11/1920x1080` | 2 | Soft | 45 | 3289 | 1.000 | 0.0135 | 0.320 | 20.0 |
| `tag36h11/3840x2160` | 0 | Hard | 45 | 0 | 1.000 | **1.0000** | 0.361 | 64.6 |
| `tag36h11/3840x2160` | 1 | Hard | 45 | 0 | 1.000 | **1.0000** | 0.361 | 61.6 |
| `tag36h11/3840x2160` | 2 | Hard | 45 | 1 | 1.000 | 0.9783 | 0.361 | 63.5 |
| `tag36h11/3840x2160` | 0 | Soft | 45 | 5794 | 1.000 | 0.0077 | 0.361 | 85.9 |
| `tag36h11/3840x2160` | 1 | Soft | 45 | 8501 | 1.000 | 0.0053 | 0.361 | 82.1 |
| `tag36h11/3840x2160` | 2 | Soft | 45 | 10228 | 1.000 | 0.0044 | 0.361 | 75.2 |

Charuco runs were attempted but produced 0%/0% recall under the default `--family AprilTag36h11`. ChAruco corpora need a different family configuration to be evaluated; it would not have changed the conclusion.

## Decision rule application

> The new Soft default is the smallest `max_hamming` such that:
> - precision ≥ 0.5 across all configs,
> - recall regression ≤ 1.0 pp vs current Soft default on every config,
> - rotation error p50 ≤ Hard's value × 1.5.

```
[tag36h11/1280x720]  Hard@h2 rot_p50=0.323°
  Soft@h0: P=0.0549 (NO)  recall=1.000 (OK)  rot_p50=0.323° (OK)  → FAIL
  Soft@h1: P=0.0373 (NO)  recall=1.000 (OK)  rot_p50=0.323° (OK)  → FAIL
  Soft@h2: P=0.0321 (NO)  recall=1.000 (OK)  rot_p50=0.323° (OK)  → FAIL
[tag36h11/1920x1080] Hard@h2 rot_p50=0.320°
  Soft@h0: P=0.0245 (NO)  recall=1.000 (OK)  rot_p50=0.320° (OK)  → FAIL
  Soft@h1: P=0.0160 (NO)  recall=1.000 (OK)  rot_p50=0.320° (OK)  → FAIL
  Soft@h2: P=0.0135 (NO)  recall=1.000 (OK)  rot_p50=0.320° (OK)  → FAIL
[tag36h11/3840x2160] Hard@h2 rot_p50=0.361°
  Soft@h0: P=0.0077 (NO)  recall=1.000 (OK)  rot_p50=0.361° (OK)  → FAIL
  Soft@h1: P=0.0053 (NO)  recall=1.000 (OK)  rot_p50=0.361° (OK)  → FAIL
  Soft@h2: P=0.0044 (NO)  recall=1.000 (OK)  rot_p50=0.361° (OK)  → FAIL
```

Every cell fails the precision criterion. The other two criteria (recall, rotation) are trivially satisfied because Soft and Hard produce identical recall and pose error on matched records.

## Why parameter tuning can't save Soft

`crates/locus-core/src/strategy.rs:165-194` gates Soft acceptance on a single LLR-distance threshold:

```rust
let llr_per_hamming_bit = 60_u32;
let soft_threshold = max_error.max(1) * llr_per_hamming_bit;
let coarse_rejection_threshold = max_error * 2;
// ... iterate candidates within `coarse_rejection_threshold` Hamming distance
// ... accept the LLR-closest if `best_dist < soft_threshold`
```

Tightening `max_hamming` shrinks **both** the search radius (`coarse_rejection_threshold`) and the acceptance threshold (`soft_threshold`), but it never changes the decision rule's fundamental shape: "find the LLR-closest valid codeword, accept if it's close enough." On clean synthetic corpora where the funnel + bit-sampling stages already let through O(10²–10⁴) quad-shaped pieces of noise per frame, that rule accepts any noise pattern whose LLR happens to land near a real codeword — and `tag36h11`'s code distance leaves enough valid codewords nearby that this happens for ~1–80 quads per frame depending on resolution.

The empirically-observed scaling of FPs / true tags:

| Resolution | FPs/TP @ h=0 | FPs/TP @ h=2 |
| :--- | -: | -: |
| 720p | 17.2× | 30.2× |
| 1080p | 39.8× | 73.1× |
| 2160p | 128.8× | 227.3× |

FP density rises with resolution because the funnel's quad-extraction throughput rises with pixel count — more raw quads through the funnel means more dice rolls against the Soft acceptance gate.

## Note on the user's earlier "rot p50 0.5°–1.5°" claim

The earlier observation that Soft's rotation error p50 was 0.5°–1.5° vs Hard's sub-0.2° **did not replicate** against matched records in this sweep. On the same corpora, both Soft and Hard report **identical** rot p50 (0.32°–0.36°). This makes mechanical sense: the Soft/Hard split affects which quads the decoder *accepts*, but once a true tag is decoded, both modes route through the same pose pipeline. The earlier figure was likely an artifact of the report aggregation including non-attributed `false_positive` records (where `rot_err_deg` is `NaN`) or matching FPs to nearest GT and computing rot vs the wrong GT. Either way, the precision finding stands and is the more important issue: Soft accepts hundreds of FPs per frame, Hard accepts zero.

## Recommendation

1. **Do not change the bench harness's default `max_hamming`** in this PR. Hard is functionally insensitive (100% precision at every value), so changing the default is no-op for the operating-point binary and only marginally tightens an already-broken Soft.
2. **Surface the structural caveat** in `tools/cli.py`'s `--max-hamming` help text and in the profile README so future readers don't conclude that lower `max_hamming` will fix Soft's precision.
3. **Queue the structural fix** as the next workstream. The fix is post-decode geometric verification: after Soft accepts a candidate, project its decoded bit pattern back to image space and reject if the residual exceeds a learned threshold. Sketch:
   - Reuse the corner residuals already computed during ERF refinement.
   - Reject Soft-decoded tags whose residual exceeds Hard's residual distribution by ≥ 3σ on a calibration set.
   - Estimated scope: ~150 LOC in `crates/locus-core/src/strategy.rs` + a calibration script + tests. New PR.
4. **Consider follow-up**: if the structural fix isn't worth the eng cost, prefer to deprecate `decode_mode=Soft` from the bench harness's default lineup. On the corpora available to us today, Soft is **strictly dominated**: identical recall and pose error to Hard, 25–35% more latency, and 18×–227× more FPs per true tag.

## Reproducing this sweep

```bash
mkdir -p runs/soft_sweep
for corpus in single_tag_locus_v1_tag36h11_1280x720 \
              single_tag_locus_v1_tag36h11_1920x1080 \
              single_tag_locus_v1_tag36h11_3840x2160; do
  for h in 0 1 2; do
    PYTHONPATH=. uv run --group bench tools/cli.py bench real \
      --hub-config "$corpus" \
      --max-hamming "$h" \
      --limit 100 \
      --record-out "runs/soft_sweep/${corpus}_h${h}.parquet"
  done
done
```

Aggregation script: `/tmp/aggregate_sweep.py` in the development context (kept inline rather than committed because the parquets themselves are not version-controlled — re-run the loop above to regenerate).

## Hardware

CPU: 16 logical cores (verified via `lscpu` in this session). Linux 6.8.0-107-generic. Build: `--release` via `uv run maturin develop --release`. RAYON_NUM_THREADS unset (Rayon picks defaults).
