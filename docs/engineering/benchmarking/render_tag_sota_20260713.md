# render-tag SOTA refresh — OpenCV 5.0 (2026-07-13)

Refreshes the detector-comparison matrix after the **OpenCV 4.13 → 5.0.0** bump,
with the OpenCV baseline **re-tuned** for its best 5.0 config. Supersedes the
`render_tag_sota_20260425.md` snapshot for the OpenCV rows.

> **2026-07-19 update (v0.7.0).** The `Locus (high_accuracy)` rows in §1/§2 are
> refreshed to the **v0.7.0 shipped config**, which enables model-edge pose
> refinement by default (`pose.pose_edge_refinement_enabled = true`). Re-measured on
> the same AMD EPYC-Milan hardware with `render_tag_sota_eval.py`; the competitor
> rows reproduced bit-for-bit (accuracy is deterministic) and are unchanged. The
> pre-refinement `high_accuracy` baseline was rot p99 0.600° (1080p) / 1.113° (2160p)
> at 13.8 / 56.4 ms. See `model_edge_refinement_20260715.md`.

## Methodology

- **Datasets:** `locus_v1_tag36h11_1920x1080` and `_3840x2160` Hub render-tag
  subsets (50 scenes, single tag36h11 each); captured via
  `tools/bench/render_tag_sota_eval.py` (set `RENDER_TAG_SOTA_CONFIG` for 2160p).
- **Single-threaded latency.** Every detector is pinned to one thread for an
  apples-to-apples comparison: `cv2.setNumThreads(1)`, Locus `threads=1`,
  AprilTag `nthreads=1`, `RAYON_NUM_THREADS=1`. Latency is the mean per-frame
  detector-only wall time, measured serially on a quiescent machine.
- **Hardware (verified, `lscpu`):** AMD EPYC-Milan, 8 vCPU (4 cores × 2 threads),
  KVM. Release build (`maturin develop --release`). Locus `git e3fa5f0`.
- **Versions:** OpenCV `cv2` **5.0.0**, `pupil_apriltags` (AprilTag-C).
- **OpenCV tuning:** `bench tune` / `bench sweep` over
  `tools/bench/tune/spaces/opencv_default.json` (+ a subpix-refinement extension),
  n≈32–48 random, `precision_floor=0.98`.

Accuracy is deterministic (independent of machine load); latency is not, hence
the single-thread quiescent protocol.

## Two OpenCV operating points

OpenCV's corner-refinement choice is a genuine speed/accuracy lever, so both are
reported:

- **`subpix`** — fast; intensity-gradient corner refinement. Tuned detection
  thresholds (not the refinement convergence knobs, which don't help) take it
  from the pre-tuning 141 mm → **66.6 mm** trans p99 at the same speed.
- **`apriltag`** — best accuracy; tag-homography-aware refinement. **55.3 mm**
  trans p99 / best rotation tail, at ~2× the subpix latency (1080p). The shipped
  `OpenCVWrapper._DEFAULTS`. **Note:** its refinement *rejects* markers whose
  local re-detection fails, so it is best only on clean imagery — see ICRA below.

## §1 Detector matrix — 1920×1080 (50 scenes, single thread)

| Detector | Recall | Prec | Trans p50 | Trans p99 | Rot p50 | Rot p99 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Locus (`high_accuracy`)** | 100 % | 100 % | **0.4 mm** | **20.1 mm** | **0.041°** | **0.249°** | **15.2 ms** |
| Locus (`standard`) | 100 % | 100 % | 3.5 mm | 50.3 mm | 0.288° | 27.248° | 32.7 ms |
| OpenCV (`subpix`) | 100 % | 100 % | 3.5 mm | 66.6 mm | 0.127° | 0.569° | 101.1 ms |
| OpenCV (`apriltag`) | 100 % | 100 % | 3.0 mm | 55.3 mm | 0.067° | 0.376° | 195.8 ms |
| AprilTag-C (pupil) | 100 % | 100 % | 2.9 mm | 54.4 mm | 0.061° | 65.365° | 78.5 ms |

Locus `high_accuracy` wins the translation tail, the rotation tail (0.249° p99,
below OpenCV `apriltag`'s 0.376° — the model-edge refinement shipped on in v0.7.0
closes the gap), **and** speed (13× faster than OpenCV `apriltag`).
AprilTag-C's rotation p99 explodes to 65° on symmetric-tag IRLS branch ambiguity.

## §2 Detector matrix — 3840×2160 (50 scenes, single thread)

| Detector | Recall | Prec | Trans p50 | Trans p99 | Rot p50 | Rot p99 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Locus (`high_accuracy`)** | 100 % | 100 % | **0.8 mm** | **37.0 mm** | **0.041°** | **0.267°** | **58.0 ms** |
| Locus (`standard`) | 100 % | 100 % | 6.9 mm | 120.8 mm | 0.302° | 108.409° | 125.4 ms |
| OpenCV (`subpix`) | 100 % | 100 % | 6.5 mm | 133.8 mm | 0.104° | 0.489° | 253.9 ms |
| OpenCV (`apriltag`) | 100 % | 100 % | 5.8 mm | 107.5 mm | 0.072° | 0.449° | 885.9 ms |
| AprilTag-C (pupil) | 100 % | 100 % | 5.7 mm | 101.3 mm | 0.072° | 65.716° | 327.2 ms |

At 4K the OpenCV `apriltag` refinement cost scales badly (886 ms single-thread).

## §3 ICRA 2020 Forward (recall + corner RMSE)

ICRA's small / distant / lower-fidelity tags invert the render-tag result:
OpenCV `apriltag` refinement **rejects** marginal detections, so it is *not*
recall-best here. Per-dataset OpenCV sweep (recall × refinement):

| OpenCV config (5.0) | Recall | Corner RMSE |
| :--- | :---: | :---: |
| `subpix` (recall-best, perim 0.005) | **52.6 %** | 0.979 px |
| `apriltag` (RMSE-best) | 29.95 % | **0.388 px** |

vs the previous OpenCV-4.13 baseline (33.2 % / 0.92 px). The README ICRA row
reports the **recall-best subpix** config (ICRA is a detection-rate benchmark);
`apriltag` refinement more than halves corner RMSE but at 30 % recall.

## Tuned OpenCV configs

- **render-tag `apriltag`** (`OpenCVWrapper._DEFAULTS`): `cornerRefinementMethod=apriltag`,
  `cornerRefinementWinSize=5`, `adaptiveThreshConstant≈9.76`, `adaptiveThreshWinSize=[5,30] step 6`,
  `minMarkerPerimeterRate≈0.022`, `minMarkerDistanceRate≈0.041`, `polygonalApproxAccuracyRate≈0.029`.
- **render-tag `subpix`** (`OpenCVWrapper._SUBPIX_ALT`): `cornerRefinementMethod=subpix`,
  `cornerRefinementWinSize=9`, `adaptiveThreshConstant=10`, `adaptiveThreshWinSize=[4,33] step 9`,
  `minMarkerPerimeterRate=0.019`, `minMarkerDistanceRate=0.028`, `polygonalApproxAccuracyRate=0.023`.
- **ICRA `subpix`**: as above but `minMarkerPerimeterRate=0.005`, `adaptiveThreshConstant=7`.
