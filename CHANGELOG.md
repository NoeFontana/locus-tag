# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Fixed

- **`regression_distortion_hub` measured the pinhole detector on fisheye frames.** The harness's
  `build_intrinsics` falls through to `CameraIntrinsics::new` (no distortion) whenever the
  declared model cannot be represented, which is exactly what happens without
  `--features non_rectified`, since the `DistortionCoeffs` variants are themselves feature-gated.
  The test target did not require that feature, so the committed baselines were blessed from
  distortion-blind runs and the entire distortion path — straight-space quad extraction, the
  distortion-aware pose LMs, the `AdaptivePpb` fallback — had **no regression coverage at all**.
  Running the suite without the feature reproduces the committed snapshots to every digit, and
  passes. Every re-blessing to date (#412, #432, and the 0.9.0 line) recorded pinhole numbers.
  - The target now declares `required-features = ["non_rectified"]`, so it disappears instead of
    blessing a meaningless baseline, and the suite asserts its own premise — checked per frame,
    against `gt.intrinsics.or(options.intrinsics)`, which is what the detector is actually
    handed. Asserting on the dataset-level fallback alone would pass while every frame ran
    pinhole.
  - Both baselines are re-blessed from true distortion-path runs. Reading the old numbers as
    evidence about distorted detection is what made the "reprojection RMSE is worse than corner
    RMSE under distortion" anomaly look like a geometry defect; it was a pinhole pose fitted to
    distorted observations. The reproj/corner ratio on the real path is ~0.96, not 2.4
    (Brown-Conrady) or 8.6 (Kannala-Brandt).
- **The distortion-aware decode route ran no corner refinement at all.**
  `decode_batch_soa_with_camera_inner` is a second implementation of the decode loop, so its tail
  never received the gradient-orthogonality corner pass (#426, #430, #431, #432), the
  marker-inset photometric calibration (#434), or the frame-clipping check (#436). Toggling
  `decoder.corner_subpix` under declared distortion was a no-op. Every pass involved is a local
  photometric or geometric test on the raw image — none needs a camera model, and a lens cannot
  make a corner stop being a corner.
  - Both loops now share one `finalize_decoded_candidate` (sub-pixel pass, marker-inset
    calibration, frame-clipping check, rotation reorder of the `corner_refined` bits, homography
    recompute). The duplication *was* the defect's mechanism, so removing it is the fix: a new
    corner pass added there now reaches both routes.
  - The distorted route also sizes its sub-pixel window from the decoder that actually matched
    (`cells` travels with the accepted match), rather than a conservative minimum over every
    registered family.
  - **Run:** `cargo test --release --features bench-internals,non_rectified --test
    regression_distortion_hub -- --test-threads=1`, profile `standard`, **pose mode Accurate**
    (the harness supplies intrinsics + `tag_size`), 50 frames per hub, `rayon` default pool,
    `RAYON_NUM_THREADS` unset. Host: AMD EPYC-Milan, x86_64, 4 cores / 8 threads (`lscpu`, same
    session). Accuracy figures are deterministic and host-independent.
  - **Comparison base is the distortion path with the shared tail inert vs active** — *not* the
    previously committed baseline, which was a pinhole run and is not comparable:

    | | corner RMSE | reproj RMSE | trans p50 | rot p50 |
    | :-- | --: | --: | --: | --: |
    | Brown-Conrady, before | 0.957 px | 0.938 px | 16.8 mm | 0.305° |
    | Brown-Conrady, after | **0.234 px** | **0.225 px** | **0.4 mm** | **0.059°** |
    | Kannala-Brandt, before | 1.257 px | 1.249 px | 6.0 mm | 0.260° |
    | Kannala-Brandt, after | **0.306 px** | **0.290 px** | **0.3 mm** | **0.053°** |

  - Recall and precision are unchanged to four decimals; the pinhole path is untouched by
    construction, and its snapshots are byte-identical. Latency cost of the tail on this route,
    one run per arm: Brown-Conrady 16.19 → 16.59 ms per frame, Kannala-Brandt 14.48 → 15.34 ms —
    **2-6 %**, with a single sample per arm, so treat the spread as indicative.
  - **Still open, and pinned rather than hidden** (an `insta` baseline fails on any movement, so
    a further regression in either is caught): the distortion route remains behind the pinhole
    route on Brown-Conrady recall (89.4 % vs 98.4 %) and corners (0.234 vs 0.142 px) on the same
    frames, while winning decisively on pose (trans p50 0.4 vs 37.1 mm, rot p50 0.059° vs
    1.72°); and Brown-Conrady keeps a ~1 % catastrophic tail (rot p99 98.6°, trans p99 0.40 m)
    with the planar two-fold-ambiguity signature, which Kannala-Brandt's wider field of view does
    not show (rot p99 1.27°). The decode-first work of #433/#435 also landed on the generic loop
    only, which is the next instalment of this unification.
- **Brown-Conrady `undistort` had not converged.** The documented "Newton refinement" was a
  fixed-point iteration (`xu ← (xd − dx(xu))/radial(xu)`), only linearly convergent, capped at
  5 steps. At the shipped hub dataset's own coefficients (`k1 = -0.28`, `k2 = 0.08`) it left
  **0.42 px** of round-trip error at the image corner — above `quad.rs`'s undistort gate, which
  silently discarded every candidate whose contour touched ~1.8 % of the frame. Replaced by a 1-D
  Newton solve of the monotone radial problem followed by a 2-D Newton polish against the
  analytic Jacobian: worst-case **1.4e-9 px** over the frame (the polish's deliberate residual
  floor, five orders under the gate it feeds), in 1-2 polish steps.
  - **Out-of-domain radii are rejected, not silently inverted onto the wrong branch.** Past the
    model's radial turning point `r_d` has no preimage on the invertible branch, but the full map
    still has the mirrored one (`g(−r) = −g(r)`), which re-distorts to ~1e-14 and would therefore
    *pass* the caller's residual gate. A first cut of this fix did exactly that (`k1 = −0.1`,
    `r_d = 1.335` returned `r_u = −3.69`, residual 4e-14, accepted — where the superseded fixed
    point left residual 0.5 and was rejected). The radial solve now restores the last on-branch
    iterate and reports non-convergence, and the polish runs only on a converged solve.
  - The polish is skipped outright when `p1 == p2 == 0`, where the radial solve is already exact
    to 2e-16 — the common calibration case. The forward map and its Jacobian are evaluated
    together so the radial monomials are computed once per step, and the Jacobian is symmetric
    (`∂xd/∂yn == ∂yd/∂xn`, now asserted), so the 2x2 solve uses `det = a·d − b²`.
  - The model unit tests previously probed only `r ≤ 0.5` with hand-picked coefficients. They now
    also pin both inverters and both Jacobians at the shipped datasets' real operating point —
    `r_d = 0.80` for Brown-Conrady, `θ = 85.1°` for Kannala-Brandt — using central differences
    with a radius-scaled step, and the tuning constants are named rather than inline (the step
    tolerance is now *relative*: an absolute `1e-15` sits below 1 ulp for `r_u > 4.5`, so the
    loop always burned its full iteration budget on wide-angle coordinates).
- **Unprojection could fail silently, and three consumers trusted it anyway.**
  `CameraModel::undistort` cannot fail by signature, so outside a model's invertible domain it
  returns a point that is not a preimage of its argument. Only `quad.rs` re-distorted and
  checked; the IPPE seed of the single-tag pose path (`pose.rs`), the ideal homography the whole
  distorted decode hangs off (`decoder.rs`), and the board DLT/IPPE seed (`board.rs`) all used
  `undistort_pixel` and could not tell. New `CameraModel::undistort_checked` and
  `CameraIntrinsics::undistort_pixel_checked` perform the round trip — rejecting non-finite
  residuals explicitly, since a NaN fails every comparison and a bare `>` would *admit* them —
  and the three consumers now handle failure: no pose, a `FailedDecode` candidate, and a skipped
  correspondence respectively, rather than geometry built on a point the lens model cannot
  explain. The residual budget has one definition (`camera::MAX_UNDISTORT_RESIDUAL`) instead of
  living in `quad.rs`.
  - Identity, and so always `Some`, for `DistortionCoeffs::None`: the pinhole path is unchanged.
  - `camera::MAX_UNDISTORT_RESIDUAL` is public: it is the bound the default
    `undistort_checked` applies, so it is observable contract for anyone implementing
    `CameraModel` rather than a private tuning knob.
- **No `regression_*` suite ran in CI.** `.config/nextest.toml`'s `default-filter` excludes
  `binary(~regression)`, the `ci` profile inherits it, and `cargo insta test` defaults to
  `--test-runner auto`, which picks the installed nextest and inherits the same filter. Most of
  those suites need a dataset, but `regression_straight_space` renders its own fisheye and
  polynomial scenes, so it is a free gate — and it is the only CI coverage of the
  distortion-aware extraction and decode routes. It now runs as its own step, and it gained a
  test asserting the sub-pixel pass is live on the distorted route (mutation-checked: disabling
  the pass fails it).

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.9.0](docs/changelogs/v0.9.0.md) - 2026-10-04
- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
