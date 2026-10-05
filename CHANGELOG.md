# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Changed

- **The distortion path is one pipeline again, and ~20 % faster.** `decode_batch_soa_with_camera_inner`
  was a second implementation of the decode loop; that duplication was the *mechanism* by which
  every corner-estimator change since #426 reached the pinhole route only. It is gone. The loop
  now holds the decode policy once — scale retries, decode-first ordering with refined-quad
  verification, border-ring budgets, the near-miss recovery search — and routes image reads
  through a `Warp`: the identity for a rectified camera, which monomorphizes to the ROI-cached
  SIMD kernels unchanged, and the lens for a distorted one, whose working plane is the ideal
  plane where a marker's edges are straight. `decoder.rs` is net shorter despite gaining the
  trait, two implementations and a per-candidate sampler.
  - The one genuinely route-dependent operation is corner refinement, because refinement fits
    edges and an edge is straight only in the frame the warp defines. `Warp::refine_seed` and
    `Warp::refine_erf` supply it, so **decode-first ordering now applies on the straight-space
    route too**: refinement runs on the tags that decode (~32 per frame on the Brown-Conrady
    hub) instead of on every surviving candidate (~250).
  - **BREAKING:** `decode_batch_soa_with_camera` takes the frame's `RadialInverseTable`.
    `RoiCache::disabled()` is new.

- **The radial inverse is tabulated once per frame instead of solved per point.** Straight-space
  extraction unprojected 83,623 contour points per frame at 122.7 ns each — 29 % of that
  stage's CPU, almost all of it iteration and division (up to eight radial Newton steps, a 2-D
  polish, a verification evaluation; around nine `f64` divisions). Because the radial forward
  map is odd, `scale(s) = r_u/r_d` is smooth in `s = r_d²`, so `RadialInverseTable` indexes by
  `s` and a lookup needs **no square root and no division**. Tangential terms are then removed
  by a fixed point that reuses the same table, also division-free. Accuracy is *verified at
  build time* at the knot midpoints against a 1e-7 normalized budget (achieved 2.1e-7 px
  Brown-Conrady, 2.0e-5 px Kannala-Brandt), and the table is also structurally safer than the
  solve it replaces: knots are filled outward from the origin along one monotone branch, so a
  lookup cannot reach the mirrored far branch that an unguarded polish converges onto.
  Outside the verified domain lookups fall back to the iterative solve.

### Fixed

- **Every quad edge was being line-fitted twice.** `refine_corner_with_camera` fitted both of a
  corner's edges, so each of the four edges was fitted once as corner `i`'s trailing edge and
  again as corner `i+1`'s leading edge — with *identical arguments*, making the dedupe exact
  rather than an approximation. This was the single most expensive part of distorted extraction,
  48.8 % of its CPU time.

- **The +0.5 px lattice expansion was wrongly gated off for distorted cameras**, on the grounds
  that straight-space RDP corners are "projectively exact intersections, not integer-midpoint
  artifacts". That mistakes where the vertices were *chosen* for what the contour *is*: the
  boundary trace follows the dark side of the outline, and unprojecting a stepped contour is a
  smooth map, which locally is affine and therefore carries the half-pixel bias straight
  through. Under decode-first it stops being cosmetic, since those corners are the quad the bit
  grid is sampled through.

- **The contour was rectified before being simplified.** `chain_approximation` rejects a point
  when its two adjacent segments are exactly collinear — meaningful only on the pixel lattice,
  where adjacent traced points are an integer apart. On the rectified contour the same test
  compared a curvature residual against an absolute epsilon and kept nearly every point. It now
  simplifies first and unprojects the survivors, which is both more correct and much cheaper.
  The reordering also lets this route use the pinhole extractor's cheap compactness pre-gate,
  with area and perimeter both in pixel space so the gate means the same thing at every field
  angle.

- **`high_accuracy` built a full-frame label image on every distorted frame for a consumer that
  can never run.** `may_use_edlines()` did not know about distortion, and EdLines is
  geometrically incompatible with a declared lens.


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

#### Measured

50 frames per hub, `standard`, pose mode Accurate, `--release --features
bench-internals,non_rectified`, `--test-threads=1`, rayon default pool, `RAYON_NUM_THREADS`
unset, AMD EPYC-Milan x86_64 4c/8t via `lscpu` in the same session. Latency is a serialised
A-B-B-A comparison of `Detector::detect` on an otherwise idle host; within-arm spread was
0.07–0.15 ms.

| | before | after |
| :-- | --: | --: |
| Brown-Conrady recall | 89.40 % | **91.39 %** |
| Brown-Conrady precision | 99.671 % | **99.825 %** |
| Brown-Conrady corner RMSE | 0.2335 px | **0.2163 px** |
| Brown-Conrady reprojection RMSE | 0.2246 px | **0.2038 px** |
| Brown-Conrady translation p99 | 0.3989 m | **0.3470 m** |
| Brown-Conrady latency | 17.96 ms | **14.27 ms** |
| Kannala-Brandt precision | 99.145 % | **99.231 %** |
| Kannala-Brandt corner RMSE | 0.3056 px | **0.2847 px** |
| Kannala-Brandt reprojection RMSE | 0.2899 px | **0.2720 px** |
| Kannala-Brandt translation p99 | 0.0439 m | **0.0347 m** |
| Kannala-Brandt rotation p99 | 1.2669 deg | **1.0241 deg** |
| Kannala-Brandt latency | 16.50 ms | **13.55 ms** |

Every pinhole regression suite is **bit-identical** throughout: `regression_render_tag`,
`regression_render_tag_robustness`, `regression_board_hub`, `regression_icra2020`,
`regression_euroc`, `regression_pose_consistency_roc`, `regression_straight_space`.

Not improved, and recorded rather than hidden: mean Hamming rose (Brown-Conrady 0.0196 →
0.1053, Kannala-Brandt 0.0069 → 0.0557) and Brown-Conrady's rotation p90/p99 rose (0.2687 →
0.2977 deg, 98.57 → 103.01 deg). About half of the Hamming rise is the shared acceptance policy
— returning on the first match that clears its budget rather than searching all three scale
retries for the lowest-Hamming one, worth 0.0182 → 0.0844 on its own at unchanged recall and
corner accuracy — and keeping a cross-scale best for one route only would reintroduce exactly
the divergence this work removed. The rest, and the rotation percentiles, are consistent with
composition: corner RMSE *improved* 7 % while recall rose 2 pp, so the newly recovered tags are
not degrading the corner population, and rotation is this pipeline's ill-conditioned degree of
freedom at the high incidence angles those tags sit at. Proving that needs per-tag matching
between the two arms, which the harness does not expose; adding it is the follow-up, not an
assertion.

Latency is now dominated by a stage the lens never touches: segmentation is 8.59 ms of
Brown-Conrady's 13.25 ms of instrumented spans (64.9 %), against quad extraction at 2.14 ms
(down from 5.49 ms) and decode at 1.43 ms. Further distortion-path latency work has little left
to take.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.9.0](docs/changelogs/v0.9.0.md) - 2026-10-04
- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
