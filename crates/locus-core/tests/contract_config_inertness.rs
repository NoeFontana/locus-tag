//! Contract: **no `DetectorConfig` field is silently inert**.
//!
//! Every field is mutated away from its default on a small deterministic
//! synthetic frame set, and the *observable* detector output — decoded
//! detections, rejected candidates, poses, covariances **and** the debug
//! telemetry maps — is required to change.
//!
//! A field that provably cannot change the output is listed in
//! [`FieldCase::inert_reason`] with a written justification. The test asserts
//! those produce *identical* output, so the allowlist is a two-way contract:
//! wiring an allowlisted field up (or letting a live field go dead) fails the
//! build until the entry is updated.
//!
//! The field list is derived from an exhaustive `let`-destructuring of
//! `DetectorConfig`, so adding a field to the struct breaks compilation here
//! until a case is written for it.
//!
//! Requires `--features bench-internals` for the dictionary accessors used by
//! the synthetic tag renderer.
#![cfg(feature = "bench-internals")]
#![allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::field_reassign_with_default,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::unwrap_used,
    missing_docs
)]

use locus_core::bench_api::family_to_decoder;
use locus_core::config::{
    AdaptivePpbConfig, CornerRefinementMode, DetectorConfig, EdLinesImbalanceGatePolicy,
    QuadExtractionMode, QuadExtractionPolicy, SegmentationConnectivity, TagFamily,
};
use locus_core::{CameraIntrinsics, DetectorBuilder, ImageView};

// ============================================================================
// Deterministic synthetic scene rendering
// ============================================================================

const CANVAS: usize = 208;

/// Tiny deterministic LCG. `rand` is a `bench-internals`-only dep and seeds
/// itself from the OS; this test needs frames that are byte-identical on every
/// run so that a field's effect (and not the noise) is what is measured.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// Signed noise in `[-amp, amp]`.
    fn noise(&mut self, amp: i32) -> i32 {
        (self.next_u32() % (2 * amp as u32 + 1)) as i32 - amp
    }
}

/// Canonical `(dim + 2) x (dim + 2)` cell grid for a tag: `true` = white.
fn tag_cells(family: TagFamily, id: u16) -> (usize, Vec<bool>) {
    let decoder = family_to_decoder(family);
    let dim = decoder.dimension();
    let outer = dim + 2;
    // Border is black; interior defaults to black and is filled from the code.
    let mut cells = vec![false; outer * outer];
    let code = decoder.get_code(id).expect("valid tag id for family");
    let d_f = outer as f64;
    for (i, p) in decoder.sample_points().iter().enumerate() {
        let gx = ((p.0 + 1.0) * d_f / 2.0 - 0.5).round() as usize;
        let gy = ((p.1 + 1.0) * d_f / 2.0 - 0.5).round() as usize;
        cells[gy * outer + gx] = (code >> i) & 1 != 0;
    }
    (outer, cells)
}

/// Solve the 8-DoF homography mapping the unit square onto `quad`
/// (TL, TR, BR, BL), via the standard normalized DLT-free closed form.
fn unit_square_to_quad(quad: &[[f64; 2]; 4]) -> [[f64; 3]; 3] {
    let (x0, y0) = (quad[0][0], quad[0][1]);
    let (x1, y1) = (quad[1][0], quad[1][1]);
    let (x2, y2) = (quad[2][0], quad[2][1]);
    let (x3, y3) = (quad[3][0], quad[3][1]);

    let dx1 = x1 - x2;
    let dx2 = x3 - x2;
    let dx3 = x0 - x1 + x2 - x3;
    let dy1 = y1 - y2;
    let dy2 = y3 - y2;
    let dy3 = y0 - y1 + y2 - y3;

    let den = dx1 * dy2 - dx2 * dy1;
    let g = (dx3 * dy2 - dx2 * dy3) / den;
    let h = (dx1 * dy3 - dx3 * dy1) / den;

    [
        [x1 - x0 + g * x1, x3 - x0 + h * x3, x0],
        [y1 - y0 + g * y1, y3 - y0 + h * y3, y0],
        [g, h, 1.0],
    ]
}

fn invert3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    let inv_det = 1.0 / det;
    [
        [
            (b[1] * c[2] - b[2] * c[1]) * inv_det,
            (a[2] * c[1] - a[1] * c[2]) * inv_det,
            (a[1] * b[2] - a[2] * b[1]) * inv_det,
        ],
        [
            (b[2] * c[0] - b[0] * c[2]) * inv_det,
            (a[0] * c[2] - a[2] * c[0]) * inv_det,
            (a[2] * b[0] - a[0] * b[2]) * inv_det,
        ],
        [
            (b[0] * c[1] - b[1] * c[0]) * inv_det,
            (a[1] * c[0] - a[0] * c[1]) * inv_det,
            (a[0] * b[1] - a[1] * b[0]) * inv_det,
        ],
    ]
}

/// A scene: one tag warped onto an arbitrary quad, over a mildly textured
/// background, rendered with 3x3 supersampling so edges land sub-pixel.
struct SceneSpec {
    family: TagFamily,
    id: u16,
    /// Tag outer-border corners in image space (TL, TR, BR, BL).
    quad: [[f64; 2]; 4],
    /// Intensity of the tag's "white" cells and of the page it sits on.
    white: u8,
    /// Intensity of the tag's "black" cells.
    black: u8,
    /// Peak-to-peak background texture amplitude.
    texture: i32,
    /// Additive per-pixel noise amplitude.
    noise: i32,
    /// Cells (gx, gy) to invert — used to inject decodable bit errors.
    corrupt: &'static [(usize, usize)],
    /// Stamp a small black square over the tag's centroid.
    center_dot: bool,
}

fn render(spec: &SceneSpec, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut img = vec![0u8; CANVAS * CANVAS];

    // Background: white page with a slow diagonal ramp plus fixed-pattern noise.
    for y in 0..CANVAS {
        for x in 0..CANVAS {
            let ramp = (x + y) as i32 * spec.texture / (2 * CANVAS as i32);
            let v = i32::from(spec.white) - ramp + rng.noise(spec.noise);
            img[y * CANVAS + x] = v.clamp(0, 255) as u8;
        }
    }

    let (outer, mut cells) = tag_cells(spec.family, spec.id);
    for &(gx, gy) in spec.corrupt {
        let idx = gy * outer + gx;
        cells[idx] = !cells[idx];
    }

    let h_inv = invert3(&unit_square_to_quad(&spec.quad));

    // Bounding box of the quad, clipped to the canvas.
    let min_x = spec
        .quad
        .iter()
        .map(|p| p[0])
        .fold(f64::MAX, f64::min)
        .floor() as isize;
    let max_x = spec
        .quad
        .iter()
        .map(|p| p[0])
        .fold(f64::MIN, f64::max)
        .ceil() as isize;
    let min_y = spec
        .quad
        .iter()
        .map(|p| p[1])
        .fold(f64::MAX, f64::min)
        .floor() as isize;
    let max_y = spec
        .quad
        .iter()
        .map(|p| p[1])
        .fold(f64::MIN, f64::max)
        .ceil() as isize;

    const SS: usize = 3;
    for py in min_y.max(0)..=max_y.min(CANVAS as isize - 1) {
        for px in min_x.max(0)..=max_x.min(CANVAS as isize - 1) {
            let mut acc = 0i32;
            let mut hits = 0i32;
            for sy in 0..SS {
                for sx in 0..SS {
                    let ix = px as f64 + (sx as f64 + 0.5) / SS as f64;
                    let iy = py as f64 + (sy as f64 + 0.5) / SS as f64;
                    let w = h_inv[2][0] * ix + h_inv[2][1] * iy + h_inv[2][2];
                    let u = (h_inv[0][0] * ix + h_inv[0][1] * iy + h_inv[0][2]) / w;
                    let v = (h_inv[1][0] * ix + h_inv[1][1] * iy + h_inv[1][2]) / w;
                    if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                        continue;
                    }
                    let gx = (u * outer as f64) as usize;
                    let gy = (v * outer as f64) as usize;
                    acc += i32::from(if cells[gy * outer + gx] {
                        spec.white
                    } else {
                        spec.black
                    });
                    hits += 1;
                }
            }
            if hits == 0 {
                continue;
            }
            let covered = f64::from(hits) / (SS * SS) as f64;
            let idx = py as usize * CANVAS + px as usize;
            let tag = f64::from(acc) / f64::from(hits);
            let bg = f64::from(img[idx]);
            img[idx] = (tag * covered + bg * (1.0 - covered))
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }

    if spec.center_dot {
        let cx = spec.quad.iter().map(|p| p[0]).sum::<f64>() / 4.0;
        let cy = spec.quad.iter().map(|p| p[1]).sum::<f64>() / 4.0;
        for y in (cy as usize - 3)..(cy as usize + 3) {
            for x in (cx as usize - 3)..(cx as usize + 3) {
                img[y * CANVAS + x] = 0;
            }
        }
    }
    img
}

/// The frame set, rendered once. Deliberately heterogeneous so that
/// gate-shaped fields (aspect ratio, fill, elongation, hamming budget,
/// contrast) all have at least one frame that sits near their decision
/// boundary.
fn frames() -> &'static [Vec<u8>] {
    static FRAMES: std::sync::OnceLock<Vec<Vec<u8>>> = std::sync::OnceLock::new();
    FRAMES.get_or_init(render_frames)
}

fn render_frames() -> Vec<Vec<u8>> {
    let specs = [
        // 0: large, near-frontal, clean — the easy baseline.
        SceneSpec {
            family: TagFamily::AprilTag36h11,
            id: 7,
            quad: [[38.0, 34.5], [170.5, 36.0], [172.0, 168.5], [36.5, 170.0]],
            white: 235,
            black: 18,
            texture: 40,
            noise: 3,
            corrupt: &[],
            center_dot: false,
        },
        // 1: strongly oblique (non-degenerate pose, anisotropic corner
        //    covariance, large reprojection residuals).
        SceneSpec {
            family: TagFamily::AprilTag36h11,
            id: 23,
            quad: [[30.0, 52.0], [176.0, 22.0], [168.0, 186.0], [46.0, 146.0]],
            white: 228,
            black: 24,
            texture: 55,
            noise: 5,
            corrupt: &[],
            center_dot: false,
        },
        // 2: small + low contrast + noisy — near the decode/contrast floor.
        SceneSpec {
            family: TagFamily::AprilTag36h11,
            id: 41,
            quad: [[74.0, 78.0], [134.5, 76.0], [136.0, 136.5], [72.5, 138.0]],
            white: 150,
            black: 92,
            texture: 30,
            noise: 6,
            corrupt: &[],
            center_dot: false,
        },
        // 3: two bit errors — only decodes while the Hamming budget allows it.
        SceneSpec {
            family: TagFamily::AprilTag36h11,
            id: 3,
            quad: [[44.0, 44.0], [164.5, 42.0], [166.0, 162.5], [42.5, 164.0]],
            white: 232,
            black: 20,
            texture: 45,
            noise: 4,
            corrupt: &[(2, 2), (5, 4)],
            center_dot: false,
        },
        // 4: the funnel-gate probe. See `FUNNEL_PROBE` below.
        FUNNEL_PROBE,
    ];
    specs
        .iter()
        .enumerate()
        .map(|(i, s)| render(s, 0x5EED_0000 + i as u64))
        .collect()
}

/// A frame built specifically for the funnel's contrast gate,
/// `tau = clamp(min(0.1 * local_range, 0.5 * decoder_min_contrast), 2.0, ..)`.
///
/// `tau` responds to `decoder_min_contrast` only while `0.5 * min_contrast` is
/// below `0.1 * local_range`, and `local_range` is read from the tile under the
/// *candidate's centroid*. An ordinary tag has both a huge `local_range` and a
/// boundary contrast around 190 — far above every reachable `tau` — so the knob
/// is unobservable on clean frames however it is set.
///
/// This probe decouples the two. The tag is rendered at 20 grey levels of
/// contrast (205 on 225), which puts its boundary contrast just above the
/// default `tau = 10` and below the saturated `tau = 0.1 * local_range`; a
/// black dot stamped on the centroid keeps `local_range` at the full 225 so the
/// saturation point stays high.
const FUNNEL_PROBE: SceneSpec = SceneSpec {
    family: TagFamily::AprilTag36h11,
    id: 17,
    quad: [[52.0, 52.0], [156.0, 52.0], [156.0, 156.0], [52.0, 156.0]],
    white: 225,
    black: 205,
    texture: 0,
    noise: 0,
    corrupt: &[],
    center_dot: true,
};

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(320.0, 320.0, CANVAS as f64 / 2.0, CANVAS as f64 / 2.0)
}

const TAG_SIZE_M: f64 = 0.1;

// ============================================================================
// Output signature
// ============================================================================

/// FNV-1a over every observable output byte: detections, rejected candidates,
/// poses, covariances and the two telemetry images.
struct Hasher(u64);

impl Hasher {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }
    fn f32(&mut self, v: f32) {
        // Canonicalise NaN so an unrelated NaN payload cannot masquerade as a
        // behavioural difference.
        let v = if v.is_nan() { f32::NAN } else { v };
        self.write(&v.to_bits().to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.write(&(v as u64).to_le_bytes());
    }
}

fn signature(config: DetectorConfig) -> u64 {
    let mut detector = DetectorBuilder::new()
        .with_config(config)
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let k = intrinsics();
    let mut h = Hasher::new();

    for frame in frames() {
        let img = ImageView::new(frame, CANVAS, CANVAS, CANVAS).expect("valid image view");
        let view = detector
            .detect(&img, Some(&k), Some(TAG_SIZE_M), true)
            .expect("detection must not fail on the synthetic frame set");

        h.usize(view.ids.len());
        for (i, &id) in view.ids.iter().enumerate() {
            h.write(&id.to_le_bytes());
            h.write(&view.payloads[i].to_le_bytes());
            h.f32(view.error_rates[i]);
            for c in &view.corners[i] {
                h.f32(c.x);
                h.f32(c.y);
            }
            h.write(
                &view.homographies[i]
                    .data
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            for v in view.poses[i].data {
                h.f32(v);
            }
            for v in view.corner_covariances[i] {
                h.f32(v);
            }
        }

        h.usize(view.rejected_corners.len());
        for (i, quad) in view.rejected_corners.iter().enumerate() {
            for c in quad {
                h.f32(c.x);
                h.f32(c.y);
            }
            h.f32(view.rejected_error_rates[i]);
            h.write(&[view.rejected_funnel_status[i] as u8]);
        }

        // Telemetry maps: `binarized` is the only consumer of the tile-validity
        // mask, so a telemetry-scoped field is still observable here.
        let telem = view.telemetry.expect("telemetry requested");
        let n = telem.height * telem.stride;
        // SAFETY: `telem.binarized_ptr` / `threshold_map_ptr` are arena slices
        // of exactly `height * stride` bytes, written by
        // `apply_threshold_with_map` earlier this frame. `view` borrows the
        // context, so the arena outlives these reads.
        let (bin, thr) = unsafe {
            (
                std::slice::from_raw_parts(telem.binarized_ptr, n),
                std::slice::from_raw_parts(telem.threshold_map_ptr, n),
            )
        };
        h.write(bin);
        h.write(thr);
    }
    h.0
}

// ============================================================================
// Field cases
// ============================================================================

struct FieldCase {
    field: &'static str,
    /// Prerequisite settings that put the field on a reachable code path.
    /// Applied to *both* sides of the comparison.
    base: fn(&mut DetectorConfig),
    /// The mutation under test.
    mutate: fn(&mut DetectorConfig),
    /// `Some(reason)` asserts the field is provably inert.
    inert_reason: Option<&'static str>,
}

fn noop(_: &mut DetectorConfig) {}

/// Prerequisite: a χ² pose-consistency gate that is *tight enough to bite* on
/// a clean synthetic tag. The default σ (1.0 px) and the default branch-ratio
/// escape (5.0) both let a well-fitted pose through no matter what the FPR is,
/// so a case that only raises `pose_consistency_fpr` would measure nothing.
fn base_gate_armed(c: &mut DetectorConfig) {
    c.pose_consistency_fpr = 0.5;
    c.pose_consistency_gate_sigma_px = 0.02;
}

/// Prerequisite for `pose_consistency_fpr`: everything the gate needs *except*
/// the FPR that arms it.
fn base_gate_tuned_but_disarmed(c: &mut DetectorConfig) {
    c.pose_consistency_gate_sigma_px = 0.02;
    c.pose_consistency_min_decisive_ratio = f64::INFINITY;
}

/// Prerequisite: EdLines extraction (incompatible with Erf refinement).
fn base_edlines(c: &mut DetectorConfig) {
    c.refinement_mode = CornerRefinementMode::None;
    c.quad_extraction_mode = QuadExtractionMode::EdLines;
}

/// Prerequisite: GWLF refinement.
fn base_gwlf(c: &mut DetectorConfig) {
    c.refinement_mode = CornerRefinementMode::Gwlf;
}

/// Reason shared by the four integral/gradient-window thresholder knobs.
const INTEGRAL_THRESHOLDER_REASON: &str = concat!(
    "Read only by `threshold::adaptive_threshold_gradient_window` / ",
    "`adaptive_threshold_integral`, whose only callers are ",
    "`benches/integral_threshold_bench.rs`. The shipped pipeline runs the ",
    "tile thresholder (`apply_threshold_with_map`), so the value cannot reach ",
    "detection or telemetry. Wiring-or-removing is owned by the ",
    "robust-threshold work; this entry exists so the decision cannot be ",
    "forgotten -- it fails the moment the field becomes live."
);

const HUBER_DELTA_REASON: &str = concat!(
    "Unreachable from `Detector::detect`. It parametrises the *unweighted* ",
    "LM (`pose::refine_pose_lm`), which runs only when ",
    "`build_lm_covariances` yields `None` -- i.e. when no image is supplied. ",
    "The pipeline always passes `refinement_img`, so the structure-tensor ",
    "weighted LM (`pose_weighted::refine_pose_lm_weighted`) takes every ",
    "detection. The field is still live for direct callers of ",
    "`pose::estimate_tag_pose_with_config(.., img = None, ..)`, which is why ",
    "it is allowlisted rather than removed; deciding its fate belongs to the ",
    "pose solver, not to a config-hygiene change."
);

const NTHREADS_REASON: &str = concat!(
    "Deliberately output-invariant. `nthreads` selects the scoped Rayon pool ",
    "for intra-frame work (`LocusEngine::run_scoped`); every parallel stage ",
    "writes disjoint index-addressed chunks, so results must be bit-identical ",
    "across thread counts. This entry is the determinism guarantee. That the ",
    "field is *live* is proven by `nthreads_selects_the_pipeline_pool`."
);

fn cases() -> Vec<FieldCase> {
    vec![
        FieldCase {
            field: "threshold_tile_size",
            base: noop,
            mutate: |c| c.threshold_tile_size = 16,
            inert_reason: None,
        },
        FieldCase {
            field: "threshold_min_range",
            base: noop,
            mutate: |c| c.threshold_min_range = 0,
            inert_reason: None,
        },
        FieldCase {
            field: "enable_sharpening",
            base: noop,
            mutate: |c| c.enable_sharpening = true,
            inert_reason: None,
        },
        FieldCase {
            field: "threshold_min_radius",
            base: noop,
            mutate: |c| c.threshold_min_radius = 6,
            inert_reason: Some(INTEGRAL_THRESHOLDER_REASON),
        },
        FieldCase {
            field: "threshold_max_radius",
            base: noop,
            mutate: |c| c.threshold_max_radius = 31,
            inert_reason: Some(INTEGRAL_THRESHOLDER_REASON),
        },
        FieldCase {
            field: "adaptive_threshold_constant",
            base: noop,
            mutate: |c| c.adaptive_threshold_constant = 60,
            inert_reason: Some(INTEGRAL_THRESHOLDER_REASON),
        },
        FieldCase {
            field: "adaptive_threshold_gradient_threshold",
            base: noop,
            mutate: |c| c.adaptive_threshold_gradient_threshold = 200,
            inert_reason: Some(INTEGRAL_THRESHOLDER_REASON),
        },
        FieldCase {
            field: "quad_min_area",
            base: noop,
            mutate: |c| c.quad_min_area = 40_000,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_max_aspect_ratio",
            base: noop,
            mutate: |c| c.quad_max_aspect_ratio = 1.05,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_min_fill_ratio",
            base: noop,
            mutate: |c| c.quad_min_fill_ratio = 0.95,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_max_fill_ratio",
            base: noop,
            mutate: |c| c.quad_max_fill_ratio = 0.11,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_min_edge_length",
            base: noop,
            mutate: |c| c.quad_min_edge_length = 400.0,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_min_edge_score",
            base: noop,
            mutate: |c| c.quad_min_edge_score = 1.0e9,
            inert_reason: None,
        },
        FieldCase {
            field: "subpixel_refinement_sigma",
            base: noop,
            mutate: |c| c.subpixel_refinement_sigma = 2.5,
            inert_reason: None,
        },
        FieldCase {
            field: "segmentation_connectivity",
            base: noop,
            mutate: |c| c.segmentation_connectivity = SegmentationConnectivity::Four,
            inert_reason: None,
        },
        FieldCase {
            field: "upscale_factor",
            base: noop,
            mutate: |c| c.upscale_factor = 2,
            inert_reason: None,
        },
        FieldCase {
            field: "decimation",
            base: noop,
            mutate: |c| c.decimation = 2,
            inert_reason: None,
        },
        FieldCase {
            field: "nthreads",
            base: noop,
            mutate: |c| c.nthreads = 3,
            inert_reason: Some(NTHREADS_REASON),
        },
        FieldCase {
            field: "decoder_min_contrast",
            base: noop,
            // Raising the knob raises `tau` until it saturates against
            // `0.1 * local_range`. Observable only on the `FUNNEL_PROBE`
            // frame, which is built for exactly this gate.
            mutate: |c| c.decoder_min_contrast = 250.0,
            inert_reason: None,
        },
        FieldCase {
            field: "refinement_mode",
            base: noop,
            mutate: |c| c.refinement_mode = CornerRefinementMode::None,
            inert_reason: None,
        },
        FieldCase {
            field: "max_hamming_error",
            base: noop,
            mutate: |c| c.max_hamming_error = Some(0),
            inert_reason: None,
        },
        FieldCase {
            field: "huber_delta_px",
            base: noop,
            mutate: |c| c.huber_delta_px = 0.02,
            inert_reason: Some(HUBER_DELTA_REASON),
        },
        FieldCase {
            field: "tikhonov_alpha_max",
            base: noop,
            mutate: |c| c.tikhonov_alpha_max = 80.0,
            inert_reason: None,
        },
        FieldCase {
            field: "sigma_n_sq",
            base: noop,
            mutate: |c| c.sigma_n_sq = 400.0,
            inert_reason: None,
        },
        FieldCase {
            field: "structure_tensor_radius",
            base: noop,
            mutate: |c| c.structure_tensor_radius = 6,
            inert_reason: None,
        },
        FieldCase {
            field: "pose_consistency_fpr",
            base: base_gate_tuned_but_disarmed,
            mutate: |c| c.pose_consistency_fpr = 0.5,
            inert_reason: None,
        },
        FieldCase {
            field: "pose_consistency_gate_sigma_px",
            base: |c| {
                c.pose_consistency_fpr = 0.5;
                c.pose_consistency_min_decisive_ratio = f64::INFINITY;
            },
            mutate: |c| c.pose_consistency_gate_sigma_px = 0.02,
            inert_reason: None,
        },
        FieldCase {
            field: "pose_consistency_min_decisive_ratio",
            base: base_gate_armed,
            // The default 5.0 lets the escape clause fire on every clean
            // synthetic tag (the IPPE branch choice is decisive), so the armed
            // gate never rejects. INFINITY disables the escape and the gate
            // bites.
            mutate: |c| c.pose_consistency_min_decisive_ratio = f64::INFINITY,
            inert_reason: None,
        },
        FieldCase {
            field: "outlier_drop_d2_threshold",
            base: noop,
            mutate: |c| c.outlier_drop_d2_threshold = 1.0e-6,
            inert_reason: None,
        },
        FieldCase {
            field: "pose_edge_refinement_enabled",
            base: noop,
            mutate: |c| c.pose_edge_refinement_enabled = true,
            inert_reason: None,
        },
        FieldCase {
            field: "gwlf_transversal_alpha",
            base: base_gwlf,
            mutate: |c| c.gwlf_transversal_alpha = 0.6,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_max_elongation",
            base: noop,
            mutate: |c| c.quad_max_elongation = 1.0,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_min_density",
            base: noop,
            mutate: |c| c.quad_min_density = 0.999,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_extraction_mode",
            base: |c| c.refinement_mode = CornerRefinementMode::None,
            mutate: |c| c.quad_extraction_mode = QuadExtractionMode::EdLines,
            inert_reason: None,
        },
        FieldCase {
            field: "edlines_imbalance_gate",
            base: base_edlines,
            mutate: |c| c.edlines_imbalance_gate = EdLinesImbalanceGatePolicy::Enabled,
            inert_reason: None,
        },
        FieldCase {
            field: "quad_extraction_policy",
            base: noop,
            mutate: |c| {
                c.quad_extraction_policy = QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
                    threshold: 3.0,
                    low_extraction: QuadExtractionMode::ContourRdp,
                    high_extraction: QuadExtractionMode::EdLines,
                    low_refinement: CornerRefinementMode::Erf,
                    high_refinement: CornerRefinementMode::None,
                });
            },
            inert_reason: None,
        },
    ]
}

/// Exhaustive destructuring: adding a field to `DetectorConfig` without adding
/// a `FieldCase` fails to compile here (no `..` rest pattern).
fn every_config_field() -> Vec<&'static str> {
    let DetectorConfig {
        threshold_tile_size: _,
        threshold_min_range: _,
        enable_sharpening: _,
        threshold_min_radius: _,
        threshold_max_radius: _,
        adaptive_threshold_constant: _,
        adaptive_threshold_gradient_threshold: _,
        quad_min_area: _,
        quad_max_aspect_ratio: _,
        quad_min_fill_ratio: _,
        quad_max_fill_ratio: _,
        quad_min_edge_length: _,
        quad_min_edge_score: _,
        subpixel_refinement_sigma: _,
        segmentation_connectivity: _,
        upscale_factor: _,
        decimation: _,
        nthreads: _,
        decoder_min_contrast: _,
        refinement_mode: _,
        max_hamming_error: _,
        huber_delta_px: _,
        tikhonov_alpha_max: _,
        sigma_n_sq: _,
        structure_tensor_radius: _,
        pose_consistency_fpr: _,
        pose_consistency_gate_sigma_px: _,
        pose_consistency_min_decisive_ratio: _,
        outlier_drop_d2_threshold: _,
        pose_edge_refinement_enabled: _,
        gwlf_transversal_alpha: _,
        quad_max_elongation: _,
        quad_min_density: _,
        quad_extraction_mode: _,
        edlines_imbalance_gate: _,
        quad_extraction_policy: _,
    } = DetectorConfig::default();

    vec![
        "threshold_tile_size",
        "threshold_min_range",
        "enable_sharpening",
        "threshold_min_radius",
        "threshold_max_radius",
        "adaptive_threshold_constant",
        "adaptive_threshold_gradient_threshold",
        "quad_min_area",
        "quad_max_aspect_ratio",
        "quad_min_fill_ratio",
        "quad_max_fill_ratio",
        "quad_min_edge_length",
        "quad_min_edge_score",
        "subpixel_refinement_sigma",
        "segmentation_connectivity",
        "upscale_factor",
        "decimation",
        "nthreads",
        "decoder_min_contrast",
        "refinement_mode",
        "max_hamming_error",
        "huber_delta_px",
        "tikhonov_alpha_max",
        "sigma_n_sq",
        "structure_tensor_radius",
        "pose_consistency_fpr",
        "pose_consistency_gate_sigma_px",
        "pose_consistency_min_decisive_ratio",
        "outlier_drop_d2_threshold",
        "pose_edge_refinement_enabled",
        "gwlf_transversal_alpha",
        "quad_max_elongation",
        "quad_min_density",
        "quad_extraction_mode",
        "edlines_imbalance_gate",
        "quad_extraction_policy",
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn every_config_field_has_an_inertness_case() {
    let mut covered: Vec<&str> = cases().iter().map(|c| c.field).collect();
    covered.sort_unstable();
    let mut expected = every_config_field();
    expected.sort_unstable();
    assert_eq!(
        covered, expected,
        "every DetectorConfig field needs a FieldCase in contract_config_inertness.rs"
    );
}

#[test]
fn the_frame_set_actually_decodes() {
    // A frame set that detects nothing would make every case trivially "inert"
    // in one direction and trivially "live" in the other. Pin the baseline.
    let mut detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let k = intrinsics();
    let mut decoded = 0usize;
    let mut with_pose = 0usize;
    for frame in frames() {
        let img = ImageView::new(frame, CANVAS, CANVAS, CANVAS).unwrap();
        let view = detector
            .detect(&img, Some(&k), Some(TAG_SIZE_M), false)
            .unwrap();
        decoded += view.ids.len();
        with_pose += view.poses.iter().filter(|p| p.data[2] != 0.0).count();
    }
    assert!(
        decoded >= 3,
        "synthetic frame set decoded only {decoded} tags; the inertness cases need a live baseline"
    );
    assert!(with_pose >= 3, "pose stage produced only {with_pose} poses");
}

#[test]
fn no_detector_config_field_is_silently_inert() {
    let mut failures: Vec<String> = Vec::new();

    for case in cases() {
        let mut base = DetectorConfig::default();
        (case.base)(&mut base);
        let mut mutated = base;
        (case.mutate)(&mut mutated);
        assert_ne!(
            base, mutated,
            "case `{}` does not actually change the config",
            case.field
        );

        let before = signature(base);
        let after = signature(mutated);

        match case.inert_reason {
            None if before == after => failures.push(format!(
                "`{}`: mutating it left the detector output bit-identical. \
                 Either wire the field up, remove it, or add an `inert_reason`.",
                case.field
            )),
            Some(reason) if before != after => failures.push(format!(
                "`{}` is allowlisted as inert but changed the output. \
                 Remove the allowlist entry. Recorded reason was: {reason}",
                case.field
            )),
            _ => {},
        }
    }

    assert!(
        failures.is_empty(),
        "config inertness violations:\n{}",
        failures.join("\n")
    );
}

#[test]
fn nthreads_selects_the_pipeline_pool() {
    // The inertness allowlist asserts `nthreads` does not change *output*.
    // This asserts it does change what the pipeline runs on — i.e. that it is
    // wired at all, which is what the field silently failed to do before.
    let global = DetectorBuilder::new().build_engine();
    assert_eq!(
        global.intra_frame_threads(),
        None,
        "nthreads == 0 must stay on the global Rayon pool"
    );

    for n in [1usize, 2, 3] {
        let engine = DetectorBuilder::new().with_threads(n).build_engine();
        assert_eq!(engine.intra_frame_threads(), Some(n));
        assert_eq!(
            engine.bench_api_pipeline_num_threads(),
            n,
            "pipeline work must execute on the scoped {n}-thread pool"
        );
    }
}

#[test]
fn detection_output_is_invariant_to_thread_count() {
    let reference = signature(DetectorConfig::default());
    for n in [1usize, 2, 8] {
        let cfg = DetectorConfig {
            nthreads: n,
            ..DetectorConfig::default()
        };
        assert_eq!(
            signature(cfg),
            reference,
            "detection output changed at nthreads = {n}; parallel stages must be deterministic"
        );
    }
}
