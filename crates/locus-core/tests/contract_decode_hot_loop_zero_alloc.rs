#![allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    missing_docs
)]
//! Zero-allocation contract test for the post-extraction hot-loop
//! per-candidate SoA write-back.
//!
//! Before this contract was enforced, three rayon-driven map+collect+drain
//! patterns in the production hot-path (`decode_batch_soa_generic`,
//! `decode_batch_soa_with_camera_inner`, and `refine_poses_soa_with_config`)
//! materialised a per-frame `Vec<TupleN>` outside the workspace arena, just
//! to be drained sequentially into the SoA columns. The fix writes
//! directly into the SoA columns from the rayon workers (rayon's `Zip`
//! proves the per-index disjointness statically).
//!
//! This test installs a thread-safe heap-allocation counter as the global
//! allocator, performs a warm-up call to amortise first-call allocations
//! (decoder construction, rayon-pool spawn, thread-local arena bootstrap),
//! then takes before/after snapshots around a second steady-state call and
//! asserts that the byte-delta is exactly zero. A regression of any kind —
//! a stray `Vec::new()` in the closure, a re-introduced collect, or even a
//! Box::leak in a downstream helper — will be caught.

use locus_core::bench_api::*;
use locus_core::{DetectorConfig, ImageView, TagFamily};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Once};

// ---------------------------------------------------------------------------
// Heap-allocation counter
// ---------------------------------------------------------------------------

/// Thread-safe heap-allocation counter installed as the global allocator
/// for this integration-test binary. Each integration test in Rust compiles
/// to its own binary, so setting `#[global_allocator]` here is safe and
/// does not affect any other test file.
struct CountingAllocator {
    inner: System,
}

static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: `CountingAllocator` is a thin wrapper around the `System`
// allocator: every method delegates directly to the inner allocator
// after incrementing two relaxed atomic counters. The wrapper does
// not change the lifetime, alignment, or layout invariants required
// by `GlobalAlloc`; it only adds bookkeeping side effects.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded directly to `System::alloc` with the same layout.
        unsafe { self.inner.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded directly to `System::dealloc` with the same ptr/layout.
        unsafe { self.inner.dealloc(ptr, layout) };
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded directly to `System::alloc_zeroed` with the same layout.
        unsafe { self.inner.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // `realloc` is treated as a heap allocation event: the underlying
        // allocator may grow in place or relocate, and either way an
        // accidental `Vec::push` past its capacity inside the hot loop
        // would manifest as a realloc, which is what we want to catch.
        let old_size = layout.size();
        if new_size > old_size {
            ALLOC_BYTES.fetch_add(new_size - old_size, Ordering::Relaxed);
        }
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded directly to `System::realloc` with the same ptr/layout/new_size.
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: CountingAllocator = CountingAllocator { inner: System };

/// Pin rayon's global pool to a single worker so each candidate-iteration
/// runs on the same thread. Without this, the contract becomes flaky: the
/// thread-local `WORKSPACE_ARENA` lives per rayon worker, and a steady-state
/// call can grow a worker's arena the first time it sees a particular
/// candidate's ROI, even if a previous warm-up call hit a different subset
/// of workers. With a 1-thread pool, the warm-up fully primes the single
/// worker's arena and the steady-state call performs zero heap allocations.
fn pin_rayon_to_one_thread() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Best-effort: if rayon's global pool has already been built by
        // another test in the same binary, this is a no-op. The contract
        // assertion still holds for both possibilities — with > 1 worker
        // we just need more warm-up passes to converge.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global();
    });
}

/// Serialise the three contract tests. The global allocator counter is
/// shared across all tests in this binary, and `cargo test`'s default
/// parallel scheduler runs them concurrently — so without this mutex,
/// allocations from one test's body leak into another test's snapshot
/// delta and the assertion goes flaky. `nextest` already forks per test,
/// but the workspace's `cargo insta test --check` uses `cargo test`, which
/// keeps tests in the same process. A single mutex is the simplest fix.
static CONTRACT_SERIAL: Mutex<()> = Mutex::new(());

fn snapshot() -> (usize, usize) {
    (
        ALLOC_BYTES.load(Ordering::Relaxed),
        ALLOC_COUNT.load(Ordering::Relaxed),
    )
}

fn delta(before: (usize, usize), after: (usize, usize)) -> (usize, usize) {
    (after.0 - before.0, after.1 - before.1)
}

/// Convergence-driven zero-allocation assertion.
///
/// `body` performs per-iteration fixture re-seed (idempotent, must not
/// allocate after warm-up) followed by the call under test. The
/// convergence loop runs `body` until **two consecutive** invocations
/// show a strict `(0, 0)` byte/count delta, or fails the contract if 32
/// attempts don't stabilise.
///
/// The previous design used a fixed 8-pass warm-up plus a `bytes == 0`
/// assertion. The review caught three holes that this design closes:
/// 1. `pin_rayon_to_one_thread()` can silently no-op if rayon's global
///    pool was already built; on a many-worker pool, 8 warm-ups may not
///    visit every worker's thread-local `WORKSPACE_ARENA`. Convergence
///    keeps running until every worker's arena stabilises.
/// 2. A `realloc` that shrinks credits 0 bytes but bumps `count`; the
///    old `bytes == 0` assertion silently passed. We now require
///    `(bytes, count) == (0, 0)`.
/// 3. Any new lazily-initialised structure (tracing context, per-decoder
///    thread-local cache, etc.) that takes >8 calls to stabilise would
///    have flaked the old test. The 32-pass ceiling gives ample margin.
///
/// Why a single closure rather than `(setup, measure)`: both halves want
/// `&mut batch`, and the borrow checker can't share that across two
/// distinct `FnMut`s held simultaneously. Folding re-seed into the
/// measured window is harmless because the re-seed is a fixed-size
/// memcpy that allocates zero bytes after the first warm-up iteration.
fn assert_zero_alloc<F: FnMut()>(label: &str, mut body: F) {
    let mut prev_zero = false;
    let mut last_delta = (0usize, 0usize);
    for _ in 0..32 {
        let before = snapshot();
        body();
        let after = snapshot();
        let d = delta(before, after);
        last_delta = d;
        if d == (0, 0) {
            if prev_zero {
                return;
            }
            prev_zero = true;
        } else {
            prev_zero = false;
        }
    }
    panic!(
        "{label} did not converge to zero allocations within 32 warm-up passes; \
         last steady-state delta = {} bytes, {} count. The hot-loop SoA write-back \
         contract is violated — likely a re-introduced `Vec::new()` / `Box::new()` / \
         `collect()` outside the workspace arena.",
        last_delta.0, last_delta.1
    );
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Build a realistic-looking decoding batch: 250 candidates, the first 50
/// marked Active with quad corners pointing into a real test image (so the
/// decoder enters the hot SIMD sampling path). Capacity matches
/// `BenchDataset::generate_bench_batch(50, 200)` used by `decoding_bench`.
fn build_decode_fixture() -> (Box<DetectionBatch>, Vec<u8>, usize) {
    // 256×256 synthetic canvas containing a single AprilTag at the centre.
    // Candidate 0 is seeded with the tag's ground-truth corner positions so
    // the decoder enters the `state == Valid` branch (rotation reorder,
    // refined-corners write-back, homography recompute) — without this,
    // every candidate took the `FailedDecode` bypass and the contract test
    // covered only one of three branches of the hot loop.
    let canvas = 256usize;
    let tag_size = 96usize;
    let (data, gt_corners) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, tag_size, canvas, 0.0);

    let mut batch = DetectionBatch::new_boxed();
    let n: usize = 250;
    let active_count: usize = 50;
    // Candidate 0: the tag's ground-truth corners — decodes into the
    // `state == Valid` write-back branch.
    batch.corners[0] = [
        Point2f {
            x: gt_corners[0][0] as f32,
            y: gt_corners[0][1] as f32,
        },
        Point2f {
            x: gt_corners[1][0] as f32,
            y: gt_corners[1][1] as f32,
        },
        Point2f {
            x: gt_corners[2][0] as f32,
            y: gt_corners[2][1] as f32,
        },
        Point2f {
            x: gt_corners[3][0] as f32,
            y: gt_corners[3][1] as f32,
        },
    ];
    batch.status_mask[0] = CandidateState::Active;
    // Candidates 1..n: distributed small quads pointing at non-tag regions,
    // routed through the `FailedDecode` bypass.
    for i in 1..n {
        let base_x = (i % 16) as f32 * 14.0 + 4.0;
        let base_y = (i / 16) as f32 * 14.0 + 4.0;
        batch.corners[i] = [
            Point2f {
                x: base_x,
                y: base_y,
            },
            Point2f {
                x: base_x + 10.0,
                y: base_y,
            },
            Point2f {
                x: base_x + 10.0,
                y: base_y + 10.0,
            },
            Point2f {
                x: base_x,
                y: base_y + 10.0,
            },
        ];
        batch.status_mask[i] = if i < active_count {
            CandidateState::Active
        } else {
            CandidateState::FailedDecode
        };
    }
    (batch, data, n)
}

/// Build a pose-refinement fixture: 50 Valid candidates with realistic
/// corner coordinates so the LM solver actually iterates.
fn build_pose_fixture() -> Box<DetectionBatch> {
    let mut batch = DetectionBatch::new_boxed();
    let v: usize = 50;
    for i in 0..v {
        let base_x = (i % 16) as f32 * 14.0 + 4.0;
        let base_y = (i / 16) as f32 * 14.0 + 4.0;
        batch.corners[i] = [
            Point2f {
                x: base_x,
                y: base_y,
            },
            Point2f {
                x: base_x + 10.0,
                y: base_y,
            },
            Point2f {
                x: base_x + 10.0,
                y: base_y + 10.0,
            },
            Point2f {
                x: base_x,
                y: base_y + 10.0,
            },
        ];
        batch.status_mask[i] = CandidateState::Valid;
        batch.ids[i] = i as u32;
    }
    batch
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `decode_batch_soa` (production pinhole entry) must not perform any
/// heap allocation on a warm steady-state call. After PR-F, the
/// per-candidate worker writes its SoA cells directly via rayon `Zip`;
/// the only ephemeral state lives in the thread-local `WORKSPACE_ARENA`.
///
/// The fixture now seeds candidate 0 with the synthetic tag's ground-truth
/// corners, so the convergence loop exercises **both** branches of the hot
/// closure — the `FailedDecode` bypass (candidates 1..50) AND the
/// `state == Valid` write-back path (rotation reorder, ERF refined corner
/// overwrite, homography recompute on candidate 0).
#[test]
fn contract_decode_batch_soa_zero_alloc_steady_state() {
    let _serial = CONTRACT_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pin_rayon_to_one_thread();
    let (mut batch, data, n) = build_decode_fixture();
    let canvas = (data.len() as f64).sqrt() as usize;
    let img = ImageView::new(&data, canvas, canvas, canvas).expect("valid image");
    let config = DetectorConfig::default();
    let decoders = vec![family_to_decoder(TagFamily::AprilTag36h11)];

    // Snapshot input state ONCE so every convergence-loop iteration sees
    // the same fixture (the decoder writes refined / rotated corners and
    // recomputes homographies on Valid candidates, so without restoring
    // these the inputs would drift across iterations and the steady-state
    // arena high-water-mark might never settle).
    let original_corners: Vec<[Point2f; 4]> = batch.corners[..n].to_vec();
    compute_homographies_soa(
        &batch.corners[0..n],
        &batch.status_mask[0..n],
        &mut batch.homographies[0..n],
    );
    let original_homographies: Vec<Matrix3x3> = batch.homographies[..n].to_vec();

    assert_zero_alloc("decode_batch_soa", || {
        // Per-iteration setup: restore the inputs the decoder may have
        // mutated on the previous iteration (refined / rotated corners,
        // recomputed homographies, Valid/FailedDecode status flips).
        for i in 0..n {
            batch.corners[i] = original_corners[i];
            batch.homographies[i] = original_homographies[i];
            batch.status_mask[i] = if i < 50 {
                CandidateState::Active
            } else {
                CandidateState::FailedDecode
            };
        }
        decode_batch_soa(&mut batch, n, &img, &decoders, &config);
    });
}

/// Companion contract for `refine_poses_soa_with_config` on the default
/// (ERF refinement) route.
#[test]
fn contract_refine_poses_soa_zero_alloc_steady_state() {
    let _serial = CONTRACT_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pin_rayon_to_one_thread();
    let mut batch = build_pose_fixture();
    let v: usize = 50;
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;
    let config = DetectorConfig::default();

    assert_zero_alloc("refine_poses_soa_with_config (ERF route)", || {
        refine_poses_soa_with_config(&mut batch, v, &intrinsics, tag_size, None, &config);
    });
}

/// Companion contract for `refine_poses_soa_with_config` on the **GWLF**
/// route. The default test above leaves `corner_covariances` at zero so
/// `compute_one`'s GWLF branch (`config.refinement_mode == Gwlf` →
/// unpack `covs_row[j*4..j*4+3]` row-major into a `Matrix2`) is never
/// touched. This test forces the GWLF route by setting
/// `refinement_mode = Gwlf` and seeding identity per-corner covariances
/// so the branch's slice-indexing arithmetic is actually exercised.
#[test]
fn contract_refine_poses_soa_zero_alloc_gwlf_route() {
    let _serial = CONTRACT_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pin_rayon_to_one_thread();
    let mut batch = build_pose_fixture();
    let v: usize = 50;
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;

    // Seed identity per-corner covariances. Layout matches `compute_one`'s
    // unpack `Matrix2::new(covs_row[j*4], covs_row[j*4+1], covs_row[j*4+2],
    // covs_row[j*4+3])` per corner j ∈ 0..4.
    for i in 0..v {
        for j in 0..4 {
            batch.corner_covariances[i][j * 4] = 1.0; // xx
            batch.corner_covariances[i][j * 4 + 1] = 0.0; // xy
            batch.corner_covariances[i][j * 4 + 2] = 0.0; // yx
            batch.corner_covariances[i][j * 4 + 3] = 1.0; // yy
        }
    }

    let config = DetectorConfig {
        refinement_mode: locus_core::CornerRefinementMode::Gwlf,
        ..DetectorConfig::default()
    };

    assert_zero_alloc("refine_poses_soa_with_config (GWLF route)", || {
        refine_poses_soa_with_config(&mut batch, v, &intrinsics, tag_size, None, &config);
    });
}

/// Companion contract for `decode_batch_soa_with_camera` on the non-rectified
/// inner path (Brown-Conrady model with mild distortion forces the inner
/// path; the rectified `PinholeModel` would short-circuit to
/// `decode_batch_soa`, which is covered by the first test).
#[cfg(feature = "non_rectified")]
#[test]
fn contract_decode_batch_soa_with_camera_inner_zero_alloc_steady_state() {
    let _serial = CONTRACT_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pin_rayon_to_one_thread();
    let (mut batch, data, n) = build_decode_fixture();
    let canvas = (data.len() as f64).sqrt() as usize;
    let img = ImageView::new(&data, canvas, canvas, canvas).expect("valid image");
    let config = DetectorConfig::default();
    let decoders = vec![family_to_decoder(TagFamily::AprilTag36h11)];

    let intrinsics = CameraIntrinsics::with_brown_conrady(
        800.0, 800.0, 128.0, 128.0, -0.12, 0.05, 0.0, 0.0, 0.0,
    );
    let model = BrownConradyModel {
        k1: -0.12,
        k2: 0.05,
        p1: 0.0,
        p2: 0.0,
        k3: 0.0,
    };

    let original_corners: Vec<[Point2f; 4]> = batch.corners[..n].to_vec();
    compute_homographies_soa(
        &batch.corners[0..n],
        &batch.status_mask[0..n],
        &mut batch.homographies[0..n],
    );
    let original_homographies: Vec<Matrix3x3> = batch.homographies[..n].to_vec();

    assert_zero_alloc("decode_batch_soa_with_camera (non_rectified inner)", || {
        for i in 0..n {
            batch.corners[i] = original_corners[i];
            batch.homographies[i] = original_homographies[i];
            batch.status_mask[i] = if i < 50 {
                CandidateState::Active
            } else {
                CandidateState::FailedDecode
            };
        }
        decode_batch_soa_with_camera(
            &mut batch,
            n,
            &img,
            &decoders,
            &config,
            Some(&intrinsics),
            &model,
        );
    });
}
