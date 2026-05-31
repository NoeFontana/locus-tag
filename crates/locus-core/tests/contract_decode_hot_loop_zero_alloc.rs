#![allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
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

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Build a realistic-looking decoding batch: 250 candidates, the first 50
/// marked Active with quad corners pointing into a real test image (so the
/// decoder enters the hot SIMD sampling path). Capacity matches
/// `BenchDataset::generate_bench_batch(50, 200)` used by `decoding_bench`.
fn build_decode_fixture() -> (Box<DetectionBatch>, Vec<u8>, usize) {
    // Use a 256×256 synthetic canvas containing a single AprilTag at the
    // centre. The non-tag candidates point at random quad corners that
    // route through the FailedDecode bypass — this exercises both the
    // bypass and the active branch of the closure.
    let canvas = 256usize;
    let tag_size = 96usize;
    let (data, _gt) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, tag_size, canvas, 0.0);

    let mut batch = Box::new(DetectionBatch::new());
    // 50 Active candidates pointing at quad corners distributed across
    // the canvas; the first is centred on the synthetic tag.
    let n: usize = 250;
    let active_count: usize = 50;
    for i in 0..n {
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
    let mut batch = Box::new(DetectionBatch::new());
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

    // Compute homographies once outside the loop (same setup as
    // `decoding_bench::bench_decoding_soa_realistic`).
    compute_homographies_soa(
        &batch.corners[0..n],
        &batch.status_mask[0..n],
        &mut batch.homographies[0..n],
    );

    // Warm-up: amortise first-call allocations (decoder construction,
    // rayon-pool spawn, thread-local arena bootstrap). The status_mask
    // for valid candidates flips Active → Valid/FailedDecode on the warm-up,
    // so we re-seed it to Active for the steady-state measurement. We
    // run several warm-up passes because rayon's `WORKSPACE_ARENA`
    // thread-local is per worker; a single warm-up may only have populated
    // a subset of the rayon-pool workers' arenas, so we keep running until
    // the arena delta stabilises.
    for _ in 0..8 {
        decode_batch_soa(&mut batch, n, &img, &decoders, &config);
        for i in 0..n {
            batch.status_mask[i] = if i < 50 {
                CandidateState::Active
            } else {
                CandidateState::FailedDecode
            };
        }
    }

    // Steady-state: take the before snapshot AFTER warm-up, run a second
    // identical call, take the after snapshot. The byte delta must be
    // zero — any drift is a regression of the contract.
    let before = snapshot();
    decode_batch_soa(&mut batch, n, &img, &decoders, &config);
    let after = snapshot();

    let (bytes, count) = delta(before, after);
    assert_eq!(
        bytes, 0,
        "decode_batch_soa allocated {bytes} bytes ({count} allocations) on a warm call; \
         contract violated. Per-frame `Vec<TupleN>` outside the workspace arena was \
         re-introduced or a new heap allocation slipped into the rayon closure."
    );
}

/// Companion contract for `refine_poses_soa_with_config`.
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

    // Warm-up: several passes so every rayon worker's thread-local arena
    // and any per-call lazily-initialised structures (tracing span
    // metadata, etc.) are fully primed.
    for _ in 0..8 {
        refine_poses_soa_with_config(&mut batch, v, &intrinsics, tag_size, None, &config);
    }

    let before = snapshot();
    refine_poses_soa_with_config(&mut batch, v, &intrinsics, tag_size, None, &config);
    let after = snapshot();

    let (bytes, count) = delta(before, after);
    assert_eq!(
        bytes, 0,
        "refine_poses_soa_with_config allocated {bytes} bytes ({count} allocations) \
         on a warm call; contract violated. Per-frame `Vec<TupleN>` outside the arena \
         was re-introduced or a new heap allocation slipped into the rayon closure."
    );
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

    compute_homographies_soa(
        &batch.corners[0..n],
        &batch.status_mask[0..n],
        &mut batch.homographies[0..n],
    );

    // Warm-up: several passes so every rayon worker's thread-local arena
    // is fully primed (see the rectified-path test for rationale).
    for _ in 0..8 {
        decode_batch_soa_with_camera(
            &mut batch,
            n,
            &img,
            &decoders,
            &config,
            Some(&intrinsics),
            &model,
        );
        for i in 0..n {
            batch.status_mask[i] = if i < 50 {
                CandidateState::Active
            } else {
                CandidateState::FailedDecode
            };
        }
    }

    let before = snapshot();
    decode_batch_soa_with_camera(
        &mut batch,
        n,
        &img,
        &decoders,
        &config,
        Some(&intrinsics),
        &model,
    );
    let after = snapshot();

    let (bytes, count) = delta(before, after);
    assert_eq!(
        bytes, 0,
        "decode_batch_soa_with_camera (non_rectified inner) allocated {bytes} bytes \
         ({count} allocations) on a warm call; contract violated."
    );
}
