#![allow(
    unsafe_code,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::unwrap_used,
    missing_docs
)]
//! Enforces the DetectionBatch phase-isolation contract defined in
//! `docs/engineering/detection-batch-contract.md §4`.
//!
//! Each phase of the detection pipeline is only permitted to mutate a specific
//! set of columns. A phase that writes outside its lane silently breaks the
//! lock-free Rayon parallelization guarantee. These tests pre-seed every
//! column with a distinctive sentinel, run exactly one phase, and assert that
//! the set of mutated columns is a subset of the phase's declared write set.
//!
//! This is a black-box integration test — phases are called through the
//! public `bench_api`, which is the same surface used by every other
//! integration test in the crate.

use bumpalo::Bump;
use locus_core::bench_api::*;
use locus_core::config::PoseEstimationMode;
use locus_core::{DetectorConfig, ImageView, TagFamily};
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Column identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Column {
    Corners,
    Homographies,
    Ids,
    Payloads,
    ErrorRates,
    Poses,
    StatusMask,
    FunnelStatus,
    CornerCovariances,
    #[cfg(feature = "bench-internals")]
    PoseConsistencyD2,
    #[cfg(feature = "bench-internals")]
    PoseConsistencyD2MaxCorner,
    #[cfg(feature = "bench-internals")]
    IppeBranchD2Ratio,
}

// ---------------------------------------------------------------------------
// Sentinels — distinctive non-default values per column. Floats are compared
// bit-for-bit via byte-level memcmp, so any replacement (including NaN) is
// detected as a write.
// ---------------------------------------------------------------------------

const F32_SENTINEL: f32 = -9_999.0;
const U32_SENTINEL: u32 = u32::MAX;
const U64_SENTINEL: u64 = u64::MAX;

fn seed_sentinels(batch: &mut DetectionBatch) {
    for i in 0..batch.capacity() {
        batch.corners[i] = [Point2f {
            x: F32_SENTINEL,
            y: F32_SENTINEL,
        }; 4];
        batch.homographies[i] = Matrix3x3 {
            data: [F32_SENTINEL; 9],
            padding: [F32_SENTINEL; 7],
        };
        batch.ids[i] = U32_SENTINEL;
        batch.payloads[i] = U64_SENTINEL;
        batch.error_rates[i] = F32_SENTINEL;
        batch.poses[i] = Pose6D {
            data: [F32_SENTINEL; 7],
            padding: F32_SENTINEL,
        };
        // Use Valid as the status sentinel: any phase that writes status_mask
        // will transition to a different state.
        batch.status_mask[i] = CandidateState::Valid;
        // Use RejectedSampling as the funnel sentinel: any funnel transition
        // (None, PassedContrast, RejectedContrast) differs.
        batch.funnel_status[i] = FunnelStatus::RejectedSampling;
        batch.corner_covariances[i] = [F32_SENTINEL; 16];
        #[cfg(feature = "bench-internals")]
        {
            batch.pose_consistency_d2[i] = F32_SENTINEL;
            batch.pose_consistency_d2_max_corner[i] = F32_SENTINEL;
            batch.ippe_branch_d2_ratio[i] = F32_SENTINEL;
        }
    }
}

// ---------------------------------------------------------------------------
// Column-level byte-equal diff. All field element types are `#[repr(C)]` or
// `#[repr(u8)]` POD with explicit padding, so byte comparison is sound and
// catches NaN writes (bit patterns differ), which `==` would mask.
// ---------------------------------------------------------------------------

fn bytes_of<T>(slice: &[T]) -> &[u8] {
    // SAFETY: DetectionBatch's column arrays are `#[repr(C)]` POD slices
    // with deterministic layouts. We only borrow read-only bytes for the
    // duration of the diff; no aliasing or mutation.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

fn changed_columns(before: &DetectionBatch, after: &DetectionBatch) -> BTreeSet<Column> {
    let mut set = BTreeSet::new();
    if bytes_of(&before.corners[..]) != bytes_of(&after.corners[..]) {
        set.insert(Column::Corners);
    }
    if bytes_of(&before.homographies[..]) != bytes_of(&after.homographies[..]) {
        set.insert(Column::Homographies);
    }
    if bytes_of(&before.ids[..]) != bytes_of(&after.ids[..]) {
        set.insert(Column::Ids);
    }
    if bytes_of(&before.payloads[..]) != bytes_of(&after.payloads[..]) {
        set.insert(Column::Payloads);
    }
    if bytes_of(&before.error_rates[..]) != bytes_of(&after.error_rates[..]) {
        set.insert(Column::ErrorRates);
    }
    if bytes_of(&before.poses[..]) != bytes_of(&after.poses[..]) {
        set.insert(Column::Poses);
    }
    if bytes_of(&before.status_mask[..]) != bytes_of(&after.status_mask[..]) {
        set.insert(Column::StatusMask);
    }
    if bytes_of(&before.funnel_status[..]) != bytes_of(&after.funnel_status[..]) {
        set.insert(Column::FunnelStatus);
    }
    if bytes_of(&before.corner_covariances[..]) != bytes_of(&after.corner_covariances[..]) {
        set.insert(Column::CornerCovariances);
    }
    #[cfg(feature = "bench-internals")]
    {
        if bytes_of(&before.pose_consistency_d2[..]) != bytes_of(&after.pose_consistency_d2[..]) {
            set.insert(Column::PoseConsistencyD2);
        }
        if bytes_of(&before.pose_consistency_d2_max_corner[..])
            != bytes_of(&after.pose_consistency_d2_max_corner[..])
        {
            set.insert(Column::PoseConsistencyD2MaxCorner);
        }
        if bytes_of(&before.ippe_branch_d2_ratio[..]) != bytes_of(&after.ippe_branch_d2_ratio[..]) {
            set.insert(Column::IppeBranchD2Ratio);
        }
    }
    set
}

/// Clone a `DetectionBatch` into a fresh heap allocation for before/after diff.
fn snapshot(batch: &DetectionBatch) -> Box<DetectionBatch> {
    let mut out = Box::new(DetectionBatch::new());
    out.corners.copy_from_slice(&batch.corners);
    out.homographies.copy_from_slice(&batch.homographies);
    out.ids.copy_from_slice(&batch.ids);
    out.payloads.copy_from_slice(&batch.payloads);
    out.error_rates.copy_from_slice(&batch.error_rates);
    out.poses.copy_from_slice(&batch.poses);
    out.status_mask.copy_from_slice(&batch.status_mask);
    out.funnel_status.copy_from_slice(&batch.funnel_status);
    out.corner_covariances
        .copy_from_slice(&batch.corner_covariances);
    #[cfg(feature = "bench-internals")]
    {
        out.pose_consistency_d2
            .copy_from_slice(&batch.pose_consistency_d2);
        out.pose_consistency_d2_max_corner
            .copy_from_slice(&batch.pose_consistency_d2_max_corner);
        out.ippe_branch_d2_ratio
            .copy_from_slice(&batch.ippe_branch_d2_ratio);
    }
    out
}

/// Assert `changed ⊆ allowed`. Lists unexpected mutations.
fn assert_writes_within(changed: &BTreeSet<Column>, allowed: &[Column], phase_name: &str) {
    let allowed_set: BTreeSet<Column> = allowed.iter().copied().collect();
    let unexpected: Vec<Column> = changed.difference(&allowed_set).copied().collect();
    assert!(
        unexpected.is_empty(),
        "Phase {phase_name} wrote outside its lane: unexpected columns {unexpected:?}. \
         Allowed set per contract: {allowed_set:?}. Changed: {changed:?}. \
         See docs/engineering/detection-batch-contract.md §4.",
    );
}

// ---------------------------------------------------------------------------
// Phase A — Contour Extraction (extract_quads_soa)
//   Allowed writes: corners, status_mask, corner_covariances
// ---------------------------------------------------------------------------

#[test]
fn contract_phase_a_empty_label_result() {
    let mut batch = Box::new(DetectionBatch::new());
    seed_sentinels(&mut batch);
    let before = snapshot(&batch);

    let width = 32usize;
    let height = 32usize;
    let data = vec![0u8; width * height];
    let img = ImageView::new(&data, width, height, width).expect("valid image");
    let config = DetectorConfig::default();
    let labels = vec![0u32; width * height];
    let label_result = LabelResult {
        labels: &labels,
        component_stats: Vec::new(),
    };

    let (n, _) = extract_quads_soa(&mut batch, &img, &label_result, &config, 1, &img, false);

    assert_eq!(n, 0);
    let changed = changed_columns(&before, &batch);
    assert_writes_within(
        &changed,
        &[
            Column::Corners,
            Column::StatusMask,
            Column::CornerCovariances,
        ],
        "A (extract_quads_soa, empty input)",
    );
    // Phase A normalises status_mask even when no candidates are found.
    assert!(
        changed.contains(&Column::StatusMask),
        "Phase A must normalise status_mask on every call",
    );
}

#[test]
fn contract_phase_a_real_tag() {
    let canvas = 128usize;
    let tag_size = 64usize;
    let (data, _gt) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, tag_size, canvas, 0.0);
    let img = ImageView::new(&data, canvas, canvas, canvas).expect("valid image");
    let config = DetectorConfig::default();

    let arena = Bump::new();
    let engine = ThresholdEngine::from_config(&config);
    let tile_stats = engine.compute_tile_stats(&arena, &img);
    let binarized = arena.alloc_slice_fill_copy(canvas * canvas, 0u8);
    let threshold_map = arena.alloc_slice_fill_copy(canvas * canvas, 0u8);
    engine.apply_threshold_with_map(&arena, &img, &tile_stats, binarized, threshold_map);
    let label_result = label_components_lsl(
        &arena,
        &img,
        threshold_map,
        config.segmentation_connectivity == locus_core::config::SegmentationConnectivity::Eight,
        config.quad_min_area,
    );

    let mut batch = Box::new(DetectionBatch::new());
    seed_sentinels(&mut batch);
    let before = snapshot(&batch);

    let (_n, _) = extract_quads_soa(&mut batch, &img, &label_result, &config, 1, &img, false);

    let changed = changed_columns(&before, &batch);
    assert_writes_within(
        &changed,
        &[
            Column::Corners,
            Column::StatusMask,
            Column::CornerCovariances,
        ],
        "A (extract_quads_soa, real tag)",
    );
}

// ---------------------------------------------------------------------------
// Phase B — Homography (compute_homographies_soa)
//   Allowed writes: homographies. The function signature takes split slices
//   from the batch; we verify end to end that no column bleeds through.
// ---------------------------------------------------------------------------

#[test]
fn contract_phase_b_compute_homographies() {
    let mut batch = Box::new(DetectionBatch::new());
    seed_sentinels(&mut batch);

    batch.corners[0] = [
        Point2f { x: 10.0, y: 10.0 },
        Point2f { x: 50.0, y: 12.0 },
        Point2f { x: 52.0, y: 54.0 },
        Point2f { x: 11.0, y: 52.0 },
    ];
    batch.status_mask[0] = CandidateState::Active;

    let before = snapshot(&batch);

    // The pipeline takes exactly this disjoint split borrow at
    // `detector.rs` — rustc's disjoint-field-borrow rule permits it without
    // `unsafe`.
    let n = 1usize;
    compute_homographies_soa(
        &batch.corners[..n],
        &batch.status_mask[..n],
        &mut batch.homographies[..n],
    );

    let changed = changed_columns(&before, &batch);
    assert_writes_within(
        &changed,
        &[Column::Homographies],
        "B (compute_homographies_soa)",
    );
    assert!(
        changed.contains(&Column::Homographies),
        "Phase B must actually write a homography for an Active candidate",
    );
}

// ---------------------------------------------------------------------------
// Phase B.5 — Fast-Path Funnel (apply_funnel_gate)
//   Allowed writes: status_mask, funnel_status
// ---------------------------------------------------------------------------

#[test]
fn contract_phase_bd5_apply_funnel_gate() {
    let mut batch = Box::new(DetectionBatch::new());
    seed_sentinels(&mut batch);

    batch.corners[0] = [
        Point2f { x: 8.0, y: 8.0 },
        Point2f { x: 56.0, y: 8.0 },
        Point2f { x: 56.0, y: 56.0 },
        Point2f { x: 8.0, y: 56.0 },
    ];
    batch.status_mask[0] = CandidateState::Active;

    let size = 64usize;
    let mut data = vec![0u8; size * size];
    for y in 12..52 {
        for x in 12..52 {
            data[y * size + x] = 255;
        }
    }
    let img = ImageView::new(&data, size, size, size).expect("valid image");
    let tile_size = 8usize;
    let tiles = (size / tile_size) * (size / tile_size);
    let tile_stats = vec![TileStats { min: 0, max: 255 }; tiles];

    let before = snapshot(&batch);

    apply_funnel_gate(&mut batch, 1, &img, &tile_stats, tile_size, 20.0, 1.0);

    let changed = changed_columns(&before, &batch);
    assert_writes_within(
        &changed,
        &[Column::StatusMask, Column::FunnelStatus],
        "B.5 (apply_funnel_gate)",
    );
    assert!(
        changed.contains(&Column::FunnelStatus),
        "Phase B.5 must write funnel_status",
    );
}

// ---------------------------------------------------------------------------
// Phase C — Batched Sampling & Decoding (decode_batch_soa)
//   Allowed writes: ids, payloads, error_rates, status_mask, corners
//     (the corners write is the documented rotation-permutation exception —
//      see `docs/engineering/detection-batch-contract.md §4 Phase C`).
// ---------------------------------------------------------------------------

#[test]
fn contract_phase_c_decode_batch_soa_empty() {
    let mut batch = Box::new(DetectionBatch::new());
    seed_sentinels(&mut batch);
    let before = snapshot(&batch);

    let size = 32usize;
    let data = vec![128u8; size * size];
    let img = ImageView::new(&data, size, size, size).expect("valid image");
    let decoders = vec![family_to_decoder(TagFamily::AprilTag36h11)];
    let config = DetectorConfig::default();

    decode_batch_soa(&mut batch, 0, &img, &decoders, &config);

    let changed = changed_columns(&before, &batch);
    assert!(
        changed.is_empty(),
        "Phase C must be a no-op for n=0 but mutated: {changed:?}",
    );
}

#[test]
fn contract_phase_c_decode_batch_soa_active() {
    let size = 64usize;
    let mut data = vec![200u8; size * size];
    for y in 20..44 {
        for x in 20..44 {
            data[y * size + x] = 40;
        }
    }
    let img = ImageView::new(&data, size, size, size).expect("valid image");
    let decoders = vec![family_to_decoder(TagFamily::AprilTag36h11)];
    let config = DetectorConfig::default();

    let mut batch = Box::new(DetectionBatch::new());
    seed_sentinels(&mut batch);
    batch.corners[0] = [
        Point2f { x: 20.0, y: 20.0 },
        Point2f { x: 44.0, y: 20.0 },
        Point2f { x: 44.0, y: 44.0 },
        Point2f { x: 20.0, y: 44.0 },
    ];
    batch.status_mask[0] = CandidateState::Active;
    // Seed a valid homography so the decoder's sampler runs the full pass.
    let dst = [[20.0f64, 20.0], [44.0, 20.0], [44.0, 44.0], [20.0, 44.0]];
    let h = Homography::from_pairs(&[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], &dst)
        .expect("non-degenerate");
    let mut h_data = [0.0f32; 9];
    for (j, val) in h.h.iter().enumerate() {
        h_data[j] = *val as f32;
    }
    batch.homographies[0] = Matrix3x3 {
        data: h_data,
        padding: [0.0; 7],
    };

    let before = snapshot(&batch);

    decode_batch_soa(&mut batch, 1, &img, &decoders, &config);

    let changed = changed_columns(&before, &batch);
    assert_writes_within(
        &changed,
        &[
            Column::Corners,
            Column::Ids,
            Column::Payloads,
            Column::ErrorRates,
            Column::StatusMask,
        ],
        "C (decode_batch_soa, n=1)",
    );
}

// ---------------------------------------------------------------------------
// Phase D — Pose Refinement (refine_poses_soa_with_config)
//   Allowed writes: poses
// ---------------------------------------------------------------------------

#[test]
fn contract_phase_d_refine_poses_soa() {
    let mut batch = Box::new(DetectionBatch::new());
    seed_sentinels(&mut batch);

    batch.corners[0] = [
        Point2f { x: 10.0, y: 10.0 },
        Point2f { x: 50.0, y: 10.0 },
        Point2f { x: 50.0, y: 50.0 },
        Point2f { x: 10.0, y: 50.0 },
    ];
    batch.status_mask[0] = CandidateState::Valid;
    // Zero covariance → unweighted path, no need for real Structure-Tensor
    // priors.
    batch.corner_covariances[0] = [0.0; 16];

    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);
    let config = DetectorConfig::default();
    let before = snapshot(&batch);

    refine_poses_soa_with_config(
        &mut batch,
        1,
        &intrinsics,
        0.1,
        None,
        PoseEstimationMode::Fast,
        &config,
    );

    let changed = changed_columns(&before, &batch);
    let allowed = {
        #[cfg(feature = "bench-internals")]
        {
            vec![
                Column::Poses,
                Column::PoseConsistencyD2,
                Column::PoseConsistencyD2MaxCorner,
                Column::IppeBranchD2Ratio,
            ]
        }
        #[cfg(not(feature = "bench-internals"))]
        {
            vec![Column::Poses]
        }
    };
    assert_writes_within(&changed, &allowed, "D (refine_poses_soa_with_config)");
    assert!(
        changed.contains(&Column::Poses),
        "Phase D must write a pose for a Valid candidate",
    );
}
