#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
use locus_core::bench_api::{CandidateState, DetectionBatch, FunnelStatus, ROUTED_TO_STATIC};

#[test]
fn test_batch_allocation() {
    let batch = DetectionBatch::new_boxed();
    assert_eq!(batch.capacity(), 1024);
}

#[test]
fn test_batch_alignment() {
    let batch = DetectionBatch::new_boxed();
    let corners_ptr = batch.corners.as_ptr() as usize;
    assert_eq!(corners_ptr % 32, 0);

    let homographies_ptr = batch.homographies.as_ptr() as usize;
    assert_eq!(homographies_ptr % 32, 0);
}

#[test]
fn test_funnel_status_initialization() {
    let batch = DetectionBatch::new_boxed();
    assert_eq!(batch.funnel_status[0], FunnelStatus::None);
}

/// `new_boxed` locks two contracts that the `Box::new_zeroed` SAFETY block
/// relies on:
///
/// 1. Zero-initialized columns land on the documented default variant
///    (`CandidateState::Empty`, `FunnelStatus::None`, `0.0`, `0`).
/// 2. Non-zero columns are explicitly patched by `slice::fill` —
///    `routed_to` to `ROUTED_TO_STATIC` and (under `bench-internals`) the
///    four telemetry columns to NaN / `u8::MAX`.
///
/// If a future cleanup PR deletes any of the `slice::fill` calls, or if a
/// new column is added without an explicit default patch, this test fails.
/// Checks both endpoints (`[0]` and `[N-1]`) so a partial-slice fill regression
/// surfaces.
#[test]
fn test_new_boxed_defaults_full_coverage() {
    let batch = DetectionBatch::new_boxed();
    let last = batch.capacity() - 1;

    // Zero-init columns (covered by `Box::<Self>::new_zeroed`).
    assert_eq!(batch.status_mask[0], CandidateState::Empty);
    assert_eq!(batch.status_mask[last], CandidateState::Empty);
    assert_eq!(batch.funnel_status[0], FunnelStatus::None);
    assert_eq!(batch.funnel_status[last], FunnelStatus::None);
    assert_eq!(batch.ids[0], 0);
    assert_eq!(batch.ids[last], 0);
    assert_eq!(batch.payloads[0], 0);
    // Compare f32 zero-defaults by bit pattern: `new_zeroed` guarantees all
    // bytes are 0, which for IEEE 754 means `+0.0`. `assert_eq!(_, 0.0)` would
    // trip `clippy::float_cmp`.
    assert_eq!(batch.error_rates[0].to_bits(), 0);
    assert_eq!(batch.ppb_estimate[0].to_bits(), 0);
    assert_eq!(batch.ppb_estimate[last].to_bits(), 0);

    // Non-zero column patched by `slice::fill`.
    assert_eq!(batch.routed_to[0], ROUTED_TO_STATIC);
    assert_eq!(batch.routed_to[last], ROUTED_TO_STATIC);
}

/// `bench-internals` telemetry columns must be NaN / `u8::MAX` after
/// construction so consumers can distinguish "the gate did not run" from "the
/// gate produced a finite score". A deleted `slice::fill(f32::NAN)` would
/// leave the columns at `0.0`, which is a valid score and would silently lie
/// to downstream calibration audits.
#[cfg(feature = "bench-internals")]
#[test]
fn test_new_boxed_bench_internals_defaults() {
    let batch = DetectionBatch::new_boxed();
    let last = batch.capacity() - 1;
    assert!(batch.pose_consistency_d2[0].is_nan());
    assert!(batch.pose_consistency_d2[last].is_nan());
    assert!(batch.pose_consistency_d2_max_corner[0].is_nan());
    assert!(batch.pose_consistency_d2_max_corner[last].is_nan());
    assert!(batch.ippe_branch_d2_ratio[0].is_nan());
    assert!(batch.ippe_branch_d2_ratio[last].is_nan());
    assert_eq!(batch.outlier_corner_idx[0], u8::MAX);
    assert_eq!(batch.outlier_corner_idx[last], u8::MAX);
}
