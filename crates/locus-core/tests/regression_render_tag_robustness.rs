#![allow(
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! Regression tests for robustness-oriented rendered single-tag datasets.
//!
//! These subsets stress the detector under conditions that the `tag36h11`
//! golden set does not probe:
//!   * `tag16h5`       — dense 16h5 family (more codes, tighter bit layout).
//!   * `high_iso`      — high-noise sensor simulation.
//!   * `low_key`       — low-key lighting / low dynamic range.
//!   * `raw_pipeline`  — raw-like pipeline variant.
//!
//! All variants are rendered at 1920×1080. See `regression_render_tag.rs` for
//! the baseline `tag36h11` suite and `regression_board_hub.rs` /
//! `regression_distortion_hub.rs` for board- and distortion-specific tests.

use locus_core::TagFamily;

mod common;

use common::hub::{RenderTagOpts, run_render_tag_test};

#[test]
fn regression_hub_tag16h5_1080p() {
    let _g = common::telemetry::init("regression_hub_tag16h5_1080p");
    run_render_tag_test(
        "locus_v1_tag16h5_1920x1080",
        TagFamily::AprilTag16h5,
        RenderTagOpts::default(),
    );
}

#[test]
fn regression_hub_high_iso_1080p() {
    let _g = common::telemetry::init("regression_hub_high_iso_1080p");
    run_render_tag_test(
        "locus_v1_high_iso_1920x1080",
        TagFamily::AprilTag36h11,
        RenderTagOpts::default(),
    );
}

#[test]
fn regression_hub_low_key_1080p() {
    let _g = common::telemetry::init("regression_hub_low_key_1080p");
    run_render_tag_test(
        "locus_v1_low_key_1920x1080",
        TagFamily::AprilTag36h11,
        RenderTagOpts::default(),
    );
}

#[test]
fn regression_hub_raw_pipeline_1080p() {
    let _g = common::telemetry::init("regression_hub_raw_pipeline_1080p");
    run_render_tag_test(
        "locus_v1_raw_pipeline_1920x1080",
        TagFamily::AprilTag36h11,
        RenderTagOpts::default(),
    );
}

// Tuned variants measure configuration-only KPI ceilings. Each fixture
// inlines every `standard` value because profile JSON does not currently
// resolve `extends` (see `crates/locus-core/src/config.rs`).

#[test]
fn regression_hub_tag16h5_1080p_tuned() {
    let _g = common::telemetry::init("regression_hub_tag16h5_1080p_tuned");
    run_render_tag_test(
        "locus_v1_tag16h5_1920x1080",
        TagFamily::AprilTag16h5,
        RenderTagOpts {
            profile_json: Some(include_str!("fixtures/robustness/tag16h5_tuned.json")),
            snapshot_suffix: "_tuned",
            ..RenderTagOpts::default()
        },
    );
}

#[test]
fn regression_hub_low_key_1080p_tuned() {
    let _g = common::telemetry::init("regression_hub_low_key_1080p_tuned");
    run_render_tag_test(
        "locus_v1_low_key_1920x1080",
        TagFamily::AprilTag36h11,
        RenderTagOpts {
            profile_json: Some(include_str!("fixtures/robustness/low_key_tuned.json")),
            snapshot_suffix: "_tuned",
            ..RenderTagOpts::default()
        },
    );
}

#[test]
fn regression_hub_raw_pipeline_1080p_tuned() {
    let _g = common::telemetry::init("regression_hub_raw_pipeline_1080p_tuned");
    run_render_tag_test(
        "locus_v1_raw_pipeline_1920x1080",
        TagFamily::AprilTag36h11,
        RenderTagOpts {
            profile_json: Some(include_str!("fixtures/robustness/raw_pipeline_tuned.json")),
            snapshot_suffix: "_tuned",
            ..RenderTagOpts::default()
        },
    );
}
