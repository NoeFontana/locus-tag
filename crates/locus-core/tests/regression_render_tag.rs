#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::type_complexity,
    clippy::unnecessary_debug_formatting,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! Regression tests for rendered `tag36h11` datasets from the hub.
//!
//! Covers the four standard resolutions across four axes of variation —
//! accuracy baseline, Fast-mode pose, corner refinement, and quad extraction.
//! Robustness variants (tag16h5, high_iso, low_key, raw_pipeline) live in
//! `regression_render_tag_robustness.rs`.

mod common;

mod accuracy_baseline {
    use super::common;
    use common::hub::{RenderTagOpts, run_render_tag_test};
    use locus_core::TagFamily;

    #[test]
    fn regression_hub_tag36h11_640x480() {
        let _g = common::telemetry::init("regression_hub_tag36h11_640x480");
        run_render_tag_test(
            "locus_v1_tag36h11_640x480",
            TagFamily::AprilTag36h11,
            RenderTagOpts::default(),
        );
    }

    #[test]
    fn regression_hub_tag36h11_720p() {
        let _g = common::telemetry::init("regression_hub_tag36h11_720p");
        run_render_tag_test(
            "locus_v1_tag36h11_1280x720",
            TagFamily::AprilTag36h11,
            RenderTagOpts::default(),
        );
    }

    #[test]
    fn regression_hub_tag36h11_1080p() {
        let _g = common::telemetry::init("regression_hub_tag36h11_1080p");
        run_render_tag_test(
            "locus_v1_tag36h11_1920x1080",
            TagFamily::AprilTag36h11,
            RenderTagOpts::default(),
        );
    }

    #[test]
    fn regression_hub_tag36h11_2160p() {
        let _g = common::telemetry::init("regression_hub_tag36h11_2160p");
        run_render_tag_test(
            "locus_v1_tag36h11_3840x2160",
            TagFamily::AprilTag36h11,
            RenderTagOpts::default(),
        );
    }
}

mod pose_mode_variants {
    use super::common;
    use common::hub::{RenderTagOpts, run_render_tag_test};
    use locus_core::{PoseEstimationMode, TagFamily};

    #[test]
    fn regression_hub_fast_tag36h11_1080p() {
        let _g = common::telemetry::init("regression_hub_fast_tag36h11_1080p");
        run_render_tag_test(
            "locus_v1_tag36h11_1920x1080",
            TagFamily::AprilTag36h11,
            RenderTagOpts {
                mode: PoseEstimationMode::Fast,
                ..Default::default()
            },
        );
    }
}

mod refinement_variants {
    use super::common;
    use common::hub::{RenderTagOpts, run_render_tag_test};
    use locus_core::{TagFamily, config::CornerRefinementMode};

    #[test]
    fn regression_hub_tag36h11_1080p_gwlf() {
        let _g = common::telemetry::init("regression_hub_tag36h11_1080p_gwlf");
        run_render_tag_test(
            "locus_v1_tag36h11_1920x1080",
            TagFamily::AprilTag36h11,
            RenderTagOpts {
                snapshot_suffix: "_gwlf",
                refinement: Some(CornerRefinementMode::Gwlf),
                ..Default::default()
            },
        );
    }

    #[test]
    fn regression_hub_tag36h11_1080p_highaccuracy() {
        let _g = common::telemetry::init("regression_hub_tag36h11_1080p_highaccuracy");
        run_render_tag_test(
            "locus_v1_tag36h11_1920x1080",
            TagFamily::AprilTag36h11,
            RenderTagOpts {
                profile: Some("high_accuracy"),
                snapshot_suffix: "_highaccuracy",
                ..Default::default()
            },
        );
    }
}

mod quad_extraction_variants {
    use super::common;
    use common::hub::{RenderTagOpts, run_render_tag_test};
    use locus_core::{
        TagFamily,
        config::{CornerRefinementMode, QuadExtractionMode},
    };

    #[test]
    fn regression_hub_tag36h11_720p_edlines_none() {
        let _g = common::telemetry::init("regression_hub_tag36h11_720p_edlines_none");
        run_render_tag_test(
            "locus_v1_tag36h11_1280x720",
            TagFamily::AprilTag36h11,
            RenderTagOpts {
                snapshot_suffix: "_edlines_none",
                refinement: Some(CornerRefinementMode::None),
                quad_mode: Some(QuadExtractionMode::EdLines),
                ..Default::default()
            },
        );
    }

    #[test]
    fn regression_hub_tag36h11_720p_edlines_gwlf() {
        let _g = common::telemetry::init("regression_hub_tag36h11_720p_edlines_gwlf");
        run_render_tag_test(
            "locus_v1_tag36h11_1280x720",
            TagFamily::AprilTag36h11,
            RenderTagOpts {
                snapshot_suffix: "_edlines_gwlf",
                refinement: Some(CornerRefinementMode::Gwlf),
                quad_mode: Some(QuadExtractionMode::EdLines),
                ..Default::default()
            },
        );
    }

    #[test]
    fn regression_hub_tag36h11_1080p_moments_culling() {
        let _g = common::telemetry::init("regression_hub_tag36h11_1080p_moments_culling");
        run_render_tag_test(
            "locus_v1_tag36h11_1920x1080",
            TagFamily::AprilTag36h11,
            RenderTagOpts {
                snapshot_suffix: "_moments_culling",
                moments_culling: Some((15.0, 0.15)),
                ..Default::default()
            },
        );
    }

    #[test]
    fn regression_hub_tag36h11_1080p_edlines() {
        let _g = common::telemetry::init("regression_hub_tag36h11_1080p_edlines");
        run_render_tag_test(
            "locus_v1_tag36h11_1920x1080",
            TagFamily::AprilTag36h11,
            RenderTagOpts {
                snapshot_suffix: "_edlines",
                quad_mode: Some(QuadExtractionMode::EdLines),
                ..Default::default()
            },
        );
    }

    #[test]
    fn regression_hub_tag36h11_1080p_edlines_moments() {
        let _g = common::telemetry::init("regression_hub_tag36h11_1080p_edlines_moments");
        run_render_tag_test(
            "locus_v1_tag36h11_1920x1080",
            TagFamily::AprilTag36h11,
            RenderTagOpts {
                snapshot_suffix: "_edlines_moments",
                quad_mode: Some(QuadExtractionMode::EdLines),
                moments_culling: Some((15.0, 0.15)),
                ..Default::default()
            },
        );
    }
}
