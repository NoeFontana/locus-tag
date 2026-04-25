#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::panic
)]
//! Pins the `render_tag_hub` 2160p contract on the three scenes (0006, 0026,
//! 0033) that previously missed: the largest GT-overlapping CCL component
//! must pass the production geometric gates, *and* the detector must return
//! at least one decoded tag. Together these guard against an upstream
//! segmentation regression and against an `extract_quads_soa` truncation /
//! ordering regression (see `pixel_count_descending_order` in `quad.rs`).

#[cfg(feature = "bench-internals")]
mod common;

#[cfg(feature = "bench-internals")]
mod render_tag_2160p {
    use super::common;
    use bumpalo::Bump;
    use common::hub::HubProvider;
    use common::hub::load_detect_options;
    use common::resolve_hub_root;
    use locus_core::Detector;
    use locus_core::DetectorConfig;
    use locus_core::ImageView;
    use locus_core::TagFamily;
    use locus_core::bench_api::{ComponentStats, ThresholdEngine, label_components_lsl};

    const HUB_CONFIG: &str = "locus_v1_tag36h11_3840x2160";
    const PINNED_SCENES: &[&str] = &[
        "scene_0006_cam_0000.png",
        "scene_0026_cam_0000.png",
        "scene_0033_cam_0000.png",
    ];

    #[test]
    fn render_tag_hub_2160p_decodes_largest_components() {
        let Ok(hub_dir) = std::env::var("LOCUS_HUB_DATASET_DIR") else {
            println!("Skipping: LOCUS_HUB_DATASET_DIR is unset.");
            return;
        };
        let dataset_path = resolve_hub_root(&hub_dir).join(HUB_CONFIG);
        if !dataset_path.is_dir() {
            println!("Skipping: dataset not in cache: {HUB_CONFIG}");
            return;
        }

        let provider = HubProvider::new(&dataset_path)
            .expect("hub provider should construct from a populated dataset directory");

        let config = DetectorConfig::from_profile("render_tag_hub");
        let mut options = load_detect_options(&dataset_path);
        options.families = vec![TagFamily::AprilTag36h11];
        let mut detector = Detector::with_config(config);
        detector.set_families(&options.families);

        for fname in PINNED_SCENES {
            let img_path = dataset_path.join("images").join(fname);
            let img = image::open(&img_path)
                .unwrap_or_else(|_| panic!("failed to open {}", img_path.display()))
                .into_luma8();
            let (w, h) = img.dimensions();
            let raw = img.into_raw();

            let view =
                ImageView::new(&raw, w as usize, h as usize, w as usize).expect("valid image view");

            let gt_corners = provider
                .gt_map
                .get(*fname)
                .and_then(|gt| gt.tags.values().next().copied())
                .unwrap_or_else(|| panic!("no GT tag for scene {fname}"));

            let report = inspect_largest_component(&config, &view, gt_corners);
            let detections = detector
                .detect(
                    &view,
                    options.intrinsics.as_ref(),
                    options.tag_size,
                    options.pose_estimation_mode,
                    false,
                )
                .expect("detector should not error on a valid image");

            println!(
                "{fname}: bbox-cov {:.1}% | comp-area {} px | comp-bbox {}x{} | fill {:.2} | aspect {:.2} | passes-gates {} | detections {}",
                report.bbox_coverage * 100.0,
                report.pixel_count,
                report.bbox_w,
                report.bbox_h,
                report.fill_ratio,
                report.aspect,
                report.passes_geometric_gates,
                detections.len(),
            );

            assert!(
                report.passes_geometric_gates,
                "{fname}: largest GT-overlapping component fails geometric gates"
            );
            assert!(
                !detections.is_empty(),
                "{fname}: detector returned zero detections at 2160p"
            );
        }
    }

    struct ComponentReport {
        bbox_coverage: f64,
        pixel_count: u32,
        bbox_w: u32,
        bbox_h: u32,
        fill_ratio: f64,
        aspect: f64,
        passes_geometric_gates: bool,
    }

    fn inspect_largest_component(
        config: &DetectorConfig,
        view: &ImageView,
        gt_corners: [[f64; 2]; 4],
    ) -> ComponentReport {
        let arena = Bump::new();

        let engine = ThresholdEngine::from_config(config);
        let tile_stats = engine.compute_tile_stats(&arena, view);

        let binarized = arena.alloc_slice_fill_copy(view.width * view.height, 0u8);
        let threshold_map = arena.alloc_slice_fill_copy(view.width * view.height, 0u8);
        engine.apply_threshold_with_map(&arena, view, &tile_stats, binarized, threshold_map);

        let labels = label_components_lsl(
            &arena,
            view,
            threshold_map,
            config.segmentation_connectivity == locus_core::config::SegmentationConnectivity::Eight,
            config.quad_min_area,
        );

        let (g_min_x, g_min_y, g_max_x, g_max_y) = bbox_from_corners(gt_corners);
        let g_min_x = g_min_x.floor() as u16;
        let g_min_y = g_min_y.floor() as u16;
        let g_max_x = g_max_x.ceil() as u16;
        let g_max_y = g_max_y.ceil() as u16;
        let gt_area =
            f64::from(g_max_x.saturating_sub(g_min_x)) * f64::from(g_max_y.saturating_sub(g_min_y));

        let mut best_overlap: u64 = 0;
        let mut best: Option<ComponentStats> = None;
        for stat in &labels.component_stats {
            if stat.max_x < g_min_x || stat.min_x > g_max_x {
                continue;
            }
            if stat.max_y < g_min_y || stat.min_y > g_max_y {
                continue;
            }
            let ix0 = u32::from(stat.min_x.max(g_min_x));
            let iy0 = u32::from(stat.min_y.max(g_min_y));
            let ix1 = u32::from(stat.max_x.min(g_max_x));
            let iy1 = u32::from(stat.max_y.min(g_max_y));
            let overlap = u64::from(ix1.saturating_sub(ix0)) * u64::from(iy1.saturating_sub(iy0));
            if overlap > best_overlap {
                best_overlap = overlap;
                best = Some(*stat);
            }
        }

        let Some(stat) = best else {
            return ComponentReport {
                bbox_coverage: 0.0,
                pixel_count: 0,
                bbox_w: 0,
                bbox_h: 0,
                fill_ratio: 0.0,
                aspect: 0.0,
                passes_geometric_gates: false,
            };
        };

        let bbox_w = u32::from(stat.max_x - stat.min_x) + 1;
        let bbox_h = u32::from(stat.max_y - stat.min_y) + 1;
        let bbox_area = bbox_w * bbox_h;
        let fill_ratio = f64::from(stat.pixel_count) / f64::from(bbox_area.max(1));
        let aspect = f64::from(bbox_w.max(bbox_h)) / f64::from(bbox_w.min(bbox_h).max(1));

        let img_area = view.width * view.height;
        let img_max = (img_area * 9 / 10) as u32;
        let area_ok = bbox_area >= config.quad_min_area && bbox_area <= img_max;
        let aspect_ok = aspect <= f64::from(config.quad_max_aspect_ratio);
        let fill_ok = fill_ratio >= f64::from(config.quad_min_fill_ratio)
            && fill_ratio <= f64::from(config.quad_max_fill_ratio);

        ComponentReport {
            bbox_coverage: if gt_area > 0.0 {
                best_overlap as f64 / gt_area
            } else {
                0.0
            },
            pixel_count: stat.pixel_count,
            bbox_w,
            bbox_h,
            fill_ratio,
            aspect,
            passes_geometric_gates: area_ok && aspect_ok && fill_ok,
        }
    }

    fn bbox_from_corners(corners: [[f64; 2]; 4]) -> (f64, f64, f64, f64) {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for [x, y] in corners {
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
        }
        (min_x, min_y, max_x, max_y)
    }
}
