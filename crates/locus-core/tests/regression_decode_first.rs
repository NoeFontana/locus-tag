//! Decode-first ordering (`quad_refine_before_decode = false`) must report the markers, and
//! the corners, that refine-first ordering reports: it only changes which candidates get
//! refined, not how a decoded marker is refined.
#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]

use locus_core::{Detector, DetectorConfig, ImageView, TagFamily};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn detect(config: DetectorConfig, img: &ImageView) -> BTreeMap<u32, [[f32; 2]; 4]> {
    let mut detector = Detector::with_config(config);
    detector.set_families(&[TagFamily::AprilTag36h11]);
    let detections = detector.detect(img, None, None, false).expect("detect");
    (0..detections.len())
        .map(|i| {
            let c = detections.corners[i];
            (detections.ids[i], c.map(|p| [p.x, p.y]))
        })
        .collect()
}

#[test]
fn decode_first_matches_refine_first_on_icra_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/icra2020/0037.png");
    let img = image::open(path).expect("fixture").into_luma8();
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let data = img.into_raw();
    let view = ImageView::new(&data, w, h, w).expect("view");

    for profile in ["standard", "grid"] {
        let decode_first = DetectorConfig::from_profile(profile);
        assert!(!decode_first.quad_refine_before_decode);
        let refine_first = DetectorConfig {
            quad_refine_before_decode: true,
            ..decode_first
        };
        let a = detect(refine_first, &view);
        let b = detect(decode_first, &view);

        assert!(
            a.len() > 100,
            "{profile}: fixture should decode, got {}",
            a.len()
        );
        assert_eq!(
            a.keys().collect::<Vec<_>>(),
            b.keys().collect::<Vec<_>>(),
            "{profile}: decoded ids differ"
        );
        // A marker that decodes under both orders gets the same refinement. The exceptions
        // are markers whose refine-first verification failed at scale 1 and fell back to the
        // quad-stage corners; decode-first verifies at the scale that decoded.
        let max_dev = |id: &u32| {
            a[id]
                .iter()
                .zip(&b[id])
                .map(|(p, q)| (p[0] - q[0]).abs().max((p[1] - q[1]).abs()))
                .fold(0.0_f32, f32::max)
        };
        let same = a.keys().filter(|id| max_dev(id) < 1e-3).count();
        assert!(
            same * 100 >= a.len() * 98,
            "{profile}: only {same}/{} markers kept refine-first corners",
            a.len()
        );
        let worst = a.keys().map(max_dev).fold(0.0_f32, f32::max);
        assert!(worst < 0.25, "{profile}: corner deviation {worst} px");
    }
}
