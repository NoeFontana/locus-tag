//! Recall autopsy: for each ground-truth tag, the pipeline stage that lost it.
//!
//! A recall number says how many tags were missed, not why, and the fixes for "the threshold
//! never saw the border", "the border fused with its neighbours" and "the quad decoded to the
//! wrong bits" have nothing in common. This replays one frame with telemetry, relabels the
//! frame's own binarized image and attributes every missed tag to the first stage that lost it:
//!
//! | Stage | Meaning |
//! |---|---|
//! | `not_foreground` | under half the border ring is below the threshold |
//! | `too_small` | the border's component is under `quad.min_area` |
//! | `fused_tags` | the border's component also holds another truth tag's border |
//! | `fused_structure` | the component's box is over twice the tag's (connectors, clutter) |
//! | `no_quad` | a tag-sized component that produced no candidate (quad gates, funnel) |
//! | `decode_failed` | a candidate over the tag that failed to decode |
//! | `wrong_id` | decoded, but to another id |
//! | `detected` | decoded to the right id |
//!
//! Truth comes either from a hub dataset (`--dataset <dir>` with `rich_truth.json` and
//! `images/`) or from a JSONL file (`--truth <file>`), one frame per line:
//! `{"image": "<path, relative to the file>", "tags": [{"id": 3, "corners": [[x, y], ...]}]}`,
//! corners in the detector's convention and order.
//!
//! ```text
//! cargo run --release --features bench-internals --example recall_autopsy -- \
//!     --dataset tests/data/hub_cache/aprilgrid_golden_v1_1920x1080 --profile grid \
//!     [--family 36h11] [--border-bits 1] [--out autopsy.jsonl]
//! ```
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    missing_docs
)]

use bumpalo::Bump;
use locus_core::bench_api::{Homography, get_dictionary, label_components_with_stats};
use locus_core::{Detector, DetectorConfig, ImageView, TagFamily};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const STAGES: [&str; 8] = [
    "detected",
    "wrong_id",
    "decode_failed",
    "no_quad",
    "fused_structure",
    "fused_tags",
    "too_small",
    "not_foreground",
];
/// Side bins (px) of the summary.
const SIDE_BINS: [f64; 5] = [0.0, 16.0, 24.0, 32.0, 48.0];

struct TruthTag {
    id: u32,
    corners: [[f64; 2]; 4],
}

struct Frame {
    image: PathBuf,
    tags: Vec<TruthTag>,
}

struct Args {
    frames: Vec<Frame>,
    profile: String,
    family: TagFamily,
    border_bits: usize,
    out: Option<PathBuf>,
}

fn family(name: &str) -> TagFamily {
    match name.trim_start_matches("tag") {
        "36h11" => TagFamily::AprilTag36h11,
        "16h5" => TagFamily::AprilTag16h5,
        "aruco4x4_50" | "4x4_50" => TagFamily::ArUco4x4_50,
        "aruco4x4_100" | "4x4_100" => TagFamily::ArUco4x4_100,
        "aruco6x6_250" | "6x6_250" => TagFamily::ArUco6x6_250,
        "mip36h12" | "36h12" => TagFamily::ArUcoMip36h12,
        other => panic!("unknown family {other}"),
    }
}

fn corners_of(v: &Value) -> [[f64; 2]; 4] {
    let c: Vec<[f64; 2]> = serde_json::from_value(v.clone()).expect("four [x, y] corners");
    [c[0], c[1], c[2], c[3]]
}

fn hub_frames(dir: &Path) -> Vec<Frame> {
    let truth: Value = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(dir.join("rich_truth.json")).expect("rich_truth.json"),
    ))
    .expect("valid rich_truth.json");
    let mut frames: BTreeMap<String, Vec<TruthTag>> = BTreeMap::new();
    for r in truth["records"].as_array().expect("records") {
        if r["record_type"] != "TAG" || !r["corners"].is_array() {
            continue;
        }
        let image = r["image_id"].as_str().expect("image_id").to_owned();
        frames.entry(image).or_default().push(TruthTag {
            id: r["tag_id"].as_u64().expect("tag_id") as u32,
            corners: corners_of(&r["corners"]),
        });
    }
    frames
        .into_iter()
        .map(|(image, tags)| Frame {
            image: dir.join("images").join(format!("{image}.png")),
            tags,
        })
        .collect()
}

fn jsonl_frames(path: &Path) -> Vec<Frame> {
    let base = path.parent().unwrap_or(Path::new("."));
    std::fs::read_to_string(path)
        .expect("truth file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: Value = serde_json::from_str(line).expect("truth line");
            Frame {
                image: base.join(v["image"].as_str().expect("image")),
                tags: v["tags"]
                    .as_array()
                    .expect("tags")
                    .iter()
                    .map(|t| TruthTag {
                        id: t["id"].as_u64().expect("id") as u32,
                        corners: corners_of(&t["corners"]),
                    })
                    .collect(),
            }
        })
        .collect()
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let (mut frames, mut profile, mut fam, mut border_bits, mut out) = (
        None,
        "standard".to_owned(),
        TagFamily::AprilTag36h11,
        1,
        None,
    );
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--dataset" => frames = Some(hub_frames(Path::new(&value()))),
            "--truth" => frames = Some(jsonl_frames(Path::new(&value()))),
            "--profile" => profile = value(),
            "--family" => fam = family(&value()),
            "--border-bits" => border_bits = value().parse().expect("--border-bits"),
            "--out" => out = Some(PathBuf::from(value())),
            other => panic!("unknown argument {other}"),
        }
    }
    Args {
        frames: frames.expect("--dataset or --truth"),
        profile,
        family: fam,
        border_bits,
        out,
    }
}

fn mean_side(c: &[[f64; 2]; 4]) -> f64 {
    (0..4)
        .map(|j| (c[j][0] - c[(j + 1) % 4][0]).hypot(c[j][1] - c[(j + 1) % 4][1]))
        .sum::<f64>()
        / 4.0
}

fn centre(c: &[[f64; 2]; 4]) -> [f64; 2] {
    [
        c.iter().map(|p| p[0]).sum::<f64>() / 4.0,
        c.iter().map(|p| p[1]).sum::<f64>() / 4.0,
    ]
}

fn bbox_area(c: &[[f64; 2]; 4]) -> f64 {
    let (x0, x1) = c
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[0]), b.max(p[0])));
    let (y0, y1) = c
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[1]), b.max(p[1])));
    (x1 - x0).max(1.0) * (y1 - y0).max(1.0)
}

fn quad_f64(c: &[locus_core::bench_api::Point2f; 4]) -> [[f64; 2]; 4] {
    c.map(|p| [f64::from(p.x), f64::from(p.y)])
}

/// Whether `p` lies inside the convex quad `q` (either winding).
fn inside(q: &[[f64; 2]; 4], p: [f64; 2]) -> bool {
    let cross = |j: usize| {
        let (a, b) = (q[j], q[(j + 1) % 4]);
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    };
    let s: [f64; 4] = core::array::from_fn(cross);
    s.iter().all(|&v| v >= 0.0) || s.iter().all(|&v| v <= 0.0)
}

/// Image points along the middle of the tag's black border ring: `border_bits` cells wide, of
/// `cells` across. Sampled through the truth homography, 8 per border cell.
fn border_ring(corners: &[[f64; 2]; 4], cells: usize, border_bits: usize) -> Vec<[f64; 2]> {
    let h = Homography::square_to_quad(corners).expect("non-degenerate truth quad");
    // Canonical [-1, 1]: the ring's centre line sits half a border in from the edge.
    let r = 1.0 - border_bits as f64 / cells as f64;
    let n = 8 * cells;
    (0..n)
        .flat_map(|k| {
            let t = -r + 2.0 * r * (k as f64 + 0.5) / n as f64;
            [[t, -r], [t, r], [-r, t], [r, t]]
        })
        .map(|p| h.project(p))
        .collect()
}

fn main() {
    let args = parse_args();
    let config = DetectorConfig::from_profile(&args.profile);
    assert!(
        config.decimation <= 1 && config.upscale_factor <= 1,
        "the binarized image must be on the input grid"
    );
    let eight =
        config.segmentation_connectivity == locus_core::config::SegmentationConnectivity::Eight;
    let min_area = config.quad_min_area;
    let cells = get_dictionary(args.family).dimension + 2 * args.border_bits;
    let mut detector = Detector::with_config(config);
    detector.set_families(&[args.family]);

    let mut out = args
        .out
        .as_ref()
        .map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("create --out")));
    // counts[side bin][stage]
    let mut counts = vec![[0usize; STAGES.len()]; SIDE_BINS.len()];

    for frame in &args.frames {
        let img = image::open(&frame.image)
            .unwrap_or_else(|e| panic!("{}: {e}", frame.image.display()))
            .into_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let raw = img.into_raw();
        let view = ImageView::new(&raw, w, h, w).expect("image view");
        let batch = detector.detect(&view, None, None, true).expect("detect");

        let valid: Vec<(u32, [[f64; 2]; 4])> = (0..batch.len())
            .map(|i| (batch.ids[i], quad_f64(&batch.corners[i])))
            .collect();
        let rejected: Vec<([[f64; 2]; 4], f32)> = batch
            .rejected_corners
            .iter()
            .zip(batch.rejected_error_rates)
            .map(|(c, &e)| (quad_f64(c), e))
            .collect();
        let telemetry = batch.telemetry.expect("telemetry was requested");
        assert_eq!((telemetry.width, telemetry.height), (w, h));
        // SAFETY: with `debug_telemetry` the binarized buffer is a `width × height` (stride
        // `width`) slice in the detector's arena, alive until the next `detect`, which this
        // loop iteration does not reach before its last use of `binary`.
        #[allow(unsafe_code)]
        let binary = unsafe { std::slice::from_raw_parts(telemetry.binarized_ptr, w * h) }.to_vec();

        let arena = Bump::new();
        let labels = label_components_with_stats(&arena, &binary, w, h, eight);
        let label_at = |p: [f64; 2]| -> u32 {
            let (x, y) = (p[0].floor(), p[1].floor());
            if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
                return 0;
            }
            labels.labels[y as usize * w + x as usize]
        };

        // Majority label of each truth tag's border ring, and its foreground fraction.
        let rings: Vec<(u32, f64)> = frame
            .tags
            .iter()
            .map(|t| {
                let pts = border_ring(&t.corners, cells, args.border_bits);
                let mut votes: BTreeMap<u32, usize> = BTreeMap::new();
                for &p in &pts {
                    let l = label_at(p);
                    if l != 0 {
                        *votes.entry(l).or_default() += 1;
                    }
                }
                let fg: usize = votes.values().sum();
                let major = votes.iter().max_by_key(|e| *e.1).map_or(0, |e| *e.0);
                (major, fg as f64 / pts.len() as f64)
            })
            .collect();

        for (k, tag) in frame.tags.iter().enumerate() {
            let side = mean_side(&tag.corners);
            let c = centre(&tag.corners);
            let over = |q: &[[f64; 2]; 4]| inside(&tag.corners, centre(q)) && inside(q, c);
            let (label, fg) = rings[k];
            let stat = (label != 0).then(|| &labels.component_stats[label as usize - 1]);
            let shared = rings
                .iter()
                .enumerate()
                .filter(|&(m, r)| m != k && label != 0 && r.0 == label)
                .count();
            let box_ratio = stat.map_or(0.0, |s| {
                f64::from(s.max_x - s.min_x + 1) * f64::from(s.max_y - s.min_y + 1)
                    / bbox_area(&tag.corners)
            });
            let decoded = valid.iter().find(|(_, q)| over(q));
            let failed = rejected
                .iter()
                .filter(|(q, _)| over(q))
                .map(|r| r.1)
                .reduce(f32::min);
            let stage = match (decoded, failed) {
                (Some((id, _)), _) if *id == tag.id => "detected",
                (Some(_), _) => "wrong_id",
                (None, Some(_)) => "decode_failed",
                _ if fg < 0.5 => "not_foreground",
                _ if stat.is_some_and(|s| s.pixel_count < min_area) => "too_small",
                _ if shared > 0 => "fused_tags",
                _ if box_ratio > 2.0 => "fused_structure",
                _ => "no_quad",
            };
            let bin = SIDE_BINS.iter().rposition(|&b| side >= b).unwrap_or(0);
            counts[bin][STAGES.iter().position(|&s| s == stage).unwrap()] += 1;
            if let Some(out) = out.as_mut() {
                let line = json!({
                    "image": frame.image.file_name().map(|f| f.to_string_lossy()),
                    "id": tag.id,
                    "side": side,
                    "stage": stage,
                    "ring_foreground": fg,
                    "component_px": stat.map(|s| s.pixel_count),
                    "component_box_ratio": box_ratio,
                    "tags_sharing_component": shared,
                    "best_rejected_error": failed,
                });
                writeln!(out, "{line}").expect("write --out");
            }
        }
    }

    let total: usize = counts.iter().flatten().sum();
    println!(
        "{} frames, {} truth tags, profile `{}`, border {} bit(s)\n",
        args.frames.len(),
        total,
        args.profile,
        args.border_bits
    );
    print!("{:>10}", "side px");
    for s in STAGES {
        print!(" {s:>15}");
    }
    println!();
    for (b, row) in counts.iter().enumerate() {
        let n: usize = row.iter().sum();
        if n == 0 {
            continue;
        }
        let label = SIDE_BINS
            .get(b + 1)
            .map_or(format!("≥{}", SIDE_BINS[b]), |hi| {
                format!("{}–{hi}", SIDE_BINS[b])
            });
        print!("{label:>10}");
        for &c in row {
            print!(
                " {:>15}",
                format!("{c} ({:.0}%)", 100.0 * c as f64 / n as f64)
            );
        }
        println!();
    }
    print!("{:>10}", "all");
    for s in 0..STAGES.len() {
        let c: usize = counts.iter().map(|r| r[s]).sum();
        print!(
            " {:>15}",
            format!("{c} ({:.1}%)", 100.0 * c as f64 / total.max(1) as f64)
        );
    }
    println!();
}
