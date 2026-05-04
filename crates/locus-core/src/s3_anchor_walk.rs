//! S3 — Gradient-anchor walk Phase 1-5 replacement for EdLines.
//!
//! Operates on the gray image directly via Sobel gradients, bypassing the
//! binary boundary tracer's integer-pixel rounding floor (the empirically
//! confirmed cause of scene_0008's 4 px corner error per
//! `docs/engineering/edlines_s1_corner_exclusion_2026-05-04.md`).
//!
//! Pipeline:
//!   - Stage A: Sobel + 4-octant NMS → anchor pixels (high-|∇| local maxima).
//!   - Stage B: smart-routing chain walking with gradient-orientation gate.
//!   - Stage C: split chains into linear segments at curvature changes.
//!   - Stage D: pick top-4 longest segments with mutual-non-parallel gate.
//!   - Stage E: sub-pixel TLS line fit per segment.
//!   - Stage G: dense perpendicular refinement of Stage E lines.
//!   - Stage F: intersect adjacent CW-ordered lines → 4 corners.
//!
//! Reference design memo:
//! `docs/engineering/edlines_s3_anchor_walk_design_2026-05-04.md`.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

use crate::Point;
use crate::image::ImageView;
use crate::quad::CornerCovariances;
use crate::segmentation::ComponentStats;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the S3 anchor-walk Phase 1-5 replacement.
pub(crate) struct S3Config {
    /// Anchor gradient-magnitude threshold |∇|.  Default 16.0
    /// (intensity units, calibrated against scene_0008 in
    /// `tools/bench/edlines_s3_day2.py`).
    pub anchor_threshold: f64,
    /// Gradient-orientation coherence gate (cosine threshold).  Anchors
    /// whose gradient angle differs by more than `acos(coherence_min)` from
    /// the chain's current cursor are not added.  Default 30° → 0.866.
    pub coherence_min: f64,
    /// Stage C — RMS perpendicular distance threshold for splitting a chain.
    /// Default 1.5 px.
    pub split_max_rms: f64,
    /// Stage C — minimum segment length after splitting.  Default 8 anchors.
    pub min_seg_len: usize,
    /// Stage D — minimum segment euclidean span (pixels).  Default
    /// max(20, bbox_short / 4).  Caller scales by bbox.
    pub min_segment_span_floor_px: f64,
    /// Stage G — perpendicular search half-width (pixels).  Default 5.0.
    pub refine_search_half_width: f64,
    /// Stage G — sample step along the line tangent.  Default 1.0.
    pub refine_sample_step: f64,
    /// Stage G — minimum gradient magnitude at the sub-pixel parabolic
    /// vertex to accept a refinement point.  Default 8.0.
    pub refine_grad_min_mag: f64,
    /// Stage G — number of refinement iterations.  Default 2.
    pub refine_iters: usize,
}

impl S3Config {
    #[must_use]
    pub fn defaults() -> Self {
        Self {
            anchor_threshold: 16.0,
            coherence_min: (30.0_f64).to_radians().cos(),
            split_max_rms: 1.5,
            min_seg_len: 8,
            min_segment_span_floor_px: 20.0,
            refine_search_half_width: 5.0,
            refine_sample_step: 1.0,
            refine_grad_min_mag: 8.0,
            refine_iters: 2,
        }
    }
}

// ── Internal types ────────────────────────────────────────────────────────────

/// Sub-pixel line in homogeneous form.  `nx² + ny² = 1`.
#[derive(Clone, Copy, Debug)]
struct Line {
    nx: f64,
    ny: f64,
    d: f64,
    cx: f64,
    cy: f64,
}

#[derive(Clone, Copy)]
struct Anchor {
    x: i32,
    y: i32,
    gx: f64,
    gy: f64,
    mag: f64,
}

// ── Stage A — anchor pixel extraction ─────────────────────────────────────────

/// Extract anchor pixels within the bbox of a connected component.
///
/// An anchor is a pixel whose gradient magnitude exceeds `anchor_threshold`
/// AND is a local maximum of `|∇|` along its gradient direction (Canny-style
/// non-maximum suppression, quantised to 4 octants for speed).
fn extract_anchors<'a>(
    arena: &'a Bump,
    gray: &ImageView,
    stat: &ComponentStats,
    cfg: &S3Config,
) -> BumpVec<'a, Anchor> {
    let mut out = BumpVec::new_in(arena);
    let w = gray.width as i32;
    let h = gray.height as i32;
    // Pad by 2 px so Sobel + NMS stays in-bounds.
    let pad = 2_i32;
    let x_lo = (i32::from(stat.min_x) - pad).max(1);
    let y_lo = (i32::from(stat.min_y) - pad).max(1);
    let x_hi = (i32::from(stat.max_x) + pad).min(w - 2);
    let y_hi = (i32::from(stat.max_y) + pad).min(h - 2);
    if x_lo >= x_hi || y_lo >= y_hi {
        return out;
    }
    let stride = gray.stride as i32;
    let data = gray.data;

    // 3×3 Sobel: gx = (p_{j-1,i+1} + 2 p_{j,i+1} + p_{j+1,i+1})
    //                 - (p_{j-1,i-1} + 2 p_{j,i-1} + p_{j+1,i-1})
    //            gy similarly along the y axis.
    #[inline]
    fn sample(data: &[u8], stride: i32, x: i32, y: i32) -> i32 {
        i32::from(data[(y * stride + x) as usize])
    }

    let tau = cfg.anchor_threshold;

    for y in y_lo..=y_hi {
        for x in x_lo..=x_hi {
            // Sobel — 9 samples.
            let p00 = sample(data, stride, x - 1, y - 1);
            let p01 = sample(data, stride, x, y - 1);
            let p02 = sample(data, stride, x + 1, y - 1);
            let p10 = sample(data, stride, x - 1, y);
            let p12 = sample(data, stride, x + 1, y);
            let p20 = sample(data, stride, x - 1, y + 1);
            let p21 = sample(data, stride, x, y + 1);
            let p22 = sample(data, stride, x + 1, y + 1);
            let gx_i = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
            let gy_i = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
            let gx = gx_i as f64;
            let gy = gy_i as f64;
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < tau {
                continue;
            }
            // 4-octant NMS: pick step direction perpendicular to the edge,
            // i.e. along the gradient.
            let abs_gx = gx.abs();
            let abs_gy = gy.abs();
            let (dx, dy) = if abs_gy > 2.0 * abs_gx {
                (0_i32, 1_i32)
            } else if abs_gx > 2.0 * abs_gy {
                (1_i32, 0_i32)
            } else if (gx >= 0.0) == (gy >= 0.0) {
                (1_i32, 1_i32)
            } else {
                (1_i32, -1_i32)
            };
            let xn = x - dx;
            let yn = y - dy;
            let xp = x + dx;
            let yp = y + dy;
            // Bounds: x_lo..x_hi, y_lo..y_hi guarantees ±1 stays in image
            // since we padded by 2 above.
            let mag_n = {
                let g = sobel_grad(data, stride, xn, yn);
                (g.0 * g.0 + g.1 * g.1).sqrt()
            };
            let mag_p = {
                let g = sobel_grad(data, stride, xp, yp);
                (g.0 * g.0 + g.1 * g.1).sqrt()
            };
            // NMS: strictly greater on the negative-direction neighbour, weakly
            // greater on the positive-direction neighbour.  Asymmetry breaks
            // the tie that occurs on synthetic step edges where both rows of
            // the transition have identical gradient magnitude.
            if mag <= mag_n || mag < mag_p {
                continue;
            }
            out.push(Anchor {
                x,
                y,
                gx,
                gy,
                mag,
            });
        }
    }
    out
}

#[inline]
fn sobel_grad(data: &[u8], stride: i32, x: i32, y: i32) -> (f64, f64) {
    let s = |dx: i32, dy: i32| -> i32 {
        i32::from(data[((y + dy) * stride + (x + dx)) as usize])
    };
    let p00 = s(-1, -1);
    let p01 = s(0, -1);
    let p02 = s(1, -1);
    let p10 = s(-1, 0);
    let p12 = s(1, 0);
    let p20 = s(-1, 1);
    let p21 = s(0, 1);
    let p22 = s(1, 1);
    let gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
    let gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
    (gx as f64, gy as f64)
}

// ── Stage B — gradient-direction chain walking ────────────────────────────────

/// Walk anchors into chains following the edge tangent direction.
///
/// Smart-routing per Akinlar/Topal: from each anchor (sorted by `|∇|`
/// descending), step along the tangent (perpendicular to gradient).  At each
/// step, three candidates are considered (ahead, ahead-left, ahead-right);
/// the highest-`|∇|` candidate that is also an anchor with a coherent
/// gradient orientation (`cos(angle) ≥ coherence_min`) is appended.
///
/// Returns a list of chains; each chain is a list of indices into the
/// `anchors` slice.
fn walk_chains<'a>(
    arena: &'a Bump,
    anchors: &[Anchor],
    gray_w: usize,
    gray_h: usize,
    cfg: &S3Config,
) -> BumpVec<'a, BumpVec<'a, u32>> {
    let n = anchors.len();
    // Spatial index: (x, y) → anchor index.  Use a flat BumpVec mapped over
    // the gray-image extent for O(1) lookup; fits in arena.
    let map_size = gray_w * gray_h;
    let mut pos_to_idx: BumpVec<'a, i32> = BumpVec::with_capacity_in(map_size, arena);
    pos_to_idx.resize(map_size, -1_i32);
    for (i, a) in anchors.iter().enumerate() {
        let idx = a.y as usize * gray_w + a.x as usize;
        pos_to_idx[idx] = i as i32;
    }

    let mut consumed: BumpVec<'a, bool> = BumpVec::with_capacity_in(n, arena);
    consumed.resize(n, false);

    // Sort anchor indices by |∇| descending.  Use a separate index vector.
    let mut order: BumpVec<'a, u32> = BumpVec::with_capacity_in(n, arena);
    order.extend((0..n as u32).collect::<Vec<_>>());
    order.sort_by(|&a, &b| anchors[b as usize].mag.partial_cmp(&anchors[a as usize].mag).unwrap());

    // 8-connected step directions in CCW order.
    const DIRS: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    fn dir_index(dx: i32, dy: i32) -> usize {
        DIRS.iter().position(|&d| d == (dx, dy)).unwrap_or(0)
    }
    fn rotate_45(dx: i32, dy: i32, sign: i32) -> (i32, i32) {
        let i = dir_index(dx, dy) as i32;
        let new_i = ((i + sign).rem_euclid(8)) as usize;
        DIRS[new_i]
    }
    fn quantised_tangent(gx: f64, gy: f64, sign: i32) -> (i32, i32) {
        // Tangent = (-gy, gx)/|g| × sign.
        let tx = -gy * sign as f64;
        let ty = gx * sign as f64;
        let ax = tx.abs();
        let ay = ty.abs();
        if ax > 2.0 * ay {
            (if tx > 0.0 { 1 } else { -1 }, 0)
        } else if ay > 2.0 * ax {
            (0, if ty > 0.0 { 1 } else { -1 })
        } else {
            (
                if tx > 0.0 { 1 } else { -1 },
                if ty > 0.0 { 1 } else { -1 },
            )
        }
    }

    let coh_min = cfg.coherence_min;
    let max_chain_len = anchors.len(); // hard cap

    let mut chains: BumpVec<'a, BumpVec<'a, u32>> = BumpVec::new_in(arena);

    for &start_u32 in order.iter() {
        let start = start_u32 as usize;
        if consumed[start] {
            continue;
        }
        consumed[start] = true;
        let mut chain: BumpVec<'a, u32> = BumpVec::with_capacity_in(64, arena);
        chain.push(start_u32);

        for direction_sign in [1_i32, -1_i32] {
            let mut cursor = start;
            let mut step = quantised_tangent(anchors[start].gx, anchors[start].gy, direction_sign);
            for _ in 0..max_chain_len {
                let cur = anchors[cursor];
                let cx = cur.x;
                let cy = cur.y;
                let candidates = [
                    (cx + step.0, cy + step.1),
                    {
                        let r = rotate_45(step.0, step.1, 1);
                        (cx + r.0, cy + r.1)
                    },
                    {
                        let r = rotate_45(step.0, step.1, -1);
                        (cx + r.0, cy + r.1)
                    },
                ];
                let cur_g_norm = (cur.gx * cur.gx + cur.gy * cur.gy).sqrt();
                let mut best_idx: i32 = -1;
                let mut best_score = -1.0_f64;
                let mut best_step = (0_i32, 0_i32);
                for c in candidates {
                    if c.0 < 0 || c.1 < 0 || c.0 >= gray_w as i32 || c.1 >= gray_h as i32 {
                        continue;
                    }
                    let map_i = (c.1 as usize) * gray_w + c.0 as usize;
                    let idx = pos_to_idx[map_i];
                    if idx < 0 {
                        continue;
                    }
                    let ci = idx as usize;
                    if consumed[ci] {
                        continue;
                    }
                    let cand = anchors[ci];
                    let cand_g_norm = (cand.gx * cand.gx + cand.gy * cand.gy).sqrt();
                    if cand_g_norm < 1e-9 || cur_g_norm < 1e-9 {
                        continue;
                    }
                    let cos_sim = (cur.gx * cand.gx + cur.gy * cand.gy).abs()
                        / (cur_g_norm * cand_g_norm);
                    if cos_sim < coh_min {
                        continue;
                    }
                    let score = cand.mag * cos_sim;
                    if score > best_score {
                        best_score = score;
                        best_idx = idx;
                        best_step = (c.0 - cx, c.1 - cy);
                    }
                }
                if best_idx < 0 {
                    break;
                }
                let bi = best_idx as usize;
                consumed[bi] = true;
                if direction_sign > 0 {
                    chain.push(best_idx as u32);
                } else {
                    chain.insert(0, best_idx as u32);
                }
                cursor = bi;
                step = best_step;
            }
        }
        chains.push(chain);
    }
    chains
}

// ── Top-level entry ───────────────────────────────────────────────────────────

/// Top-level S3 pipeline.  Returns `None` on any per-stage failure (caller
/// falls back to the current EdLines path or rejects the candidate).
pub(crate) fn run_anchor_walk(
    arena: &Bump,
    gray: &ImageView,
    stat: &ComponentStats,
    cfg: &S3Config,
) -> Option<([Point; 4], CornerCovariances)> {
    let anchors = extract_anchors(arena, gray, stat, cfg);
    if anchors.len() < 32 {
        return None;
    }
    let chains = walk_chains(arena, anchors.as_slice(), gray.width, gray.height, cfg);
    let _ = chains;
    // Stages C-G live in subsequent commits.
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    fn make_synthetic_square(canvas: usize, x0: usize, y0: usize, size: usize) -> Vec<u8> {
        let mut img = vec![170_u8; canvas * canvas];
        for y in y0..(y0 + size) {
            for x in x0..(x0 + size) {
                img[y * canvas + x] = 32;
            }
        }
        img
    }

    fn stats_for_square(x0: u16, y0: u16, size: u16) -> ComponentStats {
        ComponentStats {
            pixel_count: u32::from(size) * u32::from(size),
            min_x: x0,
            min_y: y0,
            max_x: x0 + size - 1,
            max_y: y0 + size - 1,
            first_pixel_x: x0,
            first_pixel_y: y0,
            m10: u64::from(x0) * u64::from(size),
            m01: u64::from(y0) * u64::from(size),
            m20: 0,
            m02: 0,
            m11: 0,
        }
    }

    #[test]
    fn chain_walking_produces_4_dominant_chains_on_square() {
        let canvas = 100;
        let x0 = 30;
        let y0 = 30;
        let size = 40;
        let pixels = make_synthetic_square(canvas, x0, y0, size);
        let img = ImageView::new(&pixels, canvas, canvas, canvas).unwrap();
        let stats = stats_for_square(x0 as u16, y0 as u16, size as u16);
        let arena = Bump::new();
        let cfg = S3Config::defaults();
        let anchors = extract_anchors(&arena, &img, &stats, &cfg);
        let chains = walk_chains(&arena, anchors.as_slice(), img.width, img.height, &cfg);
        // Sort chains by length, take top 4.
        let mut lens: Vec<usize> = chains.iter().map(BumpVec::len).collect();
        lens.sort_unstable_by(|a, b| b.cmp(a));
        // The 4 longest chains should each cover roughly one edge (~size
        // anchors per edge). Allow wide tolerance for synthetic boundary
        // effects.
        assert!(lens.len() >= 4, "fewer than 4 chains, got {}", lens.len());
        for &l in &lens[..4] {
            assert!(
                l >= 20,
                "expected each top-4 chain ≥ 20 anchors on synthetic square, got {l}"
            );
        }
    }

    #[test]
    fn anchor_extraction_finds_square_edges() {
        let canvas = 100;
        let x0 = 30;
        let y0 = 30;
        let size = 40;
        let pixels = make_synthetic_square(canvas, x0, y0, size);
        let img = ImageView::new(&pixels, canvas, canvas, canvas).unwrap();
        let stats = stats_for_square(x0 as u16, y0 as u16, size as u16);
        let arena = Bump::new();
        let cfg = S3Config::defaults();
        let anchors = extract_anchors(&arena, &img, &stats, &cfg);
        // For a clean 40-px synthetic square (sharp edges), Stage A should
        // produce ~4*size = ~160 anchors after NMS (one per row/col on each
        // edge, plus corners).  Allow a wide tolerance band.
        assert!(
            anchors.len() >= 100 && anchors.len() <= 250,
            "expected 100-250 anchors on synthetic square, got {}",
            anchors.len()
        );
    }
}
