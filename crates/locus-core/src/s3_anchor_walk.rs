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

// ── Stage C — chain → linear segments ─────────────────────────────────────────

/// TLS line through 2D points (sub-pixel-aware).  Returns the line and its
/// per-point RMS perpendicular distance.
fn fit_line_tls(pts: &[(f64, f64)]) -> Option<Line> {
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    for &(x, y) in pts {
        sx += x;
        sy += y;
    }
    let cx = sx / n;
    let cy = sy / n;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for &(x, y) in pts {
        let dx = x - cx;
        let dy = y - cy;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    sxx /= n;
    syy /= n;
    sxy /= n;
    // Smallest-eigenvalue eigenvector of the 2×2 covariance.
    let trace = sxx + syy;
    let disc = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
    let lam_min = 0.5 * (trace - disc);
    let (nx, ny) = if sxy.abs() > 1e-12 {
        // Eigenvector of (sxx-λ, sxy; sxy, syy-λ) for λ=λ_min.
        let vx = sxy;
        let vy = lam_min - sxx;
        let len = (vx * vx + vy * vy).sqrt();
        if len < 1e-12 {
            (1.0, 0.0)
        } else {
            (vx / len, vy / len)
        }
    } else if sxx < syy {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let d = -(nx * cx + ny * cy);
    Some(Line { nx, ny, d, cx, cy })
}

fn line_rms(line: &Line, pts: &[(f64, f64)]) -> f64 {
    let mut s = 0.0_f64;
    for &(x, y) in pts {
        let r = line.nx * x + line.ny * y + line.d;
        s += r * r;
    }
    (s / pts.len() as f64).sqrt()
}

/// Iteratively split a chain at its worst-fitting point until each segment
/// has RMS perpendicular distance ≤ `max_rms` or is shorter than `2 * min_seg_len`.
///
/// `chain_pts` are the integer-anchor (x, y) positions for one chain.
/// Returns a list of (start, end) byte ranges into `chain_pts` for the
/// surviving segments.
fn split_chain<'a>(
    arena: &'a Bump,
    chain_pts: &[(f64, f64)],
    max_rms: f64,
    min_seg_len: usize,
) -> BumpVec<'a, (usize, usize)> {
    let mut out: BumpVec<'a, (usize, usize)> = BumpVec::new_in(arena);
    fn recurse(
        out: &mut BumpVec<(usize, usize)>,
        all: &[(f64, f64)],
        start: usize,
        end: usize,
        max_rms: f64,
        min_seg_len: usize,
    ) {
        let len = end - start;
        if len < 2 * min_seg_len {
            if len >= min_seg_len {
                out.push((start, end));
            }
            return;
        }
        let pts = &all[start..end];
        let line = match fit_line_tls(pts) {
            Some(l) => l,
            None => return,
        };
        if line_rms(&line, pts) <= max_rms {
            out.push((start, end));
            return;
        }
        let mut worst = 0_usize;
        let mut worst_r = -1.0_f64;
        for (i, &(x, y)) in pts.iter().enumerate() {
            let r = (line.nx * x + line.ny * y + line.d).abs();
            if r > worst_r {
                worst_r = r;
                worst = i;
            }
        }
        let split = start + worst;
        if worst < min_seg_len {
            recurse(out, all, split + 1, end, max_rms, min_seg_len);
        } else if worst > len - min_seg_len {
            recurse(out, all, start, split, max_rms, min_seg_len);
        } else {
            recurse(out, all, start, split, max_rms, min_seg_len);
            recurse(out, all, split + 1, end, max_rms, min_seg_len);
        }
    }
    recurse(&mut out, chain_pts, 0, chain_pts.len(), max_rms, min_seg_len);
    out
}

// ── Stage D — top-4 segment selection ────────────────────────────────────────

/// Pick the 4 longest segments such that at least one pair has angular
/// difference ≥ 30° (rejects an all-parallel set).
///
/// Each segment is identified by `(chain_index, start, end)` into the
/// caller's chain list.  Returns indices into the input `segments` slice.
fn select_top4(segments: &[Segment], min_span_px: f64) -> Option<[usize; 4]> {
    // Filter by Euclidean span between segment endpoints.
    let mut candidates: Vec<(usize, f64)> = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let dx = seg.end_x - seg.start_x;
        let dy = seg.end_y - seg.start_y;
        let span = (dx * dx + dy * dy).sqrt();
        if span >= min_span_px {
            candidates.push((i, span));
        }
    }
    if candidates.len() < 4 {
        return None;
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top4 = [
        candidates[0].0,
        candidates[1].0,
        candidates[2].0,
        candidates[3].0,
    ];
    // Confirm at least one pairwise angular distance ≥ 30°.
    let mut max_diff = 0.0_f64;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let a = segments[top4[i]].angle_deg;
            let b = segments[top4[j]].angle_deg;
            let mut d = (a - b).abs();
            if d > 90.0 {
                d = 180.0 - d;
            }
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    if max_diff < 30.0 {
        return None;
    }
    Some(top4)
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    chain_idx: usize,
    start: usize,
    end: usize,
    start_x: f64,
    start_y: f64,
    end_x: f64,
    end_y: f64,
    angle_deg: f64,
}

// ── Stage E — sub-pixel TLS line fit ─────────────────────────────────────────

/// Adjust an anchor to the sub-pixel max of |∇| along its gradient direction
/// via 3-point parabolic vertex of integer-pixel gradient magnitudes.
fn subpixel_adjust_anchor(gray: &ImageView, a: &Anchor) -> (f64, f64) {
    let g_norm = (a.gx * a.gx + a.gy * a.gy).sqrt();
    if g_norm < 1e-9 {
        return (a.x as f64 + 0.5, a.y as f64 + 0.5);
    }
    let nx = a.gx / g_norm;
    let ny = a.gy / g_norm;
    let w = gray.width as i32;
    let h = gray.height as i32;
    let stride = gray.stride as i32;
    let in_bounds = |xf: f64, yf: f64| {
        let xi = xf.round() as i32;
        let yi = yf.round() as i32;
        xi >= 1 && xi < w - 1 && yi >= 1 && yi < h - 1
    };
    let grad_mag_at = |xf: f64, yf: f64| -> f64 {
        let xi = xf.round() as i32;
        let yi = yf.round() as i32;
        if !(xi >= 1 && xi < w - 1 && yi >= 1 && yi < h - 1) {
            return 0.0;
        }
        let g = sobel_grad(gray.data, stride, xi, yi);
        (g.0 * g.0 + g.1 * g.1).sqrt()
    };
    let xf = a.x as f64;
    let yf = a.y as f64;
    if !(in_bounds(xf - nx, yf - ny) && in_bounds(xf + nx, yf + ny)) {
        return (xf + 0.5, yf + 0.5);
    }
    let g_m = grad_mag_at(xf - nx, yf - ny);
    let g_0 = grad_mag_at(xf, yf);
    let g_p = grad_mag_at(xf + nx, yf + ny);
    let denom = g_p + g_m - 2.0 * g_0;
    if denom.abs() < 1e-9 {
        return (xf + 0.5, yf + 0.5);
    }
    let mut delta = (g_m - g_p) / (2.0 * denom);
    if delta < -0.5 {
        delta = -0.5;
    }
    if delta > 0.5 {
        delta = 0.5;
    }
    // Anchor position in pixel-centre coordinates is (a.x + 0.5, a.y + 0.5);
    // adjustment moves along the unit gradient direction.
    (xf + 0.5 + delta * nx, yf + 0.5 + delta * ny)
}

fn fit_segment_subpixel<'a>(
    arena: &'a Bump,
    gray: &ImageView,
    anchors: &[Anchor],
    chain: &[u32],
    seg: (usize, usize),
) -> Option<Line> {
    let (s, e) = seg;
    let mut sub_pts: BumpVec<'a, (f64, f64)> = BumpVec::with_capacity_in(e - s, arena);
    for &ai in &chain[s..e] {
        sub_pts.push(subpixel_adjust_anchor(gray, &anchors[ai as usize]));
    }
    fit_line_tls(sub_pts.as_slice())
}

// ── Stage G — perpendicular refinement (Phase-3-style dense sweep) ────────────

#[inline]
fn bilinear_sample(gray: &ImageView, sx: f64, sy: f64) -> Option<f64> {
    let w = gray.width;
    let h = gray.height;
    if sx < 0.5 || sy < 0.5 || sx >= (w as f64 - 0.5) || sy >= (h as f64 - 0.5) {
        return None;
    }
    let xi = sx.floor() as usize;
    let yi = sy.floor() as usize;
    let ax = sx - xi as f64;
    let ay = sy - yi as f64;
    let stride = gray.stride;
    let i00 = f64::from(gray.data[yi * stride + xi]);
    let i10 = f64::from(gray.data[yi * stride + xi + 1]);
    let i01 = f64::from(gray.data[(yi + 1) * stride + xi]);
    let i11 = f64::from(gray.data[(yi + 1) * stride + xi + 1]);
    Some(
        (1.0 - ax) * (1.0 - ay) * i00
            + ax * (1.0 - ay) * i10
            + (1.0 - ax) * ay * i01
            + ax * ay * i11,
    )
}

fn perpendicular_refine(
    arena: &Bump,
    gray: &ImageView,
    line: &Line,
    bbox: (f64, f64, f64, f64),
    cfg: &S3Config,
) -> Option<Line> {
    let nx = line.nx;
    let ny = line.ny;
    let cx = line.cx;
    let cy = line.cy;
    // Tangent = perpendicular to normal.
    let tx = -ny;
    let ty = nx;
    // Project bbox corners onto tangent for the full edge span.
    let (bx0, by0, bx1, by1) = bbox;
    let bbox_corners = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)];
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for (bx, by) in bbox_corners {
        let t = (bx - cx) * tx + (by - cy) * ty;
        if t < t_min {
            t_min = t;
        }
        if t > t_max {
            t_max = t;
        }
    }
    let big_k = (cfg.refine_search_half_width.ceil() as i32) + 1;
    let n_ints = (2 * big_k + 1) as usize;
    let n_grads = (2 * big_k - 1) as usize;
    let mut sub_pts: BumpVec<(f64, f64)> = BumpVec::new_in(arena);

    let mut t = t_min;
    while t <= t_max {
        let px = cx + t * tx;
        let py = cy + t * ty;
        let mut intensities = Vec::with_capacity(n_ints);
        let mut all_in = true;
        for ki in 0..n_ints {
            let k = ki as i32 - big_k;
            let sx = px + (k as f64) * nx;
            let sy = py + (k as f64) * ny;
            match bilinear_sample(gray, sx, sy) {
                Some(v) => intensities.push(v),
                None => {
                    all_in = false;
                    break;
                }
            }
        }
        if all_in {
            let mut grads = Vec::with_capacity(n_grads);
            for ki in 0..n_grads {
                grads.push((intensities[ki + 2] - intensities[ki]) * 0.5);
            }
            // Find argmax of |grad|.
            let mut argmax = 0_usize;
            let mut max_abs = 0.0_f64;
            for (i, &g) in grads.iter().enumerate() {
                let a = g.abs();
                if a > max_abs {
                    max_abs = a;
                    argmax = i;
                }
            }
            if max_abs >= cfg.refine_grad_min_mag && argmax > 0 && argmax < n_grads - 1 {
                let g_m = grads[argmax - 1].abs();
                let g_0 = grads[argmax].abs();
                let g_p = grads[argmax + 1].abs();
                let denom = g_p + g_m - 2.0 * g_0;
                let delta = if denom.abs() > 1e-6 {
                    let mut d = (g_m - g_p) / (2.0 * denom);
                    if d < -0.5 {
                        d = -0.5;
                    }
                    if d > 0.5 {
                        d = 0.5;
                    }
                    d
                } else {
                    0.0
                };
                let mut k_star = (argmax as f64 + 1.0) - big_k as f64 + delta;
                if k_star > cfg.refine_search_half_width {
                    k_star = cfg.refine_search_half_width;
                }
                if k_star < -cfg.refine_search_half_width {
                    k_star = -cfg.refine_search_half_width;
                }
                sub_pts.push((px + k_star * nx, py + k_star * ny));
            }
        }
        t += cfg.refine_sample_step;
    }
    if sub_pts.len() < 4 {
        return Some(*line);
    }
    fit_line_tls(sub_pts.as_slice())
}

// ── Stage F — corner extraction ──────────────────────────────────────────────

fn intersect_lines(l1: &Line, l2: &Line) -> Option<(f64, f64)> {
    let det = l1.nx * l2.ny - l1.ny * l2.nx;
    if det.abs() < 1e-9 {
        return None;
    }
    let x = (l1.ny * l2.d - l2.ny * l1.d) / det;
    let y = (l2.nx * l1.d - l1.nx * l2.d) / det;
    Some(-x).map(|x_neg| (-x_neg, y))
}

fn order_lines_cw(lines: &[Line; 4]) -> [usize; 4] {
    let mut centres = [(0.0_f64, 0.0_f64); 4];
    for i in 0..4 {
        centres[i] = (lines[i].cx, lines[i].cy);
    }
    let mid_x = (centres[0].0 + centres[1].0 + centres[2].0 + centres[3].0) * 0.25;
    let mid_y = (centres[0].1 + centres[1].1 + centres[2].1 + centres[3].1) * 0.25;
    let mut idx_angle: [(usize, f64); 4] = [
        (0, (centres[0].1 - mid_y).atan2(centres[0].0 - mid_x)),
        (1, (centres[1].1 - mid_y).atan2(centres[1].0 - mid_x)),
        (2, (centres[2].1 - mid_y).atan2(centres[2].0 - mid_x)),
        (3, (centres[3].1 - mid_y).atan2(centres[3].0 - mid_x)),
    ];
    idx_angle.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    [
        idx_angle[0].0,
        idx_angle[1].0,
        idx_angle[2].0,
        idx_angle[3].0,
    ]
}

fn extract_corners(lines: &[Line; 4]) -> Option<[(f64, f64); 4]> {
    let order = order_lines_cw(lines);
    let mut corners = [(0.0, 0.0); 4];
    for i in 0..4 {
        corners[i] = intersect_lines(&lines[order[i]], &lines[order[(i + 1) % 4]])?;
    }
    Some(corners)
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

    // Stage C — split each chain into linear segments.
    let mut all_segments: BumpVec<Segment> = BumpVec::new_in(arena);
    for (ci, chain) in chains.iter().enumerate() {
        if chain.len() < cfg.min_seg_len {
            continue;
        }
        // Build sub-pixel position list for the chain (use anchor x+0.5, y+0.5).
        let mut chain_pts: BumpVec<(f64, f64)> = BumpVec::with_capacity_in(chain.len(), arena);
        for &ai in chain.iter() {
            let a = &anchors[ai as usize];
            chain_pts.push((a.x as f64 + 0.5, a.y as f64 + 0.5));
        }
        let splits = split_chain(arena, chain_pts.as_slice(), cfg.split_max_rms, cfg.min_seg_len);
        for (s, e) in splits.iter().copied() {
            if e - s < cfg.min_seg_len {
                continue;
            }
            // Compute Euclidean span + angle from start/end pts.
            let (sx, sy) = chain_pts[s];
            let (ex, ey) = chain_pts[e - 1];
            let dx = ex - sx;
            let dy = ey - sy;
            let angle_deg = (dy.atan2(dx).to_degrees() + 180.0).rem_euclid(180.0);
            all_segments.push(Segment {
                chain_idx: ci,
                start: s,
                end: e,
                start_x: sx,
                start_y: sy,
                end_x: ex,
                end_y: ey,
                angle_deg,
            });
        }
    }

    // Stage D — top-4 selection.
    let bbox_w = (i32::from(stat.max_x) - i32::from(stat.min_x)) as f64;
    let bbox_h = (i32::from(stat.max_y) - i32::from(stat.min_y)) as f64;
    let bbox_short = bbox_w.min(bbox_h);
    let min_span = cfg.min_segment_span_floor_px.max(bbox_short / 4.0);
    let top4_indices = select_top4(all_segments.as_slice(), min_span)?;

    // Stage E — sub-pixel TLS line fit per segment.
    let mut top4_lines: [Line; 4] = [Line {
        nx: 0.0,
        ny: 0.0,
        d: 0.0,
        cx: 0.0,
        cy: 0.0,
    }; 4];
    for (slot, &si) in top4_indices.iter().enumerate() {
        let seg = all_segments[si];
        let chain = &chains[seg.chain_idx];
        let line = fit_segment_subpixel(arena, gray, anchors.as_slice(), chain, (seg.start, seg.end))?;
        top4_lines[slot] = line;
    }

    // Stage G — perpendicular refinement.  Iterate twice.
    let bbox = (
        i32::from(stat.min_x) as f64,
        i32::from(stat.min_y) as f64,
        i32::from(stat.max_x) as f64,
        i32::from(stat.max_y) as f64,
    );
    for _ in 0..cfg.refine_iters {
        let mut refined = top4_lines;
        for slot in 0..4 {
            if let Some(rline) = perpendicular_refine(arena, gray, &top4_lines[slot], bbox, cfg) {
                refined[slot] = rline;
            }
        }
        top4_lines = refined;
    }

    // Stage F — corner extraction.
    let corners_xy = extract_corners(&top4_lines)?;
    let corners = [
        Point {
            x: corners_xy[0].0,
            y: corners_xy[0].1,
        },
        Point {
            x: corners_xy[1].0,
            y: corners_xy[1].1,
        },
        Point {
            x: corners_xy[2].0,
            y: corners_xy[2].1,
        },
        Point {
            x: corners_xy[3].0,
            y: corners_xy[3].1,
        },
    ];
    // S3 v1 leaves corner covariances at zero (downstream pose code treats
    // that as "no covariance prior").  Future work could derive Σ from
    // line residuals.
    let covs: CornerCovariances = [[0.0_f32; 4]; 4];
    Some((corners, covs))
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
    fn end_to_end_returns_4_corners_within_2px_of_synthetic_square() {
        // Sharp synthetic 40-px black square on white canvas at (30, 30).
        // Expected corners (pixel-centre coords): (30.5, 30.5), (69.5, 30.5),
        // (69.5, 69.5), (30.5, 69.5) — but Sobel + sub-pixel adjustment may
        // shift by ±0.5 on hard step edges.  Allow 2 px tolerance.
        let canvas = 100;
        let x0 = 30;
        let y0 = 30;
        let size = 40;
        let pixels = make_synthetic_square(canvas, x0, y0, size);
        let img = ImageView::new(&pixels, canvas, canvas, canvas).unwrap();
        let stats = stats_for_square(x0 as u16, y0 as u16, size as u16);
        let arena = Bump::new();
        let cfg = S3Config::defaults();
        let result = run_anchor_walk(&arena, &img, &stats, &cfg);
        assert!(result.is_some(), "S3 returned None on clean synthetic square");
        let (corners, _) = result.unwrap();
        // Match each detected corner to the closest expected corner.
        let expected = [
            (30.5, 30.5),
            (69.5, 30.5),
            (69.5, 69.5),
            (30.5, 69.5),
        ];
        for (cx, cy) in expected {
            let mut best = f64::INFINITY;
            for c in &corners {
                let d = ((c.x - cx).powi(2) + (c.y - cy).powi(2)).sqrt();
                if d < best {
                    best = d;
                }
            }
            assert!(
                best < 2.0,
                "expected corner near ({cx}, {cy}); nearest detected at distance {best:.3} px"
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
