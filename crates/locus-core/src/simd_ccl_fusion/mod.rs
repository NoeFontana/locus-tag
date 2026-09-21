//! SIMD-Accelerated Connected Components Labeling (CCL) with Fused Thresholding.
//!
//! This module implements a high-performance segmentation pipeline that defeats the "memory wall"
//! by fusing adaptive thresholding with Run-Length Encoding (RLE) extraction. It processes images
//! in 1D segments rather than individual pixels, drastically reducing memory bandwidth requirements
//! and branch mispredictions.

// `RunSink::push_run` is the innermost statement of the SIMD scanline kernels.
// Unless it inlines into them, making the kernels generic over the sink buys
// nothing and every run costs an indirect call through a trait object shape
// LLVM cannot see into, so all three implementations force the inline.
#![allow(clippy::inline_always)]

use crate::image::ImageView;
use crate::segmentation::{ComponentStats, LabelResult, UnionFind};
use bumpalo::Bump;
use rayon::prelude::*;

/// A 1D Run-Length Encoded (RLE) segment representing contiguous foreground pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RleSegment {
    /// The row (Y coordinate) of the segment.
    pub y: u16,
    /// The starting column (X coordinate) of the segment (inclusive).
    pub start_x: u16,
    /// The ending column (X coordinate) of the segment (exclusive).
    pub end_x: u16,
    /// The component label assigned to this segment. Uninitialized/unassigned is 0.
    pub label: u32,
}

impl RleSegment {
    /// Create a new RleSegment with an unassigned label.
    #[must_use]
    pub const fn new(y: u16, start_x: u16, end_x: u16) -> Self {
        Self {
            y,
            start_x,
            end_x,
            label: 0,
        }
    }
}

/// Architecture-specific SIMD scanline processing.
pub mod simd_scanline;

/// Receiver for the foreground runs a scanline kernel finds.
///
/// The kernels are generic over this trait so that *counting* runs and
/// *materialising* them share one implementation. The parallel extractor needs
/// both — a counting pass to size the output, then a writing pass — and any
/// divergence between the two would be a memory-safety-adjacent correctness
/// bug, so there is deliberately only one copy of the scan logic.
///
/// `push_run` is called once per run, in left-to-right order within a row, and
/// rows are visited in increasing `y`.
pub trait RunSink {
    /// Record one foreground run `[start_x, end_x)` on row `y`.
    fn push_run(&mut self, y: u16, start_x: u16, end_x: u16);
}

impl RunSink for Vec<RleSegment> {
    #[inline(always)]
    fn push_run(&mut self, y: u16, start_x: u16, end_x: u16) {
        self.push(RleSegment::new(y, start_x, end_x));
    }
}

/// Counts runs without materialising them (pass 1 of the parallel extractor).
pub struct CountSink(pub u32);

impl RunSink for CountSink {
    #[inline(always)]
    fn push_run(&mut self, _y: u16, _start_x: u16, _end_x: u16) {
        self.0 += 1;
    }
}

/// Writes runs into a pre-sized slice, numbering them from `first_index`
/// (pass 2 of the parallel extractor).
///
/// `first_index` is the run's global index, which doubles as its Union-Find id —
/// the sequential extractor assigned the same ids in a separate loop.
pub struct SliceSink<'a> {
    /// Destination, exactly as long as the row's run count.
    pub out: &'a mut [RleSegment],
    /// Number of runs written so far.
    pub written: usize,
    /// Global index of the row's first run.
    pub first_index: u32,
}

impl RunSink for SliceSink<'_> {
    #[inline(always)]
    fn push_run(&mut self, y: u16, start_x: u16, end_x: u16) {
        let index = self.first_index + self.written as u32;
        if let Some(slot) = self.out.get_mut(self.written) {
            *slot = RleSegment {
                y,
                start_x,
                end_x,
                label: index,
            };
            self.written += 1;
        }
    }
}

/// Baseline scalar implementation of the fused Threshold + RLE extraction kernel.
/// Extracts runs of black pixels (value < threshold) from the image.
#[allow(dead_code)]
#[must_use]
pub fn extract_rle_segments_scalar(img: &ImageView, threshold_map: &[u8]) -> Vec<RleSegment> {
    let mut segments = Vec::new();
    let height = img.height as u16;

    for y in 0..height {
        let y_usize = y as usize;
        let src_row = img.get_row(y_usize);
        let thresh_row = &threshold_map[y_usize * img.width..(y_usize + 1) * img.width];

        process_row_scalar(src_row, thresh_row, y, &mut segments);
    }

    segments
}

/// Scalar fallback for a single row.
pub fn process_row_scalar<S: RunSink>(src_row: &[u8], thresh_row: &[u8], y: u16, sink: &mut S) {
    let mut in_segment = false;
    let mut start_x = 0;
    let width = src_row.len();

    for (x, (&s, &t)) in src_row.iter().zip(thresh_row.iter()).enumerate() {
        let is_foreground = s < t;

        if is_foreground && !in_segment {
            in_segment = true;
            start_x = x as u16;
        } else if !is_foreground && in_segment {
            in_segment = false;
            sink.push_run(y, start_x, x as u16);
        }
    }

    if in_segment {
        sink.push_run(y, start_x, width as u16);
    }
}

/// Walks the (now immutable) Union-Find forest to `i`'s root.
///
/// Read-only, so it is safe to call from several threads at once. Omitting path
/// compression is bit-exact — compression only shortens future walks, it never
/// changes which index is the root — and pointless here anyway, since every
/// index is resolved exactly once.
#[inline]
fn resolve_root(parent: &[u32], i: u32) -> u32 {
    let mut root = i;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    root
}

/// Folds one run into its component's statistics.
///
/// Every term is order-independent (min / max / integer sum) except
/// `first_pixel_*`, which is taken from the first run that reaches an untouched
/// accumulator — i.e. the component's first run in scan order, exactly the run
/// the previous implementation's first-encounter branch picked. A run is never
/// empty, so `pixel_count == 0` is a reliable "untouched" marker.
#[inline]
fn accumulate_run(stats: &mut ComponentStats, run: &RleSegment) {
    if stats.pixel_count == 0 {
        stats.first_pixel_x = run.start_x;
        stats.first_pixel_y = run.y;
    }
    stats.min_x = stats.min_x.min(run.start_x);
    stats.max_x = stats.max_x.max(run.end_x - 1);
    stats.min_y = stats.min_y.min(run.y);
    stats.max_y = stats.max_y.max(run.y);
    stats.pixel_count += u32::from(run.end_x - run.start_x);
    // Accumulate spatial moments using closed-form per-run sums.
    // Run covers x in [a, b) exclusive (end_x is exclusive).
    let a = u64::from(run.start_x);
    let b = u64::from(run.end_x);
    let yu = u64::from(run.y);
    // NOTE: `a - 1` and `2*a - 1` are always multiplied by `a`, so their value
    // is irrelevant when a = 0. saturating_sub avoids u64 underflow in debug
    // builds (release wraps, but the `* a` factor zeros the term anyway).
    stats.m10 += b * (b - 1) / 2 - a * a.saturating_sub(1) / 2;
    stats.m01 += yu * (b - a);
    stats.m20 +=
        (b - 1) * b * (2 * b - 1) / 6 - a.saturating_sub(1) * a * (2 * a).saturating_sub(1) / 6;
    stats.m02 += yu * yu * (b - a);
    stats.m11 += yu * (b * (b - 1) / 2 - a * a.saturating_sub(1) / 2);
}

/// Rows per leaf task in the parallel RLE extraction.
///
/// A leaf reads `ROWS_PER_RLE_TASK * width` bytes of image *and* threshold map
/// (256 KB total at 4K), which stays inside a core's L2 while still producing
/// ~135 tasks at 2160 rows so rayon can balance rows of very different run
/// density.
const ROWS_PER_RLE_TASK: usize = 16;

/// One-pass extraction, unchanged from the previous implementation: scan every
/// row into a growable vector, then number the runs.
///
/// Used when the rayon pool has a single worker, where the parallel extractor's
/// counting pre-pass is pure overhead (+6.1 ms on a 1.7 M-run 4K frame) with no
/// second worker to pay for it. The growable vector stays on the system
/// allocator here — as it was on `main` — because in the arena its growth
/// copies measured ~0.9 ms slower per 4K frame than `realloc`; the hot-path
/// allocation is removed on the multi-worker path, which is the one production
/// takes.
fn extract_runs_sequential(img: &ImageView, threshold_map: &[u8]) -> Vec<RleSegment> {
    let width = img.width;
    let mut runs = Vec::new();
    for y in 0..img.height {
        simd_scanline::process_row_simd(
            img.get_row(y),
            &threshold_map[y * width..(y + 1) * width],
            y as u16,
            &mut runs,
        );
    }
    for (index, run) in runs.iter_mut().enumerate() {
        run.label = index as u32;
    }
    runs
}

/// Fused threshold + RLE extraction, parallelised over image rows.
///
/// Rows are independent, but the output must stay in scan order and runs must
/// keep dense global indices, so this runs the scanline kernel twice: once with
/// a [`CountSink`] to size each row, then — after an exclusive prefix sum — once
/// with a [`SliceSink`] writing straight into each row's slot in the final
/// array. Both passes use the *same* kernel, so they cannot disagree.
///
/// The result is bit-identical to `simd_scanline::extract_rle_segments`
/// followed by the index-assignment loop.
fn extract_runs_parallel<'a>(
    arena: &'a Bump,
    img: &ImageView,
    threshold_map: &[u8],
) -> &'a mut [RleSegment] {
    let width = img.width;
    let height = img.height;

    // Pass 1: runs per row.
    let offsets = arena.alloc_slice_fill_copy(height + 1, 0u32);
    offsets[1..]
        .par_iter_mut()
        .enumerate()
        .for_each(|(y, count)| {
            let mut sink = CountSink(0);
            simd_scanline::process_row_simd(
                img.get_row(y),
                &threshold_map[y * width..(y + 1) * width],
                y as u16,
                &mut sink,
            );
            *count = sink.0;
        });

    // Exclusive prefix sum -> `offsets[y]` is row y's first global run index.
    let mut total: u32 = 0;
    for slot in offsets.iter_mut() {
        total += *slot;
        *slot = total;
    }
    debug_assert_eq!(offsets[0], 0);

    let runs = arena.alloc_slice_fill_copy(total as usize, RleSegment::new(0, 0, 0));
    let offsets: &[u32] = offsets;
    write_runs_rows(runs, img, threshold_map, offsets, 0, height);
    runs
}

/// Writes the runs of image rows `[y0, y1)` into `out`, which covers exactly
/// `offsets[y0]..offsets[y1]`.
///
/// The recursion bisects the row range and splits `out` at the matching run
/// offset, so the two halves own disjoint memory and `split_at_mut` proves it —
/// no `unsafe` and no shared-mutable state.
fn write_runs_rows(
    out: &mut [RleSegment],
    img: &ImageView,
    threshold_map: &[u8],
    offsets: &[u32],
    y0: usize,
    y1: usize,
) {
    if y1 <= y0 {
        return;
    }
    if y1 - y0 > ROWS_PER_RLE_TASK {
        let ym = y0 + (y1 - y0) / 2;
        let split = (offsets[ym] - offsets[y0]) as usize;
        let (top, bottom) = out.split_at_mut(split);
        rayon::join(
            || write_runs_rows(top, img, threshold_map, offsets, y0, ym),
            || write_runs_rows(bottom, img, threshold_map, offsets, ym, y1),
        );
        return;
    }

    let width = img.width;
    let mut rest = out;
    for y in y0..y1 {
        let row_len = (offsets[y + 1] - offsets[y]) as usize;
        let (row_out, tail) = rest.split_at_mut(row_len);
        rest = tail;
        let mut sink = SliceSink {
            out: row_out,
            written: 0,
            first_index: offsets[y],
        };
        simd_scanline::process_row_simd(
            img.get_row(y),
            &threshold_map[y * width..(y + 1) * width],
            y as u16,
            &mut sink,
        );
        debug_assert_eq!(
            sink.written, row_len,
            "counting and writing passes disagreed on row {y}"
        );
    }
}

/// Performs Light-Speed Labeling (LSL) / Run-based Union-Find on the extracted RLE segments.
/// Fully resolves equivalences and outputs the 2D label map expected by the rest of the pipeline.
#[allow(clippy::too_many_lines)]
#[tracing::instrument(skip_all, name = "pipeline::segmentation")]
pub fn label_components_lsl<'a>(
    arena: &'a Bump,
    img: &ImageView,
    threshold_map: &[u8],
    use_8_connectivity: bool,
    min_area: u32,
) -> LabelResult<'a> {
    // Runs come out in scan order with `label` already set to the global index
    // the Union-Find uses as its id. Both extractors produce byte-identical
    // output; only a multi-worker pool can amortise the parallel one's counting
    // pre-pass, so a single-worker pool keeps the one-pass scan.
    let mut owned_runs = Vec::new();
    let arena_runs = if rayon::current_num_threads() > 1 {
        Some(extract_runs_parallel(arena, img, threshold_map))
    } else {
        owned_runs = extract_runs_sequential(img, threshold_map);
        None
    };
    let runs: &[RleSegment] = arena_runs.as_deref().unwrap_or(&owned_runs);

    if runs.is_empty() {
        return LabelResult {
            labels: arena.alloc_slice_fill_copy(img.width * img.height, 0u32),
            component_stats: Vec::new(),
        };
    }

    let mut uf = UnionFind::new_in(arena, runs.len());
    let mut curr_row_range = 0..0;
    let mut i = 0;

    while i < runs.len() {
        let y = runs[i].y;
        let start = i;
        while i < runs.len() && runs[i].y == y {
            i += 1;
        }
        let prev_row_range = curr_row_range;
        curr_row_range = start..i;

        if y > 0 && !prev_row_range.is_empty() && runs[prev_row_range.start].y == y - 1 {
            let mut p_idx = prev_row_range.start;
            for c_idx in curr_row_range.clone() {
                let curr = &runs[c_idx];
                if use_8_connectivity {
                    // 8-connectivity: [start_x, end_x)
                    // overlap diagonally if prev.end_x >= curr.start_x and prev.start_x <= curr.end_x
                    while p_idx < prev_row_range.end && runs[p_idx].end_x < curr.start_x {
                        p_idx += 1;
                    }
                    let mut temp_p = p_idx;
                    while temp_p < prev_row_range.end && runs[temp_p].start_x <= curr.end_x {
                        uf.union(curr.label, runs[temp_p].label);
                        temp_p += 1;
                    }
                } else {
                    // 4-connectivity
                    while p_idx < prev_row_range.end && runs[p_idx].end_x <= curr.start_x {
                        p_idx += 1;
                    }
                    let mut temp_p = p_idx;
                    while temp_p < prev_row_range.end && runs[temp_p].start_x < curr.end_x {
                        uf.union(curr.label, runs[temp_p].label);
                        temp_p += 1;
                    }
                }
            }
        }
    }

    // ---- Root resolution -------------------------------------------------
    // The equivalence forest is final from here on, so `parent` becomes
    // read-only and every run's root can be resolved by a pure walk. Skipping
    // path compression is bit-exact: compression changes how fast a root is
    // reached, never which index it is.
    let parent = uf.parents();

    // Dense component slots, numbered in ascending-root order. The previous
    // implementation numbered slots in first-encounter order and then walked a
    // `runs.len()`-sized table in ascending-root order to assign final labels;
    // numbering by root up front yields the same final label sequence while
    // shrinking that walk from O(runs) to O(components).
    let slot = arena.alloc_slice_fill_copy(runs.len(), 0u32);
    let mut num_components = 0u32;
    for (i, p) in parent.iter().enumerate() {
        if *p as usize == i {
            slot[i] = num_components;
            num_components += 1;
        }
    }
    let slot: &[u32] = slot;

    // Per-run component slot. Resolving it is the random-access half of the old
    // stats and fill loops (a `find` plus a table lookup per run). With more
    // than one worker it pays to hoist it into its own parallel pass, leaving
    // both consumers purely sequential; with one worker that extra pass over a
    // multi-MB array is pure cost, so it stays fused into the stats loop (which
    // is still one `find` pass fewer than the previous implementation, because
    // the fill below reads `comp` instead of resolving again).
    let comp = arena.alloc_slice_fill_copy(runs.len(), 0u32);
    let stats_by_slot =
        arena.alloc_slice_fill_copy(num_components as usize, ComponentStats::default());
    let parallel = rayon::current_num_threads() > 1;

    if parallel {
        comp.par_iter_mut().enumerate().for_each(|(i, c)| {
            *c = slot[resolve_root(parent, i as u32) as usize];
        });
    }

    // ---- Stats accumulation ---------------------------------------------
    for (i, run) in runs.iter().enumerate() {
        let c = if parallel {
            comp[i]
        } else {
            let c = slot[resolve_root(parent, i as u32) as usize];
            comp[i] = c;
            c
        };
        accumulate_run(&mut stats_by_slot[c as usize], run);
    }
    let comp: &[u32] = comp;

    // ---- Final labelling -------------------------------------------------
    let mut component_stats = Vec::with_capacity(num_components as usize);
    let slot_to_final_label = arena.alloc_slice_fill_copy(num_components as usize, 0u32);
    let mut next_label = 1u32;

    for (c, s) in stats_by_slot.iter().enumerate() {
        if s.pixel_count >= min_area {
            component_stats.push(*s);
            slot_to_final_label[c] = next_label;
            next_label += 1;
        }
    }
    let slot_to_final_label: &[u32] = slot_to_final_label;

    let labels = arena.alloc_slice_fill_copy(img.width * img.height, 0u32);

    fill_label_rows(
        labels,
        runs,
        comp,
        slot_to_final_label,
        img.width,
        0,
        img.height,
        0,
        runs.len(),
    );

    LabelResult {
        labels,
        component_stats,
    }
}

/// Rows per leaf task in the parallel label-buffer fill.
///
/// A leaf owns `ROWS_PER_FILL_TASK * width * 4` bytes of the label buffer
/// (1 MB at 4K), which keeps each task's writes inside a core's private cache
/// while still producing ~34 tasks for a 2160-row frame — enough for rayon to
/// balance across 8 workers even when rows differ wildly in run density.
const ROWS_PER_FILL_TASK: usize = 64;

/// Writes the final label of every run into the (already zeroed) label buffer.
///
/// `labels` covers image rows `[y0, y1)` and `runs[run_lo..run_hi]` are exactly
/// the runs on those rows. Runs are emitted in scan order, so the row range can
/// be bisected and the two halves filled independently: the recursion splits
/// both the label rows and the run range at the same row boundary, which makes
/// the two tasks touch disjoint memory and lets `split_at_mut` prove it without
/// `unsafe`.
///
/// Bit-exactness: runs never overlap, so writing them in a different order
/// produces the same buffer as the original sequential loop.
#[allow(clippy::too_many_arguments)]
fn fill_label_rows(
    labels: &mut [u32],
    runs: &[RleSegment],
    comp: &[u32],
    slot_to_final_label: &[u32],
    width: usize,
    y0: usize,
    y1: usize,
    run_lo: usize,
    run_hi: usize,
) {
    if run_lo >= run_hi {
        return;
    }
    if y1 - y0 > ROWS_PER_FILL_TASK {
        let ym = y0 + (y1 - y0) / 2;
        // Runs are sorted by row, so the split point is a partition point.
        let run_mid = run_lo + runs[run_lo..run_hi].partition_point(|r| (r.y as usize) < ym);
        let (top, bottom) = labels.split_at_mut((ym - y0) * width);
        rayon::join(
            || {
                fill_label_rows(
                    top,
                    runs,
                    comp,
                    slot_to_final_label,
                    width,
                    y0,
                    ym,
                    run_lo,
                    run_mid,
                );
            },
            || {
                fill_label_rows(
                    bottom,
                    runs,
                    comp,
                    slot_to_final_label,
                    width,
                    ym,
                    y1,
                    run_mid,
                    run_hi,
                );
            },
        );
        return;
    }

    for (run, &c) in runs[run_lo..run_hi].iter().zip(comp[run_lo..run_hi].iter()) {
        let label = slot_to_final_label[c as usize];
        if label > 0 {
            let row_off = (run.y as usize - y0) * width;
            labels[row_off + run.start_x as usize..row_off + run.end_x as usize].fill(label);
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_label_components_lsl_red_phase() {
        let arena = Bump::new();
        let width = 8;
        let height = 4;

        // Image with two separated 2x2 blocks:
        // row 0: . x x . . x x .
        // row 1: . x x . . x x .
        // row 2: . . . . . . . .
        // row 3: . . . . . . . .

        let mut pixels = vec![200u8; width * height];
        pixels[1] = 50;
        pixels[2] = 50;
        pixels[5] = 50;
        pixels[6] = 50;
        pixels[8 + 1] = 50;
        pixels[8 + 2] = 50;
        pixels[8 + 5] = 50;
        pixels[8 + 6] = 50;

        let threshold_map = vec![128u8; width * height];
        let img = ImageView::new(&pixels, width, height, width).expect("Valid test image");

        let result = label_components_lsl(&arena, &img, &threshold_map, true, 1);

        assert_eq!(result.component_stats.len(), 2);

        // Assuming label 1 is left block and label 2 is right block
        let mut found_labels = std::collections::HashSet::new();
        found_labels.insert(result.labels[1]);
        found_labels.insert(result.labels[5]);

        assert_eq!(found_labels.len(), 2);
        assert!(!found_labels.contains(&0));

        // Verify stats
        assert_eq!(result.component_stats[0].pixel_count, 4);
        assert_eq!(result.component_stats[1].pixel_count, 4);
    }

    #[test]
    fn test_extract_rle_segments_scalar() {
        let width = 8;
        let height = 2;
        // Image with two black segments on first row, one on second row.
        // Black is < threshold (e.g. 100 vs 128)
        let pixels = vec![
            200, 50, 50, 200, 50, 200, 200, 200, // Row 0: RLE at [1, 3) and [4, 5)
            50, 50, 50, 50, 200, 200, 200, 200, // Row 1: RLE at [0, 4)
        ];
        let threshold_map = vec![128u8; width * height];

        let img = ImageView::new(&pixels, width, height, width).expect("Valid test image");
        let segments = extract_rle_segments_scalar(&img, &threshold_map);

        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], RleSegment::new(0, 1, 3));
        assert_eq!(segments[1], RleSegment::new(0, 4, 5));
        assert_eq!(segments[2], RleSegment::new(1, 0, 4));
    }
}

/// Differential tests pinning `label_components_lsl` to the pre-parallelisation
/// reference implementation.
///
/// The parallel version must be **bit-exact**, not merely equivalent: the label
/// numbering decides `component_stats` order, which decides quad-candidate
/// order and therefore the `MAX_CANDIDATES` cut, so any renumbering would move
/// the ICRA / render-tag snapshots.
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod differential_tests {
    use super::*;

    /// Verbatim copy of `label_components_lsl` as it stood before the parallel
    /// rewrite (`origin/main`, pre `perf/parallel-ccl`). Deliberately *not*
    /// refactored: it is the oracle, so it must keep drifting-free provenance.
    #[allow(clippy::too_many_lines)]
    fn label_components_lsl_reference<'a>(
        arena: &'a Bump,
        img: &ImageView,
        threshold_map: &[u8],
        use_8_connectivity: bool,
        min_area: u32,
    ) -> LabelResult<'a> {
        let mut runs = simd_scanline::extract_rle_segments(img, threshold_map);

        for (id, run) in runs.iter_mut().enumerate() {
            run.label = id as u32;
        }

        if runs.is_empty() {
            return LabelResult {
                labels: arena.alloc_slice_fill_copy(img.width * img.height, 0u32),
                component_stats: Vec::new(),
            };
        }

        let mut uf = UnionFind::new_in(arena, runs.len());
        let mut curr_row_range = 0..0;
        let mut i = 0;

        while i < runs.len() {
            let y = runs[i].y;
            let start = i;
            while i < runs.len() && runs[i].y == y {
                i += 1;
            }
            let prev_row_range = curr_row_range;
            curr_row_range = start..i;

            if y > 0 && !prev_row_range.is_empty() && runs[prev_row_range.start].y == y - 1 {
                let mut p_idx = prev_row_range.start;
                for c_idx in curr_row_range.clone() {
                    let curr = &runs[c_idx];
                    if use_8_connectivity {
                        while p_idx < prev_row_range.end && runs[p_idx].end_x < curr.start_x {
                            p_idx += 1;
                        }
                        let mut temp_p = p_idx;
                        while temp_p < prev_row_range.end && runs[temp_p].start_x <= curr.end_x {
                            uf.union(curr.label, runs[temp_p].label);
                            temp_p += 1;
                        }
                    } else {
                        while p_idx < prev_row_range.end && runs[p_idx].end_x <= curr.start_x {
                            p_idx += 1;
                        }
                        let mut temp_p = p_idx;
                        while temp_p < prev_row_range.end && runs[temp_p].start_x < curr.end_x {
                            uf.union(curr.label, runs[temp_p].label);
                            temp_p += 1;
                        }
                    }
                }
            }
        }

        let mut root_to_temp_idx = vec![usize::MAX; runs.len()];
        let mut temp_stats = Vec::new();

        for run in &runs {
            let root = uf.find(run.label) as usize;
            if root_to_temp_idx[root] == usize::MAX {
                root_to_temp_idx[root] = temp_stats.len();
                temp_stats.push(ComponentStats {
                    first_pixel_x: run.start_x,
                    first_pixel_y: run.y,
                    ..ComponentStats::default()
                });
            }
            let s_idx = root_to_temp_idx[root];
            let stats = &mut temp_stats[s_idx];
            stats.min_x = stats.min_x.min(run.start_x);
            stats.max_x = stats.max_x.max(run.end_x - 1);
            stats.min_y = stats.min_y.min(run.y);
            stats.max_y = stats.max_y.max(run.y);
            stats.pixel_count += u32::from(run.end_x - run.start_x);
            let a = u64::from(run.start_x);
            let b = u64::from(run.end_x);
            let yu = u64::from(run.y);
            stats.m10 += b * (b - 1) / 2 - a * a.saturating_sub(1) / 2;
            stats.m01 += yu * (b - a);
            stats.m20 += (b - 1) * b * (2 * b - 1) / 6
                - a.saturating_sub(1) * a * (2 * a).saturating_sub(1) / 6;
            stats.m02 += yu * yu * (b - a);
            stats.m11 += yu * (b * (b - 1) / 2 - a * a.saturating_sub(1) / 2);
        }

        let mut component_stats = Vec::with_capacity(temp_stats.len());
        let mut root_to_final_label = vec![0u32; runs.len()];
        let mut next_label = 1u32;

        for (root, root_to_temp) in root_to_temp_idx.iter().enumerate() {
            if *root_to_temp != usize::MAX {
                let s = temp_stats[*root_to_temp];
                if s.pixel_count >= min_area {
                    component_stats.push(s);
                    root_to_final_label[root] = next_label;
                    next_label += 1;
                }
            }
        }

        let labels = arena.alloc_slice_fill_copy(img.width * img.height, 0u32);
        let width = img.width;

        for run in &runs {
            let root = uf.find(run.label) as usize;
            let label = root_to_final_label[root];
            if label > 0 {
                let row_off = run.y as usize * width;
                labels[row_off + run.start_x as usize..row_off + run.end_x as usize].fill(label);
            }
        }

        LabelResult {
            labels,
            component_stats,
        }
    }

    fn stats_tuple(
        s: &ComponentStats,
    ) -> (u16, u16, u16, u16, u32, u16, u16, u64, u64, u64, u64, u64) {
        (
            s.min_x,
            s.max_x,
            s.min_y,
            s.max_y,
            s.pixel_count,
            s.first_pixel_x,
            s.first_pixel_y,
            s.m10,
            s.m01,
            s.m20,
            s.m02,
            s.m11,
        )
    }

    /// Runs both implementations on one frame and asserts bit-exact equality of
    /// the label buffer and of every component statistic, in order.
    fn assert_bit_exact(
        name: &str,
        pixels: &[u8],
        threshold_map: &[u8],
        width: usize,
        height: usize,
        use_8_connectivity: bool,
        min_area: u32,
    ) {
        let img = ImageView::new(pixels, width, height, width).expect("valid test image");

        let arena_ref = Bump::new();
        let reference = label_components_lsl_reference(
            &arena_ref,
            &img,
            threshold_map,
            use_8_connectivity,
            min_area,
        );

        let arena_new = Bump::new();
        let actual = label_components_lsl(
            &arena_new,
            &img,
            threshold_map,
            use_8_connectivity,
            min_area,
        );

        assert_eq!(
            actual.component_stats.len(),
            reference.component_stats.len(),
            "{name}: component count (conn8={use_8_connectivity}, min_area={min_area})"
        );
        for (idx, (a, r)) in actual
            .component_stats
            .iter()
            .zip(reference.component_stats.iter())
            .enumerate()
        {
            assert_eq!(
                stats_tuple(a),
                stats_tuple(r),
                "{name}: component {idx} stats (conn8={use_8_connectivity}, min_area={min_area})"
            );
        }

        assert_eq!(
            actual.labels.len(),
            reference.labels.len(),
            "{name}: label len"
        );
        let first_diff = actual
            .labels
            .iter()
            .zip(reference.labels.iter())
            .position(|(a, r)| a != r);
        assert!(
            first_diff.is_none(),
            "{name}: label buffer differs at pixel index {first_diff:?} \
             (conn8={use_8_connectivity}, min_area={min_area}, width={width}, height={height})",
        );
    }

    /// Deterministic xorshift, so the adversarial frames are reproducible
    /// everywhere without pulling an RNG crate into the test.
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }
    }

    /// The frame zoo: each entry is (name, pixels, threshold, width, height).
    #[allow(clippy::type_complexity)]
    fn frames() -> Vec<(String, Vec<u8>, Vec<u8>, usize, usize)> {
        let mut out: Vec<(String, Vec<u8>, Vec<u8>, usize, usize)> = Vec::new();

        // Widths straddling the AVX2 (32 B) and AVX-512 (64 B) block sizes so
        // the SIMD tail is exercised together with the labelling.
        let shapes = [
            (1usize, 1usize),
            (7, 5),
            (31, 3),
            (32, 32),
            (33, 17),
            (64, 64),
            (65, 9),
            (127, 71),
            (256, 181),
            (640, 480),
        ];

        for &(w, h) in &shapes {
            let t = vec![128u8; w * h];

            // All background / all foreground.
            out.push((
                format!("empty_{w}x{h}"),
                vec![200u8; w * h],
                t.clone(),
                w,
                h,
            ));
            out.push((format!("full_{w}x{h}"), vec![10u8; w * h], t.clone(), w, h));

            // Checkerboard: maximal run count, and the case where 4- and
            // 8-connectivity disagree on every pixel.
            let mut checker = vec![200u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    if (x + y) % 2 == 0 {
                        checker[y * w + x] = 10;
                    }
                }
            }
            out.push((format!("checker_{w}x{h}"), checker, t.clone(), w, h));

            // Anti-diagonal staircase: a single 8-connected component that is
            // 4-disconnected, and a maximally deep union chain.
            let mut diag = vec![200u8; w * h];
            for y in 0..h {
                let x = y % w;
                diag[y * w + x] = 10;
            }
            out.push((format!("diag_{w}x{h}"), diag, t.clone(), w, h));

            // Comb: vertical teeth joined by a bottom bar — forces many unions
            // into one root on the final row.
            let mut comb = vec![200u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    if x % 3 == 0 || y == h - 1 {
                        comb[y * w + x] = 10;
                    }
                }
            }
            out.push((format!("comb_{w}x{h}"), comb, t.clone(), w, h));

            // Dense and sparse pseudo-random speckle.
            for (tag, cut) in [("noise_sparse", 24u32), ("noise_dense", 180)] {
                let mut rng = Lcg(0x2026_0920 ^ ((w as u64) << 20) ^ (h as u64));
                let mut noise = vec![0u8; w * h];
                for p in &mut noise {
                    *p = if rng.next_u32() % 256 < cut { 10 } else { 200 };
                }
                out.push((format!("{tag}_{w}x{h}"), noise, t.clone(), w, h));
            }

            // Spatially varying threshold map with a marker-like blob: the
            // production regime (tile thresholds differ across the frame).
            let mut rng = Lcg(0xDEAD_BEEF ^ (w as u64));
            let mut px = vec![0u8; w * h];
            let mut tm = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    px[y * w + x] = (rng.next_u32() % 256) as u8;
                    tm[y * w + x] = (((x / 8 + y / 8) % 5) * 40) as u8;
                }
            }
            for y in h / 4..(h * 3) / 4 {
                for x in w / 4..(w * 3) / 4 {
                    px[y * w + x] = 0;
                    tm[y * w + x] = 200;
                }
            }
            out.push((format!("tiles_{w}x{h}"), px, tm, w, h));
        }

        out
    }

    /// Runs `body` once in a single-worker pool and once in a 4-worker pool, so
    /// both the sequential and the parallel run extractor (and both the
    /// one-task and the split `rayon::join` fill) are covered.
    fn in_both_pools(body: impl Fn() + Sync) {
        for threads in [1usize, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            pool.install(&body);
        }
    }

    #[test]
    fn parallel_ccl_is_bit_exact_vs_reference() {
        let zoo = frames();
        in_both_pools(|| {
            for (name, pixels, tm, w, h) in &zoo {
                for conn8 in [true, false] {
                    for min_area in [0u32, 1, 4, 16, 1000] {
                        assert_bit_exact(name, pixels, tm, *w, *h, conn8, min_area);
                    }
                }
            }
        });
    }

    /// The two run extractors must be interchangeable: same runs, same order,
    /// same global indices, and both equal to the legacy `Vec`-push extractor.
    #[test]
    fn run_extractors_agree() {
        for (name, pixels, tm, w, h) in frames() {
            let img = ImageView::new(&pixels, w, h, w).expect("valid test image");

            let seq = extract_runs_sequential(&img, &tm);
            let arena_par = Bump::new();
            let par = extract_runs_parallel(&arena_par, &img, &tm);
            assert_eq!(
                seq.as_slice(),
                par,
                "{name}: sequential vs parallel extraction"
            );

            let mut legacy = simd_scanline::extract_rle_segments(&img, &tm);
            for (index, run) in legacy.iter_mut().enumerate() {
                run.label = index as u32;
            }
            assert_eq!(seq, legacy.as_slice(), "{name}: vs legacy extraction");
        }
    }

    /// One frame big enough to cross `ROWS_PER_FILL_TASK`, so the recursive
    /// `rayon::join` split (and its run-range bisection) is actually taken.
    #[test]
    fn parallel_ccl_is_bit_exact_on_multi_task_frame() {
        let (w, h) = (1024usize, 1024usize);
        let mut rng = Lcg(0x5EED_1234);
        let mut pixels = vec![0u8; w * h];
        let mut tm = vec![128u8; w * h];
        for p in &mut pixels {
            *p = (rng.next_u32() % 256) as u8;
        }
        // A few large blobs that span many fill tasks, on top of the speckle.
        for (cx, cy, r) in [
            (200usize, 150usize, 120usize),
            (700, 800, 200),
            (512, 512, 64),
        ] {
            for y in cy.saturating_sub(r)..(cy + r).min(h) {
                for x in cx.saturating_sub(r)..(cx + r).min(w) {
                    pixels[y * w + x] = 0;
                    tm[y * w + x] = 255;
                }
            }
        }
        in_both_pools(|| {
            for conn8 in [true, false] {
                for min_area in [0u32, 16, 4096] {
                    assert_bit_exact("multi_task_1024", &pixels, &tm, w, h, conn8, min_area);
                }
            }
        });
    }
}
