//! Adaptive thresholding and image binarization.
//!
//! This module provides the first stage of the detection pipeline, converting grayscale
//! input into binary images while adapting to local lighting conditions.
//!
//! Segmentation consumes the per-pixel **threshold map** this module writes: a
//! pixel is foreground when `pixel < threshold_map[pixel]`, so a threshold of
//! `0` means "never foreground". [`crate::config::ThresholdMode`] selects how
//! the map is built:
//! 1. **Tile-based** (`TileMidExtreme`, the default): one threshold per
//!    `tile_size` tile from the min/max over its 3×3 tile neighbourhood, then
//!    expanded to pixels. Cheap — stats are computed once per tile, not per
//!    pixel — but the threshold follows the local *extremes*, which speckles
//!    flat regions and lets a dark background fuse with a marker.
//! 2. **Local mean** (`LocalMean`): a true per-pixel local mean over a
//!    `(2r+1)²` window, minus a constant, from a sliding column-sum
//!    accumulator. Tracks the local background level instead of the extremes.
//!
//! The integral-image kernels lower in this file are benchmark references
//! (`benches/integral_threshold_bench.rs`); no detection path calls them.

#![allow(unsafe_code, clippy::cast_sign_loss)]
use crate::config::{DetectorConfig, ThresholdMode};
use crate::image::ImageView;
use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use multiversion::multiversion;
use rayon::prelude::*;

/// Statistics for a single threshold tile.
#[derive(Clone, Copy, Debug, Default)]
pub struct TileStats {
    /// Minimum pixel value in the tile.
    pub min: u8,
    /// Maximum pixel value in the tile.
    pub max: u8,
}

/// Adaptive thresholding engine using tile-based stats.
pub struct ThresholdEngine {
    /// Size of the tiles used for local thresholding statistics.
    pub tile_size: usize,
    /// Minimum intensity range for a tile to be considered valid.
    pub min_range: u8,
    /// How the per-pixel foreground threshold is built.
    pub mode: ThresholdMode,
    /// Window radius (pixels) for [`ThresholdMode::LocalMean`].
    pub local_mean_radius: usize,
    /// Constant subtracted from the local mean by [`ThresholdMode::LocalMean`].
    pub constant: i16,
}

impl Default for ThresholdEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ThresholdEngine {
    /// Create a new ThresholdEngine with default settings.
    #[must_use]
    pub fn new() -> Self {
        let d = DetectorConfig::default();
        Self {
            tile_size: 8,  // Standard 8x8 tiles
            min_range: 10, // Match DetectorConfig::default()
            mode: d.threshold_mode,
            local_mean_radius: d.threshold_local_mean_radius,
            constant: d.adaptive_threshold_constant,
        }
    }

    /// Create a ThresholdEngine from detector configuration.
    #[must_use]
    pub fn from_config(config: &DetectorConfig) -> Self {
        Self {
            tile_size: config.threshold_tile_size,
            min_range: config.threshold_min_range,
            mode: config.threshold_mode,
            local_mean_radius: config.threshold_local_mean_radius,
            constant: config.adaptive_threshold_constant,
        }
    }

    /// Compute min/max statistics for each tile in the image.
    /// Optimized with SIMD-friendly memory access patterns and subsampling (stride 2).
    #[must_use]
    #[tracing::instrument(skip_all, name = "pipeline::threshold_compute_stats")]
    pub fn compute_tile_stats<'a>(
        &self,
        arena: &'a Bump,
        img: &ImageView,
    ) -> BumpVec<'a, TileStats> {
        let ts = self.tile_size;
        let tiles_wide = img.width / ts;
        let tiles_high = img.height / ts;
        let mut stats = BumpVec::with_capacity_in(tiles_wide * tiles_high, arena);
        stats.resize(tiles_wide * tiles_high, TileStats { min: 255, max: 0 });

        stats
            .par_chunks_mut(tiles_wide)
            .enumerate()
            .for_each(|(ty, stats_row)| {
                // Subsampling: Only process every other row within a tile (stride 2)
                // This statistically approximates the min/max sufficient for thresholding
                for dy in 0..ts {
                    let py = ty * ts + dy;
                    let src_row = img.get_row(py);

                    // Process all tiles in this row with SIMD-friendly min/max
                    compute_row_tile_stats_simd(src_row, stats_row, ts);
                }
            });
        stats
    }

    /// Apply adaptive thresholding to the image.
    /// Optimized with pre-expanded threshold maps and vectorized row processing.
    #[expect(
        clippy::too_many_lines,
        reason = "one cohesive adaptive-threshold routine (tile threshold + validity, propagation, row expansion); splitting it would fragment the data flow"
    )]
    #[allow(dead_code)]
    pub fn apply_threshold(
        &self,
        arena: &Bump,
        img: &ImageView,
        stats: &[TileStats],
        output: &mut [u8],
    ) {
        let ts = self.tile_size;
        let tiles_wide = img.width / ts;
        let tiles_high = img.height / ts;

        let mut tile_thresholds = BumpVec::with_capacity_in(tiles_wide * tiles_high, arena);
        tile_thresholds.resize(tiles_wide * tiles_high, 0u8);
        let mut tile_valid = BumpVec::with_capacity_in(tiles_wide * tiles_high, arena);
        tile_valid.resize(tiles_wide * tiles_high, 0u8);

        tile_thresholds
            .par_chunks_mut(tiles_wide)
            .enumerate()
            .for_each(|(ty, t_row)| {
                let y_start = ty.saturating_sub(1);
                let y_end = (ty + 1).min(tiles_high - 1);

                for tx in 0..tiles_wide {
                    let mut nmin = 255u8;
                    let mut nmax = 0u8;

                    let x_start = tx.saturating_sub(1);
                    let x_end = (tx + 1).min(tiles_wide - 1);

                    for ny in y_start..=y_end {
                        let row_off = ny * tiles_wide;
                        for nx in x_start..=x_end {
                            let s = stats[row_off + nx];
                            if s.min < nmin {
                                nmin = s.min;
                            }
                            if s.max > nmax {
                                nmax = s.max;
                            }
                        }
                    }

                    let t_idx = tx;
                    let res = ((u16::from(nmin) + u16::from(nmax)) >> 1) as u8;

                    t_row[t_idx] = res;
                }
            });

        // Compute tile_valid (can be done in same loop above or separate)
        for ty in 0..tiles_high {
            for tx in 0..tiles_wide {
                let mut nmin = 255;
                let mut nmax = 0;
                let y_start = ty.saturating_sub(1);
                let y_end = (ty + 1).min(tiles_high - 1);
                let x_start = tx.saturating_sub(1);
                let x_end = (tx + 1).min(tiles_wide - 1);

                for ny in y_start..=y_end {
                    let row_off = ny * tiles_wide;
                    for nx in x_start..=x_end {
                        let s = stats[row_off + nx];
                        if s.min < nmin {
                            nmin = s.min;
                        }
                        if s.max > nmax {
                            nmax = s.max;
                        }
                    }
                }
                let idx = ty * tiles_wide + tx;
                tile_valid[idx] = if nmax.saturating_sub(nmin) < self.min_range {
                    0
                } else {
                    255
                };
            }
        }

        // --- Propagation Pass ---
        // Fill thresholds for invalid tiles from their neighbors to stay
        // consistent within large uniform regions.
        for _ in 0..2 {
            // 2 iterations are usually enough for local consistency
            for ty in 0..tiles_high {
                for tx in 0..tiles_wide {
                    let idx = ty * tiles_wide + tx;
                    if tile_valid[idx] == 0 {
                        let mut sum_thresh = 0u32;
                        let mut count = 0u32;

                        let y_start = ty.saturating_sub(1);
                        let y_end = (ty + 1).min(tiles_high - 1);
                        let x_start = tx.saturating_sub(1);
                        let x_end = (tx + 1).min(tiles_wide - 1);

                        for ny in y_start..=y_end {
                            let row_off = ny * tiles_wide;
                            for nx in x_start..=x_end {
                                let n_idx = row_off + nx;
                                if tile_valid[n_idx] > 0 {
                                    sum_thresh += u32::from(tile_thresholds[n_idx]);
                                    count += 1;
                                }
                            }
                        }

                        if count > 0 {
                            tile_thresholds[idx] = (sum_thresh / count) as u8;
                            tile_valid[idx] = 128; // Partial valid (propagated)
                        }
                    }
                }
            }
        }

        let thresholds_slice = tile_thresholds.as_slice();
        let valid_slice = tile_valid.as_slice();

        output
            .par_chunks_mut(ts * img.width)
            .enumerate()
            .for_each_init(
                || (vec![0u8; img.width], vec![0u8; img.width]),
                |(row_thresholds, row_valid), (ty, output_tile_rows)| {
                    if ty >= tiles_high {
                        return;
                    }
                    // Zero-fill the trailing pixels beyond `tiles_wide * ts`: the
                    // per-tile loop below only writes indices `[0, tiles_wide * ts)`
                    // but `threshold_row_simd` reads the full `img.width`.
                    row_thresholds.fill(0);
                    row_valid.fill(0);

                    // Expand tile stats to row buffers
                    for tx in 0..tiles_wide {
                        let idx = ty * tiles_wide + tx;
                        let thresh = thresholds_slice[idx];
                        let valid = valid_slice[idx];
                        for i in 0..ts {
                            row_thresholds[tx * ts + i] = thresh;
                            row_valid[tx * ts + i] = valid;
                        }
                    }

                    for dy in 0..ts {
                        let py = ty * ts + dy;
                        let src_row = img.get_row(py);
                        let dst_row = &mut output_tile_rows[dy * img.width..(dy + 1) * img.width];

                        threshold_row_simd(src_row, dst_row, row_thresholds, row_valid);
                    }
                },
            );
    }

    /// Apply adaptive thresholding and return both binary and threshold maps.
    ///
    /// This is needed for threshold-model-aware segmentation, which uses the
    /// per-pixel threshold values to connect pixels by their deviation sign.
    ///
    /// The exact rule is selected by [`ThresholdMode`]; segmentation reads
    /// `threshold_output` alone, so a threshold of `0` means "this pixel can
    /// never be foreground".
    #[expect(
        clippy::needless_range_loop,
        reason = "tx indexes both the tile-neighbourhood min/max scan and the t_row write, so the range loop is clearer than a zipped iterator here"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one cohesive routine: mode dispatch, tile threshold, tile validity and row expansion share the tile-grid geometry computed at the top"
    )]
    #[tracing::instrument(skip_all, name = "pipeline::threshold_apply_map")]
    pub fn apply_threshold_with_map(
        &self,
        arena: &Bump,
        img: &ImageView,
        stats: &[TileStats],
        binary_output: &mut [u8],
        threshold_output: &mut [u8],
    ) {
        if self.mode == ThresholdMode::LocalMean {
            self.apply_local_mean(arena, img, binary_output, threshold_output);
            return;
        }
        let ts = self.tile_size;
        let tiles_wide = img.width / ts;
        let tiles_high = img.height / ts;

        let mut tile_thresholds = BumpVec::with_capacity_in(tiles_wide * tiles_high, arena);
        tile_thresholds.resize(tiles_wide * tiles_high, 0u8);
        let mut tile_valid = BumpVec::with_capacity_in(tiles_wide * tiles_high, arena);
        tile_valid.resize(tiles_wide * tiles_high, 0u8);

        tile_thresholds
            .par_chunks_mut(tiles_wide)
            .enumerate()
            .for_each(|(ty, t_row)| {
                let y_start = ty.saturating_sub(1);
                let y_end = (ty + 1).min(tiles_high - 1);

                for tx in 0..tiles_wide {
                    let mut nmin = 255u8;
                    let mut nmax = 0u8;

                    let x_start = tx.saturating_sub(1);
                    let x_end = (tx + 1).min(tiles_wide - 1);

                    for ny in y_start..=y_end {
                        let row_off = ny * tiles_wide;
                        for nx in x_start..=x_end {
                            let s = stats[row_off + nx];
                            if s.min < nmin {
                                nmin = s.min;
                            }
                            if s.max > nmax {
                                nmax = s.max;
                            }
                        }
                    }

                    t_row[tx] = ((u16::from(nmin) + u16::from(nmax)) >> 1) as u8;
                }
            });

        // Compute tile_valid
        for ty in 0..tiles_high {
            for tx in 0..tiles_wide {
                let mut nmin = 255;
                let mut nmax = 0;
                let y_start = ty.saturating_sub(1);
                let y_end = (ty + 1).min(tiles_high - 1);
                let x_start = tx.saturating_sub(1);
                let x_end = (tx + 1).min(tiles_wide - 1);

                for ny in y_start..=y_end {
                    let row_off = ny * tiles_wide;
                    for nx in x_start..=x_end {
                        let s = stats[row_off + nx];
                        if s.min < nmin {
                            nmin = s.min;
                        }
                        if s.max > nmax {
                            nmax = s.max;
                        }
                    }
                }
                let idx = ty * tiles_wide + tx;
                tile_valid[idx] = if nmax.saturating_sub(nmin) < self.min_range {
                    0
                } else {
                    255
                };
            }
        }

        // Write thresholds and binary output in parallel
        let thresholds_slice = tile_thresholds.as_slice();
        let valid_slice = tile_valid.as_slice();

        binary_output
            .par_chunks_mut(ts * img.width)
            .enumerate()
            .for_each_init(
                || (vec![0u8; img.width], vec![0u8; img.width]),
                |(row_thresholds, row_valid), (ty, bin_tile_rows)| {
                    if ty >= tiles_high {
                        return;
                    }
                    // SAFETY: `binary_output.par_chunks_mut(ts * img.width)`
                    // yields one `ty` per worker; the parallel `threshold_output`
                    // tile slice for the same `ty` is therefore disjoint from
                    // every other worker's write set. The `ty < tiles_high`
                    // guard above keeps `ty * ts * img.width + ts * img.width`
                    // within the original `threshold_output` length.
                    let thresh_tile_rows = unsafe {
                        let ptr = threshold_output.as_ptr().cast_mut();
                        std::slice::from_raw_parts_mut(ptr.add(ty * ts * img.width), ts * img.width)
                    };

                    row_thresholds.fill(0);
                    row_valid.fill(0);

                    for tx in 0..tiles_wide {
                        let idx = ty * tiles_wide + tx;
                        let thresh = thresholds_slice[idx];
                        let valid = valid_slice[idx];
                        for i in 0..ts {
                            row_thresholds[tx * ts + i] = thresh;
                            row_valid[tx * ts + i] = valid;
                        }
                    }

                    for dy in 0..ts {
                        let py = ty * ts + dy;
                        let src_row = img.get_row(py);

                        // Write binary output
                        let bin_row = &mut bin_tile_rows[dy * img.width..(dy + 1) * img.width];
                        threshold_row_simd(src_row, bin_row, row_thresholds, row_valid);

                        // Write threshold map
                        thresh_tile_rows[dy * img.width..(dy + 1) * img.width]
                            .copy_from_slice(row_thresholds);
                    }
                },
            );
    }

    /// Per-pixel local-mean threshold, `t(x,y) = mean_{(2r+1)²}(x,y) - constant`.
    ///
    /// Computed with a sliding column-sum rather than an integral image: the
    /// only auxiliary buffer is one `u32` column accumulator per row-strip
    /// (`strips * width * 4` bytes, ≈ 0.4 MB at 4K), against 33–66 MB for a
    /// full-frame integral image. The accumulator is exact integer arithmetic
    /// and every strip re-initialises it from the image, so the output does
    /// not depend on the strip size or on the number of rayon workers.
    #[expect(
        clippy::many_single_char_names,
        reason = "w/h/r/c are the conventional image-kernel names (width, height, window radius, OpenCV's C offset) and read better here than spelled-out aliases"
    )]
    fn apply_local_mean(
        &self,
        arena: &Bump,
        img: &ImageView,
        binary_output: &mut [u8],
        threshold_output: &mut [u8],
    ) {
        let w = img.width;
        let h = img.height;
        if w == 0 || h == 0 {
            return;
        }
        let r = self.local_mean_radius.max(1);
        let c = i32::from(self.constant);

        // A strip re-scans `2r+1` rows to prime its column sums, so keep strips
        // comfortably taller than the window; still aim for ≥ 4 strips per
        // worker so rayon can balance.
        let target = h.div_ceil((rayon::current_num_threads() * 4).max(1));
        let strip_rows = target.max(4 * r + 1).min(h).max(1);
        let n_strips = h.div_ceil(strip_rows);
        let col_sums = arena.alloc_slice_fill_copy(n_strips * w, 0u32);

        threshold_output[..w * h]
            .par_chunks_mut(strip_rows * w)
            .zip(binary_output[..w * h].par_chunks_mut(strip_rows * w))
            .zip(col_sums.par_chunks_mut(w))
            .enumerate()
            .for_each(|(strip, ((t_chunk, b_chunk), cols))| {
                let y_begin = strip * strip_rows;
                let rows = t_chunk.len() / w;

                // Prime the column sums for the window of the strip's first row.
                let mut y0 = y_begin.saturating_sub(r);
                let mut y1 = (y_begin + r + 1).min(h);
                cols.fill(0);
                for y in y0..y1 {
                    accumulate_row(cols, img.get_row(y), true);
                }

                for dy in 0..rows {
                    let y = y_begin + dy;
                    if dy > 0 {
                        // The window moves down by exactly one row, so at most
                        // one row enters and one row leaves.
                        let ny1 = (y + r + 1).min(h);
                        if ny1 > y1 {
                            accumulate_row(cols, img.get_row(y1), true);
                            y1 = ny1;
                        }
                        let ny0 = y.saturating_sub(r);
                        if ny0 > y0 {
                            accumulate_row(cols, img.get_row(y0), false);
                            y0 = ny0;
                        }
                    }

                    let rows_in_window = (y1 - y0) as u32;
                    let src_row = img.get_row(y);
                    let t_row = &mut t_chunk[dy * w..(dy + 1) * w];
                    let b_row = &mut b_chunk[dy * w..(dy + 1) * w];
                    local_mean_row(src_row, cols, t_row, b_row, r, rows_in_window, c);
                }
            });
    }
}

/// Add (`add = true`) or subtract a source row from the column accumulator.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn accumulate_row(cols: &mut [u32], src: &[u8], add: bool) {
    if add {
        for (col, &p) in cols.iter_mut().zip(src.iter()) {
            *col += u32::from(p);
        }
    } else {
        for (col, &p) in cols.iter_mut().zip(src.iter()) {
            *col -= u32::from(p);
        }
    }
}

/// Emit one row of `threshold = local_mean - c` plus its binarization.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn local_mean_row(
    src: &[u8],
    cols: &[u32],
    thresholds: &mut [u8],
    binary: &mut [u8],
    r: usize,
    rows_in_window: u32,
    c: i32,
) {
    let w = src.len();
    // Running horizontal sum over the column accumulator: `sum` always holds
    // the box sum over columns `[x0, x1)`.
    let mut x1 = (r + 1).min(w);
    let mut x0 = 0usize;
    // `u64` so a full-width window on a very large frame cannot overflow:
    // `(2r+1) · 255 · w` exceeds `u32` beyond ~16.8 Mpx.
    let mut sum: u64 = cols[..x1].iter().map(|&v| u64::from(v)).sum();

    let mut last_area = 0u32;
    let mut inv_area = 0u32;

    for x in 0..w {
        let area = rows_in_window * (x1 - x0) as u32;
        if area != last_area {
            // `area` only changes within `r` pixels of the left/right edge,
            // so this reciprocal is recomputed ~2r times per row, not per pixel.
            inv_area = ((1u64 << 31) / u64::from(area)) as u32;
            last_area = area;
        }
        let mean = ((sum * u64::from(inv_area)) >> 31) as i32;

        let t = (mean - c).clamp(0, 255) as u8;
        thresholds[x] = t;
        binary[x] = if src[x] < t { 0 } else { 255 };

        // Slide the window one column to the right.
        if x + r + 1 < w {
            sum += u64::from(cols[x + r + 1]);
            x1 += 1;
        }
        if x >= r {
            sum -= u64::from(cols[x - r]);
            x0 += 1;
        }
    }
}

#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn compute_row_tile_stats_simd(src_row: &[u8], stats: &mut [TileStats], tile_size: usize) {
    let chunks = src_row.chunks_exact(tile_size);
    for (chunk, stat) in chunks.zip(stats.iter_mut()) {
        let mut rmin = stat.min;
        let mut rmax = stat.max;

        // Hint to compiler for vectorization
        for &p in chunk {
            rmin = rmin.min(p);
            rmax = rmax.max(p);
        }

        stat.min = rmin;
        stat.max = rmax;
    }
}

#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn threshold_row_simd(src: &[u8], dst: &mut [u8], thresholds: &[u8], valid_mask: &[u8]) {
    let len = src.len();
    // Use chunks_exact to help compiler vectorize (e.g. into 16 or 32-byte chunks)
    let src_chunks = src.chunks_exact(16);
    let dst_chunks = dst.chunks_exact_mut(16);
    let thresh_chunks = thresholds.chunks_exact(16);
    let mask_chunks = valid_mask.chunks_exact(16);

    for (((s_c, d_c), t_c), m_c) in src_chunks
        .zip(dst_chunks)
        .zip(thresh_chunks)
        .zip(mask_chunks)
    {
        for i in 0..16 {
            let s = s_c[i];
            let t = t_c[i];
            let v = m_c[i];
            // Nested if structure often helps compiler with CMV/Masks
            d_c[i] = if v > 0 {
                if s < t { 0 } else { 255 }
            } else {
                255
            };
        }
    }

    // Handle tail
    let processed = (len / 16) * 16;
    for i in processed..len {
        let s = src[i];
        let t = thresholds[i];
        let v = valid_mask[i];
        dst[i] = if v > 0 {
            if s < t { 0 } else { 255 }
        } else {
            255
        };
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::naive_bytecount,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_threshold_invariants(data in prop::collection::vec(0..=255u8, 16)) {
            let mut min = 255u8;
            let mut max = 0u8;
            for &b in &data {
                if b < min { min = b; }
                if b > max { max = b; }
            }
            let (rmin, rmax) = compute_min_max_simd(&data);
            assert_eq!(rmin, min);
            assert_eq!(rmax, max);
        }

        #[test]
        fn test_binarization_invariants(src in prop::collection::vec(0..=255u8, 16), thresh in 0..=255u8) {
            let mut dst = vec![0u8; 16];
            let valid = vec![255u8; 16];
            threshold_row_simd(&src, &mut dst, &[thresh; 16], &valid);

            for (i, &s) in src.iter().enumerate() {
                if s >= thresh {
                    assert_eq!(dst[i], 255);
                } else {
                    assert_eq!(dst[i], 0);
                }
            }

            // Test invalid tile (mask=0) should be white (255)
            let invalid = vec![0u8; 16];
            threshold_row_simd(&src, &mut dst, &[thresh; 16], &invalid);
            for d in dst {
                assert_eq!(d, 255);
            }
        }
    }

    #[test]
    fn test_threshold_engine_e2e() {
        let width = 16;
        let height = 16;
        let mut data = vec![128u8; width * height];
        // Draw a black square in a white background area to test adaptive threshold
        for y in 4..12 {
            for x in 4..12 {
                data[y * width + x] = 50;
            }
        }
        for y in 0..height {
            for x in 0..width {
                if !(2..=14).contains(&x) || !(2..=14).contains(&y) {
                    data[y * width + x] = 200;
                }
            }
        }

        let img = ImageView::new(&data, width, height, width).unwrap();
        let engine = ThresholdEngine::new();
        let arena = Bump::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let mut output = vec![0u8; width * height];
        engine.apply_threshold(&arena, &img, &stats, &mut output);

        // At (8,8), it should be black (0) because it's 50 and thresh should be around (50+200)/2 = 125
        assert_eq!(output[8 * width + 8], 0);
        // At (1,1), it should be white (255)
        assert_eq!(output[width + 1], 255);
    }

    #[test]
    fn test_threshold_with_decimation() {
        let width = 32;
        let height = 32;
        let mut data = vec![200u8; width * height];
        // Draw a black square (16x16) in the center
        for y in 8..24 {
            for x in 8..24 {
                data[y * width + x] = 50;
            }
        }

        let img = ImageView::new(&data, width, height, width).unwrap();

        // Decimate by 2 -> 16x16
        let mut decimated_data = vec![0u8; 16 * 16];
        let decimated_img = img
            .decimate_to(2, &mut decimated_data)
            .expect("decimation failed");

        assert_eq!(decimated_img.width, 16);
        assert_eq!(decimated_img.height, 16);
        let arena = Bump::new();
        let engine = ThresholdEngine::new();
        let stats = engine.compute_tile_stats(&arena, &decimated_img);
        let mut output = vec![0u8; 16 * 16];
        engine.apply_threshold(&arena, &decimated_img, &stats, &mut output);

        // At (4,4) in decimated image (which is 8,8 in original), it should be black (0)
        assert_eq!(output[4 * 16 + 4], 0);
        // At (0,0) in decimated image, it should be white (255)
        assert_eq!(output[0], 255);
    }

    // ========================================================================
    // THRESHOLD MODE TESTS
    // ========================================================================

    /// Deterministic pseudo-random image (no `rand` dependency — see
    /// `docs/engineering/constraints.md` §5).
    fn lcg_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// Exact `O(r²)` box mean, the reference the sliding accumulator must match.
    #[allow(clippy::many_single_char_names)]
    fn naive_box_mean(data: &[u8], w: usize, h: usize, x: usize, y: usize, r: usize) -> u32 {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(h);
        let x0 = x.saturating_sub(r);
        let x1 = (x + r + 1).min(w);
        let mut sum = 0u32;
        for yy in y0..y1 {
            for xx in x0..x1 {
                sum += u32::from(data[yy * w + xx]);
            }
        }
        sum / ((y1 - y0) * (x1 - x0)) as u32
    }

    fn run_mode(
        mode: ThresholdMode,
        radius: usize,
        constant: i16,
        data: &[u8],
        w: usize,
        h: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let img = ImageView::new(data, w, h, w).unwrap();
        let engine = ThresholdEngine {
            tile_size: 8,
            min_range: 10,
            mode,
            local_mean_radius: radius,
            constant,
        };
        let arena = Bump::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let mut binary = vec![0u8; w * h];
        let mut map = vec![0u8; w * h];
        engine.apply_threshold_with_map(&arena, &img, &stats, &mut binary, &mut map);
        (binary, map)
    }

    /// The sliding column accumulator must reproduce the exact box mean.
    ///
    /// Tolerance 1 covers the fixed-point reciprocal (`(sum * ⌊2³¹/area⌋) >> 31`
    /// can land one below the exact quotient); the accumulator itself is exact.
    #[test]
    fn local_mean_matches_naive_box_mean() {
        for &(w, h) in &[(64usize, 48usize), (37, 29), (8, 8), (129, 5)] {
            let data = lcg_image(w, h, 7);
            for &r in &[1usize, 3, 12, 40] {
                let (binary, map) = run_mode(ThresholdMode::LocalMean, r, 0, &data, w, h);
                for y in 0..h {
                    for x in 0..w {
                        let expect = naive_box_mean(&data, w, h, x, y, r).min(255);
                        let got = u32::from(map[y * w + x]);
                        assert!(
                            expect.abs_diff(got) <= 1,
                            "w={w} h={h} r={r} at ({x},{y}): got {got}, exact {expect}"
                        );
                        // The binary map is always the map applied to the source.
                        let want_bin = if data[y * w + x] < map[y * w + x] {
                            0
                        } else {
                            255
                        };
                        assert_eq!(binary[y * w + x], want_bin);
                    }
                }
            }
        }
    }

    /// `constant` shifts the threshold down, so raising it can only ever turn
    /// foreground pixels into background — the noise-suppression knob.
    #[test]
    fn local_mean_constant_is_monotone() {
        let (w, h) = (96usize, 64usize);
        let data = lcg_image(w, h, 11);
        let (_, no_offset) = run_mode(ThresholdMode::LocalMean, 8, 0, &data, w, h);
        let (_, with_offset) = run_mode(ThresholdMode::LocalMean, 8, 10, &data, w, h);
        for i in 0..w * h {
            assert!(with_offset[i] <= no_offset[i]);
        }
    }

    /// A perfectly flat frame has no structure. The local mean equals the grey
    /// level everywhere, so any positive `constant` drives the threshold below
    /// it and nothing is foreground — the noise-suppression property.
    ///
    /// The historical mode is the one that speckles: `t = mid(97, 97) = 97` is
    /// published to segmentation even though the tile carries no signal.
    #[test]
    fn local_mean_suppresses_flat_regions() {
        let (w, h) = (64usize, 64usize);
        let data = vec![97u8; w * h];

        let (binary, map) = run_mode(ThresholdMode::LocalMean, 8, 5, &data, w, h);
        // 97 - 5, within the 1-unit slack of the fixed-point reciprocal.
        assert!(
            map.iter().all(|&t| (91..=92).contains(&t)),
            "{:?}",
            &map[..8]
        );
        assert!(
            binary.iter().all(|&b| b == 255),
            "flat frame produced foreground"
        );

        let (_, map) = run_mode(ThresholdMode::TileMidExtreme, 8, 5, &data, w, h);
        assert!(map.contains(&97));
    }

    /// The binary map and the threshold map segmentation reads must agree pixel
    /// for pixel under `LocalMean`. (`TileMidExtreme` deliberately does not:
    /// it suppresses flat tiles in the binary map only, which is the split this
    /// mode exists to close.)
    #[test]
    fn local_mean_keeps_binary_and_threshold_maps_consistent() {
        let (w, h) = (80usize, 56usize);
        let mut data = lcg_image(w, h, 3);
        for y in 16..40 {
            data[y * w + 16..y * w + 48].fill(0);
        }
        let (binary, map) = run_mode(ThresholdMode::LocalMean, 6, 2, &data, w, h);
        for i in 0..w * h {
            let want = if data[i] < map[i] { 0 } else { 255 };
            assert_eq!(binary[i], want, "disagreed at {i}");
        }
    }

    /// A thick black square on a bright background: the local mean must mark the
    /// *whole* square as foreground, including its flat interior, which the tile
    /// midpoint cannot do (`mid(0, 0) = 0` there).
    #[test]
    fn local_mean_fills_a_thick_flat_interior() {
        let (w, h) = (64usize, 64usize);
        let mut data = vec![200u8; w * h];
        for y in 16..48 {
            data[y * w + 16..y * w + 48].fill(0);
        }
        let centre = 32 * w + 32;
        let (_, tile) = run_mode(ThresholdMode::TileMidExtreme, 8, 0, &data, w, h);
        let (_, local) = run_mode(ThresholdMode::LocalMean, 24, 15, &data, w, h);
        assert_eq!(
            tile[centre], 0,
            "tile mode already thresholded the interior"
        );
        assert!(
            local[centre] > 0,
            "local mean left the flat interior unthresholded"
        );
    }

    // ========================================================================
    // THRESHOLD ROBUSTNESS TESTS
    // ========================================================================

    use crate::config::TagFamily;
    use crate::test_utils::{
        TestImageParams, generate_test_image_with_params, measure_border_integrity,
    };

    /// Test threshold preserves tag structure at varying sizes (distance proxy).
    /// Note: AprilTag 36h11 has 8x8 cells, so 4px/bit = 32px minimum.
    #[test]
    fn test_threshold_preserves_tag_structure_at_varying_sizes() {
        let canvas_size = 640;
        // Minimum 32px for 4 pixels per bit (AprilTag 36h11 = 8x8 cells)
        let tag_sizes = [32, 48, 64, 100, 150, 200, 300];

        for tag_size in tag_sizes {
            let params = TestImageParams {
                family: TagFamily::AprilTag36h11,
                id: 0,
                tag_size,
                canvas_size,
                ..Default::default()
            };

            let (data, corners) = generate_test_image_with_params(&params);
            let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

            let engine = ThresholdEngine::new();
            let arena = Bump::new();
            let stats = engine.compute_tile_stats(&arena, &img);
            let mut binary = vec![0u8; canvas_size * canvas_size];
            engine.apply_threshold(&arena, &img, &stats, &mut binary);

            let integrity = measure_border_integrity(&binary, canvas_size, &corners);

            // For tags >= 32px (4px/bit), we expect good binarization (>50% border detected)
            assert!(
                integrity > 0.5,
                "Tag size {} failed: border integrity = {:.2}% (expected >50%)",
                tag_size,
                integrity * 100.0
            );

            println!(
                "Tag size {:>3}px: border integrity = {:.1}%",
                tag_size,
                integrity * 100.0
            );
        }
    }

    /// Test threshold robustness to brightness and contrast variations.
    #[test]
    fn test_threshold_robustness_brightness_contrast() {
        let canvas_size = 320;
        let tag_size = 120;
        let brightness_offsets = [-50, -25, 0, 25, 50];
        let contrast_scales = [0.50, 0.75, 1.0, 1.25, 1.50];

        for &brightness in &brightness_offsets {
            for &contrast in &contrast_scales {
                let params = TestImageParams {
                    family: TagFamily::AprilTag36h11,
                    id: 0,
                    tag_size,
                    canvas_size,
                    brightness_offset: brightness,
                    contrast_scale: contrast,
                    ..Default::default()
                };

                let (data, corners) = generate_test_image_with_params(&params);
                let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

                let engine = ThresholdEngine::new();
                let arena = Bump::new();
                let stats = engine.compute_tile_stats(&arena, &img);
                let mut binary = vec![0u8; canvas_size * canvas_size];
                engine.apply_threshold(&arena, &img, &stats, &mut binary);

                let integrity = measure_border_integrity(&binary, canvas_size, &corners);

                // For moderate conditions, expect good integrity
                let is_moderate = brightness.abs() <= 25 && contrast >= 0.75;
                if is_moderate {
                    assert!(
                        integrity > 0.4,
                        "Brightness {}, Contrast {:.2}: integrity {:.1}% too low",
                        brightness,
                        contrast,
                        integrity * 100.0
                    );
                }

                println!(
                    "Brightness {:>3}, Contrast {:.2}: integrity = {:.1}%",
                    brightness,
                    contrast,
                    integrity * 100.0
                );
            }
        }
    }

    /// Test threshold robustness to varying noise levels.
    #[test]
    fn test_threshold_robustness_noise() {
        let canvas_size = 320;
        let tag_size = 120;
        let noise_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0];

        for &noise_sigma in &noise_levels {
            let params = TestImageParams {
                family: TagFamily::AprilTag36h11,
                id: 0,
                tag_size,
                canvas_size,
                noise_sigma,
                ..Default::default()
            };

            let (data, corners) = generate_test_image_with_params(&params);
            let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

            let engine = ThresholdEngine::new();
            let arena = Bump::new();
            let stats = engine.compute_tile_stats(&arena, &img);
            let mut binary = vec![0u8; canvas_size * canvas_size];
            engine.apply_threshold(&arena, &img, &stats, &mut binary);

            let integrity = measure_border_integrity(&binary, canvas_size, &corners);

            // For noise <= 15, expect reasonable integrity
            if noise_sigma <= 15.0 {
                assert!(
                    integrity > 0.45,
                    "Noise σ={:.1}: integrity {:.1}% too low",
                    noise_sigma,
                    integrity * 100.0
                );
            }

            println!(
                "Noise σ={:>4.1}: integrity = {:.1}%",
                noise_sigma,
                integrity * 100.0
            );
        }
    }

    proptest! {
        /// Fuzz test threshold with random combinations of parameters.
        /// Minimum tag size 32px to ensure 4 pixels per bit.
        #[test]
        fn test_threshold_combined_conditions_no_panic(
            tag_size in 32_usize..200,  // 32px = 4px/bit for AprilTag 36h11
            brightness in -40_i16..40,
            contrast in 0.6_f32..1.4,
            noise in 0.0_f32..20.0
        ) {
            let canvas_size = 320;

            // Skip invalid combinations (tag too big for canvas)
            if tag_size >= canvas_size - 40 {
                return Ok(());
            }

            let params = TestImageParams {
                family: TagFamily::AprilTag36h11,
                id: 0,
                tag_size,
                canvas_size,
                noise_sigma: noise,
                brightness_offset: brightness,
                contrast_scale: contrast,
            };

            let (data, _corners) = generate_test_image_with_params(&params);
            let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

            let engine = ThresholdEngine::new();
            let arena = Bump::new();
            let stats = engine.compute_tile_stats(&arena, &img);
            let mut binary = vec![0u8; canvas_size * canvas_size];

            // Should not panic
            engine.apply_threshold(&arena, &img, &stats, &mut binary);

            // Basic sanity: output should have both black and white pixels
            let black_count = binary.iter().filter(|&&p| p == 0).count();
            let white_count = binary.iter().filter(|&&p| p == 255).count();

            // Valid binary image should have both colors (only 0 and 255)
            prop_assert!(black_count + white_count == binary.len(),
                "Binary output contains non-binary values");

            // With a tag present, we expect some black and white pixels
            prop_assert!(black_count > 0, "No black pixels in output");
            prop_assert!(white_count > 0, "No white pixels in output");
        }
    }
}

#[multiversion(targets = "simd")]
fn compute_min_max_simd(data: &[u8]) -> (u8, u8) {
    let mut min = 255u8;
    let mut max = 0u8;
    for &b in data {
        min = min.min(b);
        max = max.max(b);
    }
    (min, max)
}

// =============================================================================
// INTEGRAL IMAGE-BASED ADAPTIVE THRESHOLD
// =============================================================================
//
// This implements OpenCV-style ADAPTIVE_THRESH_MEAN_C using integral images:
// 1. Compute integral image in O(W*H)
// 2. For each pixel, compute local mean in O(1) using integral image
// 3. Threshold: pixel < (local_mean - C) ? black : white
//
// This produces per-pixel adaptive thresholds for small tag detection.

/// Compute integral image (cumulative sum) for fast box filter computation.
///
/// Uses a 2-pass parallel implementation for maximum throughput on modern multicore CPUs.
/// The `integral` buffer must have size `(img.width + 1) * (img.height + 1)`.
#[expect(
    clippy::needless_range_loop,
    clippy::items_after_statements,
    reason = "the first-row zero-init writes integral[x] by flat index to match the surrounding integral-buffer index arithmetic, and const BLOCK_SIZE is declared at its point of use in the second (vertical) pass"
)]
#[allow(dead_code)]
pub fn compute_integral_image(img: &ImageView, integral: &mut [u64]) {
    let w = img.width;
    let h = img.height;
    let stride = w + 1;

    // Zero the first row efficiently
    for x in 0..stride {
        integral[x] = 0;
    }

    // use rayon::prelude::*;

    // 1st Pass: Compute horizontal cumulative sums (prefix sum per row)
    // This part is perfectly parallel.
    integral
        .par_chunks_exact_mut(stride)
        .enumerate()
        .skip(1)
        .for_each(|(y_idx, row)| {
            let y = y_idx - 1;
            let src_row = img.get_row(y);
            let mut sum = 0u64;
            // row[0] is already 0
            for x in 0..w {
                sum += u64::from(src_row[x]);
                row[x + 1] = sum;
            }
        });

    // 2nd Pass: Vertical cumulative sums
    // For large images, we process in vertical blocks to stay in cache.
    const BLOCK_SIZE: usize = 128;
    let num_blocks = stride.div_ceil(BLOCK_SIZE);

    (0..num_blocks).into_par_iter().for_each(|b| {
        let start_x = b * BLOCK_SIZE;
        let end_x = (start_x + BLOCK_SIZE).min(stride);

        // Initialize cumulative sum for this column block
        // We use a small on-stack or small-vec if needed, but since BLOCK_SIZE is small (128),
        // we can just use a fixed-size array if we want to avoid allocation entirely.
        let mut col_sums = [0u64; BLOCK_SIZE];

        // SAFETY: `(0..num_blocks).into_par_iter()` partitions the column
        // range into BLOCK_SIZE-wide stripes; each rayon worker handles one
        // `b`, so the `[start_x, end_x)` column window for a given `b` is
        // disjoint from every other worker's window. `integral` length is
        // `(h + 1) * stride`, so `y * stride + start_x + i` for
        // `y ∈ [1, h], i < end_x - start_x` stays in-bounds. The outer
        // `par_iter` borrows `integral` mutably for its full lifetime, so
        // no other reader exists.
        unsafe {
            let base_ptr = integral.as_ptr().cast_mut();
            for y in 1..=h {
                let row_ptr = base_ptr.add(y * stride + start_x);
                for (i, val) in col_sums.iter_mut().enumerate().take(end_x - start_x) {
                    let old_val = *row_ptr.add(i);
                    let new_sum = old_val + *val;
                    *row_ptr.add(i) = new_sum;
                    *val = new_sum;
                }
            }
        }
    });
}

/// Apply per-pixel adaptive threshold using integral image.
///
/// Optimized with parallel processing and branchless thresholding.
#[multiversion(targets = "simd")]
/// Apply per-pixel adaptive threshold using integral image.
///
/// Optimized with parallel processing, interior-loop vectorization, and fixed-point arithmetic.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[tracing::instrument(skip_all, name = "pipeline::threshold_integral")]
pub fn adaptive_threshold_integral(
    img: &ImageView,
    integral: &[u64],
    output: &mut [u8],
    radius: usize,
    c: i16,
) {
    let w = img.width;
    let h = img.height;
    let stride = w + 1;

    // use rayon::prelude::*;

    // Precompute interior area inverse (fixed-point 1.31)
    let side = (2 * radius + 1) as u32;
    let area = side * side;
    let inv_area_fixed = ((1u64 << 31) / u64::from(area)) as u32;

    (0..h).into_par_iter().for_each(|y| {
        let y_offset = y * w;
        let src_row = img.get_row(y);

        // SAFETY: `(0..h).into_par_iter()` yields each `y` exactly once
        // across rayon workers, so the `[y * w, y * w + w)` slice is
        // disjoint from every other worker's slice. `output` length is
        // `h * w`, so the slice is in-bounds. The outer `par_iter` borrows
        // `output` mutably for its lifetime, so no concurrent reader exists.
        let dst_row = unsafe {
            let ptr = output.as_ptr().cast_mut();
            std::slice::from_raw_parts_mut(ptr.add(y_offset), w)
        };

        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(h);

        // Define interior region for this row
        let x_start = radius;
        let x_end = w.saturating_sub(radius + 1);

        // 1. Process Left Border
        for x in 0..x_start.min(w) {
            let x0 = 0; // saturating_sub(radius) is 0
            let x1 = (x + radius + 1).min(w);
            let actual_area = (x1 - x0) * (y1 - y0);

            let i00 = integral[y0 * stride + x0];
            let i01 = integral[y0 * stride + x1];
            let i10 = integral[y1 * stride + x0];
            let i11 = integral[y1 * stride + x1];

            let sum = (i11 + i00) - (i01 + i10);
            let mean = (sum / actual_area as u64) as i16;
            let threshold = (mean - c).max(0) as u8;
            dst_row[x] = if src_row[x] < threshold { 0 } else { 255 };
        }

        // 2. Process Interior (Vectorizable)
        if x_end > x_start && y >= radius && y + radius < h {
            let row00 = &integral[y0 * stride + (x_start - radius)..];
            let row01 = &integral[y0 * stride + (x_start + radius + 1)..];
            let row10 = &integral[y1 * stride + (x_start - radius)..];
            let row11 = &integral[y1 * stride + (x_start + radius + 1)..];

            let interior_src = &src_row[x_start..x_end];
            let interior_dst = &mut dst_row[x_start..x_end];

            for i in 0..(x_end - x_start) {
                let sum = (row11[i] + row00[i]) - (row01[i] + row10[i]);
                // Fixed-point division: (sum * inv_area) >> 31
                let mean = ((sum * u64::from(inv_area_fixed)) >> 31) as i16;
                let threshold = (mean - c).max(0) as u8;
                interior_dst[i] = if interior_src[i] < threshold { 0 } else { 255 };
            }
        } else if x_end > x_start {
            // Interior X but border Y
            for x in x_start..x_end {
                let x0 = x - radius;
                let x1 = x + radius + 1;
                let actual_area = (x1 - x0) * (y1 - y0);

                let i00 = integral[y0 * stride + x0];
                let i01 = integral[y0 * stride + x1];
                let i10 = integral[y1 * stride + x0];
                let i11 = integral[y1 * stride + x1];

                let sum = (i11 + i00) - (i01 + i10);
                let mean = (sum / actual_area as u64) as i16;
                let threshold = (mean - c).max(0) as u8;
                dst_row[x] = if src_row[x] < threshold { 0 } else { 255 };
            }
        }

        // 3. Process Right Border
        for x in x_end.max(x_start)..w {
            let x0 = x.saturating_sub(radius);
            let x1 = w; // (x + radius + 1).min(w)
            let actual_area = (x1 - x0) * (y1 - y0);

            let i00 = integral[y0 * stride + x0];
            let i01 = integral[y0 * stride + x1];
            let i10 = integral[y1 * stride + x0];
            let i11 = integral[y1 * stride + x1];

            let sum = (i11 + i00) - (i01 + i10);
            let mean = (sum / actual_area as u64) as i16;
            let threshold = (mean - c).max(0) as u8;
            dst_row[x] = if src_row[x] < threshold { 0 } else { 255 };
        }
    });
}

/// Apply per-pixel adaptive threshold with gradient-based window sizing.
///
/// Highly optimized using Parallel processing, precomputed LUTs, and branchless logic.
#[expect(
    clippy::too_many_arguments,
    reason = "per-pixel thresholding kernel; the image/gradient/integral buffers plus the radius, gradient-threshold and offset knobs mirror the OpenCV adaptiveThreshold parameter list, and grouping them adds indirection on this hot path"
)]
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[tracing::instrument(skip_all, name = "pipeline::threshold_gradient_window")]
pub fn adaptive_threshold_gradient_window(
    img: &ImageView,
    gradient_map: &[u8],
    integral: &[u64],
    output: &mut [u8],
    min_radius: usize,
    max_radius: usize,
    gradient_threshold: u8,
    c: i16,
) {
    let w = img.width;
    let h = img.height;
    let stride = w + 1;

    // Precompute radius and area reciprocal LUTs (fixed-point 1.31)
    let mut radius_lut = [0usize; 256];
    let mut inv_area_lut = [0u32; 256];
    let grad_thresh_f32 = f32::from(gradient_threshold);

    for g in 0..256 {
        let r = if g as u8 >= gradient_threshold {
            min_radius
        } else {
            let t = g as f32 / grad_thresh_f32;
            let r = max_radius as f32 * (1.0 - t) + min_radius as f32 * t;
            r as usize
        };
        radius_lut[g] = r;
        let side = (2 * r + 1) as u32;
        let area = side * side;
        inv_area_lut[g] = ((1u64 << 31) / u64::from(area)) as u32;
    }

    // use rayon::prelude::*;

    (0..h).into_par_iter().for_each(|y| {
        let y_offset = y * w;
        let src_row = img.get_row(y);

        // SAFETY: `(0..h).into_par_iter()` yields each `y` exactly once
        // across rayon workers, so the `[y * w, y * w + w)` slice is
        // disjoint from every other worker's slice. `output` length is
        // `h * w`, so the slice is in-bounds. The outer `par_iter` borrows
        // `output` mutably for its lifetime, so no concurrent reader exists.
        let dst_row = unsafe {
            let ptr = output.as_ptr().cast_mut();
            std::slice::from_raw_parts_mut(ptr.add(y_offset), w)
        };

        for x in 0..w {
            let grad = gradient_map[y_offset + x];
            let radius = radius_lut[grad as usize];

            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius + 1).min(h);
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(w);

            let i00 = integral[y0 * stride + x0];
            let i01 = integral[y0 * stride + x1];
            let i10 = integral[y1 * stride + x0];
            let i11 = integral[y1 * stride + x1];

            let sum = (i11 + i00) - (i01 + i10);

            // Fixed-point mean computation
            let mean = if x >= radius && x + radius < w && y >= radius && y + radius < h {
                ((sum * u64::from(inv_area_lut[grad as usize])) >> 31) as i16
            } else {
                let actual_area = (x1 - x0) * (y1 - y0);
                (sum / actual_area as u64) as i16
            };

            let threshold = (mean - c).max(0) as u8;
            dst_row[x] = if src_row[x] < threshold { 0 } else { 255 };
        }
    });
}

/// Compute a map of local mean values.
///
/// Optimized with parallelism and vectorization.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[expect(
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    reason = "mean values are clamped to 0..=255 (and areas are usize products) before the unsigned casts, so no sign is lost; the interior loop index i addresses five parallel integral-row slices (row00/01/10/11 and interior_dst), not a single iterated slice"
)]
pub(crate) fn compute_threshold_map(
    img: &ImageView,
    integral: &[u64],
    output: &mut [u8],
    radius: usize,
    c: i16,
) {
    let w = img.width;
    let h = img.height;
    let stride = w + 1;

    // use rayon::prelude::*; // Already imported at module level

    // Precompute interior area inverse
    let side = (2 * radius + 1) as u32;
    let area = side * side;
    let inv_area_fixed = ((1u64 << 31) / u64::from(area)) as u32;

    (0..h).into_par_iter().for_each(|y| {
        let y_offset = y * w;

        // SAFETY: `(0..h).into_par_iter()` yields each `y` exactly once
        // across rayon workers, so the `[y * w, y * w + w)` slice is
        // disjoint from every other worker's slice. `output` length is
        // `h * w`, so the slice is in-bounds. The outer `par_iter` borrows
        // `output` mutably for its lifetime, so no concurrent reader exists.
        let dst_row = unsafe {
            let ptr = output.as_ptr().cast_mut();
            std::slice::from_raw_parts_mut(ptr.add(y_offset), w)
        };

        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(h);

        let x_start = radius;
        let x_end = w.saturating_sub(radius + 1);

        // 1. Process Left Border
        for x in 0..x_start.min(w) {
            let x0 = 0;
            let x1 = (x + radius + 1).min(w);
            let actual_area = (x1 - x0) * (y1 - y0);
            let sum = (integral[y1 * stride + x1] + integral[y0 * stride + x0])
                - (integral[y0 * stride + x1] + integral[y1 * stride + x0]);
            let mean = (sum / actual_area as u64) as i16;
            dst_row[x] = (mean - c).clamp(0, 255) as u8;
        }

        // 2. Process Interior (Vectorizable)
        if x_end > x_start && y >= radius && y + radius < h {
            let row00 = &integral[y0 * stride + (x_start - radius)..];
            let row01 = &integral[y0 * stride + (x_start + radius + 1)..];
            let row10 = &integral[y1 * stride + (x_start - radius)..];
            let row11 = &integral[y1 * stride + (x_start + radius + 1)..];

            let interior_dst = &mut dst_row[x_start..x_end];

            for i in 0..(x_end - x_start) {
                let sum = (row11[i] + row00[i]) - (row01[i] + row10[i]);
                let mean = ((sum * u64::from(inv_area_fixed)) >> 31) as i16;
                interior_dst[i] = (mean - c).clamp(0, 255) as u8;
            }
        }

        // 3. Process Right Border
        for x in x_end.max(x_start)..w {
            let x0 = x.saturating_sub(radius);
            let x1 = w;
            let actual_area = (x1 - x0) * (y1 - y0);
            let sum = (integral[y1 * stride + x1] + integral[y0 * stride + x0])
                - (integral[y0 * stride + x1] + integral[y1 * stride + x0]);
            let mean = (sum / actual_area as u64) as i16;
            dst_row[x] = (mean - c).clamp(0, 255) as u8;
        }
    });
}
