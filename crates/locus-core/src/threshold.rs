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

/// Floor of the noise-calibrated local-mean offset (grey levels). Below it the 8-bit
/// quantisation step, not sensor noise, decides foreground on clean or synthetic frames.
pub const NOISE_OFFSET_MIN: i32 = 2;
/// Ceiling of the noise-calibrated offset: past ~20 grey levels low-contrast markers
/// (low-key exposure, long range) stop reaching the foreground at all, so a very noisy
/// frame is better served by speckle the quad stage rejects than by missing markers.
pub const NOISE_OFFSET_MAX: i32 = 20;

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
    /// Noise-calibrated offset `k` ([`DetectorConfig::threshold_noise_k`]); `0.0` = use
    /// [`Self::constant`].
    pub noise_k: f32,
    /// Noise σ of the image being thresholded, when the caller knows it better than an
    /// estimate on that image can (see [`Self::with_noise_sigma`]).
    pub noise_sigma: Option<f64>,
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
            noise_k: d.threshold_noise_k,
            noise_sigma: None,
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
            noise_k: config.threshold_noise_k,
            noise_sigma: None,
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
    /// never be foreground". `binary_output` may be empty when the binarized image (telemetry)
    /// is not needed: the local-mean thresholder then skips it, the tile thresholder writes it
    /// to arena scratch.
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
        // The tile kernel produces the binarized image as a by-product of the same SIMD pass
        // and drives its rows from that buffer: back an empty one with scratch so the threshold
        // map is still written.
        let binary_output = if binary_output.is_empty() {
            arena.alloc_slice_fill_copy(img.width * img.height, 0u8)
        } else {
            binary_output
        };
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

    /// Supply the noise σ of the image that will be thresholded.
    ///
    /// The Immerkær estimator assumes white noise, so it is only calibrated on the raw
    /// sensor image. When a linear pre-filter (resampling, sharpening) sits between the
    /// sensor and the thresholder, estimate on the raw image and scale by the filter's
    /// white-noise gain ([`prefilter_noise_gain`]) instead of estimating on the filtered,
    /// spatially correlated result.
    #[must_use]
    pub fn with_noise_sigma(mut self, sigma: f64) -> Self {
        self.noise_sigma = Some(sigma);
        self
    }

    /// Offset subtracted from the local mean: [`Self::constant`], or, when
    /// [`Self::noise_k`] is set, `clamp(round(k · σ), NOISE_OFFSET_MIN, NOISE_OFFSET_MAX)`
    /// with σ from [`Self::with_noise_sigma`], else estimated on `img`.
    #[must_use]
    pub fn local_mean_offset(&self, img: &ImageView) -> i32 {
        if self.noise_k <= 0.0 {
            return i32::from(self.constant);
        }
        let sigma = self.noise_sigma.unwrap_or_else(|| {
            let stride = crate::gradient::noise_stride(img.width, img.height);
            crate::gradient::estimate_noise_sigma(img, stride)
        });
        let offset = (f64::from(self.noise_k) * sigma).round() as i32;
        tracing::debug!(sigma, offset, "threshold::noise_calibrated_offset");
        offset.clamp(NOISE_OFFSET_MIN, NOISE_OFFSET_MAX)
    }

    /// Per-pixel local-mean threshold, `t(x,y) = ⌊box_sum / area⌋ − offset` over the
    /// `(2r+1)²` window clipped to the image ([`Self::local_mean_offset`] gives the offset).
    ///
    /// Vertical sums come from a sliding column accumulator (one fused enter/leave pass per
    /// row), horizontal sums from a per-row prefix scan, so the interior of every row is a
    /// branch-free, vectorisable `prefix[x + r + 1] − prefix[x − r]`. Scratch is two `u32`
    /// rows per strip (≈ 0.8 MB at 4K) and every strip re-primes its accumulator from the
    /// image, so the output does not depend on the strip size or the rayon worker count.
    ///
    /// `binary_output` may be empty, in which case only the threshold map is written (the
    /// binarized image is telemetry; segmentation reads the threshold map).
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
        let r = self.local_mean_radius.clamp(1, MAX_LOCAL_MEAN_RADIUS);
        let c = self.local_mean_offset(img);
        let write_binary = !binary_output.is_empty();

        // A strip re-scans `2r + 1` rows to prime its column sums. Aim for ≥ 4 strips per
        // worker so rayon can balance, but keep strips at least twice the window so priming
        // costs at most half of the strip's own accumulation.
        let target = h.div_ceil((rayon::current_num_threads() * 4).max(1));
        let strip_rows = target.max(2 * (2 * r + 1)).min(h);
        let n_strips = h.div_ceil(strip_rows);
        let scratch = arena.alloc_slice_fill_copy(n_strips * (2 * w + 1), 0u32);

        let process = |strip: usize,
                       t_chunk: &mut [u8],
                       mut b_chunk: Option<&mut [u8]>,
                       scratch: &mut [u32]| {
            let (cols, prefix) = scratch.split_at_mut(w);
            let mut border_divs = BorderDivs::new();
            let y_begin = strip * strip_rows;
            let rows = t_chunk.len() / w;

            // Prime the column sums for the window of the strip's first row.
            let mut y0 = y_begin.saturating_sub(r);
            let mut y1 = (y_begin + r + 1).min(h);
            for y in y0..y1 {
                slide_columns(cols, Some(img.get_row(y)), None);
            }

            for dy in 0..rows {
                let y = y_begin + dy;
                if dy > 0 {
                    // The window moves down one row: at most one row enters and one leaves.
                    let ny1 = (y + r + 1).min(h);
                    let ny0 = y.saturating_sub(r);
                    let enter = (ny1 > y1).then(|| img.get_row(y1));
                    let leave = (ny0 > y0).then(|| img.get_row(y0));
                    slide_columns(cols, enter, leave);
                    y1 = ny1;
                    y0 = ny0;
                }
                let t_row = &mut t_chunk[dy * w..(dy + 1) * w];
                local_mean_row(
                    cols,
                    prefix,
                    t_row,
                    &mut border_divs,
                    r,
                    (y1 - y0) as u32,
                    c,
                );
                if let Some(b) = b_chunk.as_deref_mut() {
                    binarize_row(img.get_row(y), t_row, &mut b[dy * w..(dy + 1) * w]);
                }
            }
        };

        let t_chunks = threshold_output[..w * h].par_chunks_mut(strip_rows * w);
        let scratch_chunks = scratch.par_chunks_mut(2 * w + 1);
        if write_binary {
            t_chunks
                .zip(binary_output[..w * h].par_chunks_mut(strip_rows * w))
                .zip(scratch_chunks)
                .enumerate()
                .for_each(|(strip, ((t, b), s))| process(strip, t, Some(b), s));
        } else {
            t_chunks
                .zip(scratch_chunks)
                .enumerate()
                .for_each(|(strip, (t, s))| process(strip, t, None, s));
        }
    }
}

/// White-noise standard-deviation gain of the detector's pre-threshold filters: area
/// decimation by `decimation` (σ/d), bilinear upscaling by `upscale` (per output pixel the
/// bilinear weights `w` scale variance by `Σw²`; averaged over the output phases, per axis),
/// then the Laplacian sharpen `5·c − Σ₄ n` (gain `√29`).
///
/// Exact for decimation and for sharpening of white noise; for upscaling followed by
/// sharpening the input to the sharpen is correlated, so the product is an approximation.
#[must_use]
pub fn prefilter_noise_gain(decimation: usize, upscale: usize, sharpening: bool) -> f64 {
    let mut gain = 1.0 / decimation.max(1) as f64;
    if decimation <= 1 && upscale > 1 {
        let u = upscale as f64;
        let per_axis_variance = (0..upscale)
            .map(|j| {
                let phase = ((j as f64 + 0.5) / u - 0.5).rem_euclid(1.0);
                (1.0 - phase).powi(2) + phase.powi(2)
            })
            .sum::<f64>()
            / u;
        // Two axes: variance factor per_axis², std factor per_axis.
        gain *= per_axis_variance;
    }
    if sharpening {
        gain *= 29f64.sqrt();
    }
    gain
}

/// Largest supported local-mean radius. It bounds the window area to `255²`, which keeps
/// every box sum below `2²⁴`: exact in the wrapping `u32` prefix arithmetic and in the
/// reciprocal-multiply local mean.
pub const MAX_LOCAL_MEAN_RADIUS: usize = 127;

/// Slide the column accumulator by one row: add `enter`, subtract `leave` (either may be
/// absent at the image border). One fused pass over `cols`.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn slide_columns(cols: &mut [u32], enter: Option<&[u8]>, leave: Option<&[u8]>) {
    match (enter, leave) {
        (Some(e), Some(l)) => {
            for ((col, &a), &b) in cols.iter_mut().zip(e).zip(l) {
                // `col + a ≥ b` always (b was added earlier), so no wrap occurs.
                *col = *col + u32::from(a) - u32::from(b);
            }
        },
        (Some(e), None) => {
            for (col, &a) in cols.iter_mut().zip(e) {
                *col += u32::from(a);
            }
        },
        (None, Some(l)) => {
            for (col, &b) in cols.iter_mut().zip(l) {
                *col -= u32::from(b);
            }
        },
        (None, None) => {},
    }
}

/// Exact `⌊n / area⌋` for every `n < 2²⁴` as `(n · m) >> shift` (Granlund–Montgomery
/// round-up method: with `l = ⌈log₂ area⌉`, `m = ⌈2^(24 + l) / area⌉ < 2²⁵`). Box sums are
/// `≤ 255 · area ≤ 255³ < 2²⁴`, so the mean is exact, and both factors fit in 32 bits, which
/// lets the interior loop use the native 32×32→64 SIMD multiply.
#[derive(Clone, Copy, Default)]
struct ExactDiv {
    m: u32,
    shift: u32,
}

impl ExactDiv {
    #[inline]
    fn new(area: u32) -> Self {
        let l = u32::BITS - (area - 1).leading_zeros(); // ⌈log₂ area⌉ (0 for area = 1)
        let shift = 24 + l;
        // `m ≤ 2²⁵`, so it fits in `u32` and the product is a 32×32→64 multiply.
        let m = (1u64 << shift).div_ceil(u64::from(area)) as u32;
        Self { m, shift }
    }

    #[inline]
    fn apply(self, n: u32) -> i32 {
        ((u64::from(n) * u64::from(self.m)) >> self.shift) as i32
    }
}

/// Dividers of the `≤ 2r` clipped border windows of a row. They depend only on the window's
/// row count (which changes only within `r` rows of the top and bottom edge) and on `x`, so a
/// strip computes them once per distinct row count instead of once per border pixel.
struct BorderDivs {
    rows: u32,
    divs: [ExactDiv; 2 * MAX_LOCAL_MEAN_RADIUS],
}

impl BorderDivs {
    fn new() -> Self {
        Self {
            rows: 0,
            divs: [ExactDiv::default(); 2 * MAX_LOCAL_MEAN_RADIUS],
        }
    }

    /// Refresh for `rows_in_window`; slot `i` covers border column `i` (left edge) or
    /// `w − (lo + hi_count) + i` (right edge), as laid out by [`local_mean_row`].
    fn refresh(&mut self, rows_in_window: u32, w: usize, r: usize, lo: usize, hi: usize) {
        if self.rows == rows_in_window {
            return;
        }
        self.rows = rows_in_window;
        let cols = |x: usize| ((x + r + 1).min(w) - x.saturating_sub(r)) as u32;
        for x in 0..lo {
            self.divs[x] = ExactDiv::new(rows_in_window * cols(x));
        }
        for (slot, x) in (hi..w).enumerate() {
            self.divs[lo + slot] = ExactDiv::new(rows_in_window * cols(x));
        }
    }
}

/// One row of `threshold = ⌊box_sum / area⌋ − c`, from the column sums `cols`.
///
/// `prefix` (length `w + 1`) receives the running sum of `cols`; window sums are prefix
/// differences in wrapping `u32` arithmetic, exact because every window sum is `< 2²⁴`.
/// The interior `[r, w − r)` has a constant area and is branch-free; only the `2r`
/// border columns recompute their (clipped) area.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn local_mean_row(
    cols: &[u32],
    prefix: &mut [u32],
    thresholds: &mut [u8],
    border_divs: &mut BorderDivs,
    r: usize,
    rows_in_window: u32,
    c: i32,
) {
    let w = cols.len();
    prefix[0] = 0;
    let mut acc = 0u32;
    for (p, &v) in prefix[1..=w].iter_mut().zip(cols) {
        acc = acc.wrapping_add(v);
        *p = acc;
    }
    // Interior: the full window fits, i.e. `x ≥ r` and `x + r + 1 ≤ w`.
    let lo = r.min(w);
    let hi = w.saturating_sub(r).max(lo);
    border_divs.refresh(rows_in_window, w, r, lo, hi);
    let divs = &border_divs.divs;
    let border = |x: usize, div: ExactDiv| {
        let sum = prefix[(x + r + 1).min(w)].wrapping_sub(prefix[x.saturating_sub(r)]);
        (div.apply(sum) - c).clamp(0, 255) as u8
    };

    for (x, t) in thresholds[..lo].iter_mut().enumerate() {
        *t = border(x, divs[x]);
    }
    if hi > lo {
        let div = ExactDiv::new(rows_in_window * (2 * r + 1) as u32);
        let (ahead, behind) = (&prefix[(lo + r + 1)..=(hi + r)], &prefix[lo - r..hi - r]);
        for ((t, &p1), &p0) in thresholds[lo..hi].iter_mut().zip(ahead).zip(behind) {
            *t = (div.apply(p1.wrapping_sub(p0)) - c).clamp(0, 255) as u8;
        }
    }
    for (slot, (x, t)) in thresholds.iter_mut().enumerate().skip(hi).enumerate() {
        *t = border(x, divs[lo + slot]);
    }
}

/// `binary = 0` where `src < threshold` (foreground), else 255. Telemetry only.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn binarize_row(src: &[u8], thresholds: &[u8], binary: &mut [u8]) {
    for ((b, &s), &t) in binary.iter_mut().zip(src).zip(thresholds) {
        *b = if s < t { 0 } else { 255 };
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
            noise_k: 0.0,
            noise_sigma: None,
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
        for &(w, h) in &[(64usize, 48usize), (37, 29), (8, 8), (129, 5), (300, 9)] {
            let data = lcg_image(w, h, 7);
            for &r in &[1usize, 3, 7, 12, 40, 127] {
                let (binary, map) = run_mode(ThresholdMode::LocalMean, r, 0, &data, w, h);
                for y in 0..h {
                    for x in 0..w {
                        let expect = naive_box_mean(&data, w, h, x, y, r).min(255);
                        let got = u32::from(map[y * w + x]);
                        assert_eq!(
                            got, expect,
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
    /// Deterministic Gaussian noise around `mean` (Box–Muller over an LCG).
    fn gaussian_image(w: usize, h: usize, mean: f64, sigma: f64, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..w * h)
            .map(|_| {
                let z = (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
                (mean + sigma * z).round().clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    fn std_dev(data: &[u8], w: usize, h: usize, margin: usize) -> f64 {
        let vals: Vec<f64> = (margin..h - margin)
            .flat_map(|y| (margin..w - margin).map(move |x| (x, y)))
            .map(|(x, y)| f64::from(data[y * w + x]))
            .collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt()
    }

    /// `ExactDiv` against integer division for every area a radius ≤ 127 window can have
    /// and the dividends where a reciprocal is most likely to slip.
    #[test]
    fn exact_div_matches_integer_division() {
        let max_area = (2 * MAX_LOCAL_MEAN_RADIUS as u32 + 1).pow(2);
        for area in 1..=max_area {
            let div = ExactDiv::new(area);
            for k in [0u32, 1, 2, 127, 254, 255] {
                for n in [k * area, (k * area).saturating_sub(1), k * area + area / 2] {
                    if n <= 255 * area {
                        assert_eq!(
                            div.apply(n),
                            i32::try_from(n / area).unwrap(),
                            "n={n} area={area}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn local_mean_threshold_map_does_not_depend_on_binary_output() {
        let (w, h) = (97usize, 61usize);
        let data = lcg_image(w, h, 11);
        let (_, with_binary) = run_mode(ThresholdMode::LocalMean, 7, 3, &data, w, h);
        let img = ImageView::new(&data, w, h, w).unwrap();
        let config = DetectorConfig {
            threshold_mode: ThresholdMode::LocalMean,
            threshold_local_mean_radius: 7,
            adaptive_threshold_constant: 3,
            ..DetectorConfig::default()
        };
        let engine = ThresholdEngine::from_config(&config);
        let arena = Bump::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let mut map = vec![0u8; w * h];
        engine.apply_threshold_with_map(&arena, &img, &stats, &mut [], &mut map);
        assert_eq!(map, with_binary);
    }

    #[test]
    fn tile_threshold_map_does_not_depend_on_binary_output() {
        let (w, h) = (96usize, 64usize);
        let data = lcg_image(w, h, 5);
        let (_, with_binary) = run_mode(ThresholdMode::TileMidExtreme, 7, 0, &data, w, h);
        let img = ImageView::new(&data, w, h, w).unwrap();
        let engine = ThresholdEngine::from_config(&DetectorConfig::default());
        let arena = Bump::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let mut map = vec![0u8; w * h];
        engine.apply_threshold_with_map(&arena, &img, &stats, &mut [], &mut map);
        assert_eq!(map, with_binary);
    }

    /// `prefilter_noise_gain` against white noise pushed through the real kernels.
    #[test]
    fn prefilter_noise_gain_matches_the_filters() {
        let (w, h, sigma) = (256usize, 256usize, 8.0);
        let data = gaussian_image(w, h, 128.0, sigma, 3);
        let img = ImageView::new(&data, w, h, w).unwrap();
        let measured = std_dev(&data, w, h, 2);
        let check = |label: &str, out: &[u8], ow: usize, oh: usize, gain: f64| {
            let got = std_dev(out, ow, oh, 4) / measured;
            assert!(
                (got - gain).abs() < 0.05 * gain,
                "{label}: measured gain {got}, predicted {gain}"
            );
        };

        let mut dec = vec![0u8; (w / 2) * (h / 2)];
        let dec_img = img.decimate_to(2, &mut dec).unwrap();
        check(
            "decimate 2",
            dec_img.data,
            w / 2,
            h / 2,
            prefilter_noise_gain(2, 1, false),
        );

        let mut up = vec![0u8; w * 2 * h * 2];
        let up_img = img.upscale_to(2, &mut up).unwrap();
        check(
            "upscale 2",
            up_img.data,
            w * 2,
            h * 2,
            prefilter_noise_gain(1, 2, false),
        );

        let mut sharp = vec![0u8; w * h];
        crate::filter::laplacian_sharpen(&img, &mut sharp);
        check("sharpen", &sharp, w, h, prefilter_noise_gain(1, 1, true));
    }

    /// The noise-calibrated offset delivers its promise: on a flat frame of white noise a
    /// pixel turns foreground with probability ≈ Φ(−k).
    #[test]
    fn noise_calibrated_offset_controls_false_foreground_rate() {
        let (w, h) = (512usize, 512usize);
        let data = gaussian_image(w, h, 128.0, 4.0, 9);
        let img = ImageView::new(&data, w, h, w).unwrap();
        let config = DetectorConfig {
            threshold_mode: ThresholdMode::LocalMean,
            threshold_local_mean_radius: 7,
            threshold_noise_k: 3.0,
            ..DetectorConfig::default()
        };
        let engine = ThresholdEngine::from_config(&config);
        let arena = Bump::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let mut binary = vec![0u8; w * h];
        let mut map = vec![0u8; w * h];
        engine.apply_threshold_with_map(&arena, &img, &stats, &mut binary, &mut map);
        let rate = binary.iter().filter(|&&b| b == 0).count() as f64 / (w * h) as f64;
        // Φ(−3) = 1.35e-3; allow the offset's integer rounding and the σ̂ error (Φ(−3.5)…Φ(−2.5)).
        assert!(
            (2.3e-4..6.2e-3).contains(&rate),
            "false-foreground rate {rate}"
        );
    }

    #[test]
    fn local_mean_suppresses_flat_regions() {
        let (w, h) = (64usize, 64usize);
        let data = vec![97u8; w * h];

        let (binary, map) = run_mode(ThresholdMode::LocalMean, 8, 5, &data, w, h);
        // Exactly 97 − 5 everywhere, including the clipped border windows.
        assert!(map.iter().all(|&t| t == 92), "{:?}", &map[..8]);
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

/// Compute integral image (cumulative sum) for fast box filter computation.
///
/// Backs OpenCV-style `ADAPTIVE_THRESH_MEAN_C`: an O(1) local-mean lookup per
/// pixel instead of an O(W*H) box filter per threshold.
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
