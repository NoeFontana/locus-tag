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
//!    `tile_size` tile from the min/max over its 3×3 tile neighbourhood, cut at
//!    [`CUT_NUM`]/[`CUT_DEN`] of the range rather than at the midpoint (see there),
//!    then expanded to pixels. Cheap — stats are computed once per tile, not per
//!    pixel — but the threshold follows the local *extremes*, which speckles
//!    flat regions and lets a dark background fuse with a marker.
//! 2. **Local mean** (`LocalMean`): a true per-pixel local mean over a
//!    `(2r+1)²` window, minus a noise-calibrated offset, from a sliding column-sum
//!    accumulator. Tracks the local background level instead of the extremes.

#![allow(clippy::cast_sign_loss)]
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

/// Numerator of the cut [`ThresholdMode::TileMidExtreme`] places between the extremes of a
/// tile's 3×3 neighbourhood: `threshold = min + CUT_NUM/CUT_DEN · (max - min)`.
///
/// The midpoint is the unbiased cut for an edge the optics resolved, and it is where this
/// stage used to cut. It is the wrong cut for *topology*, because segmentation reads this map
/// to decide which pixels are one marker. Two markers whose dark regions touch — diagonally
/// adjacent ones on a calibration board, or any two whose blurred skirts overlap — are a
/// single connected component at the midpoint, and nothing downstream can take them apart: a
/// component is traced once and reduced to one quadrilateral. Cutting below the midpoint
/// shrinks every dark region by a fraction of the blur width, which breaks those contacts.
///
/// How far below is a stated assumption about the thinnest dark stroke that must survive, and
/// it is measured, not derived: the `standard` profile's recall over the SOTA benchmarks is
/// flat from the midpoint down to about 0.45 on the suites that have no touching markers, and
/// rises steeply on the ones that do (EuRoC cam_april 86.0 → 92.0 %, Liu4K 37.2 → 44.8 %,
/// render-tag low-key 66 → 96 %, 2026-10-05). Below 0.45 small and low-contrast markers start
/// to lose their one-module borders — render-tag 640 falls off at 0.40, tag16h5 at 0.42 — so
/// the gains past it are not free and are not taken here. See
/// `docs/explanation/pipeline.md`.
///
/// The cut is applied with rounding, not truncation. A truncated `CUT_NUM·range/CUT_DEN`
/// biases the cut downward by up to one grey level, and that bias is proportionally largest
/// where the range is smallest — the low-contrast tiles where erosion costs a marker its
/// border. At a range of 2 it reaches the whole fraction: the threshold would equal the
/// neighbourhood minimum, and a `pixel < threshold` rule makes that tile background entirely.
///
/// Public because it is observable contract — it decides which markers are one component —
/// not a private tuning knob.
pub const CUT_NUM: u16 = 9;
/// Denominator of [`CUT_NUM`]. Twenty expresses the measured frontier at the granularity it
/// was measured on (0.05 steps) and halves exactly, so the rounded cut below needs no wider
/// arithmetic than the `u16` the tile loop already multiplies in. The cut is *not* exact for
/// most ranges — `9·range` is a multiple of 20 only one time in twenty — but it is integer
/// throughout, so the map is bit-identical across targets and SIMD widths.
pub const CUT_DEN: u16 = 20;

/// A mis-tuned cut must be a compile error, not a frame with no foreground in it: the tile
/// loop multiplies in `u16` and divides by [`CUT_DEN`], and a numerator above the denominator
/// would put the threshold above the neighbourhood maximum (every pixel foreground).
const _: () = assert!(
    CUT_DEN != 0
        && CUT_NUM <= CUT_DEN
        && (CUT_NUM as u32) * 255 + (CUT_DEN as u32) / 2 <= u16::MAX as u32,
    "the cut must be a fraction in [0, 1] whose numerator times the widest 8-bit range fits a u16"
);

/// Minimum intensity range over a 3×3-tile neighbourhood for the centre tile to count as
/// valid in [`ThresholdMode::TileMidExtreme`]. Telemetry-scoped: flat (invalid) tiles are
/// forced to background in the binarized debug image only; the threshold map segmentation
/// reads is written for every tile, so this value never changes detections.
const TILE_MIN_RANGE: u8 = 10;

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
    /// How the per-pixel foreground threshold is built.
    pub mode: ThresholdMode,
    /// Window radius (pixels) for [`ThresholdMode::LocalMean`].
    pub local_mean_radius: usize,
    /// Noise-calibrated offset `k` of [`ThresholdMode::LocalMean`]
    /// ([`DetectorConfig::threshold_noise_k`]).
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

/// Fill `thresholds` and `valid` from the extremes over each tile's 3x3 tile neighbourhood,
/// cutting at [`CUT_NUM`]/[`CUT_DEN`] of the range. `valid` marks the tiles whose range clears
/// [`TILE_MIN_RANGE`] and is read only by the binarized telemetry image.
fn tile_cut_and_validity(
    arena: &Bump,
    stats: &[TileStats],
    tiles_wide: usize,
    tiles_high: usize,
    thresholds: &mut [u8],
    valid: &mut [u8],
) {
    // The 3x3 tile reduction, as three vectorisable passes over packed `(min, !max)` pairs.
    //
    // `!max` is `255 - max`, which turns the maximum into a *minimum*, so both reductions
    // become one byte-wise minimum and a single `pminub` lane carries both fields. And the
    // minimum over a clamped 3x3 window is exactly the minimum over the clamped rows
    // followed by the minimum over the clamped columns, so the window separates into a
    // horizontal and a vertical 3-tap over contiguous bytes. Reducing `TileStats` where it
    // lies cannot vectorise: it interleaves the two fields, so every lane would want a
    // strided gather, and the maximum would want the opposite instruction from the minimum.
    let stride = 2 * tiles_wide;
    let mut packed = BumpVec::with_capacity_in(stride * tiles_high, arena);
    packed.resize(stride * tiles_high, 0u8);
    packed
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(ty, row)| {
            let src = &stats[ty * tiles_wide..(ty + 1) * tiles_wide];
            for (pair, s) in row.chunks_exact_mut(2).zip(src) {
                pair[0] = s.min;
                pair[1] = !s.max;
            }
        });

    let mut horiz = BumpVec::with_capacity_in(stride * tiles_high, arena);
    horiz.resize(stride * tiles_high, 0u8);
    {
        let packed_rows = packed.as_slice();
        horiz
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(ty, h_row)| {
                let p = &packed_rows[ty * stride..(ty + 1) * stride];
                if stride == 2 {
                    h_row.copy_from_slice(p);
                    return;
                }
                // Clamped ends, then the interior as one three-input minimum.
                h_row[0] = p[0].min(p[2]);
                h_row[1] = p[1].min(p[3]);
                h_row[stride - 2] = p[stride - 4].min(p[stride - 2]);
                h_row[stride - 1] = p[stride - 3].min(p[stride - 1]);
                if stride > 4 {
                    min3_into(
                        &mut h_row[2..stride - 2],
                        &p[0..stride - 4],
                        &p[2..stride - 2],
                        &p[4..stride],
                    );
                }
            });
    }

    // Vertical 3-tap, back into `packed`: the pass above has fully consumed it, and each
    // worker writes only its own row while reading `horiz` immutably.
    {
        let horiz_rows = horiz.as_slice();
        packed
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(ty, v_row)| {
                let row = |i: usize| &horiz_rows[i * stride..(i + 1) * stride];
                min3_into(
                    v_row,
                    row(ty.saturating_sub(1)),
                    row(ty),
                    row((ty + 1).min(tiles_high - 1)),
                );
            });
    }

    let extremes = packed.as_slice();
    thresholds
        .par_chunks_mut(tiles_wide)
        .zip(valid.par_chunks_mut(tiles_wide))
        .enumerate()
        .for_each(|(ty, (t_row, v_row))| {
            let row = &extremes[ty * stride..(ty + 1) * stride];
            for ((t, v), pair) in t_row
                .iter_mut()
                .zip(v_row.iter_mut())
                .zip(row.chunks_exact(2))
            {
                let nmin = pair[0];
                let nmax = !pair[1];
                // `stats` is caller-supplied and its unfilled sentinel is
                // `min: 255, max: 0`, so saturate rather than underflow a `u8`.
                let range = u16::from(nmax.saturating_sub(nmin));
                *t = (u16::from(nmin) + (CUT_NUM * range + CUT_DEN / 2) / CUT_DEN) as u8;
                *v = if range < u16::from(TILE_MIN_RANGE) {
                    0
                } else {
                    255
                };
            }
        });
}

impl ThresholdEngine {
    /// Create a ThresholdEngine with the settings of [`DetectorConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::from_config(&DetectorConfig::default())
    }

    /// Create a ThresholdEngine from detector configuration.
    #[must_use]
    pub fn from_config(config: &DetectorConfig) -> Self {
        Self {
            tile_size: config.threshold_tile_size,
            mode: config.threshold_mode,
            local_mean_radius: config.threshold_local_mean_radius,
            noise_k: config.threshold_noise_k,
            noise_sigma: None,
        }
    }

    /// Compute min/max statistics for each tile in the image (every pixel of every tile
    /// row; the per-row scan is vectorised).
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
        if tiles_wide == 0 || tiles_high == 0 {
            // Fewer than `tile_size` pixels across or down: no tile grid, and `par_chunks_mut`
            // rejects a chunk size of zero.
            return stats;
        }

        stats
            .par_chunks_mut(tiles_wide)
            .enumerate()
            .for_each(|(ty, stats_row)| {
                for dy in 0..ts {
                    let py = ty * ts + dy;
                    let src_row = img.get_row(py);

                    // Process all tiles in this row with SIMD-friendly min/max
                    compute_row_tile_stats_simd(src_row, stats_row, ts);
                }
            });
        stats
    }

    /// Binarize `img` into `output` (`0` = foreground) exactly as the detector does, through
    /// [`Self::apply_threshold_with_map`], discarding the threshold map.
    #[cfg(any(test, feature = "bench-internals"))]
    pub fn apply_threshold(
        &self,
        arena: &Bump,
        img: &ImageView,
        stats: &[TileStats],
        output: &mut [u8],
    ) {
        let threshold_map = arena.alloc_slice_fill_copy(img.width * img.height, 0u8);
        self.apply_threshold_with_map(arena, img, stats, output, threshold_map);
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
        if tiles_wide == 0 || tiles_high == 0 {
            // No whole tile fits, so there are no statistics to threshold against. Leave the
            // map as the caller supplied it — zero is "never foreground" — rather than ask
            // rayon for chunks of size zero, which panics.
            return;
        }

        let mut tile_thresholds = BumpVec::with_capacity_in(tiles_wide * tiles_high, arena);
        tile_thresholds.resize(tiles_wide * tiles_high, 0u8);
        let mut tile_valid = BumpVec::with_capacity_in(tiles_wide * tiles_high, arena);
        tile_valid.resize(tiles_wide * tiles_high, 0u8);

        tile_cut_and_validity(
            arena,
            stats,
            tiles_wide,
            tiles_high,
            &mut tile_thresholds,
            &mut tile_valid,
        );

        let thresholds_slice = tile_thresholds.as_slice();
        let valid_slice = tile_valid.as_slice();
        let w = img.width;

        // Expand one tile band of the threshold map. The band's rows are identical, so build
        // the first one in place and replicate it: writing straight into the map keeps the
        // production path free of the per-worker scratch row the old loop allocated, and lets
        // the map — not the binarized image — drive the iteration.
        let expand_band = |ty: usize, thresh_band: &mut [u8]| {
            let (first, rest) = thresh_band.split_at_mut(w);
            let t_row = &thresholds_slice[ty * tiles_wide..(ty + 1) * tiles_wide];
            for (tx, &thresh) in t_row.iter().enumerate() {
                first[tx * ts..(tx + 1) * ts].fill(thresh);
            }
            // Columns past the last whole tile have no statistics: never foreground.
            first[tiles_wide * ts..].fill(0);
            for dy in 1..ts {
                rest[(dy - 1) * w..dy * w].copy_from_slice(first);
            }
        };

        if binary_output.is_empty() {
            // Production. Segmentation reads the threshold map alone, so the binarized image,
            // the full-frame buffer that holds it and the compare-and-store pass that fills it
            // are all skipped — as is the tile-validity expansion, which only that pass reads.
            threshold_output
                .par_chunks_mut(ts * w)
                .enumerate()
                .for_each(|(ty, thresh_band)| {
                    if ty < tiles_high {
                        expand_band(ty, thresh_band);
                    }
                });
            return;
        }

        // Telemetry. `par_chunks_mut` on both buffers gives each worker its own band of each,
        // so the two writes are disjoint by construction rather than by a safety argument.
        threshold_output
            .par_chunks_mut(ts * w)
            .zip(binary_output.par_chunks_mut(ts * w))
            .enumerate()
            .for_each_init(
                || vec![0u8; w],
                |row_valid, (ty, (thresh_band, bin_band))| {
                    if ty >= tiles_high {
                        return;
                    }
                    expand_band(ty, thresh_band);
                    let v_row = &valid_slice[ty * tiles_wide..(ty + 1) * tiles_wide];
                    for (tx, &valid) in v_row.iter().enumerate() {
                        row_valid[tx * ts..(tx + 1) * ts].fill(valid);
                    }
                    row_valid[tiles_wide * ts..].fill(0);

                    let row_thresholds = &thresh_band[..w];
                    for dy in 0..ts {
                        let src_row = img.get_row(ty * ts + dy);
                        let bin_row = &mut bin_band[dy * w..(dy + 1) * w];
                        threshold_row_simd(src_row, bin_row, row_thresholds, row_valid);
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

    /// Offset subtracted from the local mean:
    /// `clamp(round(k · σ), NOISE_OFFSET_MIN, NOISE_OFFSET_MAX)` with `k` = [`Self::noise_k`]
    /// and σ from [`Self::with_noise_sigma`], else estimated on `img`.
    #[must_use]
    pub fn local_mean_offset(&self, img: &ImageView) -> i32 {
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

/// `dst[i] = min(a[i], b[i], c[i])` over equal-length byte runs — one `pminub` chain per
/// vector lane. The three-input form is what the separable tile reduction needs, and a clamped
/// edge of the window is expressed by passing the same row twice, since `min(a, a, b)` is
/// `min(a, b)`.
#[multiversion(targets = "simd")]
fn min3_into(dst: &mut [u8], a: &[u8], b: &[u8], c: &[u8]) {
    let n = dst.len();
    assert!(a.len() == n && b.len() == n && c.len() == n);
    for i in 0..n {
        dst[i] = a[i].min(b[i]).min(c[i]);
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

        // At (8,8) it should be foreground (0): the pixel is 50 and the cut over a
        // neighbourhood of (50, 200) is 50 + round(9 * 150 / 20) = 118.
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

    /// Threshold `data` with `mode`; the local-mean offset is exactly `offset` grey levels
    /// (`k = offset` at a supplied σ of 1, inside the `[2, 20]` clamp).
    fn run_mode(
        mode: ThresholdMode,
        radius: usize,
        offset: i32,
        data: &[u8],
        w: usize,
        h: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        assert!((NOISE_OFFSET_MIN..=NOISE_OFFSET_MAX).contains(&offset));
        let img = ImageView::new(data, w, h, w).unwrap();
        let engine = ThresholdEngine {
            tile_size: 8,
            mode,
            local_mean_radius: radius,
            noise_k: offset as f32,
            noise_sigma: Some(1.0),
        };
        let arena = Bump::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let mut binary = vec![0u8; w * h];
        let mut map = vec![0u8; w * h];
        engine.apply_threshold_with_map(&arena, &img, &stats, &mut binary, &mut map);
        (binary, map)
    }

    /// Blur `data` in place with `passes` applications of the 5-tap cross
    /// `{centre 1/2, each 4-neighbour 1/8}` — the mean of a horizontal and a vertical
    /// `[1, 2, 1]/4`, which is *not* their separable product. It stands in for a lens
    /// point-spread function roughly one pixel wide per pass.
    fn blur(data: &mut [u8], w: usize, h: usize, passes: usize) {
        for _ in 0..passes {
            let src = data.to_vec();
            for y in 0..h {
                let (ym, yp) = (y.saturating_sub(1), (y + 1).min(h - 1));
                for x in 0..w {
                    let (xm, xp) = (x.saturating_sub(1), (x + 1).min(w - 1));
                    let at = |xx: usize, yy: usize| u32::from(src[yy * w + xx]);
                    let hsum = at(xm, y) + 2 * at(x, y) + at(xp, y);
                    let vsum = at(x, ym) + 2 * at(x, y) + at(x, yp);
                    data[y * w + x] = ((hsum + vsum + 4) / 8) as u8;
                }
            }
        }
    }

    /// Per-pixel threshold map for an arbitrary `(nmin, nmax) -> threshold` rule, by the same
    /// tile geometry as [`ThresholdEngine::apply_threshold_with_map`]. It exists so a fixture
    /// can be scored against cuts the detector does *not* ship — above all the midpoint this
    /// stage used to use — and `counterfactual_map_reproduces_the_shipped_cut` pins it to the
    /// real thing so it cannot drift away from what production does.
    fn counterfactual_map(data: &[u8], w: usize, h: usize, cut: impl Fn(u8, u8) -> u8) -> Vec<u8> {
        let ts = 8usize;
        let img = ImageView::new(data, w, h, w).unwrap();
        let engine = ThresholdEngine {
            tile_size: ts,
            mode: ThresholdMode::TileMidExtreme,
            local_mean_radius: 2,
            noise_k: 2.0,
            noise_sigma: Some(1.0),
        };
        let arena = Bump::new();
        let stats = engine.compute_tile_stats(&arena, &img);
        let (tiles_wide, tiles_high) = (w / ts, h / ts);
        let mut map = vec![0u8; w * h];
        for ty in 0..tiles_high {
            for tx in 0..tiles_wide {
                let (mut nmin, mut nmax) = (255u8, 0u8);
                for ny in ty.saturating_sub(1)..=(ty + 1).min(tiles_high - 1) {
                    for nx in tx.saturating_sub(1)..=(tx + 1).min(tiles_wide - 1) {
                        let st = stats[ny * tiles_wide + nx];
                        nmin = nmin.min(st.min);
                        nmax = nmax.max(st.max);
                    }
                }
                let t = cut(nmin, nmax);
                for dy in 0..ts {
                    let row = (ty * ts + dy) * w + tx * ts;
                    map[row..row + ts].fill(t);
                }
            }
        }
        map
    }

    /// The cut this stage ships, as a `(nmin, nmax)` rule.
    fn cut_at(num: u16) -> impl Fn(u8, u8) -> u8 {
        move |nmin, nmax| {
            let range = u16::from(nmax.saturating_sub(nmin));
            (u16::from(nmin) + (num * range + CUT_DEN / 2) / CUT_DEN) as u8
        }
    }

    /// The rule this stage used before [`CUT_NUM`]: the midpoint of the neighbourhood extremes,
    /// written exactly as it was (`(nmin + nmax) >> 1`), so the comparisons below are against
    /// what shipped and not against a rounded restatement of it.
    fn midpoint_cut(nmin: u8, nmax: u8) -> u8 {
        ((u16::from(nmin) + u16::from(nmax)) >> 1) as u8
    }

    /// [`counterfactual_map`] must agree with the detector wherever the detector has tiles,
    /// or every comparison built on it is measuring the test's arithmetic instead of the
    /// shipped cut.
    ///
    /// It also pins the separable reduction in `tile_cut_and_validity` to the naive 3x3 scan
    /// written here: the packed `(min, !max)` passes must be bit-identical to it, including on
    /// grids one and two tiles wide, where the horizontal 3-tap is all clamped ends and no
    /// interior.
    #[test]
    fn counterfactual_map_reproduces_the_shipped_cut() {
        // Tile size is 8, so these cover grids 1, 2, 3, 8 and 40 tiles wide and 1..6 tall,
        // plus dimensions that are not whole multiples of the tile size.
        let shapes = [
            (64usize, 48usize),
            (8, 8),
            (16, 8),
            (24, 16),
            (8, 48),
            (320, 24),
            (71, 43),
            (17, 9),
        ];
        for (w, h) in shapes {
            for seed in [7u32, 11, 29] {
                let data = lcg_image(w, h, seed);
                let (_, production) = run_mode(ThresholdMode::TileMidExtreme, 2, 2, &data, w, h);
                let mirror = counterfactual_map(&data, w, h, cut_at(CUT_NUM));
                assert_eq!(production, mirror, "{w}x{h} seed {seed}");
            }
        }
    }

    /// A frame with no whole tile in it has no statistics to threshold against. It must come
    /// back as "never foreground" rather than ask rayon for zero-sized chunks, which panics.
    #[test]
    fn a_frame_smaller_than_one_tile_is_not_a_panic() {
        for (w, h) in [
            (1usize, 1usize),
            (4, 4),
            (7, 7),
            (4, 8),
            (8, 4),
            (16, 3),
            (3, 16),
        ] {
            let data = lcg_image(w, h, 5);
            let (binary, map) = run_mode(ThresholdMode::TileMidExtreme, 2, 2, &data, w, h);
            assert!(
                map.iter().all(|&t| t == 0),
                "{w}x{h}: threshold map must stay at 'never foreground'"
            );
            assert_eq!(binary.len(), w * h);
        }
    }

    /// Two dark squares separated by a bright gap one pixel wide, over a sweep of contrasts and
    /// point-spread widths. A gap narrower than the PSF never reaches the bright level, so past
    /// some PSF width the midpoint rule puts the gap on the dark side of the cut and the two
    /// squares arrive at segmentation as one component — and one component is reduced to one
    /// quadrilateral, so the pair is lost.
    ///
    /// The assertion is that the shipped cut separates **strictly more** of the sweep than the
    /// midpoint does, and never fewer. It is deliberately not "all of it": the widest PSF here
    /// closes a one-pixel gap for any cut, which is the limit of choosing a single scalar and
    /// the reason the frontier below [`CUT_NUM`] is not free either. Reverting the cut to the
    /// midpoint makes the two counts equal and fails this test.
    #[test]
    fn the_cut_holds_open_gaps_the_midpoint_closed() {
        let (w, h) = (64usize, 64usize);
        let half = 7usize;
        let gap_x = 20 + half;
        let (probe_y, interior_x) = (28usize, 23usize);

        let mut separated_by_cut = 0usize;
        let mut separated_by_midpoint = 0usize;
        let mut fused_by_midpoint = 0usize;
        let mut cases = 0usize;

        for dark in [10u8, 20, 30, 40] {
            for bright in [200u8, 220, 240] {
                for passes in 1..=4 {
                    let mut data = vec![bright; w * h];
                    for y in 20..36 {
                        data[y * w + 20..y * w + gap_x].fill(dark);
                        data[y * w + gap_x + 1..y * w + gap_x + 1 + half].fill(dark);
                    }
                    blur(&mut data, w, h, passes);

                    let shipped = counterfactual_map(&data, w, h, cut_at(CUT_NUM));
                    let midpoint = counterfactual_map(&data, w, h, midpoint_cut);
                    let gap = probe_y * w + gap_x;
                    let interior = probe_y * w + interior_x;

                    cases += 1;
                    // Background at the gap is what keeps the two squares apart.
                    separated_by_cut += usize::from(data[gap] >= shipped[gap]);
                    separated_by_midpoint += usize::from(data[gap] >= midpoint[gap]);
                    fused_by_midpoint += usize::from(data[gap] < midpoint[gap]);

                    // The other half of the contract: eroding the gap open must not erode the
                    // squares themselves away, at any contrast or PSF width in the sweep.
                    assert!(
                        data[interior] < shipped[interior],
                        "square interior stopped being foreground at dark={dark} \
                         bright={bright} passes={passes} ({} vs {})",
                        data[interior],
                        shipped[interior]
                    );
                }
            }
        }

        assert!(
            fused_by_midpoint > 0,
            "premise gone: no case in the sweep fuses the pair at the midpoint, so this \
             fixture family no longer exercises the failure the cut exists to fix"
        );
        assert!(
            separated_by_cut > separated_by_midpoint,
            "the shipped cut separates {separated_by_cut}/{cases} of the sweep and the midpoint \
             separates {separated_by_midpoint}/{cases}: the cut is buying nothing here"
        );
        assert!(
            separated_by_cut >= separated_by_midpoint,
            "the shipped cut re-fused a pair the midpoint held open, which a lower cut cannot do"
        );
    }

    /// The other side of the trade, bracketed. A one-pixel dark stroke under a two-pixel PSF
    /// survives at [`CUT_NUM`]/[`CUT_DEN`] and is **lost** one twentieth lower, so this test
    /// pins the shipped cut as the lowest one that still keeps a stroke this thin — which is
    /// what the one-module borders of a small marker are made of.
    ///
    /// The stroke is placed in the same 3×3 tile neighbourhood as a solid dark block, so the
    /// neighbourhood minimum comes from the block. Without it the stroke *is* the minimum and
    /// the assertion degenerates into `nmin < nmin + something`, which holds for every cut.
    #[test]
    fn the_cut_is_the_lowest_that_keeps_a_one_pixel_dark_stroke() {
        let (w, h) = (64usize, 64usize);
        let mut data = vec![220u8; w * h];
        for y in 8..56 {
            data[y * w + 32] = 20;
            data[y * w + 40..y * w + 48].fill(20);
        }
        blur(&mut data, w, h, 2);

        let (_, production) = run_mode(ThresholdMode::TileMidExtreme, 2, 2, &data, w, h);
        let lower = counterfactual_map(&data, w, h, cut_at(CUT_NUM - 1));

        let mut lost_one_lower = 0usize;
        for y in 16..48 {
            let idx = y * w + 32;
            assert!(
                data[idx] < production[idx],
                "stroke row {y} is not foreground at the shipped cut ({} vs {})",
                data[idx],
                production[idx]
            );
            lost_one_lower += usize::from(data[idx] >= lower[idx]);
            // The block that supplies the neighbourhood minimum stays foreground either way.
            let block = y * w + 44;
            assert!(data[block] < production[block], "block row {y} lost");
        }
        assert_eq!(
            lost_one_lower,
            32,
            "the stroke survives a cut of {}/{CUT_DEN} too, so this fixture does not bracket \
             the shipped cut from below and would not notice it being lowered",
            CUT_NUM - 1
        );
    }

    /// The sliding column accumulator and the [`ExactDiv`] reciprocal reproduce the exact
    /// box mean (minus the offset).
    #[test]
    fn local_mean_matches_naive_box_mean() {
        for &(w, h) in &[(64usize, 48usize), (37, 29), (8, 8), (129, 5), (300, 9)] {
            let data = lcg_image(w, h, 7);
            for &r in &[1usize, 3, 7, 12, 40, 127] {
                let (binary, map) = run_mode(ThresholdMode::LocalMean, r, 2, &data, w, h);
                for y in 0..h {
                    for x in 0..w {
                        let expect = naive_box_mean(&data, w, h, x, y, r).saturating_sub(2);
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

    /// The offset shifts the threshold down, so raising it can only ever turn
    /// foreground pixels into background — the noise-suppression knob.
    #[test]
    fn local_mean_offset_is_monotone() {
        let (w, h) = (96usize, 64usize);
        let data = lcg_image(w, h, 11);
        let (_, small_offset) = run_mode(ThresholdMode::LocalMean, 8, 2, &data, w, h);
        let (_, large_offset) = run_mode(ThresholdMode::LocalMean, 8, 12, &data, w, h);
        for i in 0..w * h {
            assert!(large_offset[i] <= small_offset[i]);
        }
    }

    /// The offset is `k · σ` rounded and clamped to `[NOISE_OFFSET_MIN, NOISE_OFFSET_MAX]`.
    #[test]
    fn local_mean_offset_is_clamped() {
        let data = vec![0u8; 16 * 16];
        let img = ImageView::new(&data, 16, 16, 16).unwrap();
        let offset = |k: f32, sigma: f64| {
            ThresholdEngine {
                noise_k: k,
                ..ThresholdEngine::new()
            }
            .with_noise_sigma(sigma)
            .local_mean_offset(&img)
        };
        assert_eq!(offset(4.0, 2.0), 8);
        assert_eq!(offset(4.0, 0.1), NOISE_OFFSET_MIN);
        assert_eq!(offset(4.0, 50.0), NOISE_OFFSET_MAX);
    }

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
            threshold_noise_k: 3.0,
            ..DetectorConfig::default()
        };
        let engine = ThresholdEngine::from_config(&config).with_noise_sigma(1.0);
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
        let (_, with_binary) = run_mode(ThresholdMode::TileMidExtreme, 7, 2, &data, w, h);
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

    /// A perfectly flat frame has no structure. The local mean equals the grey
    /// level everywhere, so any positive offset drives the threshold below
    /// it and nothing is foreground — the noise-suppression property.
    ///
    /// The tile mode is the one that speckles: `t = mid(97, 97) = 97` is
    /// published to segmentation even though the tile carries no signal.
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
        let (_, tile) = run_mode(ThresholdMode::TileMidExtreme, 8, 2, &data, w, h);
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
        // Minimum 32px for 4 pixels per bit (AprilTag 36h11 = 8x8 cells). From ~300 px a
        // border cell spans several flat 8 px tiles, which the tile thresholder leaves
        // unthresholded (the hollow interior `ThresholdMode::LocalMean` exists to fill).
        let tag_sizes = [32, 48, 64, 100, 150, 200];

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
