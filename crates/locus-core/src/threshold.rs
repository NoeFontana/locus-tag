use crate::image::ImageView;
use multiversion::multiversion;

/// Statistics for a single tile.
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
        Self { tile_size: 8 }
    }

    /// Compute min/max statistics for each tile in the image.
    /// Optimized with SIMD-friendly memory access patterns.
    #[must_use]
    pub fn compute_tile_stats(&self, img: &ImageView) -> Vec<TileStats> {
        let ts = self.tile_size;
        let tiles_wide = img.width / ts;
        let tiles_high = img.height / ts;
        let mut stats = vec![TileStats { min: 255, max: 0 }; tiles_wide * tiles_high];

        for ty in 0..tiles_high {
            let stats_row = &mut stats[ty * tiles_wide..(ty + 1) * tiles_wide];

            for dy in 0..ts {
                let py = ty * ts + dy;
                let src_row = img.get_row(py);

                // Process all tiles in this row with SIMD-friendly min/max
                compute_row_tile_stats_simd(src_row, stats_row, ts);
            }
        }
        stats
    }

    /// Apply adaptive thresholding to the image.
    /// Optimized with pre-expanded threshold maps and vectorized row processing.
    pub fn apply_threshold(&self, img: &ImageView, stats: &[TileStats], output: &mut [u8]) {
        let ts = self.tile_size;
        let tiles_wide = img.width / ts;
        let tiles_high = img.height / ts;

        let mut tile_thresholds = vec![0u8; tiles_wide * tiles_high];
        let mut tile_valid = vec![0u8; tiles_wide * tiles_high];

        for ty in 0..tiles_high {
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

                let idx = ty * tiles_wide + tx;
                if nmax.saturating_sub(nmin) < 10 {
                    tile_valid[idx] = 0;
                } else {
                    tile_valid[idx] = 255;
                    tile_thresholds[idx] = ((u16::from(nmin) + u16::from(nmax)) >> 1) as u8;
                }
            }
        }

        // Expanded buffers are reused across rows
        let mut row_thresh = vec![0u8; img.width];
        let mut row_valid = vec![0u8; img.width];

        for ty in 0..tiles_high {
            // Expand tile thresholds to pixel-level for this tile-row
            let tile_row_thresh = &tile_thresholds[ty * tiles_wide..(ty + 1) * tiles_wide];
            let tile_row_valid = &tile_valid[ty * tiles_wide..(ty + 1) * tiles_wide];

            for (tx, (&thresh, &valid)) in tile_row_thresh
                .iter()
                .zip(tile_row_valid.iter())
                .enumerate()
            {
                let x_start = tx * ts;
                row_thresh[x_start..x_start + ts].fill(thresh);
                row_valid[x_start..x_start + ts].fill(valid);
            }

            for dy in 0..ts {
                let py = ty * ts + dy;
                let src_row = img.get_row(py);
                let dst_start = py * img.width;
                let dst_row = &mut output[dst_start..dst_start + img.width];
                threshold_row_simd(src_row, dst_row, &row_thresh, &row_valid);
            }
        }
    }
}

/// SIMD-optimized row tile stats computation.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn compute_row_tile_stats_simd(src_row: &[u8], stats: &mut [TileStats], tile_size: usize) {
    let tiles = stats.len();
    for (tx, stat) in stats.iter_mut().enumerate().take(tiles) {
        let x = tx * tile_size;
        let chunk = &src_row[x..x + tile_size];

        let mut rmin = 255u8;
        let mut rmax = 0u8;
        for &p in chunk {
            rmin = rmin.min(p);
            rmax = rmax.max(p);
        }

        stat.min = stat.min.min(rmin);
        stat.max = stat.max.max(rmax);
    }
}

/// SIMD-optimized thresholding for a full row.
#[multiversion(targets = "simd")]
fn threshold_row_simd(src: &[u8], dst: &mut [u8], thresholds: &[u8], valid_mask: &[u8]) {
    let len = src.len();
    for i in 0..len {
        let s = src[i];
        let t = thresholds[i];
        let m = valid_mask[i];
        // Branchless: (s > t) produces 0 or 1, multiply by 255
        let pass = u8::from(s > t).wrapping_neg(); // 0xFF if true, 0x00 if false
        dst[i] = pass & m;
    }
}

#[cfg(test)]
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
                if s > thresh {
                    assert_eq!(dst[i], 255);
                } else {
                    assert_eq!(dst[i], 0);
                }
            }
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
