use crate::image::ImageView;
use multiversion::multiversion;

/// Statistics for a single tile.
#[derive(Clone, Copy, Debug, Default)]
pub struct TileStats {
    pub min: u8,
    pub max: u8,
}

pub struct ThresholdEngine {
    pub tile_size: usize,
}

impl ThresholdEngine {
    pub fn new() -> Self {
        Self { tile_size: 4 }
    }

    /// Compute min/max statistics for each 4x4 tile in the image.
    pub fn compute_tile_stats(&self, img: &ImageView) -> Vec<TileStats> {
        let tiles_wide = img.width / self.tile_size;
        let tiles_high = img.height / self.tile_size;
        let mut stats = vec![TileStats { min: 255, max: 0 }; tiles_wide * tiles_high];

        use rayon::prelude::*;

        stats
            .par_chunks_exact_mut(tiles_wide)
            .enumerate()
            .for_each(|(ty, stats_row)| {
                for dy in 0..self.tile_size {
                    let py = ty * self.tile_size + dy;
                    let src_row = img.get_row(py);
                    for tx in 0..tiles_wide {
                        let x_off = tx * 4;
                        let tile_pixels = &src_row[x_off..x_off + 4];
                        let (rmin, rmax) = compute_min_max_simd(tile_pixels);
                        if rmin < stats_row[tx].min {
                            stats_row[tx].min = rmin;
                        }
                        if rmax > stats_row[tx].max {
                            stats_row[tx].max = rmax;
                        }
                    }
                }
            });
        stats
    }

    /// Apply adaptive thresholding to the image.
    /// Highly optimized for SIMD and cache locality.
    pub fn apply_threshold(&self, img: &ImageView, stats: &[TileStats], output: &mut [u8]) {
        let tiles_wide = img.width / self.tile_size;
        let tiles_high = img.height / self.tile_size;

        // Using a stack buffer for thresholds/mask to avoid heap allocation in the hot loop
        // for most common resolutions (up to 4K).
        let mut tile_thresholds = vec![0u8; tiles_wide * tiles_high];
        let mut tile_valid_mask = vec![0u8; tiles_wide * tiles_high];

        use rayon::prelude::*;

        tile_thresholds
            .par_chunks_exact_mut(tiles_wide)
            .zip(tile_valid_mask.par_chunks_exact_mut(tiles_wide))
            .enumerate()
            .for_each(|(ty, (t_row, v_row))| {
                for tx in 0..tiles_wide {
                    let mut nmin = 255u8;
                    let mut nmax = 0u8;

                    let y_start = if ty > 0 { ty - 1 } else { 0 };
                    let y_end = (ty + 1).min(tiles_high - 1);
                    let x_start = if tx > 0 { tx - 1 } else { 0 };
                    let x_end = (tx + 1).min(tiles_wide - 1);

                    for ny in y_start..=y_end {
                        let n_row_offset = ny * tiles_wide;
                        for nx in x_start..=x_end {
                            let s = stats[n_row_offset + nx];
                            if s.min < nmin {
                                nmin = s.min;
                            }
                            if s.max > nmax {
                                nmax = s.max;
                            }
                        }
                    }

                    if nmax - nmin < 10 {
                        v_row[tx] = 0;
                    } else {
                        v_row[tx] = 255;
                        t_row[tx] = ((nmin as u16 + nmax as u16) >> 1) as u8;
                    }
                }
            });

        // Apply thresholding row-by-row in parallel
        output
            .par_chunks_exact_mut(img.width * self.tile_size)
            .enumerate()
            .for_each(|(ty, output_tiles_blocks)| {
                let mut row_thresholds = [0u8; 4096];
                let mut row_valid = [0u8; 4096];

                let tile_row_idx = ty * tiles_wide;
                for tx in 0..tiles_wide {
                    let val = tile_thresholds[tile_row_idx + tx];
                    let valid = tile_valid_mask[tile_row_idx + tx];
                    let x_off = tx * 4;
                    row_thresholds[x_off] = val;
                    row_thresholds[x_off + 1] = val;
                    row_thresholds[x_off + 2] = val;
                    row_thresholds[x_off + 3] = val;
                    row_valid[x_off] = valid;
                    row_valid[x_off + 1] = valid;
                    row_valid[x_off + 2] = valid;
                    row_valid[x_off + 3] = valid;
                }

                for dy in 0..4 {
                    let py = ty * 4 + dy;
                    let src_row = img.get_row(py);
                    let dst_row = &mut output_tiles_blocks[dy * img.width..(dy + 1) * img.width];
                    threshold_row_simd(
                        src_row,
                        dst_row,
                        &row_thresholds[..img.width],
                        &row_valid[..img.width],
                    );
                }
            });
    }
}

#[multiversion(targets = "simd")]
fn threshold_row_simd(src: &[u8], dst: &mut [u8], thresholds: &[u8], valid_mask: &[u8]) {
    for i in 0..src.len() {
        // Branching-free: (src[i] > thresholds[i]) as u8 * 255 & valid_mask[i]
        let pass = if src[i] > thresholds[i] { 255 } else { 0 };
        dst[i] = pass & valid_mask[i];
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
            threshold_row_simd(&src, &mut dst, &vec![thresh; 16], &valid);

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
        if b < min {
            min = b;
        }
        if b > max {
            max = b;
        }
    }
    (min, max)
}
