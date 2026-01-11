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
        Self { tile_size: 8 } // Larger tiles for faster processing
    }

    /// Compute min/max statistics for each tile in the image.
    pub fn compute_tile_stats(&self, img: &ImageView) -> Vec<TileStats> {
        let tiles_wide = img.width / self.tile_size;
        let tiles_high = img.height / self.tile_size;
        let mut stats = vec![TileStats { min: 255, max: 0 }; tiles_wide * tiles_high];

        // Single-threaded for small images to avoid thread pool overhead
        for ty in 0..tiles_high {
            let stats_offset = ty * tiles_wide;
            for dy in 0..self.tile_size {
                let py = ty * self.tile_size + dy;
                let src_row = img.get_row(py);
                for tx in 0..tiles_wide {
                    let x = tx * self.tile_size;
                    let mut rmin = 255u8;
                    let mut rmax = 0u8;
                    for dx in 0..self.tile_size {
                        let p = src_row[x + dx];
                        if p < rmin {
                            rmin = p;
                        }
                        if p > rmax {
                            rmax = p;
                        }
                    }
                    let s = &mut stats[stats_offset + tx];
                    if rmin < s.min {
                        s.min = rmin;
                    }
                    if rmax > s.max {
                        s.max = rmax;
                    }
                }
            }
        }
        stats
    }

    /// Apply adaptive thresholding to the image.
    pub fn apply_threshold(&self, img: &ImageView, stats: &[TileStats], output: &mut [u8]) {
        let tiles_wide = img.width / self.tile_size;
        let tiles_high = img.height / self.tile_size;
        let ts = self.tile_size;

        // Compute adaptive thresholds
        let mut tile_thresholds = vec![0u8; tiles_wide * tiles_high];
        let mut tile_valid_mask = vec![0u8; tiles_wide * tiles_high];

        for ty in 0..tiles_high {
            for tx in 0..tiles_wide {
                let mut nmin = 255u8;
                let mut nmax = 0u8;

                let y_start = if ty > 0 { ty - 1 } else { 0 };
                let y_end = (ty + 1).min(tiles_high - 1);
                let x_start = if tx > 0 { tx - 1 } else { 0 };
                let x_end = (tx + 1).min(tiles_wide - 1);

                for ny in y_start..=y_end {
                    for nx in x_start..=x_end {
                        let s = stats[ny * tiles_wide + nx];
                        if s.min < nmin {
                            nmin = s.min;
                        }
                        if s.max > nmax {
                            nmax = s.max;
                        }
                    }
                }

                let idx = ty * tiles_wide + tx;
                if nmax - nmin < 10 {
                    tile_valid_mask[idx] = 0;
                } else {
                    tile_valid_mask[idx] = 255;
                    tile_thresholds[idx] = ((nmin as u16 + nmax as u16) >> 1) as u8;
                }
            }
        }

        // Apply thresholding
        for ty in 0..tiles_high {
            for dy in 0..ts {
                let py = ty * ts + dy;
                let src_row = img.get_row(py);
                let dst_start = py * img.width;
                let dst_row = &mut output[dst_start..dst_start + img.width];

                for tx in 0..tiles_wide {
                    let tile_idx = ty * tiles_wide + tx;
                    let thresh = tile_thresholds[tile_idx];
                    let valid = tile_valid_mask[tile_idx];
                    let x_start = tx * ts;

                    for dx in 0..ts {
                        let x = x_start + dx;
                        let pass = if src_row[x] > thresh { 255 } else { 0 };
                        dst_row[x] = pass & valid;
                    }
                }
            }
        }
    }
}

#[multiversion(targets = "simd")]
fn threshold_row_simd(src: &[u8], dst: &mut [u8], thresholds: &[u8], valid_mask: &[u8]) {
    // Simple loop for better autovectorization
    let len = src.len();
    for i in 0..len {
        let s = src[i];
        let t = thresholds[i];
        let m = valid_mask[i];
        let pass = if s > t { 255 } else { 0 };
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
