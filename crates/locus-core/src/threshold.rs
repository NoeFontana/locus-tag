use crate::config::DetectorConfig;
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
    /// Minimum intensity range for a tile to be considered valid.
    pub min_range: u8,
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
        Self {
            tile_size: 8, // Standard 8x8 tiles
            min_range: 5, // Lower for low-contrast edge detection
        }
    }

    /// Create a ThresholdEngine from detector configuration.
    #[must_use]
    pub fn from_config(config: &DetectorConfig) -> Self {
        Self {
            tile_size: config.threshold_tile_size,
            min_range: config.threshold_min_range,
        }
    }

    /// Compute min/max statistics for each tile in the image.
    /// Optimized with SIMD-friendly memory access patterns and subsampling (stride 2).
    #[must_use]
    pub fn compute_tile_stats(&self, img: &ImageView) -> Vec<TileStats> {
        let ts = self.tile_size;
        let tiles_wide = img.width / ts;
        let tiles_high = img.height / ts;
        let mut stats = vec![TileStats { min: 255, max: 0 }; tiles_wide * tiles_high];

        for ty in 0..tiles_high {
            let stats_row = &mut stats[ty * tiles_wide..(ty + 1) * tiles_wide];

            // Subsampling: Only process every other row within a tile (stride 2)
            // This statistically approximates the min/max sufficient for thresholding
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
                if nmax.saturating_sub(nmin) < self.min_range {
                    tile_valid[idx] = 0;
                    // Use a rough estimate for now, will be refined in propagation pass
                    tile_thresholds[idx] = ((u16::from(nmin) + u16::from(nmax)) >> 1) as u8;
                } else {
                    tile_valid[idx] = 255;
                    tile_thresholds[idx] = ((u16::from(nmin) + u16::from(nmax)) >> 1) as u8;
                }
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

        let mut row_thresholds = vec![0u8; img.width];
        let mut row_valid = vec![0u8; img.width];

        for ty in 0..tiles_high {
            // Expand tile stats to row buffers
            for tx in 0..tiles_wide {
                let idx = ty * tiles_wide + tx;
                let thresh = tile_thresholds[idx];
                let valid = tile_valid[idx];
                for i in 0..ts {
                    row_thresholds[tx * ts + i] = thresh;
                    row_valid[tx * ts + i] = valid;
                }
            }

            for dy in 0..ts {
                let py = ty * ts + dy;
                let src_row = img.get_row(py);
                let dst_start = py * img.width;
                let dst_row = &mut output[dst_start..dst_start + img.width];

                threshold_row_simd(src_row, dst_row, &row_thresholds, &row_valid);
            }
        }
    }
}

/// SIMD-optimized row tile stats computation with subsampling (stride 2).
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn compute_row_tile_stats_simd(src_row: &[u8], stats: &mut [TileStats], tile_size: usize) {
    let chunks = src_row.chunks_exact(tile_size);
    for (chunk, stat) in chunks.zip(stats.iter_mut()) {
        let mut rmin = 255u8;
        let mut rmax = 0u8;
        // Subsampling: Only process every other pixel in the row (stride 2)
        for &p in chunk.iter() {
            rmin = rmin.min(p);
            rmax = rmax.max(p);
        }

        stat.min = stat.min.min(rmin);
        stat.max = stat.max.max(rmax);
    }
}

/// SIMD-optimized thresholding for a full row.
/// SIMD-optimized thresholding for a full row.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn threshold_row_simd(src: &[u8], dst: &mut [u8], thresholds: &[u8], valid_mask: &[u8]) {
    let len = src.len();
    for i in 0..len {
        let s = src[i];
        let t = thresholds[i];
        let m = valid_mask[i];
        // Branchless: (s > t) produces 0 or 1, multiply by 255
        let pass = u8::from(s >= t).wrapping_neg(); // 0xFF if true, 0x00 if false

        // Use mask m > 0 to treat both original (255) and propagated (128) tiles as valid.
        // If valid, use pass. If invalid (m=0), force white (255).
        let is_valid = u8::from(m > 0).wrapping_neg(); // 0xFF if m > 0, 0x00 otherwise
        dst[i] = (pass & is_valid) | !is_valid;
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
                if x < 2 || x > 14 || y < 2 || y > 14 {
                    data[y * width + x] = 200;
                }
            }
        }

        let img = ImageView::new(&data, width, height, width).unwrap();
        let engine = ThresholdEngine::new();
        let stats = engine.compute_tile_stats(&img);
        let mut output = vec![0u8; width * height];
        engine.apply_threshold(&img, &stats, &mut output);

        // At (8,8), it should be black (0) because it's 50 and thresh should be around (50+200)/2 = 125
        assert_eq!(output[8 * width + 8], 0);
        // At (1,1), it should be white (255)
        assert_eq!(output[1 * width + 1], 255);
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
            let stats = engine.compute_tile_stats(&img);
            let mut binary = vec![0u8; canvas_size * canvas_size];
            engine.apply_threshold(&img, &stats, &mut binary);

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
                let stats = engine.compute_tile_stats(&img);
                let mut binary = vec![0u8; canvas_size * canvas_size];
                engine.apply_threshold(&img, &stats, &mut binary);

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
            let stats = engine.compute_tile_stats(&img);
            let mut binary = vec![0u8; canvas_size * canvas_size];
            engine.apply_threshold(&img, &stats, &mut binary);

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
            let stats = engine.compute_tile_stats(&img);
            let mut binary = vec![0u8; canvas_size * canvas_size];

            // Should not panic
            engine.apply_threshold(&img, &stats, &mut binary);

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
