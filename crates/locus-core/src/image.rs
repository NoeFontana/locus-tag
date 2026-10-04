//! Stride-aware image view for zero-copy ingestion.
#![allow(clippy::inline_always)]
#![allow(clippy::cast_sign_loss)]
#![allow(unsafe_code)]

use rayon::prelude::*;

/// A view into an image buffer with explicit stride support.
/// This allows handling NumPy arrays with padding or non-standard layouts.
#[derive(Copy, Clone)]
pub struct ImageView<'a> {
    /// The raw image data slice.
    pub data: &'a [u8],
    /// The width of the image in pixels.
    pub width: usize,
    /// The height of the image in pixels.
    pub height: usize,
    /// The stride (bytes per row) of the image.
    pub stride: usize,
}

impl<'a> ImageView<'a> {
    /// Create a new ImageView after validating that the buffer size matches the dimensions and stride.
    pub fn new(data: &'a [u8], width: usize, height: usize, stride: usize) -> Result<Self, String> {
        if stride < width {
            return Err(format!(
                "Stride ({stride}) cannot be less than width ({width})"
            ));
        }
        let required_size = if height > 0 {
            (height - 1) * stride + width
        } else {
            0
        };
        if data.len() < required_size {
            return Err(format!(
                "Buffer size ({}) is too small for {}x{} image with stride {} (required: {})",
                data.len(),
                width,
                height,
                stride,
                required_size
            ));
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
        })
    }

    /// Returns true if the image buffer has sufficient padding for safe SIMD gather operations.
    ///
    /// Some SIMD kernels (e.g. AVX2 gather) may perform 32-bit loads on 8-bit data,
    /// which can read up to 3 bytes past the end of the logical buffer.
    #[must_use]
    pub fn has_simd_padding(&self) -> bool {
        let required_size = if self.height > 0 {
            (self.height - 1) * self.stride + self.width
        } else {
            0
        };
        self.data.len() >= required_size + 3
    }

    /// Safe accessor for a specific row.
    #[inline(always)]
    #[must_use]
    pub fn get_row(&self, y: usize) -> &[u8] {
        assert!(y < self.height, "Row index {y} out of bounds");
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// Get a pixel value at (x, y) with boundary clamping.
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        // SAFETY: clamping ensures bounds
        unsafe { *self.data.get_unchecked(y * self.stride + x) }
    }

    /// Get a pixel value at (x, y) without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `x < width` and `y < height`.
    #[inline(always)]
    #[must_use]
    pub unsafe fn get_pixel_unchecked(&self, x: usize, y: usize) -> u8 {
        // debug_assert ensures we catch violations in debug mode
        debug_assert!(x < self.width, "x {} out of bounds {}", x, self.width);
        debug_assert!(y < self.height, "y {} out of bounds {}", y, self.height);
        // SAFETY: Caller guarantees that (x, y) are within the image dimensions.
        unsafe { *self.data.get_unchecked(y * self.stride + x) }
    }

    /// Sample pixel value with bilinear interpolation at sub-pixel coordinates.
    #[must_use]
    pub fn sample_bilinear(&self, x: f64, y: f64) -> f64 {
        let x = x - 0.5;
        let y = y - 0.5;

        if x < 0.0 || x >= (self.width - 1) as f64 || y < 0.0 || y >= (self.height - 1) as f64 {
            return f64::from(
                self.get_pixel(x.round().max(0.0) as usize, y.round().max(0.0) as usize),
            );
        }

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let dx = x - x0 as f64;
        let dy = y - y0 as f64;

        let v00 = f64::from(self.get_pixel(x0, y0));
        let v10 = f64::from(self.get_pixel(x1, y0));
        let v01 = f64::from(self.get_pixel(x0, y1));
        let v11 = f64::from(self.get_pixel(x1, y1));

        let v0 = v00 * (1.0 - dx) + v10 * dx;
        let v1 = v01 * (1.0 - dx) + v11 * dx;

        v0 * (1.0 - dy) + v1 * dy
    }

    /// Whether every point of the box `[lo, hi]` (Locus pixel-centre coordinates) may go to
    /// [`Self::sample_bilinear_unchecked`]. Keeps a margin of half a pixel beyond the
    /// sampler's contract on each side; NaN bounds fail.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bilinear_box_is_safe(&self, lo: [f64; 2], hi: [f64; 2]) -> bool {
        lo[0] >= 1.0
            && lo[1] >= 1.0
            && hi[0] <= self.width as f64 - 2.0
            && hi[1] <= self.height as f64 - 2.0
    }

    /// Sample pixel value with bilinear interpolation at sub-pixel coordinates without bounds checking.
    ///
    /// Same convention and result as [`Self::sample_bilinear`] inside the image: pixel `(i, j)`
    /// has its centre at `(i + 0.5, j + 0.5)`.
    ///
    /// # Safety
    /// Caller must ensure `0.5 <= x < width - 0.5` and `0.5 <= y < height - 0.5` (finite), so
    /// the two bilinear taps `floor(x - 0.5)` and `floor(x - 0.5) + 1` (likewise in `y`) are
    /// valid pixel indices. Every point of a box accepted by [`Self::bilinear_box_is_safe`]
    /// satisfies this.
    #[inline(always)]
    #[must_use]
    pub unsafe fn sample_bilinear_unchecked(&self, x: f64, y: f64) -> f64 {
        let x = x - 0.5;
        let y = y - 0.5;
        let x0 = x as usize; // Truncate is effectively floor for positive numbers
        let y0 = y as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        debug_assert!(x1 < self.width, "x1 {} out of bounds {}", x1, self.width);
        debug_assert!(y1 < self.height, "y1 {} out of bounds {}", y1, self.height);

        let dx = x - x0 as f64;
        let dy = y - y0 as f64;

        // Use unchecked pixel access
        // We know strides and offsets are valid because of the assertions (in debug) and caller contract (in release)
        // SAFETY: Caller guarantees that floor(x), floor(x)+1, floor(y), floor(y)+1 are within bounds.
        let row0 = unsafe { self.get_row_unchecked(y0) };
        // SAFETY: Same as above.
        let row1 = unsafe { self.get_row_unchecked(y1) };

        // We can access x0/x1 directly from the row slice
        // SAFETY: x0 and x1 are within bounds guaranteed by the caller.
        unsafe {
            let v00 = f64::from(*row0.get_unchecked(x0));
            let v10 = f64::from(*row0.get_unchecked(x1));
            let v01 = f64::from(*row1.get_unchecked(x0));
            let v11 = f64::from(*row1.get_unchecked(x1));

            let v0 = v00 * (1.0 - dx) + v10 * dx;
            let v1 = v01 * (1.0 - dx) + v11 * dx;

            v0 * (1.0 - dy) + v1 * dy
        }
    }

    /// Compute the gradient [gx, gy] at sub-pixel coordinates using bilinear interpolation.
    #[must_use]
    pub fn sample_gradient_bilinear(&self, x: f64, y: f64) -> [f64; 2] {
        let x = x - 0.5;
        let y = y - 0.5;

        // Optimization: Sample [gx, gy] directly using a 3x3 or 4x4 neighborhood
        // instead of 4 separate bilinear samples.
        // For a high-quality sub-pixel gradient, we sample the 4 nearest integer locations
        // and interpolate their finite-difference gradients.

        if x < 1.0 || x >= (self.width - 2) as f64 || y < 1.0 || y >= (self.height - 2) as f64 {
            // `sample_bilinear` takes pixel-centre-at-0.5 coordinates: undo the shift above.
            let (u, v) = (x + 0.5, y + 0.5);
            let gx = (self.sample_bilinear(u + 1.0, v) - self.sample_bilinear(u - 1.0, v)) * 0.5;
            let gy = (self.sample_bilinear(u, v + 1.0) - self.sample_bilinear(u, v - 1.0)) * 0.5;
            return [gx, gy];
        }

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let dx = x - x0 as f64;
        let dy = y - y0 as f64;

        // Fetch 4x4 neighborhood to compute central differences at 4 grid points
        // (x0, y0), (x0+1, y0), (x0, y0+1), (x0+1, y0+1)
        // Indices needed: x0-1..x0+2, y0-1..y0+2
        let mut g00 = [0.0, 0.0];
        let mut g10 = [0.0, 0.0];
        let mut g01 = [0.0, 0.0];
        let mut g11 = [0.0, 0.0];

        // SAFETY: Bounds are checked above (x >= 1.0, y >= 1.0, etc.)
        unsafe {
            for j in 0..2 {
                for i in 0..2 {
                    let cx = x0 + i;
                    let cy = y0 + j;

                    let gx = (f64::from(self.get_pixel_unchecked(cx + 1, cy))
                        - f64::from(self.get_pixel_unchecked(cx - 1, cy)))
                        * 0.5;
                    let gy = (f64::from(self.get_pixel_unchecked(cx, cy + 1))
                        - f64::from(self.get_pixel_unchecked(cx, cy - 1)))
                        * 0.5;

                    match (i, j) {
                        (0, 0) => g00 = [gx, gy],
                        (1, 0) => g10 = [gx, gy],
                        (0, 1) => g01 = [gx, gy],
                        (1, 1) => g11 = [gx, gy],
                        _ => unreachable!(),
                    }
                }
            }
        }

        let gx = (g00[0] * (1.0 - dx) + g10[0] * dx) * (1.0 - dy)
            + (g01[0] * (1.0 - dx) + g11[0] * dx) * dy;
        let gy = (g00[1] * (1.0 - dx) + g10[1] * dx) * (1.0 - dy)
            + (g01[1] * (1.0 - dx) + g11[1] * dx) * dy;

        [gx, gy]
    }

    /// Unsafe accessor for a specific row.
    #[inline(always)]
    pub(crate) unsafe fn get_row_unchecked(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        // SAFETY: Caller guarantees y < height. Width and stride are invariants.
        unsafe { &self.data.get_unchecked(start..start + self.width) }
    }

    /// Create a decimated copy of the image by averaging each `factor × factor` block
    /// (area resampling: anti-aliased, and every source pixel contributes).
    ///
    /// Output pixel `j` covers source pixels `[j·factor, (j + 1)·factor)`; map coordinates
    /// back with [`decimated_to_full`]. Trailing rows/columns that do not fill a block are
    /// dropped. The `output` buffer must have size at least `(width/factor) * (height/factor)`.
    pub fn decimate_to<'b>(
        &self,
        factor: usize,
        output: &'b mut [u8],
    ) -> Result<ImageView<'b>, String> {
        let factor = factor.max(1);
        if factor == 1 {
            let len = self.data.len();
            if output.len() < len {
                return Err(format!(
                    "Output buffer too small: {} < {}",
                    output.len(),
                    len
                ));
            }
            output[..len].copy_from_slice(self.data);
            return ImageView::new(&output[..len], self.width, self.height, self.width);
        }

        let new_w = self.width / factor;
        let new_h = self.height / factor;

        if output.len() < new_w * new_h {
            return Err(format!(
                "Output buffer too small for decimation: {} < {}",
                output.len(),
                new_w * new_h
            ));
        }

        output
            .par_chunks_exact_mut(new_w)
            .enumerate()
            .take(new_h)
            .for_each(|(y, out_row)| {
                let area = (factor * factor) as u32;
                let half = area / 2;
                for (x, out) in out_row.iter_mut().enumerate() {
                    let mut sum = 0u32;
                    for dy in 0..factor {
                        let row = self.get_row(y * factor + dy);
                        sum += row[x * factor..(x + 1) * factor]
                            .iter()
                            .map(|&v| u32::from(v))
                            .sum::<u32>();
                    }
                    // Round to nearest; the mean of u8 values always fits in u8.
                    *out = ((sum + half) / area) as u8;
                }
            });

        ImageView::new(&output[..new_w * new_h], new_w, new_h, new_w)
    }

    /// Create an upscaled copy of the image using bilinear interpolation.
    ///
    /// The `output` buffer must have size at least `(width*factor) * (height*factor)`.
    pub fn upscale_to<'b>(
        &self,
        factor: usize,
        output: &'b mut [u8],
    ) -> Result<ImageView<'b>, String> {
        let factor = factor.max(1);
        if factor == 1 {
            let len = self.data.len();
            if output.len() < len {
                return Err(format!(
                    "Output buffer too small: {} < {}",
                    output.len(),
                    len
                ));
            }
            output[..len].copy_from_slice(self.data);
            return ImageView::new(&output[..len], self.width, self.height, self.width);
        }

        let new_w = self.width * factor;
        let new_h = self.height * factor;

        if output.len() < new_w * new_h {
            return Err(format!(
                "Output buffer too small for upscaling: {} < {}",
                output.len(),
                new_w * new_h
            ));
        }

        let scale = 1.0 / factor as f64;

        // Centre-aware mapping: output pixel `x` has its centre at `x + 0.5`
        // in output space, i.e. `(x + 0.5) / factor` in source space
        // (`sample_bilinear` takes pixel-centre-at-0.5 coordinates): a pure scaling,
        // inverted by `upscaled_to_full`.
        output
            .par_chunks_exact_mut(new_w)
            .enumerate()
            .take(new_h)
            .for_each(|(y, out_row)| {
                let src_y = (y as f64 + 0.5) * scale;
                for (x, val) in out_row.iter_mut().enumerate() {
                    let src_x = (x as f64 + 0.5) * scale;
                    // We can use unchecked version for speed if we are confident,
                    // but sample_bilinear handles bounds checks.
                    // Given we are inside image bounds, it should be fine.
                    // To maximize perf we might want a localized optimized loop here,
                    // but for now reusing sample_bilinear is safe and clean.
                    // Round to nearest: truncation would darken every pixel by 0.5 grey.
                    *val = (self.sample_bilinear(src_x, src_y) + 0.5) as u8;
                }
            });

        ImageView::new(&output[..new_w * new_h], new_w, new_h, new_w)
    }
}

/// Maps a coordinate on the [`ImageView::decimate_to`] grid back to the full-resolution grid.
///
/// Coordinates are continuous with pixel centres at `+0.5` (the project-wide convention), so
/// pixel `j` of the decimated grid spans `[j, j + 1)` there and `[j·d, (j + 1)·d)` at full
/// resolution: the map is a pure scaling. (The `(v + 0.5)·d − 0.5` form is the same map in
/// *integer-centre* index units, i.e. OpenCV's convention, and is wrong here by `(d − 1)/2`.)
#[must_use]
pub fn decimated_to_full(v: f64, factor: usize) -> f64 {
    v * factor as f64
}

/// Maps a coordinate on the [`ImageView::upscale_to`] grid back to the original grid: the
/// inverse of the pure scaling [`ImageView::upscale_to`] samples with (see
/// [`decimated_to_full`] for the convention).
#[must_use]
pub fn upscaled_to_full(v: f64, factor: usize) -> f64 {
    v / factor as f64
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_image_view_stride() {
        let data = vec![
            1, 2, 3, 0, // row 0 + padding
            4, 5, 6, 0, // row 1 + padding
        ];
        let view = ImageView::new(&data, 3, 2, 4).expect("Valid image creation");
        assert_eq!(view.get_row(0), &[1, 2, 3]);
        assert_eq!(view.get_row(1), &[4, 5, 6]);
        assert_eq!(view.get_pixel(1, 1), 5);
    }

    /// Ramp image `I(x, y) = slope * x` (pixel centre at `x + 0.5`).
    fn ramp(width: usize, height: usize, slope: usize) -> Vec<u8> {
        (0..height)
            .flat_map(|_| (0..width).map(move |x| (slope * x) as u8))
            .collect()
    }

    /// A resampler and its coordinate mapping must agree: on a ramp, the value a resampled
    /// pixel holds identifies the source coordinate it was taken from, and mapping that
    /// pixel's centre back to the source grid must land on that same coordinate.
    #[test]
    fn decimation_mapping_matches_resampler() {
        let (w, h, slope) = (60, 12, 4);
        let data = ramp(w, h, slope);
        let img = ImageView::new(&data, w, h, w).expect("valid image");
        for d in [2usize, 3] {
            let mut out = vec![0u8; (w / d) * (h / d)];
            let dec = img.decimate_to(d, &mut out).expect("decimate");
            for j in 1..dec.width - 1 {
                let value = f64::from(dec.get_pixel(j, 2));
                let source_u = value / slope as f64 + 0.5;
                let mapped = decimated_to_full(j as f64 + 0.5, d);
                assert!(
                    (mapped - source_u).abs() < 0.5 / slope as f64 + 1e-9,
                    "d={d} j={j}: mapping gives {mapped}, resampler took {source_u}"
                );
            }
        }
    }

    #[test]
    fn upscale_mapping_matches_resampler() {
        let (w, h, slope) = (30, 8, 8);
        let data = ramp(w, h, slope);
        let img = ImageView::new(&data, w, h, w).expect("valid image");
        for u in [2usize, 3] {
            let mut out = vec![0u8; w * u * h * u];
            let up = img.upscale_to(u, &mut out).expect("upscale");
            // Stay clear of the border clamp.
            for j in 2 * u..up.width - 2 * u {
                let value = f64::from(up.get_pixel(j, 3 * u));
                let source_u = value / slope as f64 + 0.5;
                let mapped = upscaled_to_full(j as f64 + 0.5, u);
                assert!(
                    (mapped - source_u).abs() < 0.5 / slope as f64 + 1e-9,
                    "U={u} j={j}: mapping gives {mapped}, resampler took {source_u}"
                );
            }
        }
    }

    /// The border fallback of `sample_gradient_bilinear` must sample the same point as the
    /// interior path: on an image that varies only in x, gx near the top border equals gx
    /// in the interior at the same x.
    #[test]
    fn gradient_border_fallback_samples_the_requested_point() {
        // q(i) = i(i-1)/2 is integer-valued, and both gradient paths are exact on it
        // (gx = u - 1 at continuous coordinate u), so any disagreement is a sampling shift.
        let (w, h) = (23usize, 32usize);
        let data: Vec<u8> = (0..h)
            .flat_map(|_| (0..w).map(|x| (x * x.saturating_sub(1) / 2) as u8))
            .collect();
        let img = ImageView::new(&data, w, h, w).expect("valid image");
        for x in [8.3, 10.75, 12.5] {
            let border = img.sample_gradient_bilinear(x, 1.0);
            let interior = img.sample_gradient_bilinear(x, 16.0);
            assert!(
                (border[0] - interior[0]).abs() < 0.02,
                "x={x}: border gx {} vs interior gx {}",
                border[0],
                interior[0]
            );
        }
    }

    #[test]
    fn test_invalid_buffer_size() {
        let data = vec![1, 2, 3];
        let result = ImageView::new(&data, 2, 2, 2);
        assert!(result.is_err());
    }

    proptest! {
        #[test]
        fn prop_image_view_creation(
            width in 0..1000usize,
            height in 0..1000usize,
            stride_extra in 0..100usize,
            has_enough_data in prop::bool::ANY
        ) {
            let stride = width + stride_extra;
            let required_size = if height > 0 {
                (height - 1) * stride + width
            } else {
                0
            };

            let data_len = if has_enough_data {
                required_size
            } else {
                required_size.saturating_sub(1)
            };

            let data = vec![0u8; data_len];
            let result = ImageView::new(&data, width, height, stride);

            if height > 0 && required_size > 0 && !has_enough_data {
                assert!(result.is_err());
            } else {
                assert!(result.is_ok());
            }
        }

        #[test]
        fn prop_get_pixel_clamping(
            width in 1..100usize,
            height in 1..100usize,
            x in 0..200usize,
            y in 0..200usize
        ) {
            let data = vec![0u8; height * width];
            let view = ImageView::new(&data, width, height, width).expect("valid creation");
            let p = view.get_pixel(x, y);
            // Clamping should prevent panic
            assert_eq!(p, 0);
        }

        #[test]
        fn prop_sample_bilinear_invariants(
            width in 2..20usize,
            height in 2..20usize,
            data in prop::collection::vec(0..=255u8, 20*20),
            x in 0.0..20.0f64,
            y in 0.0..20.0f64
        ) {
            let real_width = width.min(20);
            let real_height = height.min(20);
            let slice = &data[..real_width * real_height];
            let view = ImageView::new(slice, real_width, real_height, real_width).expect("valid creation");

            let x = x % real_width as f64;
            let y = y % real_height as f64;

            let val = view.sample_bilinear(x, y);

            // Result should be within [0, 255]
            assert!((0.0..=255.0).contains(&val));

            // If inside 2x2 neighborhood, val should be within min/max of those 4 pixels
            // The pixels are centered at i + 0.5, so sample (x, y) is between
            // floor(x-0.5) and floor(x-0.5)+1.
            let x0 = (x - 0.5).max(0.0).floor() as usize;
            let y0 = (y - 0.5).max(0.0).floor() as usize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            if x1 < real_width && y1 < real_height {
                let v00 = view.get_pixel(x0, y0);
                let v10 = view.get_pixel(x1, y0);
                let v01 = view.get_pixel(x0, y1);
                let v11 = view.get_pixel(x1, y1);

                let min = f64::from(v00.min(v10).min(v01).min(v11));
                let max = f64::from(v00.max(v10).max(v01).max(v11));

                assert!(val >= min - 1e-9 && val <= max + 1e-9, "Value {val} not in [{min}, {max}] for x={x}, y={y}");
            }
        }
    }
}
