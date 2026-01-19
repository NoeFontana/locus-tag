//! Stride-aware image view for zero-copy ingestion.
#![allow(clippy::inline_always)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
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
        // SAFETY: Caller guarantees bounds
        unsafe { *self.data.get_unchecked(y * self.stride + x) }
    }

    /// Sample pixel value with bilinear interpolation at sub-pixel coordinates.
    #[must_use]
    pub fn sample_bilinear(&self, x: f64, y: f64) -> f64 {
        if x < 0.0 || x >= (self.width - 1) as f64 || y < 0.0 || y >= (self.height - 1) as f64 {
            return f64::from(self.get_pixel(x.round() as usize, y.round() as usize));
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

    /// Sample pixel value with bilinear interpolation at sub-pixel coordinates without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `0.0 <= x <= width - 1.001` and `0.0 <= y <= height - 1.001`
    /// such that floor(x), floor(x)+1, floor(y), floor(y)+1 are all valid indices.
    #[inline(always)]
    #[must_use]
    pub unsafe fn sample_bilinear_unchecked(&self, x: f64, y: f64) -> f64 {
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
        // SAFETY: Caller guarantees checks.
        let row0 = unsafe { self.get_row_unchecked(y0) };
        let row1 = unsafe { self.get_row_unchecked(y1) };

        // We can access x0/x1 directly from the row slice
        // SAFETY: Caller guarantees checks.
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
        // Optimization: Sample [gx, gy] directly using a 3x3 or 4x4 neighborhood
        // instead of 4 separate bilinear samples.
        // For a high-quality sub-pixel gradient, we sample the 4 nearest integer locations
        // and interpolate their finite-difference gradients.
        
        if x < 1.0 || x >= (self.width - 2) as f64 || y < 1.0 || y >= (self.height - 2) as f64 {
            let gx = (self.sample_bilinear(x + 1.0, y) - self.sample_bilinear(x - 1.0, y)) * 0.5;
            let gy = (self.sample_bilinear(x, y + 1.0) - self.sample_bilinear(x, y - 1.0)) * 0.5;
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

        unsafe {
            for j in 0..2 {
                for i in 0..2 {
                    let cx = x0 + i;
                    let cy = y0 + j;
                    
                    let gx = (f64::from(self.get_pixel_unchecked(cx + 1, cy)) - f64::from(self.get_pixel_unchecked(cx - 1, cy))) * 0.5;
                    let gy = (f64::from(self.get_pixel_unchecked(cx, cy + 1)) - f64::from(self.get_pixel_unchecked(cx, cy - 1))) * 0.5;
                    
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

        let gx = (g00[0] * (1.0 - dx) + g10[0] * dx) * (1.0 - dy) + (g01[0] * (1.0 - dx) + g11[0] * dx) * dy;
        let gy = (g00[1] * (1.0 - dx) + g10[1] * dx) * (1.0 - dy) + (g01[1] * (1.0 - dx) + g11[1] * dx) * dy;
        
        [gx, gy]
    }

    /// Unsafe accessor for a specific row.
    #[inline(always)]
    pub(crate) unsafe fn get_row_unchecked(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        // SAFETY: Caller guarantees y < height. Width and stride are invariants.
        unsafe { &self.data.get_unchecked(start..start + self.width) }
    }

    /// Create a decimated copy of the image by subsampling every `factor` pixels.
    ///
    /// The `output` buffer must have size at least `(width/factor) * (height/factor)`.
    pub fn decimate_to<'b>(&self, factor: usize, output: &'b mut [u8]) -> Result<ImageView<'b>, String> {
        let factor = factor.max(1);
        if factor == 1 {
            let len = self.data.len();
            if output.len() < len {
                return Err(format!("Output buffer too small: {} < {}", output.len(), len));
            }
            output[..len].copy_from_slice(self.data);
            return ImageView::new(&output[..len], self.width, self.height, self.width);
        }

        let new_w = self.width / factor;
        let new_h = self.height / factor;

        if output.len() < new_w * new_h {
            return Err(format!("Output buffer too small for decimation: {} < {}", output.len(), new_w * new_h));
        }

        output.par_chunks_exact_mut(new_w).enumerate().take(new_h).for_each(|(y, out_row)| {
            let src_y = y * factor;
            let src_row = self.get_row(src_y);
            for x in 0..new_w {
                out_row[x] = src_row[x * factor];
            }
        });

        ImageView::new(&output[..new_w * new_h], new_w, new_h, new_w)
    }
}

#[cfg(test)]
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

            if height > 0 && !has_enough_data {
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
            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
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
