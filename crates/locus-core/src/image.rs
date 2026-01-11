//! Stride-aware image view for zero-copy ingestion.
#![allow(clippy::inline_always)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(unsafe_code)]

/// A view into an image buffer with explicit stride support.
/// This allows handling NumPy arrays with padding or non-standard layouts.
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

    /// Unsafe accessor for a specific row.
    #[inline(always)]
    unsafe fn get_row_unchecked(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        // SAFETY: Caller guarantees y < height. Width and stride are invariants.
        unsafe { &self.data.get_unchecked(start..start + self.width) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
