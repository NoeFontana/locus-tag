//! Stride-aware image view for zero-copy ingestion.

/// A view into an image buffer with explicit stride support.
/// This allows handling NumPy arrays with padding or non-standard layouts.
pub struct ImageView<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}

impl<'a> ImageView<'a> {
    /// Create a new ImageView after validating that the buffer size matches the dimensions and stride.
    pub fn new(data: &'a [u8], width: usize, height: usize, stride: usize) -> Result<Self, String> {
        if stride < width {
            return Err(format!("Stride ({}) cannot be less than width ({})", stride, width));
        }
        let required_size = if height > 0 {
            (height - 1) * stride + width
        } else {
            0
        };
        if data.len() < required_size {
            return Err(format!(
                "Buffer size ({}) is too small for {}x{} image with stride {} (required: {})",
                data.len(), width, height, stride, required_size
            ));
        }
        Ok(Self { data, width, height, stride })
    }

    /// Safe accessor for a specific row.
    #[inline(always)]
    pub fn get_row(&self, y: usize) -> &[u8] {
        assert!(y < self.height, "Row index {} out of bounds", y);
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// Safe accessor for a specific pixel.
    #[inline(always)]
    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        assert!(x < self.width, "Column index {} out of bounds", x);
        self.get_row(y)[x]
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
        let view = ImageView::new(&data, 3, 2, 4).unwrap();
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
