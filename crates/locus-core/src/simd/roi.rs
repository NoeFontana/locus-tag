//! Region of Interest (ROI) caching for fast sampling.

use crate::image::ImageView;
use bumpalo::Bump;

/// A cache for a small region of the image to improve L1 cache hits during sampling.
#[allow(clippy::large_enum_variant)]
pub(crate) enum RoiCache<'a> {
    /// Small ROI stored on the stack.
    Stack {
        /// The cached pixel data.
        data: [u8; 1024],
        /// The bounding box of the ROI in the original image.
        min_x: usize,
        /// The bounding box of the ROI in the original image.
        min_y: usize,
        /// Width of the ROI.
        width: usize,
        /// Height of the ROI.
        height: usize,
    },
    /// Larger ROI stored in the arena.
    Arena {
        /// The cached pixel data.
        data: &'a [u8],
        /// The bounding box of the ROI in the original image.
        min_x: usize,
        /// The bounding box of the ROI in the original image.
        min_y: usize,
        /// Width of the ROI.
        width: usize,
        /// Height of the ROI.
        height: usize,
    },
}

impl<'a> RoiCache<'a> {
    /// Create a new ROI cache by copying a region from the image.
    ///
    /// If the region fits in 1024 bytes, it is stored on the stack.
    /// Otherwise, it is allocated from the provided arena.
    #[must_use]
    pub fn new(
        img: &ImageView,
        arena: &'a Bump,
        min_x: usize,
        min_y: usize,
        max_x: usize,
        max_y: usize,
    ) -> Self {
        let width = (max_x - min_x + 1).min(img.width - min_x);
        let height = (max_y - min_y + 1).min(img.height - min_y);
        let size = width * height;

        if size <= 1024 {
            let mut data = [0u8; 1024];
            for y in 0..height {
                let src_offset = (min_y + y) * img.stride + min_x;
                let dst_offset = y * width;
                data[dst_offset..dst_offset + width]
                    .copy_from_slice(&img.data[src_offset..src_offset + width]);
            }
            RoiCache::Stack {
                data,
                min_x,
                min_y,
                width,
                height,
            }
        } else {
            let dst = arena.alloc_slice_fill_default(size);
            for y in 0..height {
                let src_offset = (min_y + y) * img.stride + min_x;
                let dst_offset = y * width;
                dst[dst_offset..dst_offset + width]
                    .copy_from_slice(&img.data[src_offset..src_offset + width]);
            }
            RoiCache::Arena {
                data: dst,
                min_x,
                min_y,
                width,
                height,
            }
        }
    }

    /// Get a pixel from the cache using global coordinates.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        match self {
            RoiCache::Stack {
                data,
                min_x,
                min_y,
                width,
                height,
                ..
            } => {
                let lx = x.saturating_sub(*min_x).min(width.saturating_sub(1));
                let ly = y.saturating_sub(*min_y).min(height.saturating_sub(1));
                data[ly * width + lx]
            },
            RoiCache::Arena {
                data,
                min_x,
                min_y,
                width,
                height,
                ..
            } => {
                let lx = x.saturating_sub(*min_x).min(width.saturating_sub(1));
                let ly = y.saturating_sub(*min_y).min(height.saturating_sub(1));
                data[ly * width + lx]
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageView;
    use bumpalo::Bump;

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_roi_cache_stack() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let img = ImageView::new(&data, 10, 10, 10).expect("valid view");
        let arena = Bump::new();

        // 3x3 = 9 bytes, should be Stack
        let cache = RoiCache::new(&img, &arena, 2, 2, 4, 4);
        assert!(matches!(cache, RoiCache::Stack { .. }));
        assert_eq!(cache.get(2, 2), 22);
        assert_eq!(cache.get(4, 4), 44);
    }

    #[test]
    fn test_roi_cache_arena() {
        let mut data = vec![0u8; 40 * 40];
        data[20 * 40 + 20] = 255;
        let img = ImageView::new(&data, 40, 40, 40).expect("valid view");
        let arena = Bump::new();

        // 33x33 = 1089 bytes, should be Arena
        let cache = RoiCache::new(&img, &arena, 0, 0, 32, 32);
        assert!(matches!(cache, RoiCache::Arena { .. }));
        assert_eq!(cache.get(20, 20), 255);
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_roi_cache_clamping() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let img = ImageView::new(&data, 10, 10, 10).expect("valid view");
        let arena = Bump::new();
        let cache = RoiCache::new(&img, &arena, 2, 2, 4, 4);

        // Should clamp to edges instead of panic
        assert_eq!(cache.get(1, 1), 22); // Clamps to (2,2)
        assert_eq!(cache.get(10, 10), 44); // Clamps to (4,4)
    }
}
