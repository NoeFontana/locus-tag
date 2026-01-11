pub mod image;
pub mod quad;
pub mod segmentation;
pub mod threshold;

use crate::image::ImageView;
use crate::threshold::ThresholdEngine;
use bumpalo::Bump;

pub struct Detection {
    pub id: u32,
    pub center: [f64; 2],
    pub corners: [[f64; 2]; 4],
    pub hamming: u32,
    pub decision_margin: f64,
}

pub struct Detector {
    arena: Bump,
    threshold_engine: ThresholdEngine,
}

impl Detector {
    /// Create a new detector instance.
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
            threshold_engine: ThresholdEngine::new(),
        }
    }

    /// Primary detection entry point.
    pub fn detect(&mut self, img: &ImageView) -> Vec<Detection> {
        self.arena.reset();

        // 1. Thresholding
        let stats = self.threshold_engine.compute_tile_stats(img);

        // Allocate binarized image in arena (zero-copy/no-temp-vec)
        let binarized = self
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);
        self.threshold_engine
            .apply_threshold(img, &stats, binarized);

        // 2. Segmentation (Connected Components)
        let labels =
            crate::segmentation::label_components(&self.arena, binarized, img.width, img.height);

        // 3. Quad Fitting
        crate::quad::extract_quads(&self.arena, img, labels)
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

pub fn core_info() -> String {
    "Locus Core v0.1.0 Engine".to_string()
}
