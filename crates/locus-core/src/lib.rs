pub mod decoder;
pub mod image;
pub mod quad;
pub mod segmentation;
pub mod threshold;

use crate::decoder::TagDecoder;
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
    decoders: Vec<Box<dyn TagDecoder + Send + Sync>>,
}

impl Detector {
    /// Create a new detector instance.
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
            threshold_engine: ThresholdEngine::new(),
            decoders: vec![Box::new(crate::decoder::AprilTag36h11)],
        }
    }

    /// Add a decoder to the pipeline.
    pub fn add_decoder(&mut self, decoder: Box<dyn TagDecoder + Send + Sync>) {
        self.decoders.push(decoder);
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
        let candidates = crate::quad::extract_quads(&self.arena, img, labels);

        // 4. Decoding
        let mut final_detections = Vec::new();

        let src_points = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

        for mut cand in candidates {
            if let Some(h) = crate::decoder::Homography::from_pairs(&src_points, &cand.corners) {
                for decoder in &self.decoders {
                    // Sample bits
                    let dim = decoder.dimension() + 2; // +2 for black border
                    let mut bits = 0u64;

                    // Sample content grid
                    for y in 0..decoder.dimension() {
                        for x in 0..decoder.dimension() {
                            let tx = -1.0 + 2.0 * (x as f64 + 1.5) / dim as f64;
                            let ty = -1.0 + 2.0 * (y as f64 + 1.5) / dim as f64;

                            let img_p = h.project([tx, ty]);
                            let val = img.sample_bilinear(img_p[0], img_p[1]);

                            if val > 128.0 {
                                bits |= 1 << (y * decoder.dimension() + x);
                            }
                        }
                    }

                    if let Some((id, hamming)) = decoder.decode(bits) {
                        cand.id = id;
                        cand.hamming = hamming;
                        final_detections.push(cand);
                        break; // Stop after first matching decoder for this quad
                    }
                }
            }
        }

        final_detections
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
