pub mod decoder;
pub mod gradient;
pub mod image;
pub mod quad;
pub mod segmentation;
pub mod test_utils;
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

#[derive(Default, Debug, Clone)]
pub struct PipelineStats {
    pub threshold_ms: f64,
    pub segmentation_ms: f64,
    pub quad_extraction_ms: f64,
    pub decoding_ms: f64,
    pub total_ms: f64,
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
        self.detect_with_stats(img).0
    }

    /// Detection with detailed timing statistics.
    pub fn detect_with_stats(&mut self, img: &ImageView) -> (Vec<Detection>, PipelineStats) {
        let mut stats = PipelineStats::default();
        let start_total = std::time::Instant::now();

        self.arena.reset();

        // 1. Thresholding
        let start_thresh = std::time::Instant::now();
        let tile_stats = self.threshold_engine.compute_tile_stats(img);
        let binarized = self
            .arena
            .alloc_slice_fill_copy(img.width * img.height, 0u8);
        self.threshold_engine
            .apply_threshold(img, &tile_stats, binarized);
        stats.threshold_ms = start_thresh.elapsed().as_secs_f64() * 1000.0;

        // 2. Segmentation (Connected Components with stats)
        let start_seg = std::time::Instant::now();
        let label_result = crate::segmentation::label_components_with_stats(
            &self.arena,
            binarized,
            img.width,
            img.height,
        );
        stats.segmentation_ms = start_seg.elapsed().as_secs_f64() * 1000.0;

        // 3. Quad Fitting (Fast path with pre-filtering)
        let start_quad = std::time::Instant::now();
        let candidates = crate::quad::extract_quads_fast(&self.arena, img, &label_result);
        stats.quad_extraction_ms = start_quad.elapsed().as_secs_f64() * 1000.0;

        // 4. Decoding (Single-threaded for low latency on small candidate sets)
        let start_decode = std::time::Instant::now();
        let src_points = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

        let mut final_detections = Vec::new();
        for mut cand in candidates {
            if let Some(h) = crate::decoder::Homography::from_pairs(&src_points, &cand.corners) {
                for decoder in &self.decoders {
                    let dim = decoder.dimension();
                    let grid_dim = (dim + 2) as f64;
                    let mut bits = 0u64;

                    for y in 0..dim {
                        for x in 0..dim {
                            let tx = -1.0 + 2.0 * (x as f64 + 1.5) / grid_dim;
                            let ty = -1.0 + 2.0 * (y as f64 + 1.5) / grid_dim;

                            let img_p = h.project([tx, ty]);
                            let val = img.sample_bilinear(img_p[0], img_p[1]);

                            if val > 128.0 {
                                bits |= 1 << (y * dim + x);
                            }
                        }
                    }

                    if let Some((id, hamming)) = decoder.decode(bits) {
                        cand.id = id;
                        cand.hamming = hamming;
                        final_detections.push(cand);
                        break;
                    }
                }
            }
        }
        stats.decoding_ms = start_decode.elapsed().as_secs_f64() * 1000.0;
        stats.total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        (final_detections, stats)
    }

    /// Fast detection using decimation (2x downsampled).
    pub fn detect_gradient(&mut self, img: &ImageView) -> Vec<Detection> {
        // Decimate the image 2x
        let new_w = img.width / 2;
        let new_h = img.height / 2;

        // Simple 2x2 averaging downsample
        let mut downsampled = vec![0u8; new_w * new_h];
        for y in 0..new_h {
            for x in 0..new_w {
                let p00 = img.get_pixel(x * 2, y * 2) as u16;
                let p10 = img.get_pixel(x * 2 + 1, y * 2) as u16;
                let p01 = img.get_pixel(x * 2, y * 2 + 1) as u16;
                let p11 = img.get_pixel(x * 2 + 1, y * 2 + 1) as u16;
                downsampled[y * new_w + x] = ((p00 + p10 + p01 + p11) / 4) as u8;
            }
        }

        let view =
            ImageView::new(&downsampled, new_w, new_h, new_w).expect("Valid downsampled image");

        // Run detection on downsampled image
        let mut detections = self.detect(&view);

        // Scale corners back to original resolution
        for det in &mut detections {
            det.center[0] *= 2.0;
            det.center[1] *= 2.0;
            for corner in &mut det.corners {
                corner[0] *= 2.0;
                corner[1] *= 2.0;
            }
        }

        detections
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
