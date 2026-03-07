use std::path::{Path, PathBuf};
use image::GenericImageView;
use locus_core::batch::{DetectionBatch, CandidateState, Point2f};

pub struct BenchDataset {
    pub raw_data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl BenchDataset {
    /// Loads a specific frame from the ICRA 2020 dataset for benchmarking.
    /// 
    /// # Panics
    /// Panics if the dataset is not found at the expected location.
    pub fn load_icra_frame(subset: &str, frame_idx: usize) -> Self {
        let filename = format!("{:04}.png", frame_idx);
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../tests/data/icra2020");
        path.push(subset);
        path.push("pure_tags_images");
        path.push(&filename);

        if !path.exists() {
            panic!(
                "Benchmark dataset frame not found at {:?}. \
                 Ensure the ICRA 2020 dataset is available in tests/data/.",
                path
            );
        }

        let img = image::open(&path)
            .expect("Failed to open ICRA image")
            .to_luma8();
        let (width, height) = img.dimensions();

        Self {
            raw_data: img.into_raw(),
            width: width as usize,
            height: height as usize,
        }
    }

    /// Convenience loader for the first 'forward' frame.
    pub fn icra_forward_0() -> Self {
        Self::load_icra_frame("forward", 0)
    }

    /// Generates a realistic DetectionBatch for late-stage benchmarking.
    /// 
    /// Populates `num_valid` candidates with Valid state and `num_active` with Active state.
    pub fn generate_bench_batch(num_valid: usize, num_active: usize) -> DetectionBatch {
        let mut batch = DetectionBatch::new();
        let total = (num_valid + num_active).min(1024);

        for i in 0..total {
            // Provide some realistic-looking quad corners.
            let base_x = (i % 32) as f32 * 40.0 + 20.0;
            let base_y = (i / 32) as f32 * 40.0 + 20.0;
            
            batch.corners[i] = [
                Point2f { x: base_x, y: base_y },
                Point2f { x: base_x + 10.0, y: base_y },
                Point2f { x: base_x + 10.0, y: base_y + 10.0 },
                Point2f { x: base_x, y: base_y + 10.0 },
            ];

            if i < num_valid {
                batch.status_mask[i] = CandidateState::Valid;
                batch.ids[i] = i as u32;
                batch.payloads[i] = 0xDEADBEEF + i as u64;
            } else {
                batch.status_mask[i] = CandidateState::Active;
            }
        }
        batch
    }
}
