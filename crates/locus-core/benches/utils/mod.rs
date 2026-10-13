#![allow(clippy::expect_used, clippy::unwrap_used, dead_code)]
use locus_core::bench_api::{CandidateState, DetectionBatch, Point2f};
use serde::de::DeserializeOwned;
use std::path::PathBuf;

/// The hub dataset cache: `$LOCUS_HUB_DATASET_DIR`, else the repository's
/// `tests/data/hub_cache`, anchored at the manifest (a bench's working directory is the package
/// root, not the repository root).
#[must_use]
pub fn hub_dir() -> PathBuf {
    std::env::var("LOCUS_HUB_DATASET_DIR").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/hub_cache"),
        PathBuf::from,
    )
}

/// The records of a hub dataset's `rich_truth.json`, in either schema: v1 is a bare list of
/// records, v2 wraps it as `{"version", "records", ...}`.
///
/// # Panics
/// Panics if the file is missing or its records do not parse as `T`.
#[must_use]
pub fn rich_truth_records<T: DeserializeOwned>(dataset: &str) -> Vec<T> {
    #[derive(serde::Deserialize)]
    #[serde(untagged, bound(deserialize = "T: DeserializeOwned"))]
    enum RichTruth<T> {
        V2 { records: Vec<T> },
        V1(Vec<T>),
    }
    let path = hub_dir().join(dataset).join("rich_truth.json");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("{}: {e}", path.display()))
        .unwrap();
    match serde_json::from_reader(std::io::BufReader::new(file)).unwrap() {
        RichTruth::V2 { records } | RichTruth::V1(records) => records,
    }
}

/// A utility struct to hold image data loaded from the ICRA 2020 dataset for benchmarking.
pub struct BenchDataset {
    /// Raw luma8 image data.
    pub raw_data: Vec<u8>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl BenchDataset {
    /// Loads a specific frame from the ICRA 2020 dataset for benchmarking.
    ///
    /// # Panics
    /// Panics if the dataset is not found at the expected location.
    #[must_use]
    pub fn load_icra_frame(subset: &str, frame_idx: usize) -> Self {
        let filename = format!("{frame_idx:04}.png");
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../tests/data/icra2020");
        path.push(subset);
        path.push("pure_tags_images");
        path.push(&filename);

        assert!(
            path.exists(),
            "Benchmark dataset frame not found at {}. \
             Ensure the ICRA 2020 dataset is available in tests/data/.",
            path.display()
        );

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
    ///
    /// # Panics
    /// Panics if the dataset is not found at the expected location.
    #[must_use]
    pub fn icra_forward_0() -> Self {
        Self::load_icra_frame("forward", 0)
    }

    /// Loads, resizes, and returns an ICRA frame for multi-resolution benchmarking.
    /// Uses Bicubic (CatmullRom) interpolation for high-fidelity frequency content.
    ///
    /// # Panics
    /// Panics if the dataset is not found at the expected location.
    #[must_use]
    pub fn load_and_resize_icra_frame(
        subset: &str,
        frame_idx: usize,
        width: usize,
        height: usize,
    ) -> Self {
        let filename = format!("{frame_idx:04}.png");
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../tests/data/icra2020");
        path.push(subset);
        path.push("pure_tags_images");
        path.push(&filename);

        assert!(
            path.exists(),
            "Benchmark dataset frame not found at {}. \
             Ensure the ICRA 2020 dataset is available in tests/data/.",
            path.display()
        );

        let img = image::open(&path)
            .expect("Failed to open ICRA image")
            .to_luma8();

        // Use CatmullRom (Bicubic) for high-quality resizing
        let resized = image::imageops::resize(
            &img,
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        Self {
            raw_data: resized.into_raw(),
            width,
            height,
        }
    }

    /// Generates a realistic DetectionBatch for late-stage benchmarking.
    ///
    /// Populates `num_valid` candidates with Valid state and `num_active` with Active state.
    #[must_use]
    pub fn generate_bench_batch(num_valid: usize, num_active: usize) -> Box<DetectionBatch> {
        let mut batch = DetectionBatch::new_boxed();
        let total = (num_valid + num_active).min(1024);

        for i in 0..total {
            // Provide some realistic-looking quad corners.
            let base_x = (i % 32) as f32 * 40.0 + 20.0;
            let base_y = (i / 32) as f32 * 40.0 + 20.0;

            batch.corners[i] = [
                Point2f {
                    x: base_x,
                    y: base_y,
                },
                Point2f {
                    x: base_x + 10.0,
                    y: base_y,
                },
                Point2f {
                    x: base_x + 10.0,
                    y: base_y + 10.0,
                },
                Point2f {
                    x: base_x,
                    y: base_y + 10.0,
                },
            ];

            if i < num_valid {
                batch.status_mask[i] = CandidateState::Valid;
                batch.ids[i] = i as u32;
                batch.payloads[i] = 0xDEAD_BEEF + i as u64;
            } else {
                batch.status_mask[i] = CandidateState::Active;
            }
        }
        batch
    }
}
