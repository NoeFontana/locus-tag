use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ImageGroundTruth {
    pub tag_ids: HashSet<u32>,
    pub corners: HashMap<u32, [[f64; 2]; 4]>,
}

/// Resolves the root directory of the ICRA 2020 dataset.
///
/// Priority:
/// 1. `LOCUS_DATASET_DIR` environment variable.
/// 2. `tests/data/icra2020` relative to workspace root.
///
/// Returns `None` if the dataset cannot be found (test should skip).
pub fn resolve_dataset_root() -> Option<PathBuf> {
    // 1. Environment variable
    if let Ok(path_str) = env::var("LOCUS_DATASET_DIR") {
        let path = PathBuf::from(path_str);
        if path.exists() {
            // Check for tags.csv or expected subfolders
            if path.join("tags.csv").exists() || path.join("forward").exists() {
                return Some(path);
            }
        }
        eprintln!(
            "Strict Warning: LOCUS_DATASET_DIR set to {:?} but it doesn't look like an ICRA2020 dataset (no tags.csv or forward/ folder).",
            path
        );
    }

    // 2. Relative path fallback
    // CARGO_MANIFEST_DIR is crates/locus-core
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Assuming repo root is 2 levels up
    let candidates = [
        manifest_dir.join("../../tests/data/icra2020"),
        manifest_dir.join("tests/data/icra2020"),
    ];

    for p in &candidates {
        if p.exists() {
            // If tags.csv exists, great.
            if p.join("tags.csv").exists() {
                return Some(p.to_path_buf());
            }
            // If standard subdirs exist, accept it too.
            if p.join("forward").exists() || p.join("rotation").exists() {
                return Some(p.to_path_buf());
            }
        }
    }

    None
}

/// Loads the ground truth `tags.csv` into a map of Filename -> GroundTruth.
pub fn load_ground_truth(dataset_root: &Path) -> Option<HashMap<String, ImageGroundTruth>> {
    let csv_path = if dataset_root.join("tags.csv").exists() {
        dataset_root.join("tags.csv")
    } else {
        return None;
    };

    let mut map: HashMap<String, ImageGroundTruth> = HashMap::new();

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)
        .expect("Failed to open tags.csv");

    for result in rdr.records() {
        let record = result.expect("Failed to parse CSV record");

        // image,tag_id,corner,ground_truth_x,ground_truth_y,...
        if record.len() < 5 {
            continue;
        }

        let filename = record[0].to_string();
        let tag_id = match record[1].trim().parse::<u32>() {
            Ok(id) => id,
            Err(_) => continue,
        };
        let corner_idx = match record[2].trim().parse::<usize>() {
            Ok(idx) if idx < 4 => idx,
            _ => continue,
        };
        let gx = match record[3].trim().parse::<f64>() {
            Ok(x) => x,
            Err(_) => continue,
        };
        let gy = match record[4].trim().parse::<f64>() {
            Ok(y) => y,
            Err(_) => continue,
        };

        let entry = map.entry(filename).or_insert_with(|| ImageGroundTruth {
            tag_ids: HashSet::new(),
            corners: HashMap::new(),
        });

        entry.tag_ids.insert(tag_id);
        let corners = entry.corners.entry(tag_id).or_insert([[0.0; 2]; 4]);
        corners[corner_idx] = [gx, gy];
    }

    Some(map)
}
