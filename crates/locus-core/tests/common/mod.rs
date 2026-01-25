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
/// This function is "lazy": it returns the path if the directory exists,
/// without validating specific contents. This allows it to support various
/// dataset layouts (split CSVs, master CSVs, etc.).
pub fn resolve_dataset_root() -> Option<PathBuf> {
    // 1. Trust the Environment Variable explicitly
    if let Ok(path_str) = env::var("LOCUS_DATASET_DIR") {
        let path = PathBuf::from(path_str);
        if path.is_dir() {
            return Some(path);
        }
        eprintln!(
            "Warning: LOCUS_DATASET_DIR is set to {} but it is not a directory.",
            path.display()
        );
    }

    // 2. Relative path fallback (Standard Repo Layout)
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Try standard locations relative to crate root
    let candidates = [
        manifest_dir.join("../../tests/data/icra2020"),
        manifest_dir.join("tests/data/icra2020"),
    ];

    for p in &candidates {
        if p.is_dir() {
            return Some(p.clone());
        }
    }

    None
}

/// Smartly loads ground truth.
///
/// Priority:
/// 1. Subfolder-specific GT: `{root}/{subfolder}/tags.csv`
/// 2. Master GT: `{root}/tags.csv`
pub fn load_ground_truth(
    dataset_root: &Path,
    subfolder: &str,
) -> Option<HashMap<String, ImageGroundTruth>> {
    // Strategy: Look for the most specific data first
    let candidates = [
        dataset_root.join(subfolder).join("tags.csv"), // Subfolder specific
        dataset_root.join("tags.csv"),                 // Dataset root master
    ];

    let csv_path = candidates.iter().find(|p| p.exists())?;

    let mut map: HashMap<String, ImageGroundTruth> = HashMap::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)
        .ok()?; // Use ok()? to propagate errors as None for simplicity

    for result in rdr.records() {
        let record = result.ok()?;
        if record.len() < 5 {
            continue;
        }

        let filename = record[0].to_string();
        // Parse fields...
        let tag_id = record[1].trim().parse::<u32>().ok()?;
        let corner_idx = record[2].trim().parse::<usize>().ok()?;
        if corner_idx >= 4 {
            continue;
        }

        let gx = record[3].trim().parse::<f64>().ok()?;
        let gy = record[4].trim().parse::<f64>().ok()?;

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
