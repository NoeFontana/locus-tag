use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};


#[derive(Debug, Clone)]
pub struct ImageGroundTruth {
    #[allow(dead_code)]
    pub filename: String,
    pub tag_ids: HashSet<u32>,
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
        eprintln!("Strict Warning: LOCUS_DATASET_DIR set to {:?} but it doesn't look like an ICRA2020 dataset (no tags.csv or forward/ folder).", path);
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
/// 
/// The CSV is expected to have columns: `filename,tag_id,...` logic depends on actual file format.
/// Since I don't see the file, I'll implement a generic parser that aggregates by filename.
pub fn load_ground_truth(dataset_root: &Path) -> Option<HashMap<String, ImageGroundTruth>> {
    let csv_path = if dataset_root.join("tags.csv").exists() {
        dataset_root.join("tags.csv")
    } else {
        return None;
    };

    let mut map: HashMap<String, ImageGroundTruth> = HashMap::new();
    
    // Using the csv crate
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true) // Assumption/Constraint
        .from_path(csv_path)
        .expect("Failed to open tags.csv");

    // We assume headers like: image_name, tag_id, ...
    // If we don't know the format, we might need to inspect it. 
    // But per instructions: "Implement a parser for tags.csv... containing filename, tag_ids"
    
    // Let's try to handle standard "filename, class_id/tag_id" format.
    // We'll read loosely.
    
    for result in rdr.records() {
        let record = result.expect("Failed to parse CSV record");
        
        // Assumption: Column 0 is filename, Column 1 is tag_id
        if record.len() < 2 { continue; }
        
        let filename = record[0].to_string();
        let tag_id_str = &record[1];
        
        // Parse tag ID
        if let Ok(tag_id) = tag_id_str.trim().parse::<u32>() {
            map.entry(filename.clone())
                .or_insert_with(|| ImageGroundTruth {
                    filename,
                    tag_ids: HashSet::new(),
                })
                .tag_ids.insert(tag_id);
        }
    }
    
    Some(map)
}
