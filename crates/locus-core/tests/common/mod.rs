#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::panic,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ImageGroundTruth {
    pub tag_ids: HashSet<u32>,
    pub corners: HashMap<u32, [[f64; 2]; 4]>,
}

/// Resolves the workspace root from the crate root.
pub fn resolve_workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
}

/// Resolves the root directory of the ICRA 2020 dataset.
///
/// When `LOCUS_ICRA_DATASET_DIR` is set, the path is honoured verbatim
/// (resolved against the workspace root if relative, mirroring
/// `resolve_hub_root`); a non-directory value panics rather than silently
/// falling back so an explicit user request never short-circuits to a
/// stub fixture. When the env var is unset, the function tries the
/// in-tree fixture and the workspace-relative dataset in turn and
/// returns `None` if neither exists — that branch is what lets ad-hoc
/// `cargo test` runs skip cleanly on machines without the dataset.
pub fn resolve_dataset_root() -> Option<PathBuf> {
    if let Ok(path_str) = env::var("LOCUS_ICRA_DATASET_DIR") {
        let raw = PathBuf::from(&path_str);
        let resolved = if raw.is_absolute() {
            raw
        } else {
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let from_workspace = manifest.join("../../").join(&raw);
            if from_workspace.is_dir() {
                std::fs::canonicalize(&from_workspace).unwrap_or(from_workspace)
            } else {
                raw
            }
        };
        if resolved.is_dir() {
            return Some(resolved);
        }
        panic!(
            "LOCUS_ICRA_DATASET_DIR='{}' is not a valid directory (resolved to '{}')",
            path_str,
            resolved.display()
        );
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("tests/fixtures/icra2020"),
        manifest_dir.join("../../tests/data/icra2020"),
    ];

    for p in &candidates {
        if p.is_dir() {
            return Some(p.clone());
        }
    }

    None
}

/// Resolves the hub dataset root directory.
///
/// Anchors relative paths to the crate manifest dir so they work regardless of CWD.
pub fn resolve_hub_root(hub_dir_raw: &str) -> PathBuf {
    let path = PathBuf::from(hub_dir_raw);
    if path.is_absolute() {
        return path;
    }
    // Relative: resolve against workspace root first (CARGO_MANIFEST_DIR is the
    // crate root at crates/locus-core/, so ../../ reaches the workspace root).
    // This handles the common case: LOCUS_HUB_DATASET_DIR=tests/data/hub_cache.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let from_workspace = manifest.join("../../").join(&path);
    if from_workspace.is_dir() {
        return std::fs::canonicalize(&from_workspace).unwrap_or(from_workspace);
    }

    // Also try directly (works if the caller passed an already-correct relative path
    // or when CWD happens to be the workspace root).
    if path.is_dir() {
        return std::fs::canonicalize(&path).unwrap_or(path);
    }

    path // fallback — will fail on `exists()` with a clear skip message
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

pub mod hub;
pub mod telemetry;
