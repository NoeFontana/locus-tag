#![allow(clippy::redundant_closure_for_method_calls, clippy::unreadable_literal)]
//! EuRoC MAV calibration dataset loader.
//!
//! Provides real stereo images of a 6×6 AprilTag 36h11 grid captured with
//! a globally-shuttered MT9V034 sensor (752×480, pinhole + radtan distortion).
//!
//! The calibration sequence (`cam_april`) is part of the EuRoC MAV Dataset:
//! <https://projects.asl.ethz.ch/datasets/euroc-mav/>
//!
//! # Usage
//!
//! Download once:
//! ```text
//! bash scripts/fetch_euroc_calibration.sh
//! ```
//!
//! Then either set `LOCUS_EUROC_DATASET_DIR` or let the loader auto-discover
//! `tests/data/euroc/` from the workspace root.

use locus_core::TagFamily;
use locus_core::board::AprilGridTopology;
use locus_core::pose::CameraIntrinsics;
use std::collections::BTreeMap;
use std::path::PathBuf;

use super::hub::{DatasetItem, DatasetProvider, GroundTruth};

// ── Known EuRoC VI-Sensor cam0 calibration (from sensor.yaml) ─────────────

/// Focal length x (pixels).
pub const CAM0_FX: f64 = 458.654;
/// Focal length y (pixels).
pub const CAM0_FY: f64 = 457.296;
/// Principal point x (pixels).
pub const CAM0_CX: f64 = 367.215;
/// Principal point y (pixels).
pub const CAM0_CY: f64 = 248.375;
/// Radial distortion k1.
pub const CAM0_K1: f64 = -0.28340811;
/// Radial distortion k2.
pub const CAM0_K2: f64 = 0.07395907;
/// Tangential distortion p1.
pub const CAM0_P1: f64 = 0.00019359;
/// Tangential distortion p2.
pub const CAM0_P2: f64 = 1.76187114e-05;
/// Image width.
pub const CAM0_WIDTH: u32 = 752;
/// Image height.
pub const CAM0_HEIGHT: u32 = 480;

/// Physical AprilGrid target: tag side length (metres).
pub const TAG_SIZE: f64 = 0.088;
/// AprilGrid spacing ratio (gap = TAG_SPACING_RATIO * TAG_SIZE).
pub const TAG_SPACING_RATIO: f64 = 0.3;
/// Grid rows.
pub const GRID_ROWS: usize = 6;
/// Grid cols.
pub const GRID_COLS: usize = 6;

// ── Constructors ──────────────────────────────────────────────────────────

/// Build cam0 intrinsics **without** distortion (ideal pinhole).
/// Use this when testing detection only (no undistortion).
pub fn cam0_intrinsics_pinhole() -> CameraIntrinsics {
    CameraIntrinsics::new(CAM0_FX, CAM0_FY, CAM0_CX, CAM0_CY)
}

/// Build cam0 intrinsics **with** Brown-Conrady distortion.
#[cfg(feature = "non_rectified")]
pub fn cam0_intrinsics_distorted() -> CameraIntrinsics {
    CameraIntrinsics::with_brown_conrady(
        CAM0_FX, CAM0_FY, CAM0_CX, CAM0_CY, CAM0_K1, CAM0_K2, CAM0_P1, CAM0_P2,
        0.0, // k3 — EuRoC uses 4-param radtan
    )
}

/// Build the 6×6 AprilGrid topology matching the EuRoC calibration target.
pub fn aprilgrid_topology() -> AprilGridTopology {
    let gap = TAG_SPACING_RATIO * TAG_SIZE;
    AprilGridTopology::new(
        GRID_ROWS,
        GRID_COLS,
        gap,
        TAG_SIZE,
        TagFamily::AprilTag36h11.max_id_count(),
    )
    .expect("6×6 grid fits in 36h11 dictionary")
}

// ── Path resolution ───────────────────────────────────────────────────────

/// Resolve the EuRoC dataset root, mirroring the project convention for
/// ICRA/Hub datasets: explicit env var → auto-discover → None (skip tests).
pub fn resolve_euroc_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LOCUS_EUROC_DATASET_DIR") {
        let raw = PathBuf::from(&dir);
        let resolved = if raw.is_absolute() {
            raw
        } else {
            let ws = super::resolve_workspace_root();
            let from_ws = ws.join(&raw);
            if from_ws.is_dir() {
                std::fs::canonicalize(&from_ws).unwrap_or(from_ws)
            } else {
                raw
            }
        };
        if resolved.is_dir() {
            return Some(resolved);
        }
        panic!(
            "LOCUS_EUROC_DATASET_DIR='{}' is not a valid directory (resolved to '{}')",
            dir,
            resolved.display()
        );
    }

    // Auto-discover from workspace root
    let ws = super::resolve_workspace_root();
    let default = ws.join("tests/data/euroc");
    default.is_dir().then_some(default)
}

// ── Dataset provider ──────────────────────────────────────────────────────

/// Provides cam0 images from the EuRoC `cam_april` calibration sequence.
pub struct EurocProvider {
    name: String,
    image_paths: Vec<PathBuf>,
    intrinsics: CameraIntrinsics,
}

impl EurocProvider {
    /// Load cam0 images with a stride (1 = every frame, 10 = every 10th, etc.).
    ///
    /// Returns `None` when the dataset isn't available, allowing tests to skip.
    pub fn cam0(stride: usize) -> Option<Self> {
        let root = resolve_euroc_root()?;
        let cam0_data = root.join("cam_april/mav0/cam0/data");
        if !cam0_data.is_dir() {
            return None;
        }

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&cam0_data)
            .ok()?
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        paths.sort();

        let sampled: Vec<PathBuf> = paths.into_iter().step_by(stride).collect();

        Some(Self {
            name: format!("euroc_cam_april_cam0_stride{stride}"),
            image_paths: sampled,
            intrinsics: cam0_intrinsics_pinhole(),
        })
    }

    /// Same as [`cam0`] but uses distorted intrinsics (requires `non_rectified`).
    #[cfg(feature = "non_rectified")]
    pub fn cam0_distorted(stride: usize) -> Option<Self> {
        let mut provider = Self::cam0(stride)?;
        provider.intrinsics = cam0_intrinsics_distorted();
        provider.name = format!("euroc_cam_april_cam0_distorted_stride{stride}");
        Some(provider)
    }

    /// The camera intrinsics used by this provider.
    pub fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }

    /// Number of images that will be iterated.
    pub fn len(&self) -> usize {
        self.image_paths.len()
    }
}

impl DatasetProvider for EurocProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn iter(&self) -> Box<dyn Iterator<Item = DatasetItem> + '_> {
        let intrinsics = self.intrinsics;
        let iter = self.image_paths.iter().filter_map(move |path| {
            let img = image::open(path).ok()?.into_luma8();
            let (w, h) = img.dimensions();
            let fname = path.file_name()?.to_string_lossy().to_string();
            // EuRoC has no per-image ground truth corners — we provide the
            // camera intrinsics and tag size via the GroundTruth struct so
            // the RegressionHarness picks them up.
            let gt = GroundTruth {
                tags: BTreeMap::new(),
                poses: BTreeMap::new(),
                intrinsics: Some(intrinsics),
                tag_size: Some(TAG_SIZE),
            };
            Some((fname, img.into_raw(), w as usize, h as usize, gt))
        });
        Box::new(iter)
    }
}
