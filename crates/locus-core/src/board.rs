//! Board-level configuration and layout utilities.

use crate::batch::{DetectionBatch, MAX_CANDIDATES, Point2f};
use crate::pose::{CameraIntrinsics, Pose, projection_jacobian, symmetrize_jtj6};
use nalgebra::{Matrix2, Matrix6, UnitQuaternion, Vector3, Vector6};

// ── LO-RANSAC Configuration ────────────────────────────────────────────────

/// Configuration for LO-RANSAC board pose estimation.
///
/// These parameters are tuned for sub-pixel, metrology-grade perception on
/// structured fiducial grids. See the accompanying architectural specification
/// for the mathematical justification behind each constant.
#[derive(Clone, Debug)]
pub struct LoRansacConfig {
    /// Minimum RANSAC iterations before dynamic stopping is permitted.
    /// Guarantees the sampler escapes spatially-correlated occlusion clusters
    /// (e.g., a cable over 4 adjacent tags) before the stopping rule fires.
    pub k_min: u32,
    /// Hard ceiling on RANSAC iterations (worst-case execution time bound).
    /// Caps runtime at ~20 µs/frame even on fundamentally degraded images.
    pub k_max: u32,
    /// Desired probability of finding at least one clean minimal sample.
    pub confidence: f64,
    /// Outer inlier threshold, **squared** (pixels²).
    ///
    /// Wide net for the initial consensus gate: a 4-point IPPE model on
    /// sub-pixel noisy data can tilt, so this threshold is intentionally
    /// generous enough to admit all geometrically consistent tags even under
    /// per-tag pose noise before the LO GN step refines things.
    pub tau_outer_sq: f64,
    /// Inner inlier threshold, **squared** (pixels²).
    ///
    /// Tight gate applied *within* the LO Gauss-Newton inner loop.
    /// Anything outside ~1 px after GN smoothing is a Hamming alias,
    /// chromatic aberration, or physical occlusion — exclude from further GN.
    pub tau_inner_sq: f64,
    /// AW-LM inlier threshold, **squared** (pixels²).
    ///
    /// Applied to the LO-verified pose to build the inlier set passed to the
    /// final Anisotropic Weighted LM step.  This should be **generous** (≥ the
    /// outer threshold) so that AW-LM can exploit the maximum number of
    /// geometrically consistent observations; its internal Huber weighting
    /// (k = 1.345) robustly downweights any residual outliers.
    pub tau_aw_lm_sq: f64,
    /// Maximum iterations for the LO inner loop.
    ///
    /// Gauss-Newton converges quadratically; if the basin is not reached within
    /// 5 steps the solver is thrashing — terminate early.
    pub lo_max_iterations: u32,
    /// Minimum outer inlier count needed to trigger the LO inner loop.
    pub min_inliers: usize,
}

impl Default for LoRansacConfig {
    fn default() -> Self {
        Self {
            k_min: 15,
            k_max: 50,
            confidence: 0.9999,
            tau_outer_sq: 100.0, // 10² px²   — raw IPPE seeds have 5-10px board error; match original
            tau_inner_sq: 1.0,   // 1.0² px²  — tight gate inside LO GN loop
            tau_aw_lm_sq: 100.0, // 10² px²   — generous AW-LM input; Huber handles outliers
            lo_max_iterations: 5,
            min_inliers: 4,
        }
    }
}

// ── Board layout ───────────────────────────────────────────────────────────

/// Configuration for a fiducial marker board (ChAruco or AprilGrid).
#[derive(Clone, Debug)]
pub struct BoardConfig {
    /// Number of rows in the grid.
    pub rows: usize,
    /// Number of columns in the grid.
    pub cols: usize,
    /// Physical length of one side of a marker (meters).
    pub marker_length: f64,
    /// 3D object points for each tag ID, indexed by tag ID.
    /// Each entry contains 4 points: [TL, TR, BR, BL] in board-local coordinates.
    pub obj_points: Vec<Option<[[f64; 3]; 4]>>,
}

impl BoardConfig {
    /// Creates a new ChAruco board configuration.
    ///
    /// ChAruco boards have markers in squares where (row + col) is even.
    /// The origin (0,0,0) is at the geometric center of the board.
    #[must_use]
    pub fn new_charuco(rows: usize, cols: usize, square_length: f64, marker_length: f64) -> Self {
        let mut obj_points = vec![None; (rows * cols).div_ceil(2)];

        let total_width = cols as f64 * square_length;
        let total_height = rows as f64 * square_length;
        let offset_x = -total_width / 2.0;
        let offset_y = -total_height / 2.0;

        let marker_padding = (square_length - marker_length) / 2.0;

        let mut marker_idx = 0;
        for r in 0..rows {
            for c in 0..cols {
                if (r + c) % 2 == 0 {
                    let x = offset_x + c as f64 * square_length + marker_padding;
                    let y = offset_y + r as f64 * square_length + marker_padding;

                    let pts = [
                        [x, y, 0.0],
                        [x + marker_length, y, 0.0],
                        [x + marker_length, y + marker_length, 0.0],
                        [x, y + marker_length, 0.0],
                    ];

                    if marker_idx < obj_points.len() {
                        obj_points[marker_idx] = Some(pts);
                        marker_idx += 1;
                    }
                }
            }
        }

        Self {
            rows,
            cols,
            marker_length,
            obj_points,
        }
    }

    /// Creates a new AprilGrid board configuration.
    ///
    /// AprilGrids have markers in every cell, separated by spacing.
    /// The origin (0,0,0) is at the geometric center of the board.
    #[must_use]
    pub fn new_aprilgrid(rows: usize, cols: usize, spacing: f64, marker_length: f64) -> Self {
        let mut obj_points = vec![None; rows * cols];
        let step = marker_length + spacing;
        let board_width = cols as f64 * marker_length + (cols - 1) as f64 * spacing;
        let board_height = rows as f64 * marker_length + (rows - 1) as f64 * spacing;

        let offset_x = -board_width / 2.0;
        let offset_y = -board_height / 2.0;

        for r in 0..rows {
            for c in 0..cols {
                let x = offset_x + c as f64 * step;
                let y = offset_y + r as f64 * step;

                let pts = [
                    [x, y, 0.0],
                    [x + marker_length, y, 0.0],
                    [x + marker_length, y + marker_length, 0.0],
                    [x, y + marker_length, 0.0],
                ];

                let idx = r * cols + c;
                if idx < obj_points.len() {
                    obj_points[idx] = Some(pts);
                }
            }
        }

        Self {
            rows,
            cols,
            marker_length,
            obj_points,
        }
    }
}

// ── Result type ────────────────────────────────────────────────────────────

/// Result of a board pose estimation.
#[derive(Clone, Debug)]
pub struct BoardPose {
    /// The estimated 6-DOF pose.
    pub pose: Pose,
    /// The 6x6 pose covariance matrix in se(3) tangent space.
    /// Order: [tx, ty, tz, rx, ry, rz]
    pub covariance: Matrix6<f64>,
}

// ── Generic Correspondence Interface ──────────────────────────────────────

/// A flat, contiguous view of M 2D-3D point correspondences for pose estimation.
///
/// Points are organised in contiguous *groups* of [`group_size`](Self::group_size)
/// elements. The inlier bitmask operated on by [`RobustPoseSolver`] tracks one
/// bit *per group*, keeping the mask size bounded at 1 024 bits regardless of
/// `group_size`.
///
/// | Pipeline        | `group_size` | Bit semantics             |
/// |-----------------|--------------|---------------------------|
/// | AprilGrid       | 4            | 1 bit = 1 tag (4 corners) |
/// | ChAruco saddles | 1            | 1 bit = 1 saddle point    |
///
/// **Lifetime**: the slices are typically backed by pre-allocated scratch
/// buffers owned by [`BoardEstimator`] or by arena memory, ensuring zero heap
/// allocation on the hot path.
pub struct PointCorrespondences<'a> {
    /// Observed 2D image points (pixels). Length = M.
    pub image_points: &'a [Point2f],
    /// Corresponding 3D model points (board frame, metres). Length = M.
    pub object_points: &'a [[f64; 3]],
    /// Pre-inverted observation covariances (information matrices). Length = M.
    /// Must be positive semi-definite; use [`Matrix2::identity`] for isotropic
    /// unit weighting.
    pub information_matrices: &'a [Matrix2<f64>],
    /// Number of consecutive points forming one logical correspondence group.
    /// `M` must be an exact multiple of `group_size`.
    pub group_size: usize,
    /// Per-group seed pose hypotheses for RANSAC initialisation.
    /// Length = M / `group_size`.  `None` signals a degenerate or occluded
    /// group that RANSAC must skip as a minimal-sample candidate.
    pub seed_poses: &'a [Option<Pose>],
}

impl PointCorrespondences<'_> {
    /// Number of correspondence groups: `M / group_size`.
    #[inline]
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.image_points.len() / self.group_size
    }
}

// ── Robust Pose Solver ─────────────────────────────────────────────────────

/// Pure mathematical engine for robust, multi-correspondence board pose
/// estimation.
///
/// Completely decoupled from [`DetectionBatch`] and tag layout.  Accepts flat
/// [`PointCorrespondences`] slices and returns a verified [`BoardPose`].
///
/// **Algorithm**: LO-RANSAC (outer) → unweighted Gauss-Newton verification
/// (inner) → Anisotropic Weighted Levenberg-Marquardt final refinement.
#[derive(Default)]
pub struct RobustPoseSolver {
    /// LO-RANSAC hyper-parameters.
    pub lo_ransac: LoRansacConfig,
}

impl RobustPoseSolver {
    /// Creates a new solver with default LO-RANSAC parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: override the LO-RANSAC configuration.
    #[must_use]
    pub fn with_lo_ransac_config(mut self, cfg: LoRansacConfig) -> Self {
        self.lo_ransac = cfg;
        self
    }

    // ── Public entry point ───────────────────────────────────────────────

    /// Estimates a board pose from flat point correspondences.
    ///
    /// Returns `None` if fewer than 4 groups are present, or if LO-RANSAC
    /// cannot find a consensus set.
    #[must_use]
    pub fn estimate(
        &self,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
    ) -> Option<BoardPose> {
        if corr.num_groups() < 4 {
            return None;
        }

        // Phase 1: LO-RANSAC → verified IPPE seed + tight inlier count.
        let (best_pose, _tight_mask) = self.lo_ransac_loop(corr, intrinsics)?;

        // Phase 2: re-evaluate inliers at the generous tau_aw_lm on the
        // LO-verified pose.  The wider window maximises the AW-LM observation
        // count; Huber weighting (k = 1.345) handles any residual mild outliers.
        let (aw_lm_mask, _) =
            self.evaluate_inliers(&best_pose, corr, intrinsics, self.lo_ransac.tau_aw_lm_sq);

        // Phase 3: final AW-LM over the verified, relaxed inlier set.
        let (refined_pose, covariance) =
            self.refine_aw_lm(&best_pose, corr, intrinsics, &aw_lm_mask);

        Some(BoardPose {
            pose: refined_pose,
            covariance,
        })
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Core LO-RANSAC loop.
    ///
    /// Outer loop: random 4-group sampling → seed pose → outer-threshold
    /// evaluation.  Inner loop (LO): unweighted Gauss-Newton refinement +
    /// tight re-evaluation with monotonicity guard.
    /// Dynamic stopping: `k` is updated after each tight-count improvement using
    /// the standard RANSAC formula `k = log(1-p) / log(1-ω⁴)` where `ω` is the
    /// verified tight inlier ratio from `lo_inner`.
    #[allow(clippy::too_many_lines, clippy::cast_sign_loss)]
    fn lo_ransac_loop(
        &self,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
    ) -> Option<(Pose, [u64; 16])> {
        let cfg = &self.lo_ransac;
        let num_groups = corr.num_groups();

        let mut global_best_tight_count = 0usize;
        let mut global_best_seed: Option<Pose> = None;
        let mut dynamic_k = cfg.k_max;

        // Deterministic XOR-shift RNG (reproducible across frames).
        let mut rng_seed = 0x1337u32;

        for iter in 0..cfg.k_max {
            // ── Draw 4 distinct groups without replacement ────────────────
            let mut sample = [0usize; 4];
            let mut found = 0usize;
            let mut attempts = 0u32;
            while found < 4 && attempts < 1000 {
                attempts += 1;
                rng_seed ^= rng_seed << 13;
                rng_seed ^= rng_seed >> 17;
                rng_seed ^= rng_seed << 5;
                let s = (rng_seed as usize) % num_groups;
                if !sample[..found].contains(&s) {
                    sample[found] = s;
                    found += 1;
                }
            }
            if found < 4 {
                continue;
            }

            // ── Try each sampled group's seed pose as a hypothesis ────────
            let mut best_outer_count = 0usize;
            let mut best_outer_mask = [0u64; 16];
            let mut best_outer_pose: Option<Pose> = None;

            for &s_val in &sample {
                let Some(pose_init) = corr.seed_poses[s_val] else {
                    continue;
                };

                let (outer_mask, outer_count) =
                    self.evaluate_inliers(&pose_init, corr, intrinsics, cfg.tau_outer_sq);

                if outer_count >= cfg.min_inliers && outer_count > best_outer_count {
                    best_outer_count = outer_count;
                    best_outer_mask = outer_mask;
                    best_outer_pose = Some(pose_init);
                }
            }

            let Some(seed_pose) = best_outer_pose else {
                continue;
            };

            // ── LO inner loop (verification gate) ────────────────────────
            // The unweighted GN pose produced inside lo_inner is discarded
            // (spec mandate) to prevent biasing the AW-LM initialisation.
            // Only tight_count governs global state and dynamic stopping.
            let (_gn_pose, _tight_mask, tight_count) =
                self.lo_inner(seed_pose, &best_outer_mask, corr, intrinsics);

            if tight_count > global_best_tight_count {
                global_best_tight_count = tight_count;
                global_best_seed = Some(seed_pose);

                let inlier_ratio = tight_count as f64 / num_groups as f64;
                if inlier_ratio >= 0.99 {
                    dynamic_k = cfg.k_min;
                } else {
                    let p_fail = 1.0 - cfg.confidence;
                    let p_good_sample = 1.0 - inlier_ratio.powi(4);
                    let k_compute = p_fail.ln() / p_good_sample.ln();
                    dynamic_k = (k_compute.max(0.0).ceil() as u32).clamp(cfg.k_min, cfg.k_max);
                }
            }

            // ── Bounded early termination ─────────────────────────────────
            if iter >= cfg.k_min && iter >= dynamic_k {
                break;
            }
        }

        let final_seed = global_best_seed?;
        Some((final_seed, [0u64; 16]))
    }

    /// LO inner loop: iteratively refines the pose with unweighted Gauss-Newton
    /// and re-evaluates inliers with the tight `tau_inner` threshold.
    ///
    /// **Monotonicity guard:** if the tight inlier count stops improving,
    /// the GN has reached the basin bottom — terminate early.
    fn lo_inner(
        &self,
        seed_pose: Pose,
        outer_mask: &[u64; 16],
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
    ) -> (Pose, [u64; 16], usize) {
        let cfg = &self.lo_ransac;

        let (init_inner_mask, init_inner_count) =
            self.evaluate_inliers(&seed_pose, corr, intrinsics, cfg.tau_inner_sq);

        let mut lo_pose = seed_pose;
        // First GN step uses the outer (wider) inlier mask for better conditioning.
        let mut lo_gn_mask = *outer_mask;
        let mut prev_inner_count = init_inner_count;

        let mut best_pose = seed_pose;
        let mut best_mask = init_inner_mask;
        let mut best_count = init_inner_count;

        for _lo_iter in 0..cfg.lo_max_iterations {
            let new_pose = self.gn_step(&lo_pose, corr, intrinsics, &lo_gn_mask);

            let (new_inner_mask, new_inner_count) =
                self.evaluate_inliers(&new_pose, corr, intrinsics, cfg.tau_inner_sq);

            // Monotonicity guard: tight consensus must strictly grow.
            if new_inner_count <= prev_inner_count {
                break;
            }

            prev_inner_count = new_inner_count;
            lo_pose = new_pose;
            // Subsequent GN steps operate on the tight inlier set.
            lo_gn_mask = new_inner_mask;
            best_pose = new_pose;
            best_mask = new_inner_mask;
            best_count = new_inner_count;
        }

        (best_pose, best_mask, best_count)
    }

    /// Projects all correspondence groups and classifies each group as an
    /// inlier if its mean squared reprojection error is below `tau_sq`.
    ///
    /// The threshold comparison avoids `sqrt`:
    /// `sum_sq < group_size * tau_sq  ⟺  mean_sq < tau_sq`.
    ///
    /// Returns a 1 024-bit group-level inlier bitmask (16 × u64) and the
    /// inlier count.
    ///
    /// If *any* point in a group has near-zero camera depth the whole group is
    /// rejected immediately (break + `valid_group = false`).  This differs from
    /// `gn_step` / `refine_aw_lm` which silently `continue` past degenerate
    /// points: those solvers accumulate whatever non-degenerate corners remain,
    /// whereas the inlier test must be conservative — a partial error sum would
    /// undercount reprojection error and admit a bad pose as an inlier.
    #[allow(clippy::unused_self)]
    fn evaluate_inliers(
        &self,
        pose: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        tau_sq: f64,
    ) -> ([u64; 16], usize) {
        let mut mask = [0u64; 16];
        let mut count = 0usize;
        let num_groups = corr.num_groups();
        let gs = corr.group_size;

        for g in 0..num_groups {
            let start = g * gs;
            let threshold = gs as f64 * tau_sq;

            let mut sum_sq = 0.0f64;
            let mut valid_group = true;

            for k in start..(start + gs) {
                let obj = corr.object_points[k];
                let p_world = Vector3::new(obj[0], obj[1], obj[2]);
                let p_cam = pose.rotation * p_world + pose.translation;
                if p_cam.z < 1e-4 {
                    valid_group = false;
                    break;
                }
                let z_inv = 1.0 / p_cam.z;
                let px = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
                let py = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;
                let dx = px - f64::from(corr.image_points[k].x);
                let dy = py - f64::from(corr.image_points[k].y);
                sum_sq += dx * dx + dy * dy;
            }

            if valid_group && sum_sq < threshold {
                count += 1;
                mask[g / 64] |= 1 << (g % 64);
            }
        }

        (mask, count)
    }

    /// One step of **unweighted** Gauss-Newton pose refinement over inlier groups.
    ///
    /// Solves `(J^T J) δ = J^T r` with the left-perturbation SE(3) Jacobian.
    /// No Marquardt damping, no information-matrix weighting.
    ///
    /// Returns the original pose unchanged if the normal equations are singular.
    #[allow(clippy::unused_self)]
    fn gn_step(
        &self,
        pose: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        inlier_mask: &[u64; 16],
    ) -> Pose {
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        let gs = corr.group_size;
        let num_groups = corr.num_groups();

        for g in 0..num_groups {
            if (inlier_mask[g / 64] & (1 << (g % 64))) == 0 {
                continue;
            }
            let start = g * gs;

            for k in start..(start + gs) {
                let obj = corr.object_points[k];
                let p_world = Vector3::new(obj[0], obj[1], obj[2]);
                let p_cam = pose.rotation * p_world + pose.translation;
                if p_cam.z < 1e-4 {
                    continue;
                }
                let z_inv = 1.0 / p_cam.z;
                let x_z = p_cam.x * z_inv;
                let y_z = p_cam.y * z_inv;

                let u = intrinsics.fx * x_z + intrinsics.cx;
                let v = intrinsics.fy * y_z + intrinsics.cy;

                let res_u = f64::from(corr.image_points[k].x) - u;
                let res_v = f64::from(corr.image_points[k].y) - v;

                // Left-perturbation SE(3) Jacobian (scalar accumulation — no
                // intermediate Matrix2x6; mirrors build_normal_equations in
                // pose_weighted.rs with identity information matrix / w=1).
                let (ju0, ju2, ju3, ju4, ju5, jv1, jv2, jv3, jv4, jv5) =
                    projection_jacobian(x_z, y_z, z_inv, intrinsics);

                jtr[0] += ju0 * res_u;
                jtr[1] += jv1 * res_v;
                jtr[2] += ju2 * res_u + jv2 * res_v;
                jtr[3] += ju3 * res_u + jv3 * res_v;
                jtr[4] += ju4 * res_u + jv4 * res_v;
                jtr[5] += ju5 * res_u + jv5 * res_v;

                jtj[(0, 0)] += ju0 * ju0;
                jtj[(0, 2)] += ju0 * ju2;
                jtj[(0, 3)] += ju0 * ju3;
                jtj[(0, 4)] += ju0 * ju4;
                jtj[(0, 5)] += ju0 * ju5;

                jtj[(1, 1)] += jv1 * jv1;
                jtj[(1, 2)] += jv1 * jv2;
                jtj[(1, 3)] += jv1 * jv3;
                jtj[(1, 4)] += jv1 * jv4;
                jtj[(1, 5)] += jv1 * jv5;

                jtj[(2, 2)] += ju2 * ju2 + jv2 * jv2;
                jtj[(2, 3)] += ju2 * ju3 + jv2 * jv3;
                jtj[(2, 4)] += ju2 * ju4 + jv2 * jv4;
                jtj[(2, 5)] += ju2 * ju5 + jv2 * jv5;

                jtj[(3, 3)] += ju3 * ju3 + jv3 * jv3;
                jtj[(3, 4)] += ju3 * ju4 + jv3 * jv4;
                jtj[(3, 5)] += ju3 * ju5 + jv3 * jv5;

                jtj[(4, 4)] += ju4 * ju4 + jv4 * jv4;
                jtj[(4, 5)] += ju4 * ju5 + jv4 * jv5;

                jtj[(5, 5)] += ju5 * ju5 + jv5 * jv5;
            }
        }
        symmetrize_jtj6(&mut jtj);

        // Solve the normal equations; return original pose if system is singular.
        if let Some(chol) = jtj.cholesky() {
            let delta = chol.solve(&jtr);
            let twist = Vector3::new(delta[3], delta[4], delta[5]);
            let dq = UnitQuaternion::from_scaled_axis(twist);
            Pose {
                rotation: (dq * UnitQuaternion::from_matrix(&pose.rotation))
                    .to_rotation_matrix()
                    .into_inner(),
                translation: pose.translation + Vector3::new(delta[0], delta[1], delta[2]),
            }
        } else {
            *pose
        }
    }

    /// Final refinement: Anisotropic Weighted Levenberg-Marquardt over the
    /// verified inlier set.
    ///
    /// Uses pre-inverted information matrices from `corr.information_matrices`
    /// and Huber weighting (k = 1.345) for robustness against mild outliers
    /// inside the wide `tau_aw_lm` window.
    #[allow(clippy::too_many_lines, clippy::similar_names, clippy::unused_self)]
    fn refine_aw_lm(
        &self,
        initial_pose: &Pose,
        corr: &PointCorrespondences<'_>,
        intrinsics: &CameraIntrinsics,
        inlier_mask: &[u64; 16],
    ) -> (Pose, Matrix6<f64>) {
        let mut pose = *initial_pose;
        let mut lambda = 1e-3;
        let mut nu = 2.0;

        let gs = corr.group_size;
        let num_groups = corr.num_groups();

        let compute_equations = |current_pose: &Pose| -> (f64, Matrix6<f64>, Vector6<f64>) {
            let mut jtj = Matrix6::<f64>::zeros();
            let mut jtr = Vector6::<f64>::zeros();
            let mut total_cost = 0.0;

            for g in 0..num_groups {
                if (inlier_mask[g / 64] & (1 << (g % 64))) == 0 {
                    continue;
                }
                let start = g * gs;

                for k in start..(start + gs) {
                    let obj = corr.object_points[k];
                    let p_world = Vector3::new(obj[0], obj[1], obj[2]);
                    let p_cam = current_pose.rotation * p_world + current_pose.translation;
                    if p_cam.z < 1e-4 {
                        total_cost += 1e6;
                        continue;
                    }
                    let z_inv = 1.0 / p_cam.z;
                    let x_z = p_cam.x * z_inv;
                    let y_z = p_cam.y * z_inv;

                    let u = intrinsics.fx * x_z + intrinsics.cx;
                    let v = intrinsics.fy * y_z + intrinsics.cy;

                    let res_u = f64::from(corr.image_points[k].x) - u;
                    let res_v = f64::from(corr.image_points[k].y) - v;

                    let info = corr.information_matrices[k];

                    let dist_sq = res_u * (info[(0, 0)] * res_u + info[(0, 1)] * res_v)
                        + res_v * (info[(1, 0)] * res_u + info[(1, 1)] * res_v);

                    let huber_k = 1.345;
                    let dist = dist_sq.sqrt();
                    let weight = if dist > huber_k { huber_k / dist } else { 1.0 };
                    total_cost += if dist > huber_k {
                        huber_k * (dist - 0.5 * huber_k)
                    } else {
                        0.5 * dist_sq
                    };

                    let (ju0, ju2, ju3, ju4, ju5, jv1, jv2, jv3, jv4, jv5) =
                        projection_jacobian(x_z, y_z, z_inv, intrinsics);

                    let w00 = info[(0, 0)] * weight;
                    let w01 = info[(0, 1)] * weight;
                    let w10 = info[(1, 0)] * weight;
                    let w11 = info[(1, 1)] * weight;

                    let k00 = ju0 * w00;
                    let k01 = ju0 * w01;
                    let k10 = jv1 * w10;
                    let k11 = jv1 * w11;
                    let k20 = ju2 * w00 + jv2 * w10;
                    let k21 = ju2 * w01 + jv2 * w11;
                    let k30 = ju3 * w00 + jv3 * w10;
                    let k31 = ju3 * w01 + jv3 * w11;
                    let k40 = ju4 * w00 + jv4 * w10;
                    let k41 = ju4 * w01 + jv4 * w11;
                    let k50 = ju5 * w00 + jv5 * w10;
                    let k51 = ju5 * w01 + jv5 * w11;

                    jtr[0] += k00 * res_u + k01 * res_v;
                    jtr[1] += k10 * res_u + k11 * res_v;
                    jtr[2] += k20 * res_u + k21 * res_v;
                    jtr[3] += k30 * res_u + k31 * res_v;
                    jtr[4] += k40 * res_u + k41 * res_v;
                    jtr[5] += k50 * res_u + k51 * res_v;

                    jtj[(0, 0)] += k00 * ju0;
                    jtj[(0, 1)] += k01 * jv1;
                    jtj[(0, 2)] += k00 * ju2 + k01 * jv2;
                    jtj[(0, 3)] += k00 * ju3 + k01 * jv3;
                    jtj[(0, 4)] += k00 * ju4 + k01 * jv4;
                    jtj[(0, 5)] += k00 * ju5 + k01 * jv5;

                    jtj[(1, 1)] += k11 * jv1;
                    jtj[(1, 2)] += k10 * ju2 + k11 * jv2;
                    jtj[(1, 3)] += k10 * ju3 + k11 * jv3;
                    jtj[(1, 4)] += k10 * ju4 + k11 * jv4;
                    jtj[(1, 5)] += k10 * ju5 + k11 * jv5;

                    jtj[(2, 2)] += k20 * ju2 + k21 * jv2;
                    jtj[(2, 3)] += k20 * ju3 + k21 * jv3;
                    jtj[(2, 4)] += k20 * ju4 + k21 * jv4;
                    jtj[(2, 5)] += k20 * ju5 + k21 * jv5;

                    jtj[(3, 3)] += k30 * ju3 + k31 * jv3;
                    jtj[(3, 4)] += k30 * ju4 + k31 * jv4;
                    jtj[(3, 5)] += k30 * ju5 + k31 * jv5;

                    jtj[(4, 4)] += k40 * ju4 + k41 * jv4;
                    jtj[(4, 5)] += k40 * ju5 + k41 * jv5;

                    jtj[(5, 5)] += k50 * ju5 + k51 * jv5;
                }
            }
            symmetrize_jtj6(&mut jtj);
            (total_cost, jtj, jtr)
        };

        let (mut cur_cost, mut cur_jtj, mut cur_jtr) = compute_equations(&pose);

        for _iter in 0..20 {
            if cur_jtr.amax() < 1e-8 {
                break;
            }

            let mut jtj_damped = cur_jtj;
            let diag = cur_jtj.diagonal();
            for i in 0..6 {
                jtj_damped[(i, i)] += lambda * (diag[i] + 1e-6);
            }

            if let Some(chol) = jtj_damped.cholesky() {
                let delta = chol.solve(&cur_jtr);
                let twist = Vector3::new(delta[3], delta[4], delta[5]);
                let dq = UnitQuaternion::from_scaled_axis(twist);
                let new_pose = Pose {
                    rotation: (dq * UnitQuaternion::from_matrix(&pose.rotation))
                        .to_rotation_matrix()
                        .into_inner(),
                    translation: pose.translation + Vector3::new(delta[0], delta[1], delta[2]),
                };

                let (new_cost, new_jtj, new_jtr) = compute_equations(&new_pose);
                let rho = (cur_cost - new_cost)
                    / (0.5 * delta.dot(&(lambda * delta + cur_jtr)).max(1e-12));

                if rho > 0.0 {
                    pose = new_pose;
                    cur_cost = new_cost;
                    cur_jtj = new_jtj;
                    cur_jtr = new_jtr;
                    lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
                    nu = 2.0;
                    if delta.norm() < 1e-7 {
                        break;
                    }
                } else {
                    lambda *= nu;
                    nu *= 2.0;
                }
            } else {
                lambda *= 10.0;
            }
        }

        let covariance = cur_jtj.try_inverse().unwrap_or_else(Matrix6::zeros);
        (pose, covariance)
    }
}

// ── Board Estimator (AprilGrid adapter) ────────────────────────────────────

/// Estimator for multi-tag board poses.
///
/// Bridges the [`DetectionBatch`] SoA layout and [`BoardConfig`] tag geometry
/// with the tag-layout-agnostic [`RobustPoseSolver`].  All heavy pose
/// mathematics lives in the solver; this struct is responsible only for
/// constructing the flat [`PointCorrespondences`] view and retaining the
/// pre-allocated scratch buffers needed to do so without heap allocation.
pub struct BoardEstimator {
    /// Configuration of the board layout.
    pub config: BoardConfig,
    /// The underlying robust pose solver (contains LO-RANSAC config).
    pub solver: RobustPoseSolver,
    // ── Pre-allocated scratch buffers (single heap allocation in new()) ──────
    // img/obj/info are per-point: MAX_CORR = MAX_CANDIDATES × CORNERS_PER_TAG.
    // seeds are per-group: MAX_CANDIDATES (one seed pose per tag, not per corner).
    scratch_img: Box<[Point2f]>,
    scratch_obj: Box<[[f64; 3]]>,
    scratch_info: Box<[Matrix2<f64>]>,
    scratch_seeds: Box<[Option<Pose>]>,
}

impl BoardEstimator {
    const CORNERS_PER_TAG: usize = 4;
    const MAX_CORR: usize = MAX_CANDIDATES * Self::CORNERS_PER_TAG;

    /// Creates a new `BoardEstimator` with default LO-RANSAC parameters.
    ///
    /// Performs a single one-time heap allocation to back the scratch buffers.
    /// Reuse the same `BoardEstimator` across frames to amortise this cost and
    /// guarantee zero per-`estimate()` allocations.
    #[must_use]
    pub fn new(config: BoardConfig) -> Self {
        Self {
            config,
            solver: RobustPoseSolver::new(),
            scratch_img: vec![Point2f { x: 0.0, y: 0.0 }; Self::MAX_CORR].into_boxed_slice(),
            scratch_obj: vec![[0.0f64; 3]; Self::MAX_CORR].into_boxed_slice(),
            scratch_info: vec![Matrix2::zeros(); Self::MAX_CORR].into_boxed_slice(),
            scratch_seeds: vec![None; MAX_CANDIDATES].into_boxed_slice(),
        }
    }

    /// Builder: override the LO-RANSAC configuration.
    #[must_use]
    pub fn with_lo_ransac_config(mut self, cfg: LoRansacConfig) -> Self {
        self.solver.lo_ransac = cfg;
        self
    }

    // ── Public entry point ───────────────────────────────────────────────

    /// Estimates the board pose from a batch of detections.
    ///
    /// Returns `None` if fewer than 4 valid tags match the board layout or if
    /// LO-RANSAC cannot find a consensus set.
    #[must_use]
    pub fn estimate(
        &mut self,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
    ) -> Option<BoardPose> {
        // Phase 1: flatten valid batch entries into the pre-allocated scratch
        // slices, inverting covariances and gathering IPPE seed poses.
        let num_groups = self.flatten_batch(batch);
        if num_groups < 4 {
            return None;
        }

        let m = num_groups * Self::CORNERS_PER_TAG;
        let corr = PointCorrespondences {
            image_points: &self.scratch_img[..m],
            object_points: &self.scratch_obj[..m],
            information_matrices: &self.scratch_info[..m],
            group_size: Self::CORNERS_PER_TAG,
            seed_poses: &self.scratch_seeds[..num_groups],
        };

        // Phase 2–4: LO-RANSAC → GN verification → AW-LM refinement.
        self.solver.estimate(&corr, intrinsics)
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Scans the batch and writes all valid, board-matched tag data into the
    /// pre-allocated scratch buffers.
    ///
    /// Returns the number of tag groups written (i.e. the value of `num_groups`
    /// for the subsequent `PointCorrespondences`).
    fn flatten_batch(&mut self, batch: &DetectionBatch) -> usize {
        let mut g = 0usize;

        for i in 0..MAX_CANDIDATES {
            if batch.status_mask[i] != crate::batch::CandidateState::Valid {
                continue;
            }
            let id = batch.ids[i] as usize;
            if id >= self.config.obj_points.len() {
                continue;
            }
            let Some(obj) = self.config.obj_points[id] else {
                continue;
            };

            let base = g * Self::CORNERS_PER_TAG;
            for (j, obj_pt) in obj.iter().enumerate() {
                self.scratch_img[base + j] = batch.corners[i][j];
                self.scratch_obj[base + j] = *obj_pt;
                // Invert the per-corner covariance into an information matrix.
                // Fall back to identity if the covariance is singular.
                self.scratch_info[base + j] = Matrix2::new(
                    f64::from(batch.corner_covariances[i][j * 4]),
                    f64::from(batch.corner_covariances[i][j * 4 + 1]),
                    f64::from(batch.corner_covariances[i][j * 4 + 2]),
                    f64::from(batch.corner_covariances[i][j * 4 + 3]),
                )
                .try_inverse()
                .unwrap_or_else(Matrix2::identity);
            }

            // Compute the seed pose before writing to scratch_seeds to avoid
            // conflicting borrows of self within the same statement.
            let seed = self.init_pose_from_batch_tag(i, batch);
            self.scratch_seeds[g] = seed;
            g += 1;
        }

        g
    }

    /// Converts a single tag's stored per-tag `Pose6D` into a board-frame `Pose`.
    ///
    /// Returns `None` if the stored pose is degenerate (NaN or near-zero depth).
    fn init_pose_from_batch_tag(&self, b_idx: usize, batch: &DetectionBatch) -> Option<Pose> {
        let data = batch.poses[b_idx].data;
        if data.iter().any(|v| v.is_nan()) || data[2].abs() < 1e-6 {
            return None;
        }

        let det_t = Vector3::new(f64::from(data[0]), f64::from(data[1]), f64::from(data[2]));
        // Quaternion layout in Pose6D: [qx, qy, qz, qw] at indices [3,4,5,6].
        let det_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            f64::from(data[6]), // w
            f64::from(data[3]), // x
            f64::from(data[4]), // y
            f64::from(data[5]), // z
        ));

        let tag_id = batch.ids[b_idx] as usize;
        let tag_obj_origin = self.config.obj_points[tag_id]?[0];
        let tag_origin = Vector3::new(tag_obj_origin[0], tag_obj_origin[1], tag_obj_origin[2]);

        // Board-frame translation: t_board = t_tag - R * origin_of_tag_in_board
        Some(Pose {
            rotation: *det_q.to_rotation_matrix().matrix(),
            translation: det_t - (det_q * tag_origin),
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    missing_docs
)]
mod tests {
    use super::*;
    use crate::batch::{CandidateState, DetectionBatch, Point2f};
    use nalgebra::Matrix3;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Standard synthetic intrinsics used across tests.
    fn test_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0)
    }

    /// Collect all corner points from a `BoardConfig` into a flat `Vec`.
    fn all_corners(config: &BoardConfig) -> Vec<[f64; 3]> {
        config
            .obj_points
            .iter()
            .filter_map(|opt| *opt)
            .flat_map(std::iter::IntoIterator::into_iter)
            .collect()
    }

    /// Compute the centroid of a set of 3D points.
    fn centroid(pts: &[[f64; 3]]) -> [f64; 3] {
        let n = pts.len() as f64;
        let (sx, sy, sz) = pts.iter().fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
            (ax + p[0], ay + p[1], az + p[2])
        });
        [sx / n, sy / n, sz / n]
    }

    /// Build a `DetectionBatch` by projecting every board marker through `pose` and
    /// `intrinsics`.  Corners are stored as f32 (matching `Point2f`); per-tag pose
    /// data encodes the camera-space position of each tag's TL corner and the board
    /// rotation quaternion so that `init_pose_from_batch_tag` recovers `pose` exactly.
    /// Identity corner covariances are set so AW-LM applies isotropic weighting.
    fn build_synthetic_batch(
        config: &BoardConfig,
        pose: &Pose,
        intrinsics: &CameraIntrinsics,
    ) -> (DetectionBatch, usize) {
        let mut batch = DetectionBatch::new();
        let mut n = 0usize;

        let q = UnitQuaternion::from_matrix(&pose.rotation);

        for (tag_id, opt_pts) in config.obj_points.iter().enumerate() {
            let Some(obj) = opt_pts else { continue };

            // Project all 4 corners into the image.
            for (j, pt) in obj.iter().enumerate() {
                let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                let proj = pose.project(&p_world, intrinsics);
                batch.corners[n][j] = Point2f {
                    x: proj[0] as f32,
                    y: proj[1] as f32,
                };
            }

            // Per-tag pose: det_t = R * tl_board + t_board (camera-space TL corner).
            let tl = Vector3::new(obj[0][0], obj[0][1], obj[0][2]);
            let det_t = pose.rotation * tl + pose.translation;
            // Layout: [tx, ty, tz, qx, qy, qz, qw]
            batch.poses[n].data = [
                det_t.x as f32,
                det_t.y as f32,
                det_t.z as f32,
                q.i as f32,
                q.j as f32,
                q.k as f32,
                q.w as f32,
            ];

            // Identity 2×2 corner covariances → isotropic unit weighting for AW-LM.
            for j in 0..4 {
                batch.corner_covariances[n][j * 4] = 1.0; // (0,0)
                batch.corner_covariances[n][j * 4 + 3] = 1.0; // (1,1)
            }

            batch.ids[n] = tag_id as u32;
            batch.status_mask[n] = CandidateState::Valid;
            n += 1;
        }

        (batch, n)
    }

    /// Flatten the first `num_valid` batch entries into `PointCorrespondences`
    /// buffers using unit information matrices.
    ///
    /// Returns the four backing `Vec`s so the caller can keep them alive for the
    /// lifetime of the `PointCorrespondences` view.
    #[allow(clippy::type_complexity)]
    fn build_correspondences_from_batch(
        config: &BoardConfig,
        batch: &DetectionBatch,
        estimator: &BoardEstimator,
        num_valid: usize,
    ) -> (
        Vec<Point2f>,
        Vec<[f64; 3]>,
        Vec<Matrix2<f64>>,
        Vec<Option<Pose>>,
    ) {
        let mut img = Vec::with_capacity(num_valid * 4);
        let mut obj = Vec::with_capacity(num_valid * 4);
        let mut info = Vec::with_capacity(num_valid * 4);
        let mut seeds = Vec::with_capacity(num_valid);

        for b_idx in 0..num_valid {
            let id = batch.ids[b_idx] as usize;
            let pts = config.obj_points[id].unwrap();
            for (j, &obj_pt) in pts.iter().enumerate() {
                img.push(batch.corners[b_idx][j]);
                obj.push(obj_pt);
                // Use identity information matrices (unit covariance) for tests.
                info.push(Matrix2::identity());
            }
            seeds.push(estimator.init_pose_from_batch_tag(b_idx, batch));
        }

        (img, obj, info, seeds)
    }

    /// Compute the per-corner mean squared reprojection error (in pixel²) for the
    /// first `num_valid` candidates in the batch.
    fn mean_reprojection_sq(
        pose: &Pose,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
        config: &BoardConfig,
        num_valid: usize,
    ) -> f64 {
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for i in 0..num_valid {
            let id = batch.ids[i] as usize;
            let obj = config.obj_points[id].unwrap();
            for (j, pt) in obj.iter().enumerate() {
                let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                let proj = pose.project(&p_world, intrinsics);
                let dx = proj[0] - f64::from(batch.corners[i][j].x);
                let dy = proj[1] - f64::from(batch.corners[i][j].y);
                sum_sq += dx * dx + dy * dy;
                count += 1;
            }
        }
        sum_sq / count.max(1) as f64
    }

    // ── Board layout tests ────────────────────────────────────────────────────

    #[test]
    fn test_charuco_board_marker_count() {
        // 6×6 grid: markers appear in cells where (r+c) is even → exactly 18 markers.
        let config = BoardConfig::new_charuco(6, 6, 0.1, 0.08);
        let count = config.obj_points.iter().filter(|o| o.is_some()).count();
        assert_eq!(count, 18);
    }

    #[test]
    fn test_charuco_board_centroid_is_origin() {
        // For a symmetric ChAruco board the geometric centroid of all marker corners
        // must coincide with the board coordinate origin.
        let config = BoardConfig::new_charuco(6, 6, 0.1, 0.08);
        let pts = all_corners(&config);
        assert!(!pts.is_empty());
        let c = centroid(&pts);
        assert!(c[0].abs() < 1e-9, "centroid x = {}", c[0]);
        assert!(c[1].abs() < 1e-9, "centroid y = {}", c[1]);
        assert!(c[2].abs() < 1e-9, "all points must lie on z = 0");
    }

    #[test]
    fn test_charuco_corner_order_tl_tr_br_bl() {
        // For every marker the corners must follow the [TL, TR, BR, BL] order:
        //   TL.x < TR.x, TR.x == BR.x, BL.x == TL.x, BL.y > TL.y
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        for opt in &config.obj_points {
            let [tl, tr, br, bl] = opt.unwrap();
            assert!(tl[0] < tr[0], "TL.x must be left of TR.x");
            assert!(
                (tl[1] - tr[1]).abs() < 1e-9,
                "TL and TR must share the same y"
            );
            assert!(
                (tr[0] - br[0]).abs() < 1e-9,
                "TR and BR must share the same x"
            );
            assert!(
                (bl[0] - tl[0]).abs() < 1e-9,
                "BL and TL must share the same x"
            );
            assert!(
                bl[1] > tl[1],
                "BL.y must be below TL.y (y increases downward)"
            );
            assert!(
                (bl[1] - br[1]).abs() < 1e-9,
                "BL and BR must share the same y"
            );
            for pt in &[tl, tr, br, bl] {
                assert!(pt[2].abs() < 1e-9, "all corners must lie on z = 0");
            }
        }
    }

    #[test]
    fn test_charuco_marker_size_matches_config() {
        // Each marker's width and height must equal `marker_length`.
        let marker_length = 0.08;
        let config = BoardConfig::new_charuco(4, 4, 0.1, marker_length);
        for opt in &config.obj_points {
            let [tl, tr, _br, bl] = opt.unwrap();
            let width = tr[0] - tl[0];
            let height = bl[1] - tl[1];
            assert!((width - marker_length).abs() < 1e-9, "marker width {width}");
            assert!(
                (height - marker_length).abs() < 1e-9,
                "marker height {height}"
            );
        }
    }

    #[test]
    fn test_charuco_board_no_marker_overlap() {
        // No two markers may share a corner position.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let mut corners: Vec<[f64; 3]> = config
            .obj_points
            .iter()
            .filter_map(|o| *o)
            .flat_map(IntoIterator::into_iter)
            .collect();
        corners.sort_by(|a, b| {
            a[0].partial_cmp(&b[0])
                .unwrap()
                .then(a[1].partial_cmp(&b[1]).unwrap())
        });
        for w in corners.windows(2) {
            assert!(
                (w[0][0] - w[1][0]).abs() > 1e-9 || (w[0][1] - w[1][1]).abs() > 1e-9,
                "duplicate corner: {:?}",
                w[0]
            );
        }
    }

    #[test]
    fn test_aprilgrid_board_marker_count() {
        // AprilGrid: every cell has a marker.
        let config = BoardConfig::new_aprilgrid(4, 4, 0.01, 0.1);
        let count = config.obj_points.iter().filter(|o| o.is_some()).count();
        assert_eq!(count, 16);
    }

    #[test]
    fn test_aprilgrid_board_centroid_is_origin() {
        let config = BoardConfig::new_aprilgrid(6, 6, 0.01, 0.1);
        let pts = all_corners(&config);
        let c = centroid(&pts);
        assert!(c[0].abs() < 1e-9, "centroid x = {}", c[0]);
        assert!(c[1].abs() < 1e-9, "centroid y = {}", c[1]);
    }

    #[test]
    fn test_aprilgrid_adjacent_marker_step() {
        // Adjacent markers in the same row must be separated by marker_length + spacing.
        let marker_length = 0.1;
        let spacing = 0.02;
        let config = BoardConfig::new_aprilgrid(2, 3, spacing, marker_length);
        let step = marker_length + spacing;

        // Col 0 → col 1 within row 0.
        let tl0 = config.obj_points[0].unwrap()[0];
        let tl1 = config.obj_points[1].unwrap()[0];
        assert!(
            (tl1[0] - tl0[0] - step).abs() < 1e-9,
            "expected step {step}, got {}",
            tl1[0] - tl0[0]
        );

        // Row 0 → row 1 within col 0.
        let tl_r0 = config.obj_points[0].unwrap()[0];
        let tl_r1 = config.obj_points[3].unwrap()[0]; // row 1, col 0 → index = 1*3+0 = 3
        assert!(
            (tl_r1[1] - tl_r0[1] - step).abs() < 1e-9,
            "expected row step {step}, got {}",
            tl_r1[1] - tl_r0[1]
        );
    }

    // ── Mathematical correctness tests ────────────────────────────────────────

    #[test]
    fn test_evaluate_inliers_perfect_pose_all_inliers() {
        // Under the exact ground-truth pose the reprojection error is sub-pixel
        // (limited only by f32 quantisation ≈ 1e-5 px); tau_sq = 1.0 must admit all tags.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, num_valid) = build_synthetic_batch(&config, &pose, &intrinsics);

        let (img, obj, info, seeds) =
            build_correspondences_from_batch(&config, &batch, &estimator, num_valid);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
            seed_poses: &seeds,
        };

        let solver = RobustPoseSolver::new();
        let (_, count) = solver.evaluate_inliers(&pose, &corr, &intrinsics, 1.0);
        assert_eq!(
            count, num_valid,
            "all tags must be inliers under perfect pose"
        );
    }

    #[test]
    fn test_evaluate_inliers_bad_pose_no_inliers() {
        // A pose shifted 0.5 m in X produces ~250 px reprojection error;
        // even the generous tau_sq = 100 (10 px²) must reject all tags.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, num_valid) = build_synthetic_batch(&config, &true_pose, &intrinsics);

        let bad_pose = Pose::new(Matrix3::identity(), Vector3::new(0.5, 0.0, 1.0));
        let (img, obj, info, seeds) =
            build_correspondences_from_batch(&config, &batch, &estimator, num_valid);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
            seed_poses: &seeds,
        };

        let solver = RobustPoseSolver::new();
        let (_, count) = solver.evaluate_inliers(&bad_pose, &corr, &intrinsics, 100.0);
        assert_eq!(
            count, 0,
            "no tags should survive under a heavily shifted pose"
        );
    }

    #[test]
    fn test_evaluate_inliers_inlier_mask_consistency() {
        // The bitmask returned by evaluate_inliers must have exactly `count` bits set.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, num_valid) = build_synthetic_batch(&config, &pose, &intrinsics);

        let (img, obj, info, seeds) =
            build_correspondences_from_batch(&config, &batch, &estimator, num_valid);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
            seed_poses: &seeds,
        };

        let solver = RobustPoseSolver::new();
        let (mask, count) = solver.evaluate_inliers(&pose, &corr, &intrinsics, 1.0);

        let bits_set: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        assert_eq!(
            bits_set, count,
            "bitmask popcount must equal reported count"
        );
    }

    #[test]
    fn test_init_pose_from_batch_tag_recovers_board_pose() {
        // init_pose_from_batch_tag must reconstruct the board pose from any single
        // tag's stored per-tag pose data.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, num_valid) = build_synthetic_batch(&config, &true_pose, &intrinsics);

        for b_idx in 0..num_valid {
            let pose = estimator
                .init_pose_from_batch_tag(b_idx, &batch)
                .expect("tag must produce a valid pose");
            let t_error = (pose.translation - true_pose.translation).norm();
            assert!(
                t_error < 1e-5,
                "tag {b_idx}: translation error {t_error} m exceeds 10 µm"
            );
        }
    }

    #[test]
    fn test_init_pose_from_batch_tag_nan_returns_none() {
        // A tag whose stored pose contains NaN must yield None.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let mut batch = DetectionBatch::new();
        batch.ids[0] = 0;
        batch.poses[0].data = [f32::NAN; 7];
        assert!(estimator.init_pose_from_batch_tag(0, &batch).is_none());
    }

    #[test]
    fn test_init_pose_from_batch_tag_near_zero_depth_returns_none() {
        // A tag at near-zero depth (Z ≈ 0) is degenerate and must yield None.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let mut batch = DetectionBatch::new();
        batch.ids[0] = 0;
        batch.poses[0].data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // z = 0
        assert!(estimator.init_pose_from_batch_tag(0, &batch).is_none());
    }

    #[test]
    fn test_gn_step_reduces_reprojection_error() {
        // A single unweighted Gauss-Newton step from a 2 cm offset must strictly
        // reduce the mean squared reprojection error.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, num_valid) = build_synthetic_batch(&config, &true_pose, &intrinsics);

        let perturbed = Pose::new(Matrix3::identity(), Vector3::new(0.02, 0.0, 1.0));
        let (img, obj, info, seeds) =
            build_correspondences_from_batch(&config, &batch, &estimator, num_valid);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
            seed_poses: &seeds,
        };
        // Mark all groups as inliers.
        let all_inliers = [u64::MAX; 16];

        let solver = RobustPoseSolver::new();
        let before = mean_reprojection_sq(&perturbed, &batch, &intrinsics, &config, num_valid);
        let stepped = solver.gn_step(&perturbed, &corr, &intrinsics, &all_inliers);
        let after = mean_reprojection_sq(&stepped, &batch, &intrinsics, &config, num_valid);

        assert!(
            after < before,
            "GN step must reduce error: {before:.6} → {after:.6} px²"
        );
    }

    #[test]
    fn test_gn_step_singular_returns_original() {
        // With no inliers the normal equations are all-zero (singular);
        // gn_step must return the input pose unchanged.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, num_valid) = build_synthetic_batch(&config, &pose, &intrinsics);

        let (img, obj, info, seeds) =
            build_correspondences_from_batch(&config, &batch, &estimator, num_valid);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
            seed_poses: &seeds,
        };
        let no_inliers = [0u64; 16];

        let solver = RobustPoseSolver::new();
        let result = solver.gn_step(&pose, &corr, &intrinsics, &no_inliers);
        assert!(
            (result.translation - pose.translation).norm() < 1e-12,
            "pose must be unchanged when normal equations are singular"
        );
    }

    #[test]
    fn test_refine_aw_lm_converges_from_small_offset() {
        // AW-LM from a 2 cm / 1 cm offset must converge to within 0.1 mm of the true
        // translation, and the covariance diagonal must be non-negative.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, num_valid) = build_synthetic_batch(&config, &true_pose, &intrinsics);

        let perturbed = Pose::new(Matrix3::identity(), Vector3::new(0.02, -0.01, 1.0));
        let (img, obj, info, seeds) =
            build_correspondences_from_batch(&config, &batch, &estimator, num_valid);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
            seed_poses: &seeds,
        };
        let all_inliers = [u64::MAX; 16];

        let solver = RobustPoseSolver::new();
        let (refined, cov) = solver.refine_aw_lm(&perturbed, &corr, &intrinsics, &all_inliers);

        let t_error = (refined.translation - true_pose.translation).norm();
        assert!(
            t_error < 1e-4,
            "translation error {t_error} m exceeds 0.1 mm"
        );

        for i in 0..6 {
            assert!(
                cov[(i, i)] >= 0.0,
                "covariance diagonal [{i},{i}] must be non-negative"
            );
        }
    }

    #[test]
    fn test_refine_aw_lm_covariance_is_symmetric() {
        // The returned covariance (J^T J)^{-1} must be symmetric.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, num_valid) = build_synthetic_batch(&config, &pose, &intrinsics);

        let (img, obj, info, seeds) =
            build_correspondences_from_batch(&config, &batch, &estimator, num_valid);
        let corr = PointCorrespondences {
            image_points: &img,
            object_points: &obj,
            information_matrices: &info,
            group_size: 4,
            seed_poses: &seeds,
        };
        let all_inliers = [u64::MAX; 16];

        let solver = RobustPoseSolver::new();
        let (_, cov) = solver.refine_aw_lm(&pose, &corr, &intrinsics, &all_inliers);

        for i in 0..6 {
            for j in (i + 1)..6 {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < 1e-12,
                    "covariance must be symmetric: [{i},{j}]={} ≠ [{j},{i}]={}",
                    cov[(i, j)],
                    cov[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_estimate_none_with_fewer_than_four_valid_tags() {
        // estimate() must return None when fewer than 4 board-matched tags are present.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let mut estimator = BoardEstimator::new(config);
        let intrinsics = test_intrinsics();

        for n_valid in 0..4 {
            let mut batch = DetectionBatch::new();
            for i in 0..n_valid {
                batch.ids[i] = i as u32;
                batch.status_mask[i] = CandidateState::Valid;
            }
            assert!(
                estimator.estimate(&batch, &intrinsics).is_none(),
                "expected None with {n_valid} valid tags"
            );
        }
    }

    #[test]
    fn test_estimate_end_to_end_recovers_translation() {
        // End-to-end: synthesise all markers of a 4×4 ChAruco board from a known pose
        // and verify that estimate() recovers the pose to within 1 mm / 0.1°.
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let mut estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let true_pose = Pose::new(Matrix3::identity(), Vector3::new(0.05, -0.03, 1.5));
        let (batch, _) = build_synthetic_batch(&config, &true_pose, &intrinsics);

        let result = estimator.estimate(&batch, &intrinsics);
        assert!(
            result.is_some(),
            "estimate() must succeed with all tags visible"
        );

        let board_pose = result.unwrap();
        let t_error = (board_pose.pose.translation - true_pose.translation).norm();
        assert!(t_error < 1e-3, "translation error {t_error} m exceeds 1 mm");

        let est_q = UnitQuaternion::from_matrix(&board_pose.pose.rotation);
        let true_q = UnitQuaternion::from_matrix(&true_pose.rotation);
        let r_error = est_q.angle_to(&true_q).to_degrees();
        assert!(r_error < 0.1, "rotation error {r_error}° exceeds 0.1°");
    }

    #[test]
    fn test_estimate_covariance_is_positive_definite() {
        // The covariance returned alongside a valid estimate must have a positive
        // diagonal (positive semi-definite is sufficient for a well-conditioned scene).
        let config = BoardConfig::new_charuco(4, 4, 0.1, 0.08);
        let mut estimator = BoardEstimator::new(config.clone());
        let intrinsics = test_intrinsics();
        let pose = Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0));
        let (batch, _) = build_synthetic_batch(&config, &pose, &intrinsics);

        let result = estimator.estimate(&batch, &intrinsics).unwrap();
        for i in 0..6 {
            assert!(
                result.covariance[(i, i)] > 0.0,
                "covariance diagonal [{i},{i}] must be positive"
            );
        }
    }
}
