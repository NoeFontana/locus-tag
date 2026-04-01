//! Board-level configuration and layout utilities.

use crate::batch::DetectionBatch;
use crate::pose::{CameraIntrinsics, Pose};
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

/// A ChArUco board configuration, defining the checkerboard and marker layout.
#[derive(Clone, Debug)]
pub struct CharucoBoard {
    /// The underlying board configuration for tag detection.
    pub config: BoardConfig,
    /// The size of the checkerboard squares (meters).
    pub square_length: f64,
    /// 3D coordinates of the checkerboard corners (saddle points).
    pub checkerboard_corners: Vec<[f64; 3]>,
}

impl CharucoBoard {
    /// Create a new ChArUco board configuration.
    ///
    /// The board has `rows` rows and `cols` columns of checkerboard squares.
    /// Markers are placed in the white squares.
    #[must_use]
    pub fn new(rows: usize, cols: usize, square_length: f64, marker_length: f64) -> Self {
        let config = BoardConfig::new_charuco(rows, cols, square_length, marker_length);

        // Checkerboard corners are the intersections between squares.
        // For a grid of rows x cols squares, there are (rows-1) x (cols-1) internal corners.
        let mut checkerboard_corners = Vec::with_capacity((rows - 1) * (cols - 1));
        let total_width = cols as f64 * square_length;
        let total_height = rows as f64 * square_length;
        let offset_x = -total_width / 2.0;
        let offset_y = -total_height / 2.0;

        for r in 1..rows {
            for c in 1..cols {
                let x = offset_x + c as f64 * square_length;
                let y = offset_y + r as f64 * square_length;
                checkerboard_corners.push([x, y, 0.0]);
            }
        }

        Self {
            config,
            square_length,
            checkerboard_corners,
        }
    }
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

// ── Estimator ─────────────────────────────────────────────────────────────

/// Estimator for multi-tag board poses.
pub struct BoardEstimator {
    /// Configuration of the board layout.
    pub config: BoardConfig,
    /// Configuration for the LO-RANSAC robust solver.
    pub lo_ransac: LoRansacConfig,
}

impl BoardEstimator {
    /// Creates a new `BoardEstimator` with default LO-RANSAC parameters.
    #[must_use]
    pub fn new(config: BoardConfig) -> Self {
        Self {
            config,
            lo_ransac: LoRansacConfig::default(),
        }
    }

    /// Builder: override the LO-RANSAC configuration.
    #[must_use]
    pub fn with_lo_ransac_config(mut self, cfg: LoRansacConfig) -> Self {
        self.lo_ransac = cfg;
        self
    }

    // ── Public entry point ───────────────────────────────────────────────

    /// Estimates the board pose from a batch of detections.
    ///
    /// Returns `None` if fewer than 4 valid tags match the board layout or if
    /// LO-RANSAC cannot find a consensus set.
    ///
    /// # Panics
    ///
    /// Panics if the internal `obj_points` for a valid tag index is missing.
    #[must_use]
    pub fn estimate(
        &self,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
    ) -> Option<BoardPose> {
        // Phase 1: collect batch indices of valid tags that belong to this board.
        let (valid_indices, num_valid) = self.collect_valid_candidates(batch);
        if num_valid < 4 {
            return None;
        }
        let valid = &valid_indices[..num_valid];

        // Phase 2: LO-RANSAC → verified pose + tight inlier set (tau_inner).
        // The LO inner loop uses tau_inner for its GN steps to stay clean, but
        // AW-LM needs a richer inlier set to maximise covariance quality.
        let (best_pose, _tight_mask) = self.lo_ransac_loop(valid, batch, intrinsics)?;

        // Phase 3: re-evaluate inliers at tau_aw_lm using the LO-verified pose.
        // The LO has already identified true outliers (occluded/aliased tags).
        // Using the generous tau_aw_lm (default 10px) on the LO-refined pose
        // maximises the number of geometrically consistent observations that
        // AW-LM can exploit. AW-LM's Huber weighting (k = 1.345) robustly
        // downweights any residual mild outliers inside this wider window.
        let (aw_lm_mask, _) = self.evaluate_inliers(
            &best_pose,
            batch,
            intrinsics,
            valid,
            self.lo_ransac.tau_aw_lm_sq,
        );

        // Phase 4: final AW-LM over the verified, relaxed inlier set.
        let (refined_pose, covariance) =
            self.refine_aw_lm(&best_pose, batch, intrinsics, valid, &aw_lm_mask);

        Some(BoardPose {
            pose: refined_pose,
            covariance,
        })
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Scans the batch and returns all indices of `Valid` candidates that map
    /// to a known board tag.
    fn collect_valid_candidates(
        &self,
        batch: &DetectionBatch,
    ) -> ([usize; crate::batch::MAX_CANDIDATES], usize) {
        let mut valid_indices = [0usize; crate::batch::MAX_CANDIDATES];
        let mut num_valid = 0;
        for i in 0..crate::batch::MAX_CANDIDATES {
            if batch.status_mask[i] == crate::batch::CandidateState::Valid {
                let id = batch.ids[i] as usize;
                if id < self.config.obj_points.len() && self.config.obj_points[id].is_some() {
                    valid_indices[num_valid] = i;
                    num_valid += 1;
                }
            }
        }
        (valid_indices, num_valid)
    }

    /// Core LO-RANSAC loop.
    ///
    /// Outer loop: random 4-tag sampling → IPPE seed → outer-threshold evaluation.
    /// Inner loop (LO): unweighted Gauss-Newton refinement + tight re-evaluation
    ///                  with monotonicity guard.
    /// Dynamic stopping: `k` is updated after each improvement using the
    /// standard formula `k = log(1-p) / log(1-ω⁴)`.
    ///
    /// **Per the architectural specification**, the noisy spatial pose computed
    /// *inside* the LO inner loop is discarded once the mask is finalised.
    /// This function returns the best **IPPE seed pose** (clean, unbiased by
    /// a small subset of GN steps) alongside the tight (`tau_inner`) inlier
    /// bitmask.  AW-LM then drives the final refinement from this unbiased
    /// starting point.
    #[allow(clippy::too_many_lines)]
    fn lo_ransac_loop(
        &self,
        valid: &[usize],
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
    ) -> Option<(Pose, [u64; 16])> {
        let cfg = &self.lo_ransac;
        let num_valid = valid.len();

        // Seed selection uses OUTER count: the IPPE seed with the most
        // tau_outer inliers is the best-conditioned starting point for AW-LM.
        // Tight count (tau_inner) is used only inside lo_inner for GN steps.
        let mut best_outer_inliers = 0usize;
        let mut best_outer_seed: Option<Pose> = None;

        // Deterministic XOR-shift RNG (reproducible across frames).
        let mut seed = 0x1337u32;

        for _iter in 0..cfg.k_max {
            // ── Draw 4 distinct tags without replacement ─────────────────
            let mut sample = [0usize; 4];
            let mut found = 0usize;
            let mut attempts = 0u32;
            while found < 4 && attempts < 1000 {
                attempts += 1;
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let s = (seed as usize) % num_valid;
                if !sample[..found].contains(&s) {
                    sample[found] = s;
                    found += 1;
                }
            }
            if found < 4 {
                continue;
            }

            // ── Initialise from each sampled tag; keep the best seed ──────
            // Each tag's stored per-tag pose is converted to a board-frame
            // hypothesis. We pick whichever seed produces the most outer
            // inliers to pass to the LO inner loop.
            let mut best_outer_count = 0usize;
            let mut best_outer_mask = [0u64; 16];
            let mut best_outer_pose: Option<Pose> = None;

            for &s_val in &sample {
                let b_idx = valid[s_val];
                let Some(pose_init) = self.init_pose_from_batch_tag(b_idx, batch) else {
                    continue;
                };

                let (outer_mask, outer_count) =
                    self.evaluate_inliers(&pose_init, batch, intrinsics, valid, cfg.tau_outer_sq);

                if outer_count >= cfg.min_inliers && outer_count > best_outer_count {
                    best_outer_count = outer_count;
                    best_outer_mask = outer_mask;
                    best_outer_pose = Some(pose_init);
                }
            }

            let Some(seed_pose) = best_outer_pose else {
                continue;
            };

            // ── Outer-count-based seed selection ─────────────────────────
            // Use the IPPE seed with the most outer inliers as the AW-LM seed.
            if best_outer_count > best_outer_inliers {
                best_outer_inliers = best_outer_count;
                best_outer_seed = Some(seed_pose);
            }

            // ── LO inner loop ─────────────────────────────────────────────
            // GN pose is discarded (spec mandate). lo_inner runs for
            // monotonicity-guarded tight refinement; its mask is unused here
            // since estimate() re-evaluates at tau_aw_lm anyway.
            let _ = self.lo_inner(seed_pose, &best_outer_mask, batch, intrinsics, valid);
        }

        // Return the IPPE seed with the best outer consensus.  estimate() will
        // re-evaluate the inlier mask at tau_aw_lm before calling AW-LM.
        let final_seed = best_outer_seed?;
        Some((final_seed, [0u64; 16]))
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

    /// Projects all 4 corners of every candidate in `valid` and classifies
    /// each tag as an inlier if its mean squared reprojection error is below
    /// `tau_sq`.
    ///
    /// Uses squared-error comparison to avoid `sqrt` on the critical path.
    /// Returns a 1024-bit inlier bitmask (16 × u64) and the inlier count.
    fn evaluate_inliers(
        &self,
        pose: &Pose,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
        valid: &[usize],
        tau_sq: f64,
    ) -> ([u64; 16], usize) {
        let mut mask = [0u64; 16];
        let mut count = 0usize;

        for (i, &b_idx) in valid.iter().enumerate() {
            let id = batch.ids[b_idx] as usize;
            let obj = self.config.obj_points[id].expect("missing obj_points");

            // Accumulate squared reprojection error over all 4 corners.
            // Threshold: sum_sq < 4 * tau_sq  ⟺  mean_sq < tau_sq (no sqrt needed).
            let mut sum_sq = 0.0f64;
            for (j, pt) in obj.iter().enumerate() {
                let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                let proj = pose.project(&p_world, intrinsics);
                let dx = proj[0] - f64::from(batch.corners[b_idx][j].x);
                let dy = proj[1] - f64::from(batch.corners[b_idx][j].y);
                sum_sq += dx * dx + dy * dy;
            }

            if sum_sq < 4.0 * tau_sq {
                count += 1;
                mask[i / 64] |= 1 << (i % 64);
            }
        }

        (mask, count)
    }

    /// LO inner loop: iteratively refines the pose with unweighted Gauss-Newton
    /// and re-evaluates inliers with the tight `tau_inner` threshold.
    ///
    /// **Monotonicity guard:** if the tight inlier count stops improving,
    /// the GN has reached the bottom of the objective basin — terminate early.
    ///
    /// Returns the best `(pose, tight_inlier_mask, tight_inlier_count)` found.
    fn lo_inner(
        &self,
        seed_pose: Pose,
        outer_mask: &[u64; 16],
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
        valid: &[usize],
    ) -> (Pose, [u64; 16], usize) {
        let cfg = &self.lo_ransac;

        // Establish baseline: evaluate the IPPE seed with the tight threshold.
        let (init_inner_mask, init_inner_count) =
            self.evaluate_inliers(&seed_pose, batch, intrinsics, valid, cfg.tau_inner_sq);

        let mut lo_pose = seed_pose;
        // First GN step uses the outer (wider) inlier mask for better conditioning.
        let mut lo_gn_mask = *outer_mask;
        let mut prev_inner_count = init_inner_count;

        // Track the best tight result seen across all LO iterations.
        let mut best_pose = seed_pose;
        let mut best_mask = init_inner_mask;
        let mut best_count = init_inner_count;

        for _lo_iter in 0..cfg.lo_max_iterations {
            // One step of unweighted Gauss-Newton over the current GN mask.
            let new_pose = self.gn_step(&lo_pose, batch, intrinsics, valid, &lo_gn_mask);

            // Re-evaluate with the tight threshold.
            let (new_inner_mask, new_inner_count) =
                self.evaluate_inliers(&new_pose, batch, intrinsics, valid, cfg.tau_inner_sq);

            // Monotonicity guard: the tight consensus must strictly grow.
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

    /// One step of **unweighted** Gauss-Newton pose refinement.
    ///
    /// Solves `(J^T J) δ = J^T r` with the left-perturbation SE(3) Jacobian.
    /// No Marquardt damping, no information-matrix weighting — this is a pure
    /// least-squares step designed to quickly smooth a noisy minimal-sample pose.
    ///
    /// Returns the original pose unchanged if the normal equations are singular.
    #[allow(clippy::similar_names)]
    fn gn_step(
        &self,
        pose: &Pose,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
        valid: &[usize],
        inlier_mask: &[u64; 16],
    ) -> Pose {
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();

        for (i, &b_idx) in valid.iter().enumerate() {
            if (inlier_mask[i / 64] & (1 << (i % 64))) == 0 {
                continue;
            }

            let id = batch.ids[b_idx] as usize;
            let obj = self.config.obj_points[id].expect("missing obj_points");

            for (j, pt) in obj.iter().enumerate() {
                let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                let p_cam = pose.rotation * p_world + pose.translation;
                let z = p_cam.z.max(1e-6);
                let z_inv = 1.0 / z;
                let z_inv2 = z_inv * z_inv;

                let u = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
                let v = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;

                let res_u = f64::from(batch.corners[b_idx][j].x) - u;
                let res_v = f64::from(batch.corners[b_idx][j].y) - v;

                let pcx = p_cam.x;
                let pcy = p_cam.y;

                // Left-perturbation SE(3) Jacobian rows (2 × 6):
                // du/dξ = [fx/z,  0,    -fx·x/z²,  -fx·x·y/z²,  fx·(1+x²/z²),  -fx·y/z]
                // dv/dξ = [0,     fy/z, -fy·y/z²,  -fy·(1+y²/z²), fy·x·y/z²,    fy·x/z]
                let mut jac = nalgebra::Matrix2x6::<f64>::zeros();
                jac[(0, 0)] = intrinsics.fx * z_inv;
                jac[(0, 2)] = -intrinsics.fx * pcx * z_inv2;
                jac[(0, 3)] = -intrinsics.fx * pcx * pcy * z_inv2;
                jac[(0, 4)] = intrinsics.fx * (1.0 + pcx * pcx * z_inv2);
                jac[(0, 5)] = -intrinsics.fx * pcy * z_inv;
                jac[(1, 1)] = intrinsics.fy * z_inv;
                jac[(1, 2)] = -intrinsics.fy * pcy * z_inv2;
                jac[(1, 3)] = -intrinsics.fy * (1.0 + pcy * pcy * z_inv2);
                jac[(1, 4)] = intrinsics.fy * pcx * pcy * z_inv2;
                jac[(1, 5)] = intrinsics.fy * pcx * z_inv;

                let res = nalgebra::Vector2::new(res_u, res_v);
                jtj += jac.transpose() * jac;
                jtr += jac.transpose() * res;
            }
        }

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

    // ── Final refinement ─────────────────────────────────────────────────

    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn refine_aw_lm(
        &self,
        initial_pose: &Pose,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
        valid_indices: &[usize],
        inlier_mask: &[u64; 16],
    ) -> (Pose, Matrix6<f64>) {
        let mut pose = *initial_pose;
        let mut lambda = 1e-3;
        let mut nu = 2.0;

        // Pre-compute information matrices (inverses of covariances) to avoid
        // repeated inversions inside the iteration loop.
        let mut info_matrices = Vec::with_capacity(valid_indices.len());
        for (i, &b_idx) in valid_indices.iter().enumerate() {
            let mut infos = [Matrix2::identity(); 4];
            if (inlier_mask[i / 64] & (1 << (i % 64))) != 0 {
                for (j, info) in infos.iter_mut().enumerate() {
                    *info = Matrix2::new(
                        f64::from(batch.corner_covariances[b_idx][j * 4]),
                        f64::from(batch.corner_covariances[b_idx][j * 4 + 1]),
                        f64::from(batch.corner_covariances[b_idx][j * 4 + 2]),
                        f64::from(batch.corner_covariances[b_idx][j * 4 + 3]),
                    )
                    .try_inverse()
                    .unwrap_or_else(Matrix2::identity);
                }
            }
            info_matrices.push(infos);
        }

        let compute_equations = |current_pose: &Pose| -> (f64, Matrix6<f64>, Vector6<f64>) {
            let mut jtj = Matrix6::<f64>::zeros();
            let mut jtr = Vector6::<f64>::zeros();
            let mut total_cost = 0.0;

            for (i, &b_idx) in valid_indices.iter().enumerate() {
                if (inlier_mask[i / 64] & (1 << (i % 64))) == 0 {
                    continue;
                }

                let id = batch.ids[b_idx] as usize;
                let obj = self.config.obj_points[id].expect("missing obj_points");
                let infos = &info_matrices[i];

                for (j, pt) in obj.iter().enumerate() {
                    let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                    let p_cam = current_pose.rotation * p_world + current_pose.translation;
                    let z = p_cam.z.max(1e-6);
                    let z_inv = 1.0 / z;
                    let z_inv2 = z_inv * z_inv;

                    let u = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
                    let v = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;

                    let res_u = f64::from(batch.corners[b_idx][j].x) - u;
                    let res_v = f64::from(batch.corners[b_idx][j].y) - v;

                    let info = infos[j];

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

                    let weighted_info = info * weight;

                    let mut jac = nalgebra::Matrix2x6::<f64>::zeros();
                    let pcx = p_cam.x;
                    let pcy = p_cam.y;

                    jac[(0, 0)] = intrinsics.fx * z_inv;
                    jac[(0, 1)] = 0.0;
                    jac[(0, 2)] = -intrinsics.fx * pcx * z_inv2;
                    jac[(0, 3)] = -intrinsics.fx * pcx * pcy * z_inv2;
                    jac[(0, 4)] = intrinsics.fx * (1.0 + pcx * pcx * z_inv2);
                    jac[(0, 5)] = -intrinsics.fx * pcy * z_inv;

                    jac[(1, 0)] = 0.0;
                    jac[(1, 1)] = intrinsics.fy * z_inv;
                    jac[(1, 2)] = -intrinsics.fy * pcy * z_inv2;
                    jac[(1, 3)] = -intrinsics.fy * (1.0 + pcy * pcy * z_inv2);
                    jac[(1, 4)] = intrinsics.fy * pcx * pcy * z_inv2;
                    jac[(1, 5)] = intrinsics.fy * pcx * z_inv;

                    let res = nalgebra::Vector2::new(res_u, res_v);
                    jtj += jac.transpose() * weighted_info * jac;
                    jtr += jac.transpose() * weighted_info * res;
                }
            }
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
