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
    /// Dynamic stopping: `k` is updated after each tight-count improvement using
    /// the standard RANSAC formula `k = log(1-p) / log(1-ω⁴)` where `ω` is the
    /// **verified** tight inlier ratio from `lo_inner`.
    ///
    /// **Per the architectural specification**, the unweighted GN pose produced
    /// *inside* `lo_inner` is discarded once the tight consensus mask is
    /// finalised.  `lo_inner` acts as the **strict verification gate**: it proves
    /// that the outer-wide consensus set is in the true global basin of attraction
    /// (tight count stays high), not a phantom induced by background clutter.
    /// The `tight_count` from that gate then collapses `dynamic_k` to enable
    /// bounded early termination.  The retained seed is the **original, clean
    /// IPPE pose** — free from the unweighted GN bias — so AW-LM starts from the
    /// best-conditioned initialisation point.
    #[allow(clippy::too_many_lines, clippy::cast_sign_loss)]
    fn lo_ransac_loop(
        &self,
        valid: &[usize],
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
    ) -> Option<(Pose, [u64; 16])> {
        let cfg = &self.lo_ransac;
        let num_valid = valid.len();

        // Global best tracked by the tight (tau_inner) inlier count produced by
        // lo_inner — not the wide outer count.  This ensures the retained seed has
        // been rigorously verified against the strict 1-px² gate.
        let mut global_best_tight_count = 0usize;
        let mut global_best_seed: Option<Pose> = None;

        // Dynamic stopping criterion, initialised to the hard ceiling.
        // Collapses toward k_min as lo_inner verifies increasingly large
        // tight consensus sets.
        let mut dynamic_k = cfg.k_max;

        // Deterministic XOR-shift RNG (reproducible across frames).
        let mut seed = 0x1337u32;

        for iter in 0..cfg.k_max {
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

            // ── LO inner loop (verification gate) ─────────────────────────
            // Run tight GN refinement to prove the outer consensus is in the
            // true basin of attraction.  The unweighted GN pose is discarded
            // (spec mandate) to prevent biasing the AW-LM initialisation.
            // Only `tight_count` is retained to govern global state and the
            // dynamic stopping criterion.
            let (_gn_pose, _tight_mask, tight_count) =
                self.lo_inner(seed_pose, &best_outer_mask, batch, intrinsics, valid);

            if tight_count > global_best_tight_count {
                global_best_tight_count = tight_count;
                // Retain the clean IPPE seed, not the unweighted GN pose.
                global_best_seed = Some(seed_pose);

                // Update dynamic stopping criterion from the verified tight ratio.
                let inlier_ratio = tight_count as f64 / num_valid as f64;
                if inlier_ratio >= 0.99 {
                    // Near-perfect board: k_min iterations are sufficient.
                    dynamic_k = cfg.k_min;
                } else {
                    let p_fail = 1.0 - cfg.confidence;
                    // Probability that a random 4-tag sample contains at least
                    // one outlier under the tight inlier ratio.
                    let p_good_sample = 1.0 - inlier_ratio.powi(4);
                    let k_compute = p_fail.ln() / p_good_sample.ln();
                    dynamic_k = (k_compute.max(0.0).ceil() as u32).clamp(cfg.k_min, cfg.k_max);
                }
            }

            // ── Bounded early termination ─────────────────────────────────
            // Guard: at least k_min iterations must complete before stopping,
            // regardless of how quickly dynamic_k collapses, to escape
            // spatially-correlated occlusion clusters.
            if iter >= cfg.k_min && iter >= dynamic_k {
                break;
            }
        }

        // Return the clean IPPE seed that survived the lo_inner tight gate.
        // estimate() re-evaluates the inlier mask at tau_aw_lm before calling
        // AW-LM, so no mask needs to be threaded through here.
        let final_seed = global_best_seed?;
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

        let valid: Vec<usize> = (0..num_valid).collect();
        let (_, count) = estimator.evaluate_inliers(&pose, &batch, &intrinsics, &valid, 1.0);
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
        let valid: Vec<usize> = (0..num_valid).collect();
        let (_, count) = estimator.evaluate_inliers(&bad_pose, &batch, &intrinsics, &valid, 100.0);
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

        let valid: Vec<usize> = (0..num_valid).collect();
        let (mask, count) = estimator.evaluate_inliers(&pose, &batch, &intrinsics, &valid, 1.0);

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
        let valid: Vec<usize> = (0..num_valid).collect();
        // Mark all candidates as inliers.
        let all_inliers = [u64::MAX; 16];

        let before = mean_reprojection_sq(&perturbed, &batch, &intrinsics, &config, num_valid);
        let stepped = estimator.gn_step(&perturbed, &batch, &intrinsics, &valid, &all_inliers);
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

        let valid: Vec<usize> = (0..num_valid).collect();
        let no_inliers = [0u64; 16];

        let result = estimator.gn_step(&pose, &batch, &intrinsics, &valid, &no_inliers);
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
        let valid: Vec<usize> = (0..num_valid).collect();
        let all_inliers = [u64::MAX; 16];

        let (refined, cov) =
            estimator.refine_aw_lm(&perturbed, &batch, &intrinsics, &valid, &all_inliers);

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

        let valid: Vec<usize> = (0..num_valid).collect();
        let all_inliers = [u64::MAX; 16];
        let (_, cov) = estimator.refine_aw_lm(&pose, &batch, &intrinsics, &valid, &all_inliers);

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
        let estimator = BoardEstimator::new(config);
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
        let estimator = BoardEstimator::new(config.clone());
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
        let estimator = BoardEstimator::new(config.clone());
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
