//! Board-level configuration and layout utilities.

use crate::batch::DetectionBatch;
use crate::pose::{CameraIntrinsics, Pose};
use nalgebra::{Matrix2, Matrix6, UnitQuaternion, Vector3, Vector6};

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

        // Center of the ENTIRE grid (including corners of squares)
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

/// Result of a board pose estimation.
#[derive(Clone, Debug)]
pub struct BoardPose {
    /// The estimated 6-DOF pose.
    pub pose: Pose,
    /// The 6x6 pose covariance matrix in se(3) tangent space.
    /// Order: [tx, ty, tz, rx, ry, rz]
    pub covariance: Matrix6<f64>,
}

/// Estimator for multi-tag board poses.
pub struct BoardEstimator {
    /// Configuration of the board layout.
    pub config: BoardConfig,
}

impl BoardEstimator {
    /// Creates a new `BoardEstimator` with the given configuration.
    #[must_use]
    pub fn new(config: BoardConfig) -> Self {
        Self { config }
    }

    /// Estimates the board pose from a batch of detections.
    ///
    /// # Panics
    ///
    /// Panics if the internal `obj_points` for a valid tag index is missing.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn estimate(
        &self,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
    ) -> Option<BoardPose> {
        let mut num_valid = 0;
        let mut valid_indices = [0usize; crate::batch::MAX_CANDIDATES];

        for i in 0..crate::batch::MAX_CANDIDATES {
            if batch.status_mask[i] == crate::batch::CandidateState::Valid {
                let id = batch.ids[i] as usize;
                if id < self.config.obj_points.len() && self.config.obj_points[id].is_some() {
                    valid_indices[num_valid] = i;
                    num_valid += 1;
                }
            }
        }

        if num_valid < 4 {
            return None;
        }

        // RANSAC Loop
        let mut best_inliers = 0;
        let mut best_pose = None;
        let mut best_inlier_mask = [0u64; 16]; // 1024 bits

        let mut seed = 0x1337u32;
        let mut next_rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };

        let iterations = 50;

        for _iter in 0..iterations {
            let mut sample = [0usize; 4];
            let mut found = 0;
            let mut attempts = 0;
            while found < 4 && attempts < 1000 {
                attempts += 1;
                let s = (next_rand() as usize) % num_valid;
                if !sample[..found].contains(&s) {
                    sample[found] = s;
                    found += 1;
                }
            }
            if found < 4 {
                continue;
            }

            // Try initialization from each tag in the sample until one works
            for &s_val in &sample {
                let b_idx = valid_indices[s_val];
                let det_pose_data = batch.poses[b_idx].data;
                if det_pose_data.iter().any(|v| v.is_nan()) || det_pose_data[2].abs() < 1e-6 {
                    continue;
                }

                let det_t = Vector3::new(
                    f64::from(det_pose_data[0]),
                    f64::from(det_pose_data[1]),
                    f64::from(det_pose_data[2]),
                );
                let det_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    f64::from(det_pose_data[6]),
                    f64::from(det_pose_data[3]),
                    f64::from(det_pose_data[4]),
                    f64::from(det_pose_data[5]),
                ));

                let tag_id = batch.ids[b_idx] as usize;
                let tag_obj_origin = self.config.obj_points[tag_id].expect("missing obj_points")[0];
                let tag_origin_vec =
                    Vector3::new(tag_obj_origin[0], tag_obj_origin[1], tag_obj_origin[2]);

                let pose_init = Pose {
                    rotation: *det_q.to_rotation_matrix().matrix(),
                    translation: det_t - (det_q * tag_origin_vec),
                };

                // Evaluate inliers
                let mut inlier_count = 0;
                let mut mask = [0u64; 16];
                for (i, &b_idx_eval) in valid_indices[..num_valid].iter().enumerate() {
                    let id_eval = batch.ids[b_idx_eval] as usize;
                    let obj = self.config.obj_points[id_eval].expect("missing obj_points");

                    let mut sum_err_fixed = 0.0;
                    for (j, pt) in obj.iter().enumerate() {
                        let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                        let proj = pose_init.project(&p_world, intrinsics);
                        let dx = proj[0] - f64::from(batch.corners[b_idx_eval][j].x);
                        let dy = proj[1] - f64::from(batch.corners[b_idx_eval][j].y);
                        sum_err_fixed += (dx * dx + dy * dy).sqrt();
                    }

                    if sum_err_fixed / 4.0 < 10.0 {
                        // 10px threshold for inliers
                        inlier_count += 1;
                        mask[i / 64] |= 1 << (i % 64);
                    }
                }

                if inlier_count > best_inliers {
                    best_inliers = inlier_count;
                    best_pose = Some(pose_init);
                    best_inlier_mask = mask;
                }

                if best_inliers > num_valid * 3 / 4 {
                    break;
                }
            }
            if best_inliers > num_valid * 3 / 4 {
                break;
            }
        }

        best_pose.map(|pose| {
            let (refined_pose, covariance) = self.ref_aw_lm(
                &pose,
                batch,
                intrinsics,
                &valid_indices[..num_valid],
                &best_inlier_mask,
            );
            BoardPose {
                pose: refined_pose,
                covariance,
            }
        })
    }

    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn ref_aw_lm(
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

                for (j, pt) in obj.iter().enumerate() {
                    let p_world = Vector3::new(pt[0], pt[1], pt[2]);
                    let p_cam = current_pose.rotation * p_world + current_pose.translation;
                    let z_inv = 1.0 / p_cam.z.max(1e-6);
                    let z_inv2 = z_inv * z_inv;

                    let u = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
                    let v = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;

                    let res_u = f64::from(batch.corners[b_idx][j].x) - u;
                    let res_v = f64::from(batch.corners[b_idx][j].y) - v;

                    let info = Matrix2::new(
                        f64::from(batch.corner_covariances[b_idx][j * 4]),
                        f64::from(batch.corner_covariances[b_idx][j * 4 + 1]),
                        f64::from(batch.corner_covariances[b_idx][j * 4 + 2]),
                        f64::from(batch.corner_covariances[b_idx][j * 4 + 3]),
                    )
                    .try_inverse()
                    .unwrap_or_else(Matrix2::identity);

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

                    // du/d_xi = [fx/z, 0, -fx*x/z^2, -fx*x*y/z^2, fx(1+x^2/z^2), -fx*y/z]
                    jac[(0, 0)] = intrinsics.fx * z_inv;
                    jac[(0, 1)] = 0.0;
                    jac[(0, 2)] = -intrinsics.fx * pcx * z_inv2;
                    jac[(0, 3)] = -intrinsics.fx * pcx * pcy * z_inv2;
                    jac[(0, 4)] = intrinsics.fx * (1.0 + pcx * pcx * z_inv2);
                    jac[(0, 5)] = -intrinsics.fx * pcy * z_inv;

                    // dv/d_xi = [0, fy/z, -fy*y/z^2, -fy(1+y^2/z^2), fy*x*y/z^2, fy*x/z]
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
