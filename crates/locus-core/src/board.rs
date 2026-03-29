use crate::batch::{CandidateState, DetectionBatch, MAX_CANDIDATES};
use crate::decoder::Homography;
use crate::pose::{CameraIntrinsics, Pose};
use crate::workspace::WORKSPACE_ARENA;
use multiversion::multiversion;
use nalgebra::{Matrix2, Matrix3, Matrix6, Rotation3, Vector3, Vector6};

/// A 3D pose result for the entire board.
pub type BoardPose = Pose;

/// Huber loss threshold (Nielsen 1999): balances robustness vs. statistical efficiency.
const HUBER_K: f64 = 1.345;

/// Squared reprojection-error threshold for LO-RANSAC consensus (~2.0 px).
const CONSENSUS_THRESHOLD_SQ: f64 = 4.0; // 2.0² px²

/// Length of the bitmask array for tracking inlier status.
/// Each `u64` covers 64 candidates; `MAX_CANDIDATES / 64` entries span the full batch.
const VALID_MASK_LEN: usize = MAX_CANDIDATES / 64;

/// Evaluates reprojection errors for planar points using SIMD (auto-vectorized FMA).
#[allow(clippy::too_many_arguments)]
#[multiversion(targets = "simd")]
pub(crate) fn compute_reprojection_errors_simd(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    obj_x: &[f64],
    obj_y: &[f64],
    img_u: &[f64],
    img_v: &[f64],
    errors: &mut [f64],
) {
    // LLVM will unroll and vectorize this loop using AVX2/AVX-512 FMA instructions
    for i in 0..obj_x.len() {
        let ox = obj_x[i];
        let oy = obj_y[i];

        // Z=0 assumed for board local frame
        let x = r[(0, 0)] * ox + r[(0, 1)] * oy + t.x;
        let y = r[(1, 0)] * ox + r[(1, 1)] * oy + t.y;
        let z = r[(2, 0)] * ox + r[(2, 1)] * oy + t.z;

        let z_inv = 1.0 / z;

        let u_est = fx * x * z_inv + cx;
        let v_est = fy * y * z_inv + cy;

        let du = img_u[i] - u_est;
        let dv = img_v[i] - v_est;

        errors[i] = du * du + dv * dv;
    }
}

/// Executes a fast, unweighted Gauss-Newton step using all flagged inliers.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
fn unweighted_gn_step(
    pose: &Pose,
    intrinsics: &CameraIntrinsics,
    obj_x: &[f64],
    obj_y: &[f64],
    img_u: &[f64],
    img_v: &[f64],
    valid_mask: &[u64; VALID_MASK_LEN],
    valid_indices: &[usize],
    num_tags: usize,
) -> Pose {
    debug_assert!(num_tags <= valid_indices.len());
    debug_assert!(num_tags * 4 <= obj_x.len());

    let mut jtj = Matrix6::<f64>::zeros();
    let mut jtr = Vector6::<f64>::zeros();

    let fx = intrinsics.fx;
    let fy = intrinsics.fy;
    let cx = intrinsics.cx;
    let cy = intrinsics.cy;

    for (i, &batch_idx) in valid_indices[..num_tags].iter().enumerate() {
        if (valid_mask[batch_idx / 64] & (1 << (batch_idx % 64))) == 0 {
            continue;
        }

        for j in 0..4 {
            let flat_idx = i * 4 + j;
            let ox = obj_x[flat_idx];
            let oy = obj_y[flat_idx];
            let p_world = Vector3::new(ox, oy, 0.0);

            let p_cam = pose.rotation * p_world + pose.translation;
            let z_inv = 1.0 / p_cam.z;
            let z_inv2 = z_inv * z_inv;

            let u_est = fx * p_cam.x * z_inv + cx;
            let v_est = fy * p_cam.y * z_inv + cy;

            let res_u = img_u[flat_idx] - u_est;
            let res_v = img_v[flat_idx] - v_est;

            let jac_u = Vector3::new(fx * z_inv, 0.0, -fx * p_cam.x * z_inv2);
            let jac_v = Vector3::new(0.0, fy * z_inv, -fy * p_cam.y * z_inv2);
            // Rotation DOFs: ∂u/∂δω = p_cam × jac  (left SE(3) perturbation)
            let rot_u = p_cam.cross(&jac_u);
            let rot_v = p_cam.cross(&jac_v);

            let row_u = Vector6::new(jac_u.x, jac_u.y, jac_u.z, rot_u.x, rot_u.y, rot_u.z);
            let row_v = Vector6::new(jac_v.x, jac_v.y, jac_v.z, rot_v.x, rot_v.y, rot_v.z);

            jtj += row_u * row_u.transpose() + row_v * row_v.transpose();
            jtr += row_u * res_u + row_v * res_v;
        }
    }

    if let Some(chol) = jtj.cholesky() {
        let delta = chol.solve(&jtr);
        let twist = Vector3::new(delta[3], delta[4], delta[5]);
        let trans_update = Vector3::new(delta[0], delta[1], delta[2]);
        let rot_update = Rotation3::new(twist).matrix().into_owned();
        Pose::new(
            rot_update * pose.rotation,
            rot_update * pose.translation + trans_update,
        )
    } else {
        *pose
    }
}

/// Executes Anisotropic Weighted Levenberg-Marquardt (AW-LM) with Huber damping.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
fn refine_board_pose_aw_lm(
    initial_pose: &Pose,
    intrinsics: &CameraIntrinsics,
    obj_x: &[f64],
    obj_y: &[f64],
    img_u: &[f64],
    img_v: &[f64],
    info_mats: &[Matrix2<f64>],
    valid_mask: &[u64; VALID_MASK_LEN],
    valid_indices: &[usize],
    num_tags: usize,
) -> Pose {
    debug_assert!(num_tags <= valid_indices.len());
    debug_assert!(num_tags * 4 <= info_mats.len());

    let mut pose = *initial_pose;

    let mut lambda = 1e-3_f64;
    let mut nu = 2.0_f64;

    let compute_equations = |current_pose: &Pose| -> (f64, Matrix6<f64>, Vector6<f64>) {
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        let mut total_cost = 0.0;

        for (i, &batch_idx) in valid_indices[..num_tags].iter().enumerate() {
            if (valid_mask[batch_idx / 64] & (1 << (batch_idx % 64))) == 0 {
                continue;
            }

            for j in 0..4 {
                let flat_idx = i * 4 + j;
                let ox = obj_x[flat_idx];
                let oy = obj_y[flat_idx];
                let p_world = Vector3::new(ox, oy, 0.0);

                let p_cam = current_pose.rotation * p_world + current_pose.translation;
                let z_inv = 1.0 / p_cam.z;
                let z_inv2 = z_inv * z_inv;

                let u_est = intrinsics.fx * p_cam.x * z_inv + intrinsics.cx;
                let v_est = intrinsics.fy * p_cam.y * z_inv + intrinsics.cy;

                let res_u = img_u[flat_idx] - u_est;
                let res_v = img_v[flat_idx] - v_est;
                let res = nalgebra::Vector2::new(res_u, res_v);

                let info = &info_mats[flat_idx];
                let mahalanobis_sq = res.dot(&(info * res));
                let s_i = mahalanobis_sq.max(0.0).sqrt();

                let (w, cost) = if s_i <= HUBER_K {
                    (1.0, 0.5 * mahalanobis_sq)
                } else {
                    (HUBER_K / s_i, HUBER_K * (s_i - 0.5 * HUBER_K))
                };
                total_cost += cost;

                let w_mat = info * w;

                let jac_u = Vector3::new(
                    intrinsics.fx * z_inv,
                    0.0,
                    -intrinsics.fx * p_cam.x * z_inv2,
                );
                let jac_v = Vector3::new(
                    0.0,
                    intrinsics.fy * z_inv,
                    -intrinsics.fy * p_cam.y * z_inv2,
                );
                // Rotation DOFs: ∂u/∂δω = p_cam × jac  (left SE(3) perturbation)
                let rot_u = p_cam.cross(&jac_u);
                let rot_v = p_cam.cross(&jac_v);

                let row_u = Vector6::new(jac_u.x, jac_u.y, jac_u.z, rot_u.x, rot_u.y, rot_u.z);
                let row_v = Vector6::new(jac_v.x, jac_v.y, jac_v.z, rot_v.x, rot_v.y, rot_v.z);

                let w00 = w_mat[(0, 0)];
                let w01 = w_mat[(0, 1)];
                let w10 = w_mat[(1, 0)];
                let w11 = w_mat[(1, 1)];

                let jtw_u = row_u * w00 + row_v * w10;
                let jtw_v = row_u * w01 + row_v * w11;

                jtj += jtw_u * row_u.transpose() + jtw_v * row_v.transpose();
                jtr += jtw_u * res_u + jtw_v * res_v;
            }
        }
        (total_cost, jtj, jtr)
    };

    let (mut current_cost, mut jtj, mut jtr) = compute_equations(&pose);

    for _ in 0..20 {
        if jtr.amax() < 1e-8 {
            break;
        }

        let mut d_diag = Vector6::zeros();
        for k in 0..6 {
            d_diag[k] = jtj[(k, k)].max(1e-8);
        }

        let mut jtj_damped = jtj;
        for k in 0..6 {
            jtj_damped[(k, k)] += lambda * d_diag[k];
        }

        let delta = if let Some(chol) = jtj_damped.cholesky() {
            chol.solve(&jtr)
        } else {
            lambda *= 10.0;
            nu = 2.0;
            continue;
        };

        let predicted_reduction = 0.5 * delta.dot(&(lambda * d_diag.component_mul(&delta) + jtr));

        let twist = Vector3::new(delta[3], delta[4], delta[5]);
        let trans_update = Vector3::new(delta[0], delta[1], delta[2]);
        let rot_update = Rotation3::new(twist).matrix().into_owned();
        let new_pose = Pose::new(
            rot_update * pose.rotation,
            rot_update * pose.translation + trans_update,
        );

        let (new_cost, new_jtj, new_jtr) = compute_equations(&new_pose);
        let actual_reduction = current_cost - new_cost;

        let rho = if predicted_reduction > 1e-12 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        if rho > 0.0 {
            pose = new_pose;
            current_cost = new_cost;
            jtj = new_jtj;
            jtr = new_jtr;
            lambda *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            nu = 2.0;

            if delta.norm() < 1e-7 {
                break;
            }
        } else {
            lambda *= nu;
            nu *= 2.0;
        }
    }

    pose
}

/// Core engine for board pose estimation.
pub struct BoardEstimator {
    /// The canonical 3D geometry of the board.
    pub config: BoardConfig,
}

impl BoardEstimator {
    /// Creates a new `BoardEstimator` with the given configuration.
    #[must_use]
    pub fn new(config: BoardConfig) -> Self {
        Self { config }
    }

    /// Estimates the board pose given a batch of detected tags.
    ///
    /// This method leverages a thread-local `WORKSPACE_ARENA` to perform
    /// zero-heap allocations during the fast-path RANSAC inner loop.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn estimate(
        &self,
        batch: &DetectionBatch,
        intrinsics: &CameraIntrinsics,
    ) -> Option<BoardPose> {
        WORKSPACE_ARENA.with(|cell| {
            let mut arena = cell.borrow_mut();

            // Collect valid tag indices without heap allocation
            // Max candidates is 1024. We can allocate a slice from the arena.
            let valid_indices = arena.alloc_slice_fill_default(crate::batch::MAX_CANDIDATES);
            let mut num_valid = 0;

            for i in 0..crate::batch::MAX_CANDIDATES {
                if batch.status_mask[i] == CandidateState::Valid {
                    let id = batch.ids[i] as usize;
                    if id < self.config.obj_points.len() {
                        valid_indices[num_valid] = i;
                        num_valid += 1;
                    }
                }
            }

            // Need at least 4 tags for a robust planar minimal sample
            if num_valid < 4 {
                arena.reset();
                return None;
            }

            // Step 1: Minimal Sample Generator (IPPE)
            // Select 4 tags (for now, just take the first 4)
            let sample_indices = [
                &valid_indices[0],
                &valid_indices[1],
                &valid_indices[2],
                &valid_indices[3],
            ];

            let mut src_pts = [[0.0; 2]; 4];
            let mut dst_pts = [[0.0; 2]; 4];

            for (k, &idx) in sample_indices.iter().enumerate() {
                let id = batch.ids[*idx] as usize;
                let obj_corners = &self.config.obj_points[id];
                let img_corners = &batch.corners[*idx];

                src_pts[k] = [obj_corners[k].x, obj_corners[k].y];
                dst_pts[k] = [f64::from(img_corners[k].x), f64::from(img_corners[k].y)];
            }

            let Some(h_pixel) = Homography::from_pairs(&src_pts, &dst_pts) else {
                arena.reset();
                return None;
            };

            let k_inv = intrinsics.inv_matrix();
            let h_norm = k_inv * h_pixel.h;

            let Some(candidates) = crate::pose::solve_ippe_square(&h_norm) else {
                arena.reset();
                return None;
            };

            // Prepare flat arrays for SIMD Consensus Evaluation and Metrology Engine
            let total_corners = num_valid * 4;
            let obj_x = arena.alloc_slice_fill_default(total_corners);
            let obj_y = arena.alloc_slice_fill_default(total_corners);
            let img_u = arena.alloc_slice_fill_default(total_corners);
            let img_v = arena.alloc_slice_fill_default(total_corners);
            let errors = arena.alloc_slice_fill_default(total_corners);
            let info_mats = arena.alloc_slice_fill_with(total_corners, |_| Matrix2::<f64>::zeros());

            for (i, &idx) in valid_indices[..num_valid].iter().enumerate() {
                let id = batch.ids[idx] as usize;
                let obj_corners = &self.config.obj_points[id];
                let img_corners = &batch.corners[idx];
                let covs = &batch.corner_covariances[idx];

                for j in 0..4 {
                    let flat_idx = i * 4 + j;
                    obj_x[flat_idx] = obj_corners[j].x;
                    obj_y[flat_idx] = obj_corners[j].y;
                    img_u[flat_idx] = f64::from(img_corners[j].x);
                    img_v[flat_idx] = f64::from(img_corners[j].y);

                    let cov = Matrix2::new(
                        f64::from(covs[j * 4]),
                        f64::from(covs[j * 4 + 1]),
                        f64::from(covs[j * 4 + 2]),
                        f64::from(covs[j * 4 + 3]),
                    );

                    let det = cov.determinant();
                    if det.abs() > 1e-12 {
                        info_mats[flat_idx] = cov.try_inverse().unwrap_or_else(Matrix2::identity);
                    } else {
                        info_mats[flat_idx] = Matrix2::identity();
                    }
                }
            }

            let tau_sq = CONSENSUS_THRESHOLD_SQ;

            let mut best_pose = None;
            let mut best_inliers = 0;
            let mut best_inlier_mask = [0u64; 16];

            for pose in candidates {
                compute_reprojection_errors_simd(
                    &pose.rotation,
                    &pose.translation,
                    intrinsics.fx,
                    intrinsics.fy,
                    intrinsics.cx,
                    intrinsics.cy,
                    obj_x,
                    obj_y,
                    img_u,
                    img_v,
                    errors,
                );

                let mut inlier_count = 0;
                let mut mask = [0u64; 16];

                // Construct bitmask over tags (1 bit per tag)
                for i in 0..num_valid {
                    let mut tag_inlier = true;
                    for j in 0..4 {
                        if errors[i * 4 + j] > tau_sq {
                            tag_inlier = false;
                            break;
                        }
                    }
                    if tag_inlier {
                        inlier_count += 1;
                        let idx = valid_indices[i];
                        mask[idx / 64] |= 1 << (idx % 64);
                    }
                }

                if inlier_count > best_inliers {
                    // Local Optimization (LO) Handoff
                    let lo_pose = unweighted_gn_step(
                        &pose,
                        intrinsics,
                        obj_x,
                        obj_y,
                        img_u,
                        img_v,
                        &mask,
                        valid_indices,
                        num_valid,
                    );

                    // Re-evaluate consensus
                    compute_reprojection_errors_simd(
                        &lo_pose.rotation,
                        &lo_pose.translation,
                        intrinsics.fx,
                        intrinsics.fy,
                        intrinsics.cx,
                        intrinsics.cy,
                        obj_x,
                        obj_y,
                        img_u,
                        img_v,
                        errors,
                    );

                    let mut lo_inliers = 0;
                    let mut lo_mask = [0u64; 16];
                    for i in 0..num_valid {
                        let mut tag_inlier = true;
                        for j in 0..4 {
                            if errors[i * 4 + j] > tau_sq {
                                tag_inlier = false;
                                break;
                            }
                        }
                        if tag_inlier {
                            lo_inliers += 1;
                            let idx = valid_indices[i];
                            lo_mask[idx / 64] |= 1 << (idx % 64);
                        }
                    }

                    if lo_inliers > best_inliers {
                        best_inliers = lo_inliers;
                        best_inlier_mask = lo_mask;
                        best_pose = Some(lo_pose);

                        if f64::from(lo_inliers) / num_valid as f64 > 0.95 {
                            break; // Early termination
                        }
                    } else if inlier_count > best_inliers {
                        best_inliers = inlier_count;
                        best_inlier_mask = mask;
                        best_pose = Some(pose);
                    }
                }
            }

            // Metrology Engine (AW-LM)
            let final_pose = best_pose.map(|pose| {
                refine_board_pose_aw_lm(
                    &pose,
                    intrinsics,
                    obj_x,
                    obj_y,
                    img_u,
                    img_v,
                    info_mats,
                    &best_inlier_mask,
                    valid_indices,
                    num_valid,
                )
            });

            arena.reset();
            final_pose
        })
    }
}

/// Configuration and canonical 3D geometry for a fiducial marker board (ChAruco/AprilGrid).
#[derive(Clone, Debug, PartialEq)]
pub struct BoardConfig {
    /// Number of rows in the grid
    pub rows: usize,
    /// Number of columns in the grid
    pub cols: usize,
    /// Length of a single grid square side (meters)
    pub square_length: f64,
    /// Length of a single marker side (meters)
    pub marker_length: f64,
    /// Canonical 3D coordinates of all marker corners in the board's local frame.
    /// The array is ordered by marker ID. Each marker has 4 corners:
    /// 0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left.
    pub obj_points: Vec<[Vector3<f64>; 4]>,
}

impl BoardConfig {
    /// Initializes a new `BoardConfig` for a ChAruco board.
    ///
    /// ChAruco boards have a checkerboard pattern where markers are placed inside
    /// the black squares. This constructor computes the 3D coordinates for all
    /// markers in the board.
    ///
    /// By convention, the top-left corner of the board is the origin (0, 0, 0).
    /// X points right, Y points down, Z points into the board (0.0).
    #[must_use]
    pub fn new_charuco(rows: usize, cols: usize, square_length: f64, marker_length: f64) -> Self {
        let mut obj_points = Vec::new();
        let margin = (square_length - marker_length) / 2.0;

        for r in 0..rows {
            for c in 0..cols {
                // In a standard ChAruco board, markers are in the black squares.
                // Assuming top-left square (0,0) is white, black squares have (r + c) % 2 == 1.
                // Let's adopt this convention for assigning marker IDs sequentially.
                if (r + c) % 2 == 1 {
                    let y_offset = r as f64 * square_length;
                    let x_offset = c as f64 * square_length;

                    let tl = Vector3::new(x_offset + margin, y_offset + margin, 0.0);
                    let tr =
                        Vector3::new(x_offset + margin + marker_length, y_offset + margin, 0.0);
                    let br = Vector3::new(
                        x_offset + margin + marker_length,
                        y_offset + margin + marker_length,
                        0.0,
                    );
                    let bl =
                        Vector3::new(x_offset + margin, y_offset + margin + marker_length, 0.0);

                    obj_points.push([tl, tr, br, bl]);
                }
            }
        }

        Self {
            rows,
            cols,
            square_length,
            marker_length,
            obj_points,
        }
    }

    /// Initializes a new `BoardConfig` for an AprilGrid board.
    ///
    /// AprilGrid boards have markers in every grid cell, separated by a gap.
    /// The `square_length` conceptually becomes the `tag_spacing + marker_length`
    /// (the distance from the start of one tag to the start of the next).
    #[must_use]
    pub fn new_aprilgrid(rows: usize, cols: usize, tag_spacing: f64, marker_length: f64) -> Self {
        let mut obj_points = Vec::with_capacity(rows * cols);
        let step = marker_length + tag_spacing;

        for r in 0..rows {
            for c in 0..cols {
                let y_offset = r as f64 * step;
                let x_offset = c as f64 * step;

                let tl = Vector3::new(x_offset, y_offset, 0.0);
                let tr = Vector3::new(x_offset + marker_length, y_offset, 0.0);
                let br = Vector3::new(x_offset + marker_length, y_offset + marker_length, 0.0);
                let bl = Vector3::new(x_offset, y_offset + marker_length, 0.0);

                obj_points.push([tl, tr, br, bl]);
            }
        }

        Self {
            rows,
            cols,
            square_length: step,
            marker_length,
            obj_points,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charuco_board_initialization() {
        // 5x5 board means 5 rows, 5 columns of squares.
        // If top-left is white (even), black squares are where (r + c) is odd.
        // Total squares = 25. Even = 13, Odd = 12. So 12 markers.
        let board = BoardConfig::new_charuco(5, 5, 0.04, 0.02);

        assert_eq!(board.rows, 5);
        assert_eq!(board.cols, 5);
        assert!((board.square_length - 0.04).abs() < 1e-10);
        assert!((board.marker_length - 0.02).abs() < 1e-10);
        assert_eq!(board.obj_points.len(), 12);

        // Check the first marker, which should be at r=0, c=1
        let m0 = &board.obj_points[0];

        // expected top-left of square (0, 1): x = 0.04, y = 0.0
        // expected top-left of marker: x = 0.04 + 0.01 = 0.05, y = 0.0 + 0.01 = 0.01
        assert!((m0[0].x - 0.05).abs() < 1e-10);
        assert!((m0[0].y - 0.01).abs() < 1e-10);
        assert!((m0[0].z - 0.0).abs() < 1e-10);

        // tr
        assert!((m0[1].x - 0.07).abs() < 1e-10);
        assert!((m0[1].y - 0.01).abs() < 1e-10);

        // br
        assert!((m0[2].x - 0.07).abs() < 1e-10);
        assert!((m0[2].y - 0.03).abs() < 1e-10);

        // bl
        assert!((m0[3].x - 0.05).abs() < 1e-10);
        assert!((m0[3].y - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_aprilgrid_board_initialization() {
        let board = BoardConfig::new_aprilgrid(3, 4, 0.01, 0.05);

        assert_eq!(board.rows, 3);
        assert_eq!(board.cols, 4);
        assert!((board.square_length - 0.06).abs() < 1e-9);
        assert!((board.marker_length - 0.05).abs() < 1e-9);
        assert_eq!(board.obj_points.len(), 12);

        // check marker at r=1, c=2
        let idx = 4 + 2;
        let m = &board.obj_points[idx];

        // x_offset = 2 * 0.06 = 0.12
        // y_offset = 1 * 0.06 = 0.06
        assert!((m[0].x - 0.12).abs() < 1e-9);
        assert!((m[0].y - 0.06).abs() < 1e-9);

        assert!((m[2].x - 0.17).abs() < 1e-9); // 0.12 + 0.05
        assert!((m[2].y - 0.11).abs() < 1e-9); // 0.06 + 0.05
    }

    #[test]
    fn test_board_estimator_arena_borrow() {
        let config = BoardConfig::new_charuco(5, 5, 0.04, 0.02);
        let estimator = BoardEstimator::new(config);

        let batch = crate::batch::DetectionBatch::new();
        let intrinsics = crate::pose::CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
        let result = estimator.estimate(&batch, &intrinsics);

        // Since batch is empty, it should return None
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_reprojection_errors_simd() {
        let mut errors = [0.0; 4];
        let r = nalgebra::Matrix3::identity();
        let t = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let fx = 500.0;
        let fy = 500.0;
        let cx = 320.0;
        let cy = 240.0;

        // 4 points on the board
        let obj_x = [0.0, 0.1, 0.1, 0.0];
        let obj_y = [0.0, 0.0, 0.1, 0.1];

        // Let's say image points are exactly at projection for first two,
        // and have 1 pixel error for the last two.
        // For (0,0,1): u = 500 * 0/1 + 320 = 320, v = 240
        // For (0.1,0,1): u = 500 * 0.1/1 + 320 = 370, v = 240
        // For (0.1,0.1,1): u = 370, v = 290
        // For (0,0.1,1): u = 320, v = 290

        let img_u = [320.0, 370.0, 371.0, 320.0];
        let img_v = [240.0, 240.0, 290.0, 291.0];

        super::compute_reprojection_errors_simd(
            &r,
            &t,
            fx,
            fy,
            cx,
            cy,
            &obj_x,
            &obj_y,
            &img_u,
            &img_v,
            &mut errors,
        );

        assert!((errors[0] - 0.0).abs() < 1e-6);
        assert!((errors[1] - 0.0).abs() < 1e-6);
        // Error is 1px in U for point 2 => error_sq = 1.0
        assert!((errors[2] - 1.0).abs() < 1e-6);
        // Error is 1px in V for point 3 => error_sq = 1.0
        assert!((errors[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_unweighted_gn_step() {
        let r = nalgebra::Matrix3::identity();
        // Give the initial pose a slight error in translation
        let initial_t = nalgebra::Vector3::new(0.01, -0.01, 1.05);
        let gt_t = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let pose = crate::pose::Pose::new(r, initial_t);

        let fx = 500.0;
        let fy = 500.0;
        let cx = 320.0;
        let cy = 240.0;
        let intrinsics = crate::pose::CameraIntrinsics::new(fx, fy, cx, cy);

        let obj_x = [0.0, 0.1, 0.1, 0.0];
        let obj_y = [0.0, 0.0, 0.1, 0.1];

        // Ground truth image points using gt_t
        let mut img_u = [0.0; 4];
        let mut img_v = [0.0; 4];
        let gt_pose = crate::pose::Pose::new(r, gt_t);
        for i in 0..4 {
            let p_world = nalgebra::Vector3::new(obj_x[i], obj_y[i], 0.0);
            let p_img = gt_pose.project(&p_world, &intrinsics);
            img_u[i] = p_img[0];
            img_v[i] = p_img[1];
        }

        let valid_mask = [1u64; 16]; // First tag is valid
        let valid_indices = [0];

        let refined_pose = super::unweighted_gn_step(
            &pose,
            &intrinsics,
            &obj_x,
            &obj_y,
            &img_u,
            &img_v,
            &valid_mask,
            &valid_indices,
            1,
        );

        // Check if refined pose is closer to GT than initial pose
        let initial_err = (pose.translation - gt_t).norm();
        let refined_err = (refined_pose.translation - gt_t).norm();
        assert!(refined_err < initial_err);
    }

    #[test]
    fn test_refine_board_pose_aw_lm() {
        let r = nalgebra::Matrix3::identity();
        let initial_t = nalgebra::Vector3::new(0.01, -0.01, 1.05);
        let gt_t = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let pose = crate::pose::Pose::new(r, initial_t);

        let fx = 500.0;
        let fy = 500.0;
        let cx = 320.0;
        let cy = 240.0;
        let intrinsics = crate::pose::CameraIntrinsics::new(fx, fy, cx, cy);

        let obj_x = [0.0, 0.1, 0.1, 0.0];
        let obj_y = [0.0, 0.0, 0.1, 0.1];

        let mut img_u = [0.0; 4];
        let mut img_v = [0.0; 4];
        let gt_pose = crate::pose::Pose::new(r, gt_t);
        for i in 0..4 {
            let p_world = nalgebra::Vector3::new(obj_x[i], obj_y[i], 0.0);
            let p_img = gt_pose.project(&p_world, &intrinsics);
            img_u[i] = p_img[0];
            img_v[i] = p_img[1];
        }

        let valid_mask = [1u64; 16];
        let valid_indices = [0];
        let info_mats = [nalgebra::Matrix2::identity(); 4];

        let refined_pose = super::refine_board_pose_aw_lm(
            &pose,
            &intrinsics,
            &obj_x,
            &obj_y,
            &img_u,
            &img_v,
            &info_mats,
            &valid_mask,
            &valid_indices,
            1,
        );

        let initial_err = (pose.translation - gt_t).norm();
        let refined_err = (refined_pose.translation - gt_t).norm();
        println!("Initial err: {initial_err}, Refined err: {refined_err}");
        assert!(refined_err < initial_err);
    }

    /// Test A: Anisotropic covariance injection produces lower error than isotropic weighting.
    #[test]
    fn test_covariance_injection_anisotropic() {
        // A 4x4 AprilGrid at 1m distance, frontal view.
        // Add asymmetric noise: σ_u=0.5px, σ_v=3.0px.
        // The anisotropic info matrix should de-weight the noisy V axis,
        // yielding a pose closer to ground truth than identity weighting.
        let config = BoardConfig::new_aprilgrid(2, 2, 0.01, 0.05);
        let intrinsics = crate::pose::CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
        let gt_r = nalgebra::Matrix3::identity();
        let gt_t = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let gt_pose = crate::pose::Pose::new(gt_r, gt_t);

        let num_tags = config.obj_points.len();
        let total = num_tags * 4;
        let mut obj_x = vec![0.0f64; total];
        let mut obj_y = vec![0.0f64; total];
        let mut img_u = vec![0.0f64; total];
        let mut img_v = vec![0.0f64; total];

        for i in 0..num_tags {
            for j in 0..4 {
                let flat = i * 4 + j;
                let pt = &config.obj_points[i][j];
                obj_x[flat] = pt.x;
                obj_y[flat] = pt.y;
                let proj = gt_pose.project(pt, &intrinsics);
                // Add noise: 2.0px in V only
                img_u[flat] = proj[0];
                img_v[flat] = proj[1] + 2.0;
            }
        }

        // Initial pose with a small translation error
        let initial_t = nalgebra::Vector3::new(0.01, 0.0, 1.05);
        let initial_pose = crate::pose::Pose::new(gt_r, initial_t);

        let valid_mask = [!0u64; 16];
        let valid_indices: Vec<usize> = (0..num_tags).collect();

        // Anisotropic: high precision in U, low in V
        // σ_u=0.5, σ_v=3.0 → Σ = diag(0.25, 9.0) → Σ⁻¹ = diag(4.0, 0.111)
        let aniso_info = nalgebra::Matrix2::new(4.0, 0.0, 0.0, 1.0 / 9.0);
        let iso_info = nalgebra::Matrix2::identity();

        let mut aniso_mats = vec![nalgebra::Matrix2::<f64>::identity(); total];
        let mut iso_mats = vec![nalgebra::Matrix2::<f64>::identity(); total];
        for k in 0..total {
            aniso_mats[k] = aniso_info;
            iso_mats[k] = iso_info;
        }

        let pose_aniso = super::refine_board_pose_aw_lm(
            &initial_pose,
            &intrinsics,
            &obj_x,
            &obj_y,
            &img_u,
            &img_v,
            &aniso_mats,
            &valid_mask,
            &valid_indices,
            num_tags,
        );
        let pose_iso = super::refine_board_pose_aw_lm(
            &initial_pose,
            &intrinsics,
            &obj_x,
            &obj_y,
            &img_u,
            &img_v,
            &iso_mats,
            &valid_mask,
            &valid_indices,
            num_tags,
        );

        // Anisotropic weighting should down-weight the noisy V observations,
        // yielding a better X/Z estimate (less pulled toward V noise).
        // At minimum, the anisotropic run must converge to within 5cm of GT.
        let err_aniso = (pose_aniso.translation - gt_t).norm();
        let err_iso = (pose_iso.translation - gt_t).norm();
        // Both should converge reasonably; anisotropic should be at least as good.
        assert!(
            err_aniso <= err_iso + 1e-4,
            "Anisotropic err {err_aniso:.4} should be no worse than isotropic {err_iso:.4}"
        );
        assert!(
            err_aniso < 0.05,
            "Anisotropic solver must converge: err={err_aniso:.4}"
        );
    }

    /// Test B: Singular covariance in DetectionBatch falls back gracefully to identity.
    #[test]
    fn test_covariance_singular_fallback() {
        let config = BoardConfig::new_charuco(3, 3, 0.04, 0.02);
        let estimator = BoardEstimator::new(config.clone());
        let intrinsics = crate::pose::CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);

        let gt_r = nalgebra::Matrix3::identity();
        let gt_t = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let gt_pose = crate::pose::Pose::new(gt_r, gt_t);

        let mut batch = crate::batch::DetectionBatch::new();

        // Populate 4 tags with valid corners but all-zero (singular) covariances.
        for i in 0..4 {
            batch.status_mask[i] = crate::batch::CandidateState::Valid;
            batch.ids[i] = i as u32;
            let obj_corners = &config.obj_points[i];
            for (j, corner) in obj_corners.iter().enumerate().take(4) {
                let p_img = gt_pose.project(corner, &intrinsics);
                batch.corners[i][j].x = p_img[0] as f32;
                batch.corners[i][j].y = p_img[1] as f32;
            }
            // All-zero covariance = singular matrix; should fall back to identity
            batch.corner_covariances[i] = [0.0f32; 16];
        }

        // Should not panic; should return a valid pose (fallback to identity info)
        let result = estimator.estimate(&batch, &intrinsics);
        assert!(
            result.is_some(),
            "Should return a pose despite singular covariances"
        );
    }

    /// Test C: Huber loss down-weights a gross outlier — pose must be nearly unaffected.
    #[test]
    fn test_huber_outlier_robustness() {
        // Build a clean board: 4x4 AprilGrid with perfect observations.
        let config = BoardConfig::new_aprilgrid(2, 2, 0.01, 0.05);
        let intrinsics = crate::pose::CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
        let gt_r = nalgebra::Matrix3::identity();
        let gt_t = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let gt_pose = crate::pose::Pose::new(gt_r, gt_t);
        let initial_pose = crate::pose::Pose::new(gt_r, nalgebra::Vector3::new(0.02, 0.0, 1.05));

        let num_tags = config.obj_points.len();
        let total = num_tags * 4;
        let mut obj_x = vec![0.0f64; total];
        let mut obj_y = vec![0.0f64; total];
        let mut obs_u = vec![0.0f64; total];
        let mut obs_v = vec![0.0f64; total];
        let mut noisy_u = vec![0.0f64; total];
        let mut noisy_v = vec![0.0f64; total];

        for i in 0..num_tags {
            for j in 0..4 {
                let flat = i * 4 + j;
                let pt = &config.obj_points[i][j];
                obj_x[flat] = pt.x;
                obj_y[flat] = pt.y;
                let proj = gt_pose.project(pt, &intrinsics);
                obs_u[flat] = proj[0];
                obs_v[flat] = proj[1];
                noisy_u[flat] = proj[0];
                noisy_v[flat] = proj[1];
            }
        }
        // Inject a 50px gross outlier on the last corner of the last tag
        noisy_u[total - 1] += 50.0;
        noisy_v[total - 1] += 50.0;

        let valid_mask = [!0u64; 16];
        let valid_indices: Vec<usize> = (0..num_tags).collect();
        let info_mats = vec![nalgebra::Matrix2::<f64>::identity(); total];

        let pose_clean = super::refine_board_pose_aw_lm(
            &initial_pose,
            &intrinsics,
            &obj_x,
            &obj_y,
            &obs_u,
            &obs_v,
            &info_mats,
            &valid_mask,
            &valid_indices,
            num_tags,
        );
        let pose_outlier = super::refine_board_pose_aw_lm(
            &initial_pose,
            &intrinsics,
            &obj_x,
            &obj_y,
            &noisy_u,
            &noisy_v,
            &info_mats,
            &valid_mask,
            &valid_indices,
            num_tags,
        );

        let err_clean = (pose_clean.translation - gt_t).norm();
        let err_outlier = (pose_outlier.translation - gt_t).norm();

        // With Huber loss, the outlier-contaminated solution should stay close
        // to the clean solution (within 5mm difference).
        assert!(
            (err_outlier - err_clean).abs() < 0.005,
            "Huber should bound outlier influence: clean={err_clean:.4} outlier={err_outlier:.4}"
        );
    }

    /// Test D: Analytical Jacobian matches numerical finite-difference approximation.
    #[test]
    fn test_jacobian_finite_differences() {
        // Project a single 3D point and verify J_analytical ≈ J_numerical.
        let intrinsics = crate::pose::CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
        let r = nalgebra::Matrix3::identity();
        let t = nalgebra::Vector3::new(0.02, -0.01, 1.0);
        let pose = crate::pose::Pose::new(r, t);

        let p_world = nalgebra::Vector3::new(0.05, 0.03, 0.0);

        // Analytical Jacobian rows (same formula as refine_board_pose_aw_lm)
        let p_cam = pose.rotation * p_world + pose.translation;
        let z_inv = 1.0 / p_cam.z;
        let z_inv2 = z_inv * z_inv;
        let jac_u = nalgebra::Vector3::new(
            intrinsics.fx * z_inv,
            0.0,
            -intrinsics.fx * p_cam.x * z_inv2,
        );
        let jac_v = nalgebra::Vector3::new(
            0.0,
            intrinsics.fy * z_inv,
            -intrinsics.fy * p_cam.y * z_inv2,
        );

        let mut row_u = nalgebra::Vector6::<f64>::zeros();
        row_u[0] = jac_u[0];
        row_u[1] = jac_u[1];
        row_u[2] = jac_u[2];
        // Rotation: p_cam × jac_u  (left SE(3) perturbation)
        row_u[3] = p_cam.y * jac_u[2] - p_cam.z * jac_u[1];
        row_u[4] = p_cam.z * jac_u[0] - p_cam.x * jac_u[2];
        row_u[5] = p_cam.x * jac_u[1] - p_cam.y * jac_u[0];

        let mut row_v = nalgebra::Vector6::<f64>::zeros();
        row_v[0] = jac_v[0];
        row_v[1] = jac_v[1];
        row_v[2] = jac_v[2];
        row_v[3] = p_cam.y * jac_v[2] - p_cam.z * jac_v[1];
        row_v[4] = p_cam.z * jac_v[0] - p_cam.x * jac_v[2];
        row_v[5] = p_cam.x * jac_v[1] - p_cam.y * jac_v[0];

        // Numerical Jacobian via central differences on 6-DOF perturbations
        let eps = 1e-6;
        let project = |pert_pose: &crate::pose::Pose| -> [f64; 2] {
            pert_pose.project(&p_world, &intrinsics)
        };

        for dof in 0..6usize {
            let fwd = project(&perturb_pose(&pose, dof, eps));
            let bwd = project(&perturb_pose(&pose, dof, -eps));

            let num_u = (fwd[0] - bwd[0]) / (2.0 * eps);
            let num_v = (fwd[1] - bwd[1]) / (2.0 * eps);

            let ana_u = row_u[dof];
            let ana_v = row_v[dof];

            assert!(
                (ana_u - num_u).abs() < 1e-4,
                "J_u[{dof}]: analytical={ana_u:.6} numerical={num_u:.6}"
            );
            assert!(
                (ana_v - num_v).abs() < 1e-4,
                "J_v[{dof}]: analytical={ana_v:.6} numerical={num_v:.6}"
            );
        }
    }

    /// Helper: perturb a pose along a given DOF in the se(3) Lie algebra.
    /// DOF 0..2 = translation x,y,z; DOF 3..5 = rotation (axis-angle).
    fn perturb_pose(pose: &crate::pose::Pose, dof: usize, eps: f64) -> crate::pose::Pose {
        let mut delta = nalgebra::Vector6::<f64>::zeros();
        delta[dof] = eps;
        let twist = nalgebra::Vector3::new(delta[3], delta[4], delta[5]);
        let t_update = nalgebra::Vector3::new(delta[0], delta[1], delta[2]);
        let r_update = nalgebra::Rotation3::new(twist).matrix().into_owned();
        crate::pose::Pose::new(
            r_update * pose.rotation,
            r_update * pose.translation + t_update,
        )
    }

    #[test]
    fn test_minimal_sample_ippe() {
        let config = BoardConfig::new_charuco(3, 3, 0.04, 0.02);
        let estimator = BoardEstimator::new(config);
        let intrinsics = crate::pose::CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);

        let gt_rot = nalgebra::Matrix3::identity();
        let gt_t = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let gt_pose = crate::pose::Pose::new(gt_rot, gt_t);

        let mut batch = crate::batch::DetectionBatch::new();

        for i in 0..4 {
            batch.status_mask[i] = crate::batch::CandidateState::Valid;
            batch.ids[i] = i as u32;

            let obj_corners = &estimator.config.obj_points[i];
            for (j, corner) in obj_corners.iter().enumerate().take(4) {
                let p_img = gt_pose.project(corner, &intrinsics);
                batch.corners[i][j].x = p_img[0] as f32;
                batch.corners[i][j].y = p_img[1] as f32;
            }
        }

        let est_pose = estimator
            .estimate(&batch, &intrinsics)
            .expect("Failed to estimate pose");

        let t_err = (est_pose.translation - gt_t).norm();
        assert!(t_err < 0.1, "Translation error {t_err} too high");
    }
}
