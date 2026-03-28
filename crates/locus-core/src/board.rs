use multiversion::multiversion;
use nalgebra::{Matrix3, Vector3, Matrix6, Vector6, Rotation3};
use crate::batch::{DetectionBatch, CandidateState};
use crate::pose::{Pose, CameraIntrinsics};
use crate::workspace::WORKSPACE_ARENA;
use crate::decoder::Homography;

/// A 3D pose result for the entire board.
pub type BoardPose = Pose;

/// Evaluates reprojection errors for planar points using SIMD (auto-vectorized FMA).
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
fn unweighted_gn_step(
    pose: &Pose,
    intrinsics: &CameraIntrinsics,
    obj_x: &[f64],
    obj_y: &[f64],
    img_u: &[f64],
    img_v: &[f64],
    valid_mask: &[u64; 16],
    valid_indices: &[usize],
    num_tags: usize,
) -> Pose {
    let mut jtj = Matrix6::<f64>::zeros();
    let mut jtr = Vector6::<f64>::zeros();
    
    let fx = intrinsics.fx;
    let fy = intrinsics.fy;
    let cx = intrinsics.cx;
    let cy = intrinsics.cy;

    for i in 0..num_tags {
        let batch_idx = valid_indices[i];
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

            let du_dp = Vector3::new(fx * z_inv, 0.0, -fx * p_cam.x * z_inv2);
            let dv_dp = Vector3::new(0.0, fy * z_inv, -fy * p_cam.y * z_inv2);

            let mut row_u = Vector6::zeros();
            row_u[0] = du_dp[0];
            row_u[1] = du_dp[1];
            row_u[2] = du_dp[2];
            row_u[3] = du_dp[1] * p_cam.z - du_dp[2] * p_cam.y;
            row_u[4] = du_dp[2] * p_cam.x - du_dp[0] * p_cam.z;
            row_u[5] = du_dp[0] * p_cam.y - du_dp[1] * p_cam.x;

            let mut row_v = Vector6::zeros();
            row_v[0] = dv_dp[0];
            row_v[1] = dv_dp[1];
            row_v[2] = dv_dp[2];
            row_v[3] = dv_dp[1] * p_cam.z - dv_dp[2] * p_cam.y;
            row_v[4] = dv_dp[2] * p_cam.x - dv_dp[0] * p_cam.z;
            row_v[5] = dv_dp[0] * p_cam.y - dv_dp[1] * p_cam.x;

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
    pub fn estimate(&self, batch: &DetectionBatch, intrinsics: &CameraIntrinsics) -> Option<BoardPose> {
        WORKSPACE_ARENA.with(|cell| {
            let mut arena = cell.borrow_mut();
            
            // Collect valid tag indices without heap allocation
            // Max candidates is 1024. We can allocate a slice from the arena.
            let valid_indices = arena.alloc_slice_fill_default(crate::batch::MAX_CANDIDATES);
            let mut num_valid = 0;
            
            for i in 0..crate::batch::MAX_CANDIDATES {
                if batch.status_mask[i] == CandidateState::Active {
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
            let sample_indices = [&valid_indices[0], &valid_indices[1], &valid_indices[2], &valid_indices[3]];
            
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

            // Prepare flat arrays for SIMD Consensus Evaluation
            let total_corners = num_valid * 4;
            let obj_x = arena.alloc_slice_fill_default(total_corners);
            let obj_y = arena.alloc_slice_fill_default(total_corners);
            let img_u = arena.alloc_slice_fill_default(total_corners);
            let img_v = arena.alloc_slice_fill_default(total_corners);
            let errors = arena.alloc_slice_fill_default(total_corners);

            for i in 0..num_valid {
                let idx = valid_indices[i];
                let id = batch.ids[idx] as usize;
                let obj_corners = &self.config.obj_points[id];
                let img_corners = &batch.corners[idx];

                for j in 0..4 {
                    let flat_idx = i * 4 + j;
                    obj_x[flat_idx] = obj_corners[j].x;
                    obj_y[flat_idx] = obj_corners[j].y;
                    img_u[flat_idx] = f64::from(img_corners[j].x);
                    img_v[flat_idx] = f64::from(img_corners[j].y);
                }
            }

            let tau_sq = 2.0 * 2.0; // relaxed geometric threshold (~2.0 pixels)
            
            let mut best_pose = None;
            let mut best_inliers = 0;
            let mut _best_inlier_mask = [0u64; 16];

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
                        &pose, intrinsics, obj_x, obj_y, img_u, img_v, &mask, valid_indices, num_valid
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
                        _best_inlier_mask = lo_mask;
                        best_pose = Some(lo_pose);
                        
                        if lo_inliers as f64 / num_valid as f64 > 0.95 {
                            break; // Early termination
                        }
                    } else if inlier_count > best_inliers {
                        best_inliers = inlier_count;
                        _best_inlier_mask = mask;
                        best_pose = Some(pose);
                    }
                }
            }

            // Next phase will use _best_inlier_mask for Local Optimization (LO) Handoff

            arena.reset();
            best_pose
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
                    let tr = Vector3::new(x_offset + margin + marker_length, y_offset + margin, 0.0);
                    let br = Vector3::new(x_offset + margin + marker_length, y_offset + margin + marker_length, 0.0);
                    let bl = Vector3::new(x_offset + margin, y_offset + margin + marker_length, 0.0);

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
        assert_eq!(board.square_length, 0.04);
        assert_eq!(board.marker_length, 0.02);
        assert_eq!(board.obj_points.len(), 12);

        // Check the first marker, which should be at r=0, c=1
        let m0 = &board.obj_points[0];
        
        // expected top-left of square (0, 1): x = 0.04, y = 0.0
        // expected top-left of marker: x = 0.04 + 0.01 = 0.05, y = 0.0 + 0.01 = 0.01
        assert_eq!(m0[0].x, 0.05);
        assert_eq!(m0[0].y, 0.01);
        assert_eq!(m0[0].z, 0.0);
        
        // tr
        assert_eq!(m0[1].x, 0.07);
        assert_eq!(m0[1].y, 0.01);
        
        // br
        assert_eq!(m0[2].x, 0.07);
        assert_eq!(m0[2].y, 0.03);

        // bl
        assert_eq!(m0[3].x, 0.05);
        assert_eq!(m0[3].y, 0.03);
    }

    #[test]
    fn test_aprilgrid_board_initialization() {
        let board = BoardConfig::new_aprilgrid(3, 4, 0.01, 0.05);
        
        assert_eq!(board.rows, 3);
        assert_eq!(board.cols, 4);
        assert!((board.square_length - 0.06).abs() < 1e-9);
        assert_eq!(board.marker_length, 0.05);
        assert_eq!(board.obj_points.len(), 12);

        // check marker at r=1, c=2
        let idx = 1 * 4 + 2;
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
            &r, &t, fx, fy, cx, cy, &obj_x, &obj_y, &img_u, &img_v, &mut errors
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
            &pose, &intrinsics, &obj_x, &obj_y, &img_u, &img_v, &valid_mask, &valid_indices, 1
        );
        
        // Check if refined pose is closer to GT than initial pose
        let initial_err = (pose.translation - gt_t).norm();
        let refined_err = (refined_pose.translation - gt_t).norm();
        assert!(refined_err < initial_err);
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
            batch.status_mask[i] = crate::batch::CandidateState::Active;
            batch.ids[i] = i as u32;
            
            let obj_corners = &estimator.config.obj_points[i];
            for j in 0..4 {
                let p_img = gt_pose.project(&obj_corners[j], &intrinsics);
                batch.corners[i][j].x = p_img[0] as f32;
                batch.corners[i][j].y = p_img[1] as f32;
            }
        }
        
        let est_pose = estimator.estimate(&batch, &intrinsics).expect("Failed to estimate pose");
        
        let t_err = (est_pose.translation - gt_t).norm();
        assert!(t_err < 0.1, "Translation error {} too high", t_err);
    }
}
