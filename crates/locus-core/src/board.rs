use nalgebra::Vector3;

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
}
