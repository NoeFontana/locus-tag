//! Board-level configuration and layout utilities.

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
    pub fn new_charuco(
        rows: usize,
        cols: usize,
        square_length: f64,
        marker_length: f64,
    ) -> Self {
        let mut obj_points = vec![None; (rows * cols + 1) / 2];
        let board_width = cols as f64 * square_length;
        let board_height = rows as f64 * square_length;

        let offset_x = -board_width / 2.0;
        let offset_y = -board_height / 2.0;
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
    pub fn new_aprilgrid(
        rows: usize,
        cols: usize,
        spacing: f64,
        marker_length: f64,
    ) -> Self {
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
