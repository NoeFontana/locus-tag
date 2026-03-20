//! SIMD-Accelerated Connected Components Labeling (CCL) with Fused Thresholding.
//!
//! This module implements a high-performance segmentation pipeline that defeats the "memory wall"
//! by fusing adaptive thresholding with Run-Length Encoding (RLE) extraction. It processes images
//! in 1D segments rather than individual pixels, drastically reducing memory bandwidth requirements
//! and branch mispredictions.

use crate::image::ImageView;
use crate::segmentation::{ComponentStats, LabelResult, UnionFind};
use bumpalo::Bump;

/// A 1D Run-Length Encoded (RLE) segment representing contiguous foreground pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RleSegment {
    /// The row (Y coordinate) of the segment.
    pub y: u16,
    /// The starting column (X coordinate) of the segment (inclusive).
    pub start_x: u16,
    /// The ending column (X coordinate) of the segment (exclusive).
    pub end_x: u16,
    /// The component label assigned to this segment. Uninitialized/unassigned is 0.
    pub label: u32,
}

impl RleSegment {
    /// Create a new RleSegment with an unassigned label.
    #[must_use]
    pub const fn new(y: u16, start_x: u16, end_x: u16) -> Self {
        Self {
            y,
            start_x,
            end_x,
            label: 0,
        }
    }
}

/// Architecture-specific SIMD scanline processing.
pub mod simd_scanline;

/// Baseline scalar implementation of the fused Threshold + RLE extraction kernel.
/// Extracts runs of black pixels (value < threshold) from the image.
#[allow(dead_code)]
#[must_use]
pub fn extract_rle_segments_scalar(img: &ImageView, threshold_map: &[u8]) -> Vec<RleSegment> {
    let mut segments = Vec::new();
    let height = img.height as u16;

    for y in 0..height {
        let y_usize = y as usize;
        let src_row = img.get_row(y_usize);
        let thresh_row = &threshold_map[y_usize * img.width..(y_usize + 1) * img.width];

        process_row_scalar(src_row, thresh_row, y, &mut segments);
    }

    segments
}

/// Scalar fallback for a single row.
pub fn process_row_scalar(
    src_row: &[u8],
    thresh_row: &[u8],
    y: u16,
    segments: &mut Vec<RleSegment>,
) {
    let mut in_segment = false;
    let mut start_x = 0;
    let width = src_row.len();

    for (x, (&s, &t)) in src_row.iter().zip(thresh_row.iter()).enumerate() {
        let is_foreground = s < t;

        if is_foreground && !in_segment {
            in_segment = true;
            start_x = x as u16;
        } else if !is_foreground && in_segment {
            in_segment = false;
            segments.push(RleSegment::new(y, start_x, x as u16));
        }
    }

    if in_segment {
        segments.push(RleSegment::new(y, start_x, width as u16));
    }
}

/// Performs Light-Speed Labeling (LSL) / Run-based Union-Find on the extracted RLE segments.
/// Fully resolves equivalences and outputs the 2D label map expected by the rest of the pipeline.
#[allow(clippy::too_many_lines)]
pub fn label_components_lsl<'a>(
    arena: &'a Bump,
    img: &ImageView,
    threshold_map: &[u8],
    use_8_connectivity: bool,
    min_area: u32,
) -> LabelResult<'a> {
    let mut runs = simd_scanline::extract_rle_segments(img, threshold_map);

    // Assign consecutive IDs for Union-Find using the label field
    for (id, run) in runs.iter_mut().enumerate() {
        run.label = id as u32;
    }

    if runs.is_empty() {
        return LabelResult {
            labels: arena.alloc_slice_fill_copy(img.width * img.height, 0u32),
            component_stats: Vec::new(),
        };
    }

    let mut uf = UnionFind::new_in(arena, runs.len());
    let mut curr_row_range = 0..0;
    let mut i = 0;

    while i < runs.len() {
        let y = runs[i].y;
        let start = i;
        while i < runs.len() && runs[i].y == y {
            i += 1;
        }
        let prev_row_range = curr_row_range;
        curr_row_range = start..i;

        if y > 0 && !prev_row_range.is_empty() && runs[prev_row_range.start].y == y - 1 {
            let mut p_idx = prev_row_range.start;
            for c_idx in curr_row_range.clone() {
                let curr = &runs[c_idx];
                if use_8_connectivity {
                    // 8-connectivity: [start_x, end_x)
                    // overlap diagonally if prev.end_x >= curr.start_x and prev.start_x <= curr.end_x
                    while p_idx < prev_row_range.end && runs[p_idx].end_x < curr.start_x {
                        p_idx += 1;
                    }
                    let mut temp_p = p_idx;
                    while temp_p < prev_row_range.end && runs[temp_p].start_x <= curr.end_x {
                        uf.union(curr.label, runs[temp_p].label);
                        temp_p += 1;
                    }
                } else {
                    // 4-connectivity
                    while p_idx < prev_row_range.end && runs[p_idx].end_x <= curr.start_x {
                        p_idx += 1;
                    }
                    let mut temp_p = p_idx;
                    while temp_p < prev_row_range.end && runs[temp_p].start_x < curr.end_x {
                        uf.union(curr.label, runs[temp_p].label);
                        temp_p += 1;
                    }
                }
            }
        }
    }

    // Collect stats per root and assign labels
    let mut root_to_temp_idx = vec![usize::MAX; runs.len()];
    let mut temp_stats = Vec::new();

    for run in &runs {
        let root = uf.find(run.label) as usize;
        if root_to_temp_idx[root] == usize::MAX {
            root_to_temp_idx[root] = temp_stats.len();
            temp_stats.push(ComponentStats {
                first_pixel_x: run.start_x,
                first_pixel_y: run.y,
                min_x: u16::MAX,
                max_x: 0,
                min_y: u16::MAX,
                max_y: 0,
                pixel_count: 0,
            });
        }
        let s_idx = root_to_temp_idx[root];
        let stats = &mut temp_stats[s_idx];
        stats.min_x = stats.min_x.min(run.start_x);
        stats.max_x = stats.max_x.max(run.end_x - 1);
        stats.min_y = stats.min_y.min(run.y);
        stats.max_y = stats.max_y.max(run.y);
        stats.pixel_count += u32::from(run.end_x - run.start_x);
    }

    let mut component_stats = Vec::with_capacity(temp_stats.len());
    let mut root_to_final_label = vec![0u32; runs.len()];
    let mut next_label = 1u32;

    for (root, root_to_temp) in root_to_temp_idx.iter().enumerate() {
        if *root_to_temp != usize::MAX {
            let s = temp_stats[*root_to_temp];
            if s.pixel_count >= min_area {
                component_stats.push(s);
                root_to_final_label[root] = next_label;
                next_label += 1;
            }
        }
    }

    let labels = arena.alloc_slice_fill_copy(img.width * img.height, 0u32);
    let width = img.width;

    for run in &runs {
        let root = uf.find(run.label) as usize;
        let label = root_to_final_label[root];
        if label > 0 {
            let row_off = run.y as usize * width;
            labels[row_off + run.start_x as usize..row_off + run.end_x as usize].fill(label);
        }
    }

    LabelResult {
        labels,
        component_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_components_lsl_red_phase() {
        let arena = Bump::new();
        let width = 8;
        let height = 4;

        // Image with two separated 2x2 blocks:
        // row 0: . x x . . x x .
        // row 1: . x x . . x x .
        // row 2: . . . . . . . .
        // row 3: . . . . . . . .

        let mut pixels = vec![200u8; width * height];
        pixels[1] = 50;
        pixels[2] = 50;
        pixels[5] = 50;
        pixels[6] = 50;
        pixels[8 + 1] = 50;
        pixels[8 + 2] = 50;
        pixels[8 + 5] = 50;
        pixels[8 + 6] = 50;

        let threshold_map = vec![128u8; width * height];
        let img = ImageView::new(&pixels, width, height, width).expect("Valid test image");

        let result = label_components_lsl(&arena, &img, &threshold_map, true, 1);

        assert_eq!(result.component_stats.len(), 2);

        // Assuming label 1 is left block and label 2 is right block
        let mut found_labels = std::collections::HashSet::new();
        found_labels.insert(result.labels[1]);
        found_labels.insert(result.labels[5]);

        assert_eq!(found_labels.len(), 2);
        assert!(!found_labels.contains(&0));

        // Verify stats
        assert_eq!(result.component_stats[0].pixel_count, 4);
        assert_eq!(result.component_stats[1].pixel_count, 4);
    }

    #[test]
    fn test_extract_rle_segments_scalar() {
        let width = 8;
        let height = 2;
        // Image with two black segments on first row, one on second row.
        // Black is < threshold (e.g. 100 vs 128)
        let pixels = vec![
            200, 50, 50, 200, 50, 200, 200, 200, // Row 0: RLE at [1, 3) and [4, 5)
            50, 50, 50, 50, 200, 200, 200, 200, // Row 1: RLE at [0, 4)
        ];
        let threshold_map = vec![128u8; width * height];

        let img = ImageView::new(&pixels, width, height, width).expect("Valid test image");
        let segments = extract_rle_segments_scalar(&img, &threshold_map);

        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], RleSegment::new(0, 1, 3));
        assert_eq!(segments[1], RleSegment::new(0, 4, 5));
        assert_eq!(segments[2], RleSegment::new(1, 0, 4));
    }
}
