use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use rayon::prelude::*;

/// A disjoint-set forest (Union-Find) with path compression and rank optimization.
pub struct UnionFind<'a> {
    parent: &'a mut [u32],
    rank: &'a mut [u8],
}

impl<'a> UnionFind<'a> {
    /// Create a new UnionFind structure backed by the provided arena.
    pub fn new_in(arena: &'a Bump, size: usize) -> Self {
        let parent = arena.alloc_slice_fill_with(size, |i| i as u32);
        let rank = arena.alloc_slice_fill_copy(size, 0u8);
        Self { parent, rank }
    }

    /// Find the representative (root) of the set containing `i`.
    #[inline]
    pub fn find(&mut self, i: u32) -> u32 {
        let mut root = i;
        while self.parent[root as usize] != root {
            self.parent[root as usize] = self.parent[self.parent[root as usize] as usize];
            root = self.parent[root as usize];
        }
        root
    }

    /// Unite the sets containing `i` and `j`.
    #[inline]
    pub fn union(&mut self, i: u32, j: u32) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            match self.rank[root_i as usize].cmp(&self.rank[root_j as usize]) {
                std::cmp::Ordering::Less => self.parent[root_i as usize] = root_j,
                std::cmp::Ordering::Greater => self.parent[root_j as usize] = root_i,
                std::cmp::Ordering::Equal => {
                    self.parent[root_i as usize] = root_j;
                    self.rank[root_j as usize] += 1;
                }
            }
        }
    }
}

/// Bounding box and statistics for a connected component.
#[derive(Clone, Copy, Debug)]
pub struct ComponentStats {
    /// Minimum x coordinate.
    pub min_x: u16,
    /// Maximum x coordinate.
    pub max_x: u16,
    /// Minimum y coordinate.
    pub min_y: u16,
    /// Maximum y coordinate.
    pub max_y: u16,
    /// Total number of pixels in the component.
    pub pixel_count: u32,
    /// First encountered pixel X (for boundary start).
    pub first_pixel_x: u16,
    /// First encountered pixel Y (for boundary start).
    pub first_pixel_y: u16,
}

impl Default for ComponentStats {
    fn default() -> Self {
        Self {
            min_x: u16::MAX,
            max_x: 0,
            min_y: u16::MAX,
            max_y: 0,
            pixel_count: 0,
            first_pixel_x: 0,
            first_pixel_y: 0,
        }
    }
}

/// Result of connected component labeling.
pub struct LabelResult<'a> {
    /// Flat array of pixel labels (row-major).
    pub labels: &'a [u32],
    /// Statistics for each component (indexed by label - 1).
    pub component_stats: Vec<ComponentStats>,
}

/// A detected run of background pixels in a row.
#[derive(Clone, Copy, Debug)]
struct Run {
    y: u32,
    x_start: u32,
    x_end: u32,
    id: u32,
}

/// Label connected components in a binary image.
pub fn label_components<'a>(
    arena: &'a Bump,
    binary: &[u8],
    width: usize,
    height: usize,
    use_8_connectivity: bool,
) -> &'a [u32] {
    label_components_with_stats(arena, binary, width, height, use_8_connectivity).labels
}

/// Label components and compute bounding box stats for each.
#[allow(clippy::too_many_lines)]
pub fn label_components_with_stats<'a>(
    arena: &'a Bump,
    binary: &[u8],
    width: usize,
    height: usize,
    use_8_connectivity: bool,
) -> LabelResult<'a> {
    // Pass 1: Extract runs - Optimized with Rayon parallel processing
    let all_runs: Vec<Vec<Run>> = binary
        .par_chunks(width)
        .enumerate()
        .map(|(y, row)| {
            let mut row_runs = Vec::new();
            let mut x = 0;
            while x < width {
                if let Some(pos) = row[x..].iter().position(|&p| p == 0) {
                    let start = x + pos;
                    let mut end = start + 1;
                    while end < width && row[end] == 0 {
                        end += 1;
                    }
                    row_runs.push(Run {
                        y: y as u32,
                        x_start: start as u32,
                        x_end: (end - 1) as u32,
                        id: 0, // Assigned correctly during flattening
                    });
                    x = end;
                } else {
                    break;
                }
            }
            row_runs
        })
        .collect();

    let total_runs: usize = all_runs.iter().map(|r| r.len()).sum();
    let mut runs: BumpVec<Run> = BumpVec::with_capacity_in(total_runs, arena);
    for (id, mut run) in all_runs.into_iter().flatten().enumerate() {
        run.id = id as u32;
        runs.push(run);
    }

    if runs.is_empty() {
        return LabelResult {
            labels: arena.alloc_slice_fill_copy(width * height, 0u32),
            component_stats: Vec::new(),
        };
    }

    let mut uf = UnionFind::new_in(arena, runs.len());
    let mut curr_row_range = 0..0; // Initialize curr_row_range
    let mut i = 0;

    // Pass 2: Link runs between adjacent rows using two-pointer linear scan
    while i < runs.len() {
        let y = runs[i].y;

        // Identify the range of runs in the current row
        let start = i;
        while i < runs.len() && runs[i].y == y {
            i += 1;
        }
        let prev_row_range = curr_row_range; // Now correctly uses the previously assigned curr_row_range
        curr_row_range = start..i;

        if y > 0 && !prev_row_range.is_empty() && runs[prev_row_range.start].y == y - 1 {
            let mut p_idx = prev_row_range.start;
            for c_idx in curr_row_range.clone() {
                let curr = &runs[c_idx];

                if use_8_connectivity {
                    // 8-connectivity check: overlap if [xs1, xe1] and [xs2-1, xe2+1] intersect
                    while p_idx < prev_row_range.end && runs[p_idx].x_end + 1 < curr.x_start {
                        p_idx += 1;
                    }
                    let mut temp_p = p_idx;
                    while temp_p < prev_row_range.end && runs[temp_p].x_start <= curr.x_end + 1 {
                        uf.union(curr.id, runs[temp_p].id);
                        temp_p += 1;
                    }
                } else {
                    // 4-connectivity check: overlap if [xs1, xe1] and [xs2, xe2] intersect
                    while p_idx < prev_row_range.end && runs[p_idx].x_end < curr.x_start {
                        p_idx += 1;
                    }
                    let mut temp_p = p_idx;
                    while temp_p < prev_row_range.end && runs[temp_p].x_start <= curr.x_end {
                        uf.union(curr.id, runs[temp_p].id);
                        temp_p += 1;
                    }
                }
            }
        }
    }

    // Pass 3: Collect stats per root and assign labels
    let mut root_to_label: Vec<u32> = vec![0; runs.len()];
    let mut component_stats: Vec<ComponentStats> = Vec::new();
    let mut next_label = 1u32;

    // Pre-resolve roots to avoid repeated find() in Pass 4
    let mut run_roots = Vec::with_capacity(runs.len());

    for run in &runs {
        let root = uf.find(run.id) as usize;
        run_roots.push(root);
        if root_to_label[root] == 0 {
            root_to_label[root] = next_label;
            next_label += 1;
            let new_stat = ComponentStats {
                first_pixel_x: run.x_start as u16,
                first_pixel_y: run.y as u16,
                ..Default::default()
            };
            component_stats.push(new_stat);
        }
        let label_idx = (root_to_label[root] - 1) as usize;
        let stats = &mut component_stats[label_idx];
        stats.min_x = stats.min_x.min(run.x_start as u16);
        stats.max_x = stats.max_x.max(run.x_end as u16);
        stats.min_y = stats.min_y.min(run.y as u16);
        stats.max_y = stats.max_y.max(run.y as u16);
        stats.pixel_count += run.x_end - run.x_start + 1;
    }

    // Pass 4: Assign labels to pixels - Optimized with slice fill
    let labels = arena.alloc_slice_fill_copy(width * height, 0u32);
    for (run, root) in runs.iter().zip(run_roots) {
        let label = root_to_label[root];
        let row_off = run.y as usize * width;
        labels[row_off + run.x_start as usize..=row_off + run.x_end as usize].fill(label);
    }

    LabelResult {
        labels,
        component_stats,
    }
}

/// Threshold-model-aware connected component labeling.
///
/// Instead of connecting pixels based on binary value (0 = black), this function
/// connects pixels based on their *signed deviation* from the local threshold.
/// This preserves the shape of small tag corners where pure binary would see noise.
///
/// # Arguments
/// * `arena` - Bump allocator for temporary storage
/// * `grayscale` - Original grayscale image
/// * `threshold_map` - Per-pixel threshold values (from adaptive thresholding)
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
/// Same `LabelResult` as `label_components_with_stats`, but with improved connectivity
/// for small features.
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_wrap)]
pub fn label_components_threshold_model<'a>(
    arena: &'a Bump,
    grayscale: &[u8],
    threshold_map: &[u8],
    width: usize,
    height: usize,
    use_8_connectivity: bool,
    min_area: u32,
    margin: i16,
) -> LabelResult<'a> {
    // Compute signed deviation: negative = dark (below threshold), positive = light
    // We only care about dark regions (tag interior) for now
    let mut runs = BumpVec::new_in(arena);

    // Pass 1: Extract runs of "consistently dark" pixels
    // A pixel is "dark" if (grayscale - threshold) < -margin
    // This is more robust than binary == 0 because it considers the local model

    let all_runs: Vec<Vec<Run>> = (0..height)
        .into_par_iter()
        .map(|y| {
            let row_gs = &grayscale[y * width..(y + 1) * width];
            let row_th = &threshold_map[y * width..(y + 1) * width];
            let mut row_runs = Vec::new();

            let mut x = 0;
            while x < width {
                // Find start of dark run: gs < threshold - margin
                let deviation = i16::from(row_gs[x]) - i16::from(row_th[x]);
                if deviation < -margin {
                    let start = x;
                    x += 1;
                    // Continue while consistently dark
                    while x < width {
                        let dev = i16::from(row_gs[x]) - i16::from(row_th[x]);
                        if dev >= -margin {
                            break;
                        }
                        x += 1;
                    }
                    row_runs.push(Run {
                        y: y as u32,
                        x_start: start as u32,
                        x_end: (x - 1) as u32,
                        id: 0, // Assigned correctly later
                    });
                } else {
                    x += 1;
                }
            }
            row_runs
        })
        .collect();

    for (id, mut run) in all_runs.into_iter().flatten().enumerate() {
        run.id = id as u32;
        runs.push(run);
    }

    if runs.is_empty() {
        return LabelResult {
            labels: arena.alloc_slice_fill_copy(width * height, 0u32),
            component_stats: Vec::new(),
        };
    }

    // Pass 2-4 are the same as label_components_with_stats
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
                    while p_idx < prev_row_range.end && runs[p_idx].x_end + 1 < curr.x_start {
                        p_idx += 1;
                    }
                    let mut temp_p = p_idx;
                    while temp_p < prev_row_range.end && runs[temp_p].x_start <= curr.x_end + 1 {
                        uf.union(curr.id, runs[temp_p].id);
                        temp_p += 1;
                    }
                } else {
                    while p_idx < prev_row_range.end && runs[p_idx].x_end < curr.x_start {
                        p_idx += 1;
                    }
                    let mut temp_p = p_idx;
                    while temp_p < prev_row_range.end && runs[temp_p].x_start <= curr.x_end {
                        uf.union(curr.id, runs[temp_p].id);
                        temp_p += 1;
                    }
                }
            }
        }
    }

    // Pass 3: Aggregate stats for ALL potential components
    let mut root_to_temp_idx = vec![usize::MAX; runs.len()];
    let mut temp_stats = Vec::new();

    for run in &runs {
        let root = uf.find(run.id) as usize;
        if root_to_temp_idx[root] == usize::MAX {
            root_to_temp_idx[root] = temp_stats.len();
            temp_stats.push(ComponentStats {
                first_pixel_x: run.x_start as u16,
                first_pixel_y: run.y as u16,
                ..Default::default()
            });
        }
        let s_idx = root_to_temp_idx[root];
        let stats = &mut temp_stats[s_idx];
        stats.min_x = stats.min_x.min(run.x_start as u16);
        stats.max_x = stats.max_x.max(run.x_end as u16);
        stats.min_y = stats.min_y.min(run.y as u16);
        stats.max_y = stats.max_y.max(run.y as u16);
        stats.pixel_count += run.x_end - run.x_start + 1;
    }

    // Pass 4: Filter by area and assign final labels
    let mut component_stats = Vec::with_capacity(temp_stats.len());
    let mut root_to_final_label = vec![0u32; runs.len()];
    let mut next_label = 1u32;

    for root in 0..runs.len() {
        let s_idx = root_to_temp_idx[root];
        if s_idx != usize::MAX {
            let s = temp_stats[s_idx];
            if s.pixel_count >= min_area {
                component_stats.push(s);
                root_to_final_label[root] = next_label;
                next_label += 1;
            }
        }
    }

    // Pass 5: Parallel label writing for surviving components
    let labels = arena.alloc_slice_fill_copy(width * height, 0u32);
    let mut runs_by_y: Vec<Vec<(usize, usize, u32)>> = vec![Vec::new(); height];

    for run in &runs {
        let root = uf.find(run.id) as usize;
        let label = root_to_final_label[root];
        if label > 0 {
            runs_by_y[run.y as usize].push((run.x_start as usize, run.x_end as usize, label));
        }
    }

    labels
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for &(x_start, x_end, label) in &runs_by_y[y] {
                row[x_start..=x_end].fill(label);
            }
        });

    LabelResult {
        labels,
        component_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;
    use proptest::prelude::*;

    #[test]
    fn test_union_find() {
        let arena = Bump::new();
        let mut uf = UnionFind::new_in(&arena, 10);

        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(5, 6);

        assert_eq!(uf.find(1), uf.find(3));
        assert_eq!(uf.find(1), uf.find(2));
        assert_ne!(uf.find(1), uf.find(5));

        uf.union(3, 5);
        assert_eq!(uf.find(1), uf.find(6));
    }

    #[test]
    fn test_label_components_simple() {
        let arena = Bump::new();
        // 6x6 image with two separate 2x2 squares that are NOT 8-connected.
        // 0 = background (black), 255 = foreground (white)
        // Tag detector looks for black components (0)
        let binary = [
            0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255,
        ];
        let width = 6;
        let height = 6;

        let result = label_components_with_stats(&arena, &binary, width, height, false);

        assert_eq!(result.component_stats.len(), 2);

        // Component 1 (top-left)
        let s1 = result.component_stats[0];
        assert_eq!(s1.pixel_count, 4);
        assert_eq!(s1.min_x, 0);
        assert_eq!(s1.max_x, 1);
        assert_eq!(s1.min_y, 0);
        assert_eq!(s1.max_y, 1);

        // Component 2 (middle-rightish)
        let s2 = result.component_stats[1];
        assert_eq!(s2.pixel_count, 4);
        assert_eq!(s2.min_x, 3);
        assert_eq!(s2.max_x, 4);
        assert_eq!(s2.min_y, 3);
        assert_eq!(s2.max_y, 4);
    }

    #[test]
    fn test_segmentation_with_decimation() {
        let arena = Bump::new();
        let width = 32;
        let height = 32;
        let mut binary = vec![255u8; width * height];
        // Draw a 10x10 black square at (10,10)
        for y in 10..20 {
            for x in 10..20 {
                binary[y * width + x] = 0;
            }
        }

        use crate::image::ImageView;
        let img = ImageView::new(&binary, width, height, width).expect("valid creation");

        // Decimate by 2 -> 16x16
        let mut decimated_data = vec![0u8; 16 * 16];
        let decimated_img = img
            .decimate_to(2, &mut decimated_data)
            .expect("decimation failed");
        // In decimated image, square should be roughly at (5,5) with size 5x5
        let result = label_components_with_stats(&arena, decimated_img.data, 16, 16, true);

        assert_eq!(result.component_stats.len(), 1);
        let s = result.component_stats[0];
        assert_eq!(s.pixel_count, 25);
        assert_eq!(s.min_x, 5);
        assert_eq!(s.max_x, 9);
        assert_eq!(s.min_y, 5);
        assert_eq!(s.max_y, 9);
    }

    proptest! {
        #[test]
        fn prop_union_find_reflexivity(size in 1..1000usize) {
            let arena = Bump::new();
            let mut uf = UnionFind::new_in(&arena, size);
            for i in 0..size as u32 {
                assert_eq!(uf.find(i), i);
            }
        }

        #[test]
        fn prop_union_find_transitivity(size in 1..1000usize, pairs in prop::collection::vec((0..1000u32, 0..1000u32), 0..100)) {
            let arena = Bump::new();
            let real_size = size.max(1001); // Ensure indices are in range
            let mut uf = UnionFind::new_in(&arena, real_size);

            for (a, b) in pairs {
                let a = a % real_size as u32;
                let b = b % real_size as u32;
                uf.union(a, b);
                assert_eq!(uf.find(a), uf.find(b));
            }
        }

        #[test]
        fn prop_label_components_no_panic(
            width in 1..64usize,
            height in 1..64usize,
            data in prop::collection::vec(0..=1u8, 64 * 64)
        ) {
            let arena = Bump::new();
            let binary: Vec<u8> = data.iter().map(|&b| if b == 0 { 0 } else { 255 }).collect();
            let real_width = width.min(64);
            let real_height = height.min(64);
            let slice = &binary[..real_width * real_height];

            let result = label_components_with_stats(&arena, slice, real_width, real_height, true);

            for stat in result.component_stats {
                assert!(stat.pixel_count > 0);
                assert!(stat.max_x < real_width as u16);
                assert!(stat.max_y < real_height as u16);
                assert!(stat.min_x <= stat.max_x);
                assert!(stat.min_y <= stat.max_y);
            }
        }
    }

    // ========================================================================
    // SEGMENTATION ROBUSTNESS TESTS
    // ========================================================================

    use crate::config::TagFamily;
    use crate::image::ImageView;
    use crate::test_utils::{TestImageParams, generate_test_image_with_params};
    use crate::threshold::ThresholdEngine;

    /// Helper: Generate a binarized tag image at the given size.
    fn generate_binarized_tag(tag_size: usize, canvas_size: usize) -> (Vec<u8>, [[f64; 2]; 4]) {
        let params = TestImageParams {
            family: TagFamily::AprilTag36h11,
            id: 0,
            tag_size,
            canvas_size,
            ..Default::default()
        };

        let (data, corners) = generate_test_image_with_params(&params);
        let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

        let engine = ThresholdEngine::new();
        let stats = engine.compute_tile_stats(&img);
        let mut binary = vec![0u8; canvas_size * canvas_size];
        engine.apply_threshold(&img, &stats, &mut binary);

        (binary, corners)
    }

    /// Test segmentation at varying tag sizes.
    #[test]
    fn test_segmentation_at_varying_tag_sizes() {
        let canvas_size = 640;
        let tag_sizes = [32, 64, 100, 200, 300];

        for tag_size in tag_sizes {
            let arena = Bump::new();
            let (binary, corners) = generate_binarized_tag(tag_size, canvas_size);

            let result =
                label_components_with_stats(&arena, &binary, canvas_size, canvas_size, true);

            assert!(
                !result.component_stats.is_empty(),
                "Tag size {}: No components found",
                tag_size
            );

            let largest = result
                .component_stats
                .iter()
                .max_by_key(|s| s.pixel_count)
                .unwrap();

            let expected_min_x = corners[0][0] as u16;
            let expected_max_x = corners[1][0] as u16;
            let tolerance = 5;

            assert!(
                (largest.min_x as i32 - expected_min_x as i32).abs() <= tolerance,
                "Tag size {}: min_x mismatch",
                tag_size
            );
            assert!(
                (largest.max_x as i32 - expected_max_x as i32).abs() <= tolerance,
                "Tag size {}: max_x mismatch",
                tag_size
            );

            println!(
                "Tag size {:>3}px: {} components, largest has {} px",
                tag_size,
                result.component_stats.len(),
                largest.pixel_count
            );
        }
    }

    /// Test component pixel counts are reasonable for clean binarization.
    #[test]
    fn test_segmentation_component_accuracy() {
        let canvas_size = 320;
        let tag_size = 120;

        let arena = Bump::new();
        let (binary, corners) = generate_binarized_tag(tag_size, canvas_size);

        let result = label_components_with_stats(&arena, &binary, canvas_size, canvas_size, true);

        let largest = result
            .component_stats
            .iter()
            .max_by_key(|s| s.pixel_count)
            .unwrap();

        let expected_min = (tag_size * tag_size / 3) as u32;
        let expected_max = (tag_size * tag_size) as u32;

        assert!(largest.pixel_count >= expected_min);
        assert!(largest.pixel_count <= expected_max);

        let gt_width = (corners[1][0] - corners[0][0]).abs() as i32;
        let gt_height = (corners[2][1] - corners[0][1]).abs() as i32;
        let bbox_width = (largest.max_x - largest.min_x) as i32;
        let bbox_height = (largest.max_y - largest.min_y) as i32;

        assert!((bbox_width - gt_width).abs() <= 2);
        assert!((bbox_height - gt_height).abs() <= 2);

        println!(
            "Component accuracy: {} pixels, bbox={}x{} (GT: {}x{})",
            largest.pixel_count, bbox_width, bbox_height, gt_width, gt_height
        );
    }

    /// Test segmentation with noisy binary boundaries.
    #[test]
    fn test_segmentation_noisy_boundaries() {
        use rand::prelude::*;

        let canvas_size = 320;
        let tag_size = 120;

        let arena = Bump::new();
        let (mut binary, _corners) = generate_binarized_tag(tag_size, canvas_size);

        let mut rng = rand::thread_rng();
        let noise_rate = 0.05;

        for y in 1..(canvas_size - 1) {
            for x in 1..(canvas_size - 1) {
                let idx = y * canvas_size + x;
                let current = binary[idx];
                let left = binary[idx - 1];
                let right = binary[idx + 1];
                let up = binary[idx - canvas_size];
                let down = binary[idx + canvas_size];

                let is_edge =
                    current != left || current != right || current != up || current != down;
                if is_edge && rng.gen_range(0.0..1.0_f32) < noise_rate {
                    binary[idx] = if current == 0 { 255 } else { 0 };
                }
            }
        }

        let result = label_components_with_stats(&arena, &binary, canvas_size, canvas_size, true);

        assert!(!result.component_stats.is_empty());

        let largest = result
            .component_stats
            .iter()
            .max_by_key(|s| s.pixel_count)
            .unwrap();

        let min_expected = (tag_size * tag_size / 4) as u32;
        assert!(largest.pixel_count >= min_expected);

        println!(
            "Noisy segmentation: {} components, largest has {} px",
            result.component_stats.len(),
            largest.pixel_count
        );
    }
}
