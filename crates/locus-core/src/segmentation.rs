use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

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
}

impl Default for ComponentStats {
    fn default() -> Self {
        Self {
            min_x: u16::MAX,
            max_x: 0,
            min_y: u16::MAX,
            max_y: 0,
            pixel_count: 0,
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
) -> &'a [u32] {
    label_components_with_stats(arena, binary, width, height).labels
}

/// Label components and compute bounding box stats for each.
pub fn label_components_with_stats<'a>(
    arena: &'a Bump,
    binary: &[u8],
    width: usize,
    height: usize,
) -> LabelResult<'a> {
    let mut runs = BumpVec::new_in(arena);

    // Pass 1: Extract runs
    for y in 0..height {
        let row_off = y * width;
        let mut x = 0;
        while x < width {
            if binary[row_off + x] == 0 {
                let start = x;
                while x < width && binary[row_off + x] == 0 {
                    x += 1;
                }
                runs.push(Run {
                    y: y as u32,
                    x_start: start as u32,
                    x_end: (x - 1) as u32,
                    id: runs.len() as u32,
                });
            } else {
                x += 1;
            }
        }
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

                // Advance prev pointer until it overlaps or passes curr
                while p_idx < prev_row_range.end && runs[p_idx].x_end < curr.x_start {
                    p_idx += 1;
                }

                // Link all overlapping runs in the previous row
                let mut temp_p = p_idx;
                while temp_p < prev_row_range.end && runs[temp_p].x_start <= curr.x_end {
                    uf.union(curr.id, runs[temp_p].id);
                    temp_p += 1;
                }
            }
        }
    }

    // Pass 3: Collect stats per root and assign labels
    let mut root_to_label: Vec<u32> = vec![0; runs.len()];
    let mut component_stats: Vec<ComponentStats> = Vec::new();
    let mut next_label = 1u32;

    for run in &runs {
        let root = uf.find(run.id) as usize;
        if root_to_label[root] == 0 {
            root_to_label[root] = next_label;
            next_label += 1;
            component_stats.push(ComponentStats::default());
        }
        let label_idx = (root_to_label[root] - 1) as usize;
        let stats = &mut component_stats[label_idx];
        stats.min_x = stats.min_x.min(run.x_start as u16);
        stats.max_x = stats.max_x.max(run.x_end as u16);
        stats.min_y = stats.min_y.min(run.y as u16);
        stats.max_y = stats.max_y.max(run.y as u16);
        stats.pixel_count += run.x_end - run.x_start + 1;
    }

    // Pass 4: Assign labels to pixels
    let labels = arena.alloc_slice_fill_copy(width * height, 0u32);
    for run in runs {
        let root = uf.find(run.id) as usize;
        let label = root_to_label[root];
        let row_off = run.y as usize * width;
        for x in run.x_start..=run.x_end {
            labels[row_off + x as usize] = label;
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
        // 4x4 image with two separate 2x2 squares
        // 0 = background (black), 255 = foreground (white)
        // Tag detector looks for black components (0)
        let binary = [
            0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0,
        ];
        let width = 4;
        let height = 4;

        let result = label_components_with_stats(&arena, &binary, width, height);

        assert_eq!(result.component_stats.len(), 2);

        // Component 1 (top-left)
        let s1 = result.component_stats[0];
        assert_eq!(s1.pixel_count, 4);
        assert_eq!(s1.min_x, 0);
        assert_eq!(s1.max_x, 1);
        assert_eq!(s1.min_y, 0);
        assert_eq!(s1.max_y, 1);

        // Component 2 (bottom-right)
        let s2 = result.component_stats[1];
        assert_eq!(s2.pixel_count, 4);
        assert_eq!(s2.min_x, 2);
        assert_eq!(s2.max_x, 3);
        assert_eq!(s2.min_y, 2);
        assert_eq!(s2.max_y, 3);
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

            let result = label_components_with_stats(&arena, slice, real_width, real_height);

            for stat in result.component_stats {
                assert!(stat.pixel_count > 0);
                assert!(stat.max_x < real_width as u16);
                assert!(stat.max_y < real_height as u16);
                assert!(stat.min_x <= stat.max_x);
                assert!(stat.min_y <= stat.max_y);
            }
        }
    }
}
