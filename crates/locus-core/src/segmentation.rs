use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

/// A disjoint-set forest (Union-Find) with path compression and rank optimization.
pub struct UnionFind<'a> {
    parent: &'a mut [u32],
    rank: &'a mut [u8],
}

impl<'a> UnionFind<'a> {
    pub fn new_in(arena: &'a Bump, size: usize) -> Self {
        let parent = arena.alloc_slice_fill_with(size, |i| i as u32);
        let rank = arena.alloc_slice_fill_copy(size, 0u8);
        Self { parent, rank }
    }

    #[inline]
    pub fn find(&mut self, i: u32) -> u32 {
        let mut root = i;
        while self.parent[root as usize] != root {
            self.parent[root as usize] = self.parent[self.parent[root as usize] as usize];
            root = self.parent[root as usize];
        }
        root
    }

    #[inline]
    pub fn union(&mut self, i: u32, j: u32) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            if self.rank[root_i as usize] < self.rank[root_j as usize] {
                self.parent[root_i as usize] = root_j;
            } else if self.rank[root_i as usize] > self.rank[root_j as usize] {
                self.parent[root_j as usize] = root_i;
            } else {
                self.parent[root_i as usize] = root_j;
                self.rank[root_j as usize] += 1;
            }
        }
    }
}

/// Bounding box and statistics for a connected component.
#[derive(Clone, Copy, Debug)]
pub struct ComponentStats {
    pub min_x: u16,
    pub max_x: u16,
    pub min_y: u16,
    pub max_y: u16,
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
    pub labels: &'a [u32],
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
    let mut prev_row_start = 0;
    let mut curr_row_start = 0;

    // Pass 2: Link runs between adjacent rows
    for i in 0..runs.len() {
        let curr = &runs[i];

        if runs[curr_row_start].y < curr.y {
            prev_row_start = curr_row_start;
            curr_row_start = i;
        }

        if curr.y > 0 {
            for j in prev_row_start..curr_row_start {
                let prev = &runs[j];
                if prev.y != curr.y - 1 {
                    continue;
                }
                if prev.x_start <= curr.x_end && curr.x_start <= prev.x_end {
                    uf.union(curr.id, prev.id);
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
