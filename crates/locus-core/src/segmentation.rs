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
            // Path halving: make every other node in path point to its grandparent
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

/// A detected run of foreground/background pixels in a row.
#[derive(Clone, Copy, Debug)]
struct Run {
    y: u32,
    x_start: u32,
    x_end: u32,
    id: u32, // Component ID (index into UnionFind)
}

pub fn label_components<'a>(
    arena: &'a Bump,
    binary: &[u8],
    width: usize,
    height: usize,
) -> &'a [u32] {
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
        return arena.alloc_slice_fill_copy(width * height, 0u32);
    }

    let mut uf = UnionFind::new_in(arena, runs.len());
    let mut prev_row_start = 0;
    let mut curr_row_start = 0;

    // Pass 2: Link runs between adjacent rows
    for i in 0..runs.len() {
        let curr = &runs[i];

        // Update row pointers
        if runs[curr_row_start].y < curr.y {
            prev_row_start = curr_row_start;
            curr_row_start = i;
        }

        // If not first row, check overlaps with previous row
        if curr.y > 0 {
            for j in prev_row_start..curr_row_start {
                let prev = &runs[j];
                if prev.y != curr.y - 1 {
                    continue;
                }

                // Check horizontal overlap for 4-connectivity (standard RLE CCL)
                // For 8-connectivity, use: prev.x_start <= curr.x_end + 1 && curr.x_start <= prev.x_end + 1
                if prev.x_start <= curr.x_end && curr.x_start <= prev.x_end {
                    uf.union(curr.id, prev.id);
                }
            }
        }
    }

    // Pass 3: Relabel pixels
    let labels = arena.alloc_slice_fill_copy(width * height, 0u32);
    for run in runs {
        let root = uf.find(run.id);
        let row_off = run.y as usize * width;
        for x in run.x_start..=run.x_end {
            labels[row_off + x as usize] = root + 1;
        }
    }

    labels
}
