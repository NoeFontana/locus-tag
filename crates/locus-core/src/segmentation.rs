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

pub fn label_components<'a>(
    arena: &'a Bump,
    binary: &[u8],
    width: usize,
    height: usize,
) -> &'a [u32] {
    let mut uf = UnionFind::new_in(arena, width * height);

    // Optimized first pass: 4-connectivity
    for y in 0..height {
        let row_off = y * width;
        for x in 0..width {
            let idx = row_off + x;
            if binary[idx] == 0 {
                if x > 0 && binary[idx - 1] == 0 {
                    uf.union(idx as u32, (idx - 1) as u32);
                }
                if y > 0 && binary[idx - width] == 0 {
                    uf.union(idx as u32, (idx - width) as u32);
                }
            }
        }
    }

    let labels = arena.alloc_slice_fill_copy(width * height, 0u32);
    for i in 0..(width * height) {
        if binary[i] == 0 {
            labels[i] = uf.find(i as u32) + 1;
        }
    }
    labels
}
