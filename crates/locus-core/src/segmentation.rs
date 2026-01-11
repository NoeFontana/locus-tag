use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

/// A disjoint-set forest (Union-Find) with path compression and rank optimization.
/// Uses a flat array of indices for cache locality.
pub struct UnionFind<'a> {
    parent: &'a mut [u32],
}

impl<'a> UnionFind<'a> {
    /// Create a new Union-Find structure in the provided arena.
    pub fn new_in(arena: &'a Bump, size: usize) -> Self {
        let parent = arena.alloc_slice_fill_with(size, |i| i as u32);
        Self { parent }
    }

    /// Find the representative of the set containing `i`.
    pub fn find(&mut self, i: u32) -> u32 {
        let mut root = i;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }

        // Path compression
        let mut curr = i;
        while self.parent[curr as usize] != root {
            let next = self.parent[curr as usize];
            self.parent[curr as usize] = root;
            curr = next;
        }
        root
    }

    /// Union the sets containing `i` and `j`.
    pub fn union(&mut self, i: u32, j: u32) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            // Simple union: we could track rank, but for image labeling,
            // path compression is often sufficient.
            self.parent[root_i as usize] = root_j;
        }
    }
}

/// Find connected components in a binarized image (255 = light, 0 = dark).
/// Typically used on the inverted binarized image to find dark components.
pub fn label_components<'a>(
    arena: &'a Bump,
    binary: &[u8],
    width: usize,
    height: usize,
) -> &'a [u32] {
    let mut uf = UnionFind::new_in(arena, width * height);

    // Single pass for union (4-connectivity)
    for y in 0..height {
        let row_start = y * width;
        for x in 0..width {
            let idx = row_start + x;
            if binary[idx] == 0 {
                // Dark pixel
                // Check neighbors: Left and Top
                if x > 0 && binary[idx - 1] == 0 {
                    uf.union(idx as u32, (idx - 1) as u32);
                }
                if y > 0 && binary[idx - width] == 0 {
                    uf.union(idx as u32, (idx - width) as u32);
                }
            }
        }
    }

    // Second pass to canonicalize labels
    let labels = arena.alloc_slice_fill_copy(width * height, 0u32);
    for i in 0..(width * height) {
        if binary[i] == 0 {
            labels[i] = uf.find(i as u32) + 1; // 1-based labels, 0 is background
        }
    }
    labels
}
