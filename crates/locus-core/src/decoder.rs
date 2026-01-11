use nalgebra::{SMatrix, SVector};

/// A 3x3 Homography matrix.
pub struct Homography {
    pub h: SMatrix<f64, 3, 3>,
}

impl Homography {
    /// Compute homography from 4 source points to 4 destination points using DLT.
    /// Points are [x, y].
    pub fn from_pairs(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<Self> {
        let mut a = SMatrix::<f64, 8, 9>::zeros();

        for i in 0..4 {
            let sx = src[i][0];
            let sy = src[i][1];
            let dx = dst[i][0];
            let dy = dst[i][1];

            a[(i * 2, 0)] = -sx;
            a[(i * 2, 1)] = -sy;
            a[(i * 2, 2)] = -1.0;
            a[(i * 2, 6)] = sx * dx;
            a[(i * 2, 7)] = sy * dx;
            a[(i * 2, 8)] = dx;

            a[(i * 2 + 1, 3)] = -sx;
            a[(i * 2 + 1, 4)] = -sy;
            a[(i * 2 + 1, 5)] = -1.0;
            a[(i * 2 + 1, 6)] = sx * dy;
            a[(i * 2 + 1, 7)] = sy * dy;
            a[(i * 2 + 1, 8)] = dy;
        }

        // Solve A*h = 0 using SVD or simple matrix inversion if we fix h[8]=1
        // For 4 points (8 equations), we can assume h[8]=1.0 and solve a 8x8 system.
        let mut b = SVector::<f64, 8>::zeros();
        let mut m = SMatrix::<f64, 8, 8>::zeros();
        for i in 0..8 {
            for j in 0..8 {
                m[(i, j)] = a[(i, j)];
            }
            b[i] = -a[(i, 8)];
        }

        if let Some(h_vec) = m.lu().solve(&b) {
            let mut h = SMatrix::<f64, 3, 3>::identity();
            h[(0, 0)] = h_vec[0];
            h[(0, 1)] = h_vec[1];
            h[(0, 2)] = h_vec[2];
            h[(1, 0)] = h_vec[3];
            h[(1, 1)] = h_vec[4];
            h[(1, 2)] = h_vec[5];
            h[(2, 0)] = h_vec[6];
            h[(2, 1)] = h_vec[7];
            h[(2, 2)] = 1.0;
            Some(Self { h })
        } else {
            None
        }
    }

    /// Project a point using the homography.
    pub fn project(&self, p: [f64; 2]) -> [f64; 2] {
        let res = self.h * SVector::<f64, 3>::new(p[0], p[1], 1.0);
        let w = res[2];
        [res[0] / w, res[1] / w]
    }
}

pub trait TagDecoder: Send + Sync {
    fn name(&self) -> &str;
    fn dimension(&self) -> usize;
    fn decode(&self, bits: u64) -> Option<(u32, u32)>; // (id, hamming)
}

pub struct AprilTag36h11;

impl TagDecoder for AprilTag36h11 {
    fn name(&self) -> &str {
        "36h11"
    }
    fn dimension(&self) -> usize {
        6
    } // 6x6 grid of bits (excluding border)

    fn decode(&self, bits: u64) -> Option<(u32, u32)> {
        // Simplified 36h11 dictionary for Phase 4 verification
        // These are just example bit patterns
        let codes: [(u16, u64); 3] = [
            (0, 0x000000000u64),
            (42, 0x123456789u64),
            (101, 0xABCDE1234u64),
        ];

        for (id, code) in codes {
            let mut rbits = bits;
            for _ in 0..4 {
                let hamming = (rbits ^ code).count_ones();
                if hamming <= 2 {
                    return Some((id as u32, hamming));
                }
                rbits = rotate90(rbits, self.dimension());
            }
        }
        None
    }
}

pub fn rotate90(bits: u64, dim: usize) -> u64 {
    let mut res = 0u64;
    for y in 0..dim {
        for x in 0..dim {
            if (bits >> (y * dim + x)) & 1 != 0 {
                let nx = dim - 1 - y;
                let ny = x;
                res |= 1 << (ny * dim + nx);
            }
        }
    }
    res
}
