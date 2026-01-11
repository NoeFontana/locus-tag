use nalgebra::{SMatrix, SVector};

/// A 3x3 Homography matrix.
pub struct Homography {
    /// The 3x3 homography matrix.
    pub h: SMatrix<f64, 3, 3>,
}

impl Homography {
    /// Compute homography from 4 source points to 4 destination points using DLT.
    /// Points are [x, y].
    #[must_use]
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
    #[must_use]
    pub fn project(&self, p: [f64; 2]) -> [f64; 2] {
        let res = self.h * SVector::<f64, 3>::new(p[0], p[1], 1.0);
        let w = res[2];
        [res[0] / w, res[1] / w]
    }
}

/// A trait for decoding binary payloads from extracted tags.
pub trait TagDecoder: Send + Sync {
    /// Returns the name of the decoder family (e.g., "AprilTag36h11").
    fn name(&self) -> &str;
    /// Returns the dimension of the tag grid (e.g., 6 for 36h11).
    fn dimension(&self) -> usize;
    /// Decodes the extracted bits into a tag ID and hamming distance.
    ///
    /// Returns `Some((id, hamming))` if decoding is successful, `None` otherwise.
    fn decode(&self, bits: u64) -> Option<(u32, u32)>; // (id, hamming)
}

/// Decoder for the AprilTag 36h11 family.
pub struct AprilTag36h11;

impl TagDecoder for AprilTag36h11 {
    fn name(&self) -> &'static str {
        "36h11"
    }
    fn dimension(&self) -> usize {
        6
    } // 6x6 grid of bits (excluding border)

    fn decode(&self, bits: u64) -> Option<(u32, u32)> {
        // Use the full 587-code dictionary with O(1) exact match + hamming search
        crate::dictionaries::APRILTAG_36H11
            .decode(bits, 2) // Allow up to 2 bit errors
            .map(|(id, hamming)| (u32::from(id), hamming))
    }
}

/// Decoder for the AprilTag 16h5 family.
pub struct AprilTag16h5;

impl TagDecoder for AprilTag16h5 {
    fn name(&self) -> &'static str {
        "16h5"
    }
    fn dimension(&self) -> usize {
        4
    } // 4x4 grid of bits (excluding border)

    fn decode(&self, bits: u64) -> Option<(u32, u32)> {
        crate::dictionaries::APRILTAG_16H5
            .decode(bits, 1) // Allow up to 1 bit error (16h5 has lower hamming distance)
            .map(|(id, hamming)| (u32::from(id), hamming))
    }
}

/// Decoder for the ArUco 4x4_50 family.
pub struct ArUco4x4_50;

impl TagDecoder for ArUco4x4_50 {
    fn name(&self) -> &'static str {
        "4X4_50"
    }
    fn dimension(&self) -> usize {
        4
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32)> {
        crate::dictionaries::ARUCO_4X4_50
            .decode(bits, 1)
            .map(|(id, hamming)| (u32::from(id), hamming))
    }
}

/// Decoder for the ArUco 4x4_100 family.
pub struct ArUco4x4_100;

impl TagDecoder for ArUco4x4_100 {
    fn name(&self) -> &'static str {
        "4X4_100"
    }
    fn dimension(&self) -> usize {
        4
    }

    fn decode(&self, bits: u64) -> Option<(u32, u32)> {
        crate::dictionaries::ARUCO_4X4_100
            .decode(bits, 1)
            .map(|(id, hamming)| (u32::from(id), hamming))
    }
}

/// Rotates the bit pattern of a tag by 90 degrees clockwise.
///
/// # Arguments
/// * `bits` - The 64-bit integer representing the tag grid.
/// * `dim` - The dimension of the tag grid (e.g., 6).
#[must_use]
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_rotation_invariants(bits in 0..u64::MAX) {
            let dim = 6;
            let r1 = rotate90(bits, dim);
            let r2 = rotate90(r1, dim);
            let r3 = rotate90(r2, dim);
            let r4 = rotate90(r3, dim);

            // Mask to dim*dim bits to avoid noise in upper bits
            let mask = (1u64 << (dim * dim)) - 1;
            prop_assert_eq!(bits & mask, r4 & mask);
        }

        #[test]
        fn test_hamming_robustness(
            id_idx in 0usize..10,
            rotation in 0..4usize,
            flip1 in 0..36usize,
            flip2 in 0..36usize
        ) {
            let decoder = AprilTag36h11;
            let dict = &*crate::dictionaries::APRILTAG_36H11;
            let orig_id = id_idx as u16;
            let orig_code = dict.get_code(orig_id).expect("valid ID");

            // Apply rotation
            let mut test_bits = orig_code;
            for _ in 0..rotation {
                test_bits = rotate90(test_bits, 6);
            }

            // Flip bits
            test_bits ^= 1 << flip1;
            test_bits ^= 1 << flip2;

            let result = decoder.decode(test_bits);
            prop_assert!(result.is_some());
            let (decoded_id, _) = result.expect("Should decode valid pattern");
            prop_assert_eq!(decoded_id, u32::from(orig_id));
        }

        #[test]
        fn test_false_positive_resistance(bits in 0..u64::MAX) {
            let decoder = AprilTag36h11;
            // Random bitstreams should rarely match any of the 587 codes
            if let Some((_id, hamming)) = decoder.decode(bits) {
                // If it decodes, it must have low hamming distance
                prop_assert!(hamming <= 2);
            }
        }
    }

    #[test]
    fn test_all_codes_decode() {
        let decoder = AprilTag36h11;
        for id in 0..587u16 {
            let code = crate::dictionaries::APRILTAG_36H11
                .get_code(id)
                .expect("valid ID");
            let result = decoder.decode(code);
            assert_eq!(result, Some((u32::from(id), 0)), "ID {} should decode", id);
        }
    }
}
