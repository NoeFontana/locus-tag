//! Tag family dictionaries.
//!
//! This module contains pre-generated code tables for AprilTag families.
//! Codes are in row-major bit ordering for efficient extraction.

#![allow(clippy::unreadable_literal, clippy::too_many_lines)]

/// A tag family dictionary.
///
/// `dimension`, `min_hamming`, and `num_codes_per_rotation` document the
/// family's shape for downstream consumers reading `pub` fields directly; the
/// Rust hot-path decoder uses `payload_length`, `codes`, and the MIH tables.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct TagDictionary {
    /// Maximum number of bits (e.g., 36 for 36h11, 41 for 41h12).
    pub payload_length: u32,
    /// Grid dimension (e.g., 6 for 6x6).
    pub dimension: usize,
    /// Minimum hamming distance of the family.
    pub min_hamming: u32,
    /// Number of distinct rotation invariant codes
    pub num_codes_per_rotation: usize,
    /// Raw code table (N * 4 rotations).
    pub codes: &'static [u64],
    /// MIH Chunk Length.
    pub mih_chunks: usize,
    /// Multi-Index Hashing offsets: (k * MIH_BUCKETS + 1) entries.
    pub mih_offsets: &'static [usize],
    /// Multi-Index Hashing data: N * k entries, flat array.
    pub mih_data: &'static [u32],
    /// MIH bucket array size per chunk (max 1 << bits_per_chunk).
    pub mih_buckets: usize,
    /// MIH bits per chunk.
    pub mih_bits_per_chunk: u32,
    /// MIH last chunk bits.
    pub mih_last_chunk_bits: u32,
}

impl TagDictionary {
    /// Get number of unique codes in dictionary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len() / 4
    }

    /// Check if dictionary is empty.
    // Paired with `len()` to satisfy `clippy::len_without_is_empty`.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get the raw base code (rotation 0) for a given ID.
    #[must_use]
    pub fn get_code(&self, id: u16) -> Option<u64> {
        self.codes.get(id as usize * 4).copied()
    }

    /// Decode bits against all 4 rotations of every code: the nearest code (lowest index on
    /// ties) if it is within `max_hamming`. Returns `(id, hamming_distance, rotation)`.
    #[must_use]
    pub fn decode(&self, bits: u64, max_hamming: u32) -> Option<(u16, u32, u8)> {
        let mask = if self.payload_length < 64 {
            (1u64 << self.payload_length) - 1
        } else {
            u64::MAX
        };
        let bits = bits & mask;
        if max_hamming > 0 && self.payload_length > 36 {
            return self.decode_indexed(bits, max_hamming);
        }
        // One branch-free pass (vectorised popcount and min) over `hamming << 16 | index`
        // keys: the minimum is the nearest code, lowest index on ties.
        let key = nearest_key(self.codes, bits)?;
        let (hamming, idx) = (key >> 16, (key & 0xFFFF) as usize);
        if hamming > max_hamming {
            return None;
        }
        Some(((idx / 4) as u16, hamming, (idx % 4) as u8))
    }

    fn decode_indexed(&self, bits: u64, max_hamming: u32) -> Option<(u16, u32, u8)> {
        let mut best: Option<(u16, u32, u8)> = None;
        for c in 0..self.mih_chunks {
            let chunk = self.extract_mih_chunk(bits, c) as usize;
            let bucket_idx = c * self.mih_buckets + chunk;
            let offset_start = self.mih_offsets[bucket_idx];
            let offset_end = self.mih_offsets[bucket_idx + 1];

            for i in offset_start..offset_end {
                let packed = self.mih_data[i];
                if let Some(&target_code) = self.codes.get(packed as usize) {
                    let hamming = (bits ^ target_code).count_ones();
                    if hamming <= max_hamming {
                        let id = (packed >> 2) as u16;
                        let rot = (packed & 0x3) as u8;
                        if let Some((_, b_h, _)) = best {
                            if hamming < b_h {
                                best = Some((id, hamming, rot));
                            }
                        } else {
                            best = Some((id, hamming, rot));
                        }
                        if hamming == 0 {
                            return best;
                        }
                    }
                }
            }
        }
        best
    }

    /// Internal method to extract a specific chunk for Multi-Index Hashing.
    fn extract_mih_chunk(&self, bits: u64, chunk_idx: usize) -> u16 {
        let chunk_size = self.mih_bits_per_chunk;
        let last_size = self.mih_last_chunk_bits;
        let start = chunk_idx as u32 * chunk_size;
        let len = if chunk_idx == self.mih_chunks - 1 {
            last_size
        } else {
            chunk_size
        };
        ((bits >> start) & ((1u64 << len) - 1)) as u16
    }
}

/// `hamming << 16 | index` of the nearest of `codes` to `bits`, lowest index on ties (`None`
/// when empty). Shipped x86-64 wheels target the baseline ISA, which has no `popcnt`, so
/// dispatch at runtime. `build.rs` bounds code tables to 2¹⁶ entries (36h11: 2348).
#[multiversion::multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+popcnt",
    "aarch64+neon"
))]
fn nearest_key(codes: &[u64], bits: u64) -> Option<u32> {
    codes
        .iter()
        .zip(0u32..)
        .map(|(&code, i)| ((bits ^ code).count_ones() << 16) | i)
        .min()
}

// Generate all static datasets using build.rs macro inclusion
include!(concat!(env!("OUT_DIR"), "/dictionaries.rs"));

/// Return dictionary instance given family config tag.
#[must_use]
pub fn get_dictionary(family: crate::config::TagFamily) -> &'static TagDictionary {
    match family {
        crate::config::TagFamily::AprilTag16h5 => &DICT_APRILTAG16H5,
        crate::config::TagFamily::AprilTag36h11 => &DICT_APRILTAG36H11,
        crate::config::TagFamily::ArUco4x4_50 => &DICT_ARUCO4X4_50,
        crate::config::TagFamily::ArUco4x4_100 => &DICT_ARUCO4X4_100,
        crate::config::TagFamily::ArUco6x6_250 => &DICT_ARUCO6X6_250,
        crate::config::TagFamily::ArUcoMip36h12 => &DICT_ARUCOMIP36H12,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TagFamily;
    use proptest::prelude::*;

    /// Nearest code by exhaustive search, lowest index on ties.
    fn reference(dict: &TagDictionary, bits: u64, max_hamming: u32) -> Option<(u16, u32, u8)> {
        let bits = bits & ((1u64 << dict.payload_length) - 1);
        let (idx, h) = dict
            .codes
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, (bits ^ c).count_ones()))
            .min_by_key(|&(i, h)| (h, i))?;
        (h <= max_hamming).then_some(((idx / 4) as u16, h, (idx % 4) as u8))
    }

    const FAMILIES: [TagFamily; 6] = [
        TagFamily::AprilTag16h5,
        TagFamily::AprilTag36h11,
        TagFamily::ArUco4x4_50,
        TagFamily::ArUco4x4_100,
        TagFamily::ArUco6x6_250,
        TagFamily::ArUcoMip36h12,
    ];

    proptest! {
        #[test]
        fn decode_matches_exhaustive_search(fam in 0usize..6, bits: u64, max_hamming in 0u32..40) {
            let dict = get_dictionary(FAMILIES[fam]);
            prop_assert_eq!(dict.decode(bits, max_hamming), reference(dict, bits, max_hamming));
        }

        #[test]
        fn decode_near_codes_matches_exhaustive_search(
            fam in 0usize..6, idx: prop::sample::Index, flips: u64, max_hamming in 0u32..8,
        ) {
            // Random bits are far from every code; also probe each code's neighbourhood.
            let dict = get_dictionary(FAMILIES[fam]);
            let code = dict.codes[idx.index(dict.codes.len())];
            let noise = flips & flips.rotate_left(17) & flips.rotate_left(31);
            prop_assert_eq!(
                dict.decode(code ^ noise, max_hamming),
                reference(dict, code ^ noise, max_hamming)
            );
        }
    }
}
