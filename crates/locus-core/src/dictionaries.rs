//! Tag family dictionaries.
//!
//! This module contains pre-generated code tables for AprilTag families.
//! Codes are in row-major bit ordering for efficient extraction.

#![allow(clippy::unreadable_literal, clippy::too_many_lines)]

/// A tag family dictionary.
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
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get the raw base code (rotation 0) for a given ID.
    #[must_use]
    pub fn get_code(&self, id: u16) -> Option<u64> {
        self.codes.get(id as usize * 4).copied()
    }

    /// Decode bits, trying all 4 rotations via O(1) lookup then Hamming search.
    /// Returns (id, hamming_distance, rotation) if found within tolerance.
    #[must_use]
    pub fn decode(&self, bits: u64, max_hamming: u32) -> Option<(u16, u32, u8)> {
        let mask = if self.payload_length < 64 {
            (1u64 << self.payload_length) - 1
        } else {
            u64::MAX
        };
        let bits = bits & mask;

        if max_hamming > 0 {
            // First check exactly
            let mut best: Option<(u16, u32, u8)> = None;
            for (idx, &code) in self.codes.iter().enumerate() {
                if bits == code {
                    return Some(((idx / 4) as u16, 0, (idx % 4) as u8));
                }
            }
            // If not found exactly, do full or indexed search
            if self.payload_length <= 36 {
                // For small dictionaries, linear search is fast enough and guaranteed optimal
                for (idx, &code) in self.codes.iter().enumerate() {
                    let hamming = (bits ^ code).count_ones();
                    if hamming <= max_hamming {
                        if let Some((_, b_h, _)) = best {
                            if hamming < b_h {
                                best = Some(((idx / 4) as u16, hamming, (idx % 4) as u8));
                            }
                        } else {
                            best = Some(((idx / 4) as u16, hamming, (idx % 4) as u8));
                        }
                        // Early exit if perfect match found
                        if hamming == 0 {
                            return best;
                        }
                    }
                }
                best
            } else {
                self.decode_indexed(bits, max_hamming)
            }
        } else {
            // Exactly matching
            for (idx, &code) in self.codes.iter().enumerate() {
                if bits == code {
                    let id = (idx / 4) as u16;
                    let rot = (idx % 4) as u8;
                    return Some((id, 0, rot));
                }
            }
            None
        }
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

    /// Executes a callback for each candidate in the dictionary within a given Hamming distance.
    pub fn for_each_candidate_within_hamming<F>(&self, bits: u64, max_hamming: u32, mut f: F)
    where
        F: FnMut(u64, u16, u8),
    {
        if max_hamming > 4 {
            for (idx, &code) in self.codes.iter().enumerate() {
                let hamming = (bits ^ code).count_ones();
                if hamming <= max_hamming {
                    f(code, (idx / 4) as u16, (idx % 4) as u8);
                }
            }
            return;
        }

        let mut visited = [0u64; 160];
        for c in 0..self.mih_chunks {
            let chunk = self.extract_mih_chunk(bits, c) as usize;
            let bucket_idx = c * self.mih_buckets + chunk;
            let offset_start = self.mih_offsets[bucket_idx];
            let offset_end = self.mih_offsets[bucket_idx + 1];

            for i in offset_start..offset_end {
                let packed = self.mih_data[i];
                let v_idx = packed as usize / 64;
                let v_bit = 1u64 << (packed % 64);
                if visited[v_idx] & v_bit == 0 {
                    visited[v_idx] |= v_bit;
                    if let Some(&target_code) = self.codes.get(packed as usize) {
                        let hamming = (bits ^ target_code).count_ones();
                        if hamming <= max_hamming {
                            f(target_code, (packed >> 2) as u16, (packed & 0x3) as u8);
                        }
                    }
                }
            }
        }
    }
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
    }
}
