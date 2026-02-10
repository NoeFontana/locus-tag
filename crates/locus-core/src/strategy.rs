//! Pluggable decoding strategies for different SNR conditions.
//!
//! This module abstracts the process of converting sampled pixel intensities
//! into tag IDs. It provides:
//! - **Hard-Decision**: Fastest mode using binary thresholds and Hamming distance.
//! - **Soft-Decision**: High-recall mode using Log-Likelihood Ratios (LLRs).

use crate::decoder::TagDecoder;

/// Trait abstracting the decoding strategy (Hard vs Soft).
pub trait DecodingStrategy: Send + Sync + 'static {
    /// The type of code extracted from the image (e.g., u64 bits or `Vec<i16>` LLRs).
    type Code: Clone + std::fmt::Debug + Send + Sync;

    /// Convert intensities and thresholds into a code.
    fn from_intensities(intensities: &[f64], thresholds: &[f64]) -> Self::Code;

    /// Compute the "distance" between the extracted code and a dictionary target.
    ///
    /// For Hard decoding, this is Hamming distance.
    /// For Soft decoding, this is the accumulated penalty of mismatching LLRs.
    fn distance(code: &Self::Code, target: u64) -> u32;

    /// Decode the code into an ID using the provided decoder.
    fn decode(
        code: &Self::Code,
        decoder: &(impl TagDecoder + ?Sized),
        max_error: u32,
    ) -> Option<(u32, u32, u8)>;

    /// Convert the code to a debug bitstream (u64).
    fn to_debug_bits(code: &Self::Code) -> u64;
}

/// Hard-decision strategy (Hamming distance).
pub struct HardStrategy;

impl DecodingStrategy for HardStrategy {
    type Code = u64;

    fn from_intensities(intensities: &[f64], thresholds: &[f64]) -> Self::Code {
        let mut bits = 0u64;
        for (i, (&val, &thresh)) in intensities.iter().zip(thresholds.iter()).enumerate() {
            if val > thresh {
                bits |= 1 << i;
            }
        }
        bits
    }

    fn distance(code: &Self::Code, target: u64) -> u32 {
        (*code ^ target).count_ones()
    }

    fn decode(
        code: &Self::Code,
        decoder: &(impl TagDecoder + ?Sized),
        max_error: u32,
    ) -> Option<(u32, u32, u8)> {
        decoder.decode_full(*code, max_error)
    }

    fn to_debug_bits(code: &Self::Code) -> u64 {
        *code
    }
}

/// Soft-decision strategy (Log-Likelihood Ratios).
pub struct SoftStrategy;

/// A stack-allocated buffer for Log-Likelihood Ratios (LLRs).
#[derive(Clone, Debug)]
pub struct SoftCode {
    /// The LLR values for each sample point.
    pub llrs: [i16; 64],
    /// The number of valid LLRs (usually dimension^2).
    pub len: usize,
}

impl SoftStrategy {
    #[inline]
    fn distance_with_limit(code: &SoftCode, target: u64, limit: u32) -> u32 {
        let mut penalty = 0u32;
        let n = code.len;

        for i in 0..n {
            let target_bit = (target >> i) & 1;
            let llr = code.llrs[i];

            if target_bit == 1 {
                if llr < 0 {
                    penalty += u32::from(llr.unsigned_abs());
                }
            } else if llr > 0 {
                penalty += u32::from(llr.unsigned_abs());
            }

            if penalty >= limit {
                return limit;
            }
        }
        penalty
    }
}

impl DecodingStrategy for SoftStrategy {
    type Code = SoftCode;

    fn from_intensities(intensities: &[f64], thresholds: &[f64]) -> Self::Code {
        let n = intensities.len().min(64);
        let mut llrs = [0i16; 64];
        for i in 0..n {
            llrs[i] = (intensities[i] - thresholds[i]) as i16;
        }
        SoftCode { llrs, len: n }
    }

    fn distance(code: &Self::Code, target: u64) -> u32 {
        Self::distance_with_limit(code, target, u32::MAX)
    }

    fn decode(
        code: &Self::Code,
        decoder: &(impl TagDecoder + ?Sized),
        max_error: u32,
    ) -> Option<(u32, u32, u8)> {
        // Fast Path: Try hard-decoding first
        let bits = Self::to_debug_bits(code);
        if let Some((id, hamming, rot)) = decoder.decode_full(bits, 0) {
            return Some((id, hamming, rot));
        }

        let _codes_count = decoder.num_codes();
        let soft_threshold = max_error.max(1) * 60;

        let mut best_id = None;
        let mut best_dist = soft_threshold;
        let mut best_rot = 0;

        for &(target_code, id, rot) in decoder.rotated_codes() {
            let dist = Self::distance_with_limit(code, target_code, best_dist);
            if dist < best_dist {
                best_dist = dist;
                best_id = Some(u32::from(id));
                best_rot = rot;

                if best_dist == 0 {
                    return Some((u32::from(id), 0, rot));
                }
            }
        }

        if best_dist < soft_threshold {
            return best_id.map(|id| {
                let equiv_hamming = best_dist / 60;
                (id, equiv_hamming, best_rot)
            });
        }

        None
    }

    fn to_debug_bits(code: &Self::Code) -> u64 {
        let mut bits = 0u64;
        for i in 0..code.len {
            if code.llrs[i] > 0 {
                bits |= 1 << i;
            }
        }
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_distance() {
        let code = 0b1010;
        let target = 0b1100;
        // Diff: 0b0110 -> 2 bits
        assert_eq!(HardStrategy::distance(&code, target), 2);
    }

    #[test]
    fn test_soft_from_intensities() {
        let intensities = vec![100.0, 50.0, 200.0];
        let thresholds = vec![80.0, 60.0, 150.0];
        // LLR: 20, -10, 50
        let code = SoftStrategy::from_intensities(&intensities, &thresholds);
        assert_eq!(code.llrs[0], 20);
        assert_eq!(code.llrs[1], -10);
        assert_eq!(code.llrs[2], 50);
        assert_eq!(code.len, 3);
    }

    #[test]
    fn test_soft_distance() {
        // Code: [20, -10, 50] (Strong 1, Weak 0, Strong 1)
        let code = SoftCode {
            llrs: {
                let mut l = [0i16; 64];
                l[0] = 20;
                l[1] = -10;
                l[2] = 50;
                l
            },
            len: 3,
        };

        // Target 1: 1 0 1 (binary 5) -> 0b101
        // i=0 (20) -> target 1 -> Penalty 0 (match)
        // i=1 (-10) -> target 0 -> Penalty 0 (match)
        // i=2 (50) -> target 1 -> Penalty 0 (match)
        assert_eq!(SoftStrategy::distance(&code, 0b101), 0);

        // Target 2: 0 1 0 (binary 2) -> 0b010
        // i=0 (20) -> target 0 -> Penalty 20 (mismatch)
        // i=1 (-10) -> target 1 -> Penalty 10 (mismatch)
        // i=2 (50) -> target 0 -> Penalty 50 (mismatch)
        // Total: 80
        assert_eq!(SoftStrategy::distance(&code, 0b010), 80);
    }

    #[test]
    fn test_to_debug_bits() {
        let code = SoftCode {
            llrs: {
                let mut l = [0i16; 64];
                l[0] = 20;
                l[1] = -10;
                l[2] = 50;
                l
            },
            len: 3,
        };
        // >0 -> 1, <=0 -> 0
        // 1 0 1 -> 5
        assert_eq!(SoftStrategy::to_debug_bits(&code), 5);
    }
}
