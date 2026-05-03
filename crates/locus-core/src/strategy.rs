//! Hard-decision Hamming decoding helpers.
//!
//! Sampled intensities are binarized against per-cell adaptive thresholds
//! to produce a `u64` code, which is then matched against each registered
//! tag family's dictionary via [`crate::decoder::TagDecoder::decode_full`].

/// Pack sampled intensities into a `u64` bit code via the per-cell
/// adaptive threshold.
///
/// Bit `i` is set when `intensities[i] > thresholds[i]`. Inputs longer than
/// 64 elements are truncated to the first 64; the dictionaries used by
/// every shipped family fit within that bound (the largest is 6×6 = 36
/// bits).
#[must_use]
pub fn bits_from_intensities(intensities: &[f64], thresholds: &[f64]) -> u64 {
    let mut bits = 0u64;
    for (i, (&val, &thresh)) in intensities.iter().zip(thresholds.iter()).enumerate() {
        if val > thresh {
            bits |= 1 << i;
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_from_intensities_thresholds_in_order() {
        let intensities = [10.0, 50.0, 100.0, 200.0];
        let thresholds = [20.0, 40.0, 80.0, 250.0];
        // Set when intensity > threshold: bit0 no, bit1 yes, bit2 yes, bit3 no.
        assert_eq!(bits_from_intensities(&intensities, &thresholds), 0b0110);
    }
}
