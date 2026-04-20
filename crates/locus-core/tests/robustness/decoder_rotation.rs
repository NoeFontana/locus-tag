#![allow(
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::items_after_statements,
    missing_docs
)]
//! Property: decoding `rotate90^k(base_bits)` returns the original tag id with
//! hamming distance 0, for every valid id in every tag family and every
//! k ∈ {0, 1, 2, 3}.
//!
//! `rotate90` is the canonical bit-grid rotation helper re-exported via
//! `bench_api` (see `src/decoder.rs:869`). The dictionary stores four code
//! entries per id — one per 90°-CW rotation of the bit grid — so this
//! property exercises the full id × rotation surface.
//!
//! This is a property-based generalisation of the inline `test_rotation_invariants`
//! in `decoder.rs` (which only asserts `rotate^4 = identity`). The failure mode it
//! catches is a decoder that accepts only the canonical orientation or that
//! assigns the wrong rotation slot to a rotated code.

use locus_core::TagFamily;
use locus_core::bench_api::{family_to_decoder, rotate90};
use proptest::prelude::*;

fn id_strategy(family: TagFamily) -> impl Strategy<Value = u16> {
    let n = family_to_decoder(family).num_codes() as u16;
    0u16..n
}

fn check_all_rotations(family: TagFamily, id: u16) -> Result<(), TestCaseError> {
    let decoder = family_to_decoder(family);
    let dim = decoder.dimension();
    let mut bits = decoder
        .get_code(id)
        .ok_or_else(|| TestCaseError::fail(format!("get_code({id}) returned None")))?;
    let base = bits;

    for k in 0..4u32 {
        let (decoded_id, hamming, _rot) = decoder
            .decode(bits)
            .ok_or_else(|| TestCaseError::fail(format!("decode failed for k={k}")))?;
        prop_assert_eq!(decoded_id, u32::from(id), "k={}: id mismatch", k);
        prop_assert_eq!(hamming, 0, "k={}: hamming should be 0 under rotation", k);
        bits = rotate90(bits, dim);
    }

    let active_mask = if decoder.bit_count() >= 64 {
        u64::MAX
    } else {
        (1u64 << decoder.bit_count()) - 1
    };
    prop_assert_eq!(
        base & active_mask,
        bits & active_mask,
        "rotate90^4 is not identity"
    );
    Ok(())
}

/// Assert that the `rotation` field returned by `decode` varies across k —
/// the 4 decoded rotations must be a permutation of {0,1,2,3}. Catches a
/// decoder that accepts every orientation but labels them incorrectly.
fn check_rotation_field(family: TagFamily, id: u16) -> Result<(), TestCaseError> {
    let decoder = family_to_decoder(family);
    let dim = decoder.dimension();
    let Some(mut bits) = decoder.get_code(id) else {
        return Ok(());
    };

    let mut seen = [false; 4];
    for k in 0..4u32 {
        let (_, _, rot) = decoder
            .decode(bits)
            .ok_or_else(|| TestCaseError::fail(format!("decode failed for k={k}")))?;
        prop_assert!((rot as usize) < 4, "rotation {} out of range", rot);
        prop_assert!(!seen[rot as usize], "rotation {} reported twice", rot);
        seen[rot as usize] = true;
        bits = rotate90(bits, dim);
    }
    prop_assert!(seen.iter().all(|&s| s), "rotations not a permutation");
    Ok(())
}

macro_rules! family_rotation_proptest {
    ($name:ident, $family:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig {
                failure_persistence: Some(Box::new(
                    proptest::test_runner::FileFailurePersistence::Direct(
                        "proptest-regressions/decoder_rotation.txt",
                    ),
                )),
                cases: 256,
                ..ProptestConfig::default()
            })]

            #[test]
            fn $name(id in id_strategy($family)) {
                check_all_rotations($family, id)?;
                check_rotation_field($family, id)?;
            }
        }
    };
}

family_rotation_proptest!(
    prop_apriltag36h11_rotation_invariance,
    TagFamily::AprilTag36h11
);
family_rotation_proptest!(
    prop_apriltag16h5_rotation_invariance,
    TagFamily::AprilTag16h5
);
family_rotation_proptest!(prop_aruco4x4_50_rotation_invariance, TagFamily::ArUco4x4_50);
family_rotation_proptest!(
    prop_aruco4x4_100_rotation_invariance,
    TagFamily::ArUco4x4_100
);
