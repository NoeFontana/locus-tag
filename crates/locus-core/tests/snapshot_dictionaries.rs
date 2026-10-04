#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
use insta::assert_debug_snapshot;
use locus_core::TagFamily;
use locus_core::bench_api::*;

/// Stable FNV-1a 64-bit hash for byte-for-byte parity checks of the rotated code tables.
fn fnv1a_hash_u64(slice: &[u64]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for &val in slice {
        for b in val.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn snapshot_dict(family: TagFamily) -> (u32, usize, String) {
    let dict = get_dictionary(family);
    (
        dict.payload_length,
        dict.codes.len(),
        fnv1a_hash_u64(dict.codes),
    )
}

#[test]
fn test_dictionary_snapshots() {
    assert_debug_snapshot!("tag16h5_codes", snapshot_dict(TagFamily::AprilTag16h5));
    assert_debug_snapshot!("tag36h11_codes", snapshot_dict(TagFamily::AprilTag36h11));
    assert_debug_snapshot!("aruco4x4_50_codes", snapshot_dict(TagFamily::ArUco4x4_50));
    assert_debug_snapshot!("aruco4x4_100_codes", snapshot_dict(TagFamily::ArUco4x4_100));
    assert_debug_snapshot!("aruco6x6_250_codes", snapshot_dict(TagFamily::ArUco6x6_250));
    assert_debug_snapshot!(
        "aruco_mip_36h12_codes",
        snapshot_dict(TagFamily::ArUcoMip36h12)
    );
}
