#![allow(
    missing_docs,
    dead_code,
    clippy::unwrap_used,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use
)]
use insta::assert_debug_snapshot;
use locus_core::TagFamily;
use locus_core::bench_api::*;

/// Stable FNV-1a 64-bit hash for byte-for-byte parity checks
fn fnv1a_hash_usize(slice: &[usize]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for &val in slice {
        for b in val.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn fnv1a_hash_u32_data(slice: &[u32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for &val in slice {
        for b in val.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn snapshot_dict(family: TagFamily) -> (usize, String, usize, String) {
    let dict = get_dictionary(family);
    (
        dict.mih_offsets.len(),
        fnv1a_hash_usize(dict.mih_offsets),
        dict.mih_data.len(),
        fnv1a_hash_u32_data(dict.mih_data),
    )
}

#[test]
fn test_dictionary_snapshots() {
    assert_debug_snapshot!("tag36h11_parity", snapshot_dict(TagFamily::AprilTag36h11));
    assert_debug_snapshot!("tag41h12_parity", snapshot_dict(TagFamily::AprilTag41h12));
    assert_debug_snapshot!("aruco4x4_50_parity", snapshot_dict(TagFamily::ArUco4x4_50));
    assert_debug_snapshot!("aruco4x4_100_parity", snapshot_dict(TagFamily::ArUco4x4_100));
}
