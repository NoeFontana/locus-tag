use insta::assert_debug_snapshot;
use locus_core::config::TagFamily;
use locus_core::dictionaries::get_dictionary;

/// Stable FNV-1a 64-bit hash for byte-for-byte parity checks
fn fvn1a_hash_usize(slice: &[usize]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for &val in slice {
        for b in val.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{:016x}", hash)
}

fn fvn1a_hash_u32_data(slice: &[u32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for &val in slice {
        for b in val.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{:016x}", hash)
}

fn snapshot_dict(family: TagFamily) -> (usize, String, usize, String) {
    let dict = get_dictionary(family);
    (
        dict.mih_offsets.len(),
        fvn1a_hash_usize(dict.mih_offsets),
        dict.mih_data.len(),
        fvn1a_hash_u32_data(dict.mih_data),
    )
}

#[test]
fn test_dictionary_snapshots_tag36h11() {
    assert_debug_snapshot!("tag36h11_parity", snapshot_dict(TagFamily::AprilTag36h11));
}

#[test]
fn test_dictionary_snapshots_tag16h5() {
    assert_debug_snapshot!("tag16h5_parity", snapshot_dict(TagFamily::AprilTag16h5));
}

#[test]
fn test_dictionary_snapshots_tag41h12() {
    assert_debug_snapshot!("tag41h12_parity", snapshot_dict(TagFamily::AprilTag41h12));
}
