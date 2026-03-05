use locus_core::bench_api::*;
use locus_core::TagFamily;
use insta::assert_debug_snapshot;

fn snapshot_dict(family: TagFamily) -> Vec<u64> {
    let dict = get_dictionary(family);
    dict.codes.to_vec()
}

#[test]
fn test_dictionary_snapshots() {
    assert_debug_snapshot!("tag36h11_codes", snapshot_dict(TagFamily::AprilTag36h11));
}
