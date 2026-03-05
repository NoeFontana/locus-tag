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

fn snapshot_dict(family: TagFamily) -> Vec<u64> {
    let dict = get_dictionary(family);
    dict.codes.to_vec()
}

#[test]
fn test_dictionary_snapshots() {
    assert_debug_snapshot!("tag36h11_codes", snapshot_dict(TagFamily::AprilTag36h11));
}
