#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]
//! `common::datasets` repeats `xtask/datasets.toml`. This makes that repetition checked.
//!
//! The test helpers cannot read the manifest at runtime without a TOML parser in
//! `dev-dependencies`, and `docs/engineering/constraints.md` §5 asks for a lean graph, so the
//! `env` and `dest` of each dataset are written down twice. Twice is tolerable only while
//! something notices when the two copies disagree -- otherwise relocating a dataset in the
//! manifest leaves the suites resolving the old path, which is the same class of failure as
//! the silent skips this module replaced: the tests go on reporting something about data they
//! are no longer reading.
//!
//! Needs no dataset, so it runs in the default (CI) test set.

mod common;

use common::datasets::{HUB, ICRA};

/// Value of `key` inside `[table]` of the manifest, as the manifest literally spells it.
fn manifest_field(manifest: &str, table: &str, key: &str) -> String {
    let body = manifest
        .split(&format!("\n[{table}]\n"))
        .nth(1)
        .unwrap_or_else(|| panic!("xtask/datasets.toml has no [{table}] table"));
    // Stop at the next table header so a later table's `dest` cannot be picked up.
    let body = body.split("\n[").next().unwrap_or(body);
    for line in body.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(v) = rest.strip_prefix('=') {
                return v.trim().trim_matches('"').to_string();
            }
        }
    }
    panic!("[{table}] has no `{key}`");
}

#[test]
fn dataset_constants_match_the_manifest() {
    let manifest_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../xtask/datasets.toml");
    let manifest = std::fs::read_to_string(manifest_path)
        .unwrap_or_else(|e| panic!("cannot read {manifest_path}: {e}"));

    for (dataset, table) in [(&HUB, "hub"), (&ICRA, "icra2020-forward")] {
        assert_eq!(
            dataset.dest,
            manifest_field(&manifest, table, "dest"),
            "common::datasets::{} has dest '{}', but [{table}] in xtask/datasets.toml says \
             otherwise. The suites resolve the constant, so they would be reading the old \
             location while the fetcher writes the new one.",
            dataset.name,
            dataset.dest
        );
        assert_eq!(
            dataset.env,
            manifest_field(&manifest, table, "env"),
            "common::datasets::{} has env '{}', which no longer matches [{table}]; the \
             variable documented in quality-gates.md would stop relocating this dataset.",
            dataset.name,
            dataset.env
        );
    }
}
