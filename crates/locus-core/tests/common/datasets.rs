//! Where a dataset lives, and whether a suite may decline to run.
//!
//! One resolver for every dataset-backed suite. Before this module the same two questions —
//! *where is the data* and *may I skip* — had five different answers in five files:
//! `common::hub::run_render_tag_test` read the env var and returned on `Err`;
//! `regression_board_hub` read it with a `tests/data/hub_cache` default;
//! `regression_distortion_hub` read it, returned on `Err`, then asserted the subdir existed;
//! `common::resolve_dataset_root` read it and otherwise preferred an in-tree stub over the
//! real dataset; `diagnose_render_tag_2160p` read it and returned on `Err`. The divergence
//! was not cosmetic — it decided whether a suite ran at all, and two of those five answers
//! silently meant "don't".
//!
//! ## Declining must not look like passing
//!
//! The Rust harness offers two verdicts, and these suites spent "pass" on "could not run".
//! Measured on `4242890`: `cargo test --test regression_render_tag` with `LOCUS_HUB_DATASET_DIR`
//! unset reports *"8 passed; finished in 0.00s"*, and `regression_icra2020` reports
//! *"9 passed in 0.09s"* against 8.44s with data. Under that cover PR #449 moved render-tag
//! corner RMSE by 25-40 % across the tag36h11 baselines and cost recall on `tag16h5` and two
//! ICRA configs, with every suite green. A gate that cannot fail is not a gate.
//!
//! So a missing dataset is a **failure** here. An environment that genuinely has no data says
//! so out loud by setting [`ALLOW_MISSING`], which keeps the vacuum in the caller's command
//! line or workflow file where a reader can see it, instead of in a `println!` nobody reads.

use std::env;
use std::path::PathBuf;

/// Let a dataset-backed suite report success without its data.
///
/// For environments that legitimately have none (CI fetches no datasets). It is deliberately
/// verbose: the suite then verifies nothing, and that should be an explicit claim.
pub const ALLOW_MISSING: &str = "LOCUS_ALLOW_MISSING_DATASETS";

/// Let a dataset-backed suite run in a debug build.
///
/// These suites are ~19x slower unoptimised (measured: one `regression_render_tag` case,
/// 10.12 s debug against 0.52 s release on `4242890`), which turns a 30-second sweep into ten
/// minutes. The snapshot values are identical either way, so this is a speed guard, not a
/// correctness one -- set it when you specifically want an unoptimised run, e.g. under a
/// debugger.
pub const ALLOW_DEBUG: &str = "LOCUS_ALLOW_DEBUG_DATASET_TESTS";

/// One external dataset, mirroring its entry in `xtask/datasets.toml`.
///
/// `dest` repeats that manifest's `dest` field, which is the one duplication left: the tests
/// cannot read the TOML without a parser in `dev-dependencies`, so the two must be kept in
/// step by hand. The default is what makes a bare `cargo nextest run --profile datasets`
/// work with no environment at all, which is the point -- an agent that has to remember two
/// env vars will sooner or later not remember them, and the failure mode used to be silence.
pub struct Dataset {
    /// Registry key in `xtask/datasets.toml`.
    pub name: &'static str,
    /// Variable that relocates the dataset, read by the Python loaders too.
    pub env: &'static str,
    /// Repository-relative default, as declared by the manifest.
    pub dest: &'static str,
}

/// Synthetic render-tag / board / distortion suites (`[hub]`).
pub const HUB: Dataset = Dataset {
    name: "hub",
    env: "LOCUS_HUB_DATASET_DIR",
    dest: "tests/data/hub_cache",
};

/// ICRA 2020 AprilTag localization-accuracy dataset (`[icra2020-*]`).
pub const ICRA: Dataset = Dataset {
    name: "icra2020",
    env: "LOCUS_ICRA_DATASET_DIR",
    dest: "tests/data/icra2020",
};

impl Dataset {
    /// The dataset root: the environment variable when set, else the manifest default.
    ///
    /// An env var pointing at something that is not a directory panics rather than falling
    /// back, so an explicit request is never quietly redirected to a different dataset.
    pub fn root(&self) -> PathBuf {
        if let Ok(raw) = env::var(self.env) {
            let resolved = super::resolve_hub_root(&raw);
            assert!(
                resolved.is_dir(),
                "{}='{}' is not a directory (resolved to '{}'). Unset it to use the default \
                 '{}', or point it at a real cache.",
                self.env,
                raw,
                resolved.display(),
                self.dest
            );
            return resolved;
        }
        super::resolve_hub_root(self.dest)
    }

    /// The path a **gate** needs, or a loud failure.
    ///
    /// `Some` in the ordinary case. `None` only when the data is absent *and* [`ALLOW_MISSING`]
    /// declared that acceptable, so `let Some(p) = ... else { return }` stays the shape at the
    /// call site while the silent branch now has to be asked for.
    ///
    /// Pass `""` for the dataset root itself.
    pub fn require(&self, subdir: &str) -> Option<PathBuf> {
        require_release_build(subdir);
        let path = self.join(subdir);
        if path.is_dir() {
            return Some(path);
        }
        let what = if subdir.is_empty() {
            self.name.to_string()
        } else {
            format!("{}/{}", self.name, subdir)
        };
        assert!(
            env::var(ALLOW_MISSING).is_ok(),
            "{what} is not at '{}', so this suite can verify nothing.\n  \
             Fetch it:  cargo xtask data fetch {}\n  \
             Relocate:  {}=<dir>\n  \
             Or set {ALLOW_MISSING}=1 to let dataset suites report success without their \
             data -- they then check nothing, which is why it has to be said out loud.",
            path.display(),
            self.name,
            self.env
        );
        println!(
            "{what}: not run (no data at '{}'); allowed by {ALLOW_MISSING}",
            path.display()
        );
        None
    }

    /// The path a **diagnostic** needs, or `None`, quietly.
    ///
    /// For the `diagnose_*` binaries, which report findings rather than gate a change and are
    /// not excluded from the default nextest filter. They may legitimately do nothing.
    pub fn optional(&self, subdir: &str) -> Option<PathBuf> {
        let path = self.join(subdir);
        path.is_dir().then_some(path)
    }

    fn join(&self, subdir: &str) -> PathBuf {
        let root = self.root();
        if subdir.is_empty() {
            root
        } else {
            root.join(subdir)
        }
    }
}

/// Refuse to spend ten minutes doing what takes thirty seconds.
///
/// See [`ALLOW_DEBUG`]. Separate from the data check so a debug invocation is told the one
/// thing it needs even when the data is also absent.
pub fn require_release_build(what: &str) {
    assert!(
        !cfg!(debug_assertions) || env::var(ALLOW_DEBUG).is_ok(),
        "dataset suites are release-only (this is {what}): unoptimised they run ~19x slower, \
         which is ten minutes instead of thirty seconds for the full sweep, and the snapshot \
         values are identical either way.\n  \
         Re-run:  cargo nextest run --profile datasets --release --features bench-internals\n  \
         Or set {ALLOW_DEBUG}=1 if an unoptimised run is what you want."
    );
}
