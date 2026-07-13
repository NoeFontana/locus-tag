//! Regenerates `locus/locus.pyi` from the `#[gen_stub_*]`-annotated pyo3
//! surface in `lib.rs`.
//!
//! pyo3-stub-gen writes `<python-root>/locus/__init__.pyi` (its mixed-layout
//! package convention). This project's compiled module is the *leaf submodule*
//! `locus.locus` (imported as `from .locus import ...`), typed by
//! `locus/locus.pyi`, with a hand-written pure-Python `locus/__init__.py`
//! layered on top. So we relocate the generated content to `locus.pyi` and
//! delete the stray `__init__.pyi` (leaving it would shadow the pure-Python
//! package for type checkers).
//!
//! Usage (under `uv run`, so the interpreter is on the linker path):
//! - `cargo run --bin stub_gen --no-default-features --features profiles,stub-gen`
//! - append `-- --check` to fail instead of writing when the stub is stale
//!
//! Needs `stub-gen` ON (for pyo3-stub-gen) and `extension-module` OFF (so the
//! binary links libpython and runs standalone). Under any other feature set the
//! real body is compiled out — see the two `main`s below — so plain
//! `cargo build`/`clippy` never link a bin against a deferred-symbol module and
//! never pull pyo3-stub-gen into the default graph.

#[cfg(all(feature = "stub-gen", not(feature = "extension-module")))]
fn main() -> pyo3_stub_gen::Result<()> {
    use std::path::{Path, PathBuf};

    /// Removes the stray generated `__init__.pyi` on *every* exit path (including
    /// `?` early-returns). Leaving it behind would shadow the hand-written
    /// pure-Python `__init__.py` for type checkers — the exact hazard this file
    /// warns about — which matters most in `--check` (read-only) mode.
    struct RemoveOnDrop(PathBuf);
    impl Drop for RemoveOnDrop {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    let check = std::env::args().any(|a| a == "--check");

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let generated = manifest_dir.join("locus/__init__.pyi");
    let target = manifest_dir.join("locus/locus.pyi");

    let stub = locus::stub_info()?;
    stub.generate()?; // writes <python-root>/locus/__init__.pyi
    let _cleanup = RemoveOnDrop(generated.clone());

    let new_content = postprocess(&std::fs::read_to_string(&generated)?);

    if check {
        let current = std::fs::read_to_string(&target).unwrap_or_default();
        if current != new_content {
            eprintln!(
                "locus/locus.pyi is out of date.\nRun `cargo run --bin stub_gen \
                 --no-default-features --features profiles,stub-gen` (under `uv run`) and commit it."
            );
            std::process::exit(1);
        }
        println!("locus/locus.pyi: up to date");
    } else {
        std::fs::write(&target, &new_content)?;
        println!("wrote {}", target.display());
    }
    Ok(())
}

/// Deterministic fixups pyo3-stub-gen cannot express itself:
///
/// - The `CornerRefinementMode::None` variant is a Python keyword, so the raw
///   `None = ...` line is a `SyntaxError`. Emit it as `None_` (a valid alias)
///   with a note; the runtime attribute is still `None` (reach it via
///   `getattr(CornerRefinementMode, "None")`).
/// - The `#[pyclass(eq_int)]` enums are int-comparable at runtime, but
///   pyo3-stub-gen emits plain `enum.Enum`; rewrite the ones in [`INT_ENUMS`]
///   to `enum.IntEnum` so `int(TagFamily.X)` type-checks (the wrapper passes
///   families as ints). The rewrite is an explicit allowlist rather than a
///   blanket `enum.Enum`→`IntEnum` sweep: a future non-`eq_int` enum must stay
///   a plain `Enum`, so it is only converted when named here.
#[cfg(all(feature = "stub-gen", not(feature = "extension-module")))]
fn postprocess(content: &str) -> String {
    /// Enums declared `#[pyclass(..., eq_int, ...)]` in `lib.rs`. Keep in sync
    /// when adding an int-comparable enum; a non-`eq_int` enum must NOT appear.
    const INT_ENUMS: &[&str] = &[
        "TagFamily",
        "SegmentationConnectivity",
        "CornerRefinementMode",
        "QuadExtractionMode",
        "EdLinesImbalanceGatePolicy",
        "DistortionModel",
    ];

    let mut out = String::with_capacity(content.len() + 256);
    for line in content.lines() {
        let as_int_enum = line
            .strip_prefix("class ")
            .and_then(|r| r.strip_suffix("(enum.Enum):"))
            .filter(|name| INT_ENUMS.contains(name));

        let rewritten = if line.trim() == "None = ..." {
            let indent = &line[..line.len() - line.trim_start().len()];
            format!("{indent}None_ = ...  # runtime attribute is `None` (a Python keyword)")
        } else if let Some(name) = as_int_enum {
            format!("class {name}(enum.IntEnum):")
        } else {
            line.to_string()
        };
        out.push_str(&rewritten);
        out.push('\n');
    }
    out
}

/// Compiled whenever the generator prerequisites are absent (`stub-gen` off, or
/// `extension-module` on — which defers libpython symbols so a standalone binary
/// cannot link). A no-op that explains the correct invocation instead of failing
/// to compile/link.
#[cfg(not(all(feature = "stub-gen", not(feature = "extension-module"))))]
fn main() {
    eprintln!(
        "stub_gen must be built with `--no-default-features --features profiles,stub-gen` \
         (stub-gen ON, extension-module OFF). See crates/locus-py/Cargo.toml."
    );
    std::process::exit(2);
}
