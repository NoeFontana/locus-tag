# locus-tag Python Conductor Guide

When executing a Conductor track that involves modifying Python code in `crates/locus-py` or `scripts/`, your primary focus must be zero-copy safety at the FFI boundary.

**Core Rules:**
* **Zero-Copy NumPy:** All image data passed to Rust must be contiguous. Rely on `PyReadonlyArray2<u8>` on the Rust side, and do not perform slicing or copying in the hot loop on the Python side.
* **Type Stubs:** Ensure `locus/locus.pyi` is perfectly synchronized with any PyO3 changes.
* **Environment:** All tools must be executed via `uv`. Do not use system `pip`.

**Execution & Quality Gates:**
Before closing a Python-related track, you must pass the centralized repository quality gates defined in **[quality-gates.md](../../.agent/rules/quality-gates.md)**.
* Linting: Run the Python linting steps from **[lint.md](../../.agent/workflows/lint.md)**.
* Formatting: Run the Python formatting steps from **[format.md](../../.agent/workflows/format.md)**.