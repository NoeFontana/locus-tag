# Python Engineering Guidelines

When modifying Python code in `crates/locus-py` or `scripts/`, the focus must be on maximizing throughput at the FFI boundary, ensuring type safety, and maintaining reproducible environments.

## 1. FFI & Zero-Copy Rules
* **No Hot-Loop Allocations:** Do not slice, copy, or instantiate large arrays inside the tight detection loop.
* **Contiguous Memory:** Ensure image data passed to Rust is contiguous. Rely on Rust's `PyReadonlyArray2<u8>` to interpret the buffer securely.

## 2. Typing & API Surface
* **Strict Typing:** All Python code must be fully type-hinted. We rely on `basedpyright` (via the `types` dependency group) to enforce static typing.
* **Stub Synchronization:** `locus/locus.pyi` is **generated** from the annotated pyo3 surface — do not hand-edit it. After changing the PyO3 interface, regenerate and commit it with `cargo run --bin stub_gen --no-default-features --features profiles,stub-gen` (under `uv run`); CI's `stub_gen --check` fails on drift. New `#[pyclass]`/`#[pymethods]`/`#[pyfunction]` items need the matching `#[gen_stub_*]` annotation to appear in the stub.

## 3. Environment & Orchestration
* **`uv` Everywhere:** Never use global `pip`. All dependency management, locking, and tool execution must be routed through `uv`.
* **PEP 735 Dependency Groups:** Utilize specific groups (`dev`, `lint`, `types`, `bench`, `docs`, `etl`) when running tasks. For example: `uv run --group bench tools/cli.py bench`.

## 4. Quality Gates
* **Linting:** `uv run ruff check . --fix`
* **Formatting:** `uv run ruff format .`
* **Type Checking:** `uv run --group types --group bench --group etl basedpyright`
* **Testing:** `uv run pytest`
