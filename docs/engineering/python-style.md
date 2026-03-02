# Python Engineering Guidelines

When modifying Python code in `crates/locus-py` or `scripts/`, the focus must be on maximizing throughput at the FFI boundary, ensuring type safety, and maintaining reproducible environments.

## 1. FFI & Zero-Copy Rules
* **No Hot-Loop Allocations:** Do not slice, copy, or instantiate large arrays inside the tight detection loop.
* **Contiguous Memory:** Ensure image data passed to Rust is contiguous. Rely on Rust's `PyReadonlyArray2<u8>` to interpret the buffer securely.

## 2. Typing & API Surface
* **Strict Typing:** All Python code must be fully type-hinted. We rely on `mypy` (via the `types` dependency group) to enforce static typing.
* **Stub Synchronization:** If the PyO3 Rust interface changes, you MUST update the corresponding `locus/locus.pyi` type stubs to match perfectly.

## 3. Environment & Orchestration
* **`uv` Everywhere:** Never use global `pip`. All dependency management, locking, and tool execution must be routed through `uv`.
* **PEP 735 Dependency Groups:** Utilize specific groups (`dev`, `lint`, `types`, `bench`, `docs`, `etl`) when running tasks. For example: `uv run --group bench python scripts/locus_bench.py`.

## 4. Quality Gates
* **Linting:** `uv run ruff check .`
* **Formatting:** `uv run ruff format .`
* **Testing:** `uv run pytest`
