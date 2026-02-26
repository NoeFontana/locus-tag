# locus-tag Rust Conductor Guide

When executing a Conductor track that involves modifying Rust code in `crates/locus-core`, you MUST strictly adhere to the project's centralized Antigravity rules. 

**Do not proceed with coding until you have ingested the following:**
1. **[Architectural Invariants](../../.agent/rules/architecture.md):** Defines the cache-locality and SIMD dispatch expectations.
2. **[Strict Constraints](../../.agent/rules/constraints.md):** Defines the zero-allocation arena rules and the strict zero-panic policy (`Result` types, `thiserror`, no `.unwrap()`).
3. **[Core Logic Guide](../../.agent/rules/core.md):** Defines the mathematical and geometric standards (e.g., using `nalgebra::SMatrix`).

**Execution & Quality Gates:**
When you reach the Refactor or Verification stages of your track, you must execute the exact commands defined in our workflows:
* To format: Execute the steps in **[format.md](../../.agent/workflows/format.md)**
* To lint: Execute the steps in **[lint.md](../../.agent/workflows/lint.md)** (Zero warnings allowed).
* To test: Execute the steps in **[test.md](../../.agent/workflows/test.md)** and **[bench.md](../../.agent/workflows/bench.md)**.