# General Code Style Principles

These principles apply universally across the Locus codebase, transcending language boundaries.

## 1. Automated Consistency
Do not argue about style in code reviews. Rely on the machines.
* **Rust:** `cargo fmt` is the absolute authority.
* **Python:** `ruff format` and `ruff check` dictate structure and linting.
* **Action:** Configure your editor to format on save.

## 2. Explicitness Over Magic
* **Avoid Hidden Control Flow:** Code should be straightforward. Prefer explicit pattern matching and clear variable names over overly clever, condensed logic.
* **Design for the Reader:** Code is read orders of magnitude more often than it is written. Optimize for readability, even if it requires slightly more verbosity.

## 3. Actionable Documentation
* **Document the "Why":** Comments should explain the reasoning behind a decision, the mathematical basis of an algorithm, or why an edge case exists—not reiterate what the code mechanically does.
* **Keep Docs Proximate:** Place documentation as close to the relevant code as possible to prevent drift.

## 4. Architectural Simplicity
* **Flat is Better than Nested:** Avoid deep class hierarchies or overly complex module trees.
* **Minimize State:** Prefer pure functions and immutable data structures where feasible. Isolate state mutation to explicitly managed contexts (like the `Bump` arena).
