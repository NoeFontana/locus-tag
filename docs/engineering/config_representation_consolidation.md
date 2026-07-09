# Design note: consolidating the config representations

**Status:** proposal / not scheduled. Written 2026-07-09 as the follow-up
deliverable to the release-prep maintainability pass (Phase 0 harness removal,
presets 4 → 3, dead-knob prune). No code change is proposed here — this note
scopes the options so the consolidation can be decided on a clean tree.

## Problem

A single detector setting today exists in **four hand-synced representations**,
plus a fifth stub and a schema:

| # | Representation | Location | Role |
| --- | --- | --- | --- |
| 1 | Flat Rust `DetectorConfig` (+ builder) | `crates/locus-core/src/config.rs` | Hot-path struct |
| 2 | Serde profile shim (`ProfileJson`/`ThresholdJson`/… + `From<ProfileJson>`) | same file | JSON boundary |
| 3 | Pydantic model | `crates/locus-py/locus/_config.py` (+ `__init__.py`) | Python-facing config |
| 4 | PyO3 `PyDetectorConfig` (+ `fn new` + two `From` impls) | `crates/locus-py/src/lib.rs` | FFI struct |
| — | `.pyi` type stub | `crates/locus-py/locus/locus.pyi` | Static-type surface |
| — | JSON Schema | `schemas/profile.schema.json` | Referee (regenerated) |

Adding or removing one scalar knob touches **6–8 sites by hand**. There is no
single tripwire that asserts the field sets agree — drift is caught only
indirectly by three separate tests (`profile_loading.rs`, `test_profiles.py`,
the schema-diff CI job), none of which asserts *set parity* across all layers.

### Empirical evidence (this maintainability pass)

The three cleanup PRs that preceded this note are a natural stress test of the
sync, and each hit the seams:

- Removing `post_decode_refinement` required edits in **8 places**; the
  hand-maintained `.pyi` stub was silently missed and only surfaced two PRs
  later when `basedpyright` flagged `_config.py` calling `PyDetectorConfig(...)`
  with a now-missing argument. `pytest` never caught it — the stub isn't
  exercised at runtime.
- Removing a single `bool` field dropped `PyDetectorConfig` below clippy's
  bool-count threshold, which turned two `#[expect(clippy::…_excessive_bools)]`
  attributes *unfulfilled* and broke the build — a spooky-action-at-a-distance
  coupling between an unrelated field and a lint suppression.
- `From<ProfileJson>` (~40 lines of `field: p.group.field,`), Pydantic
  `_to_ffi_config` (~70 lines), and the ~40-parameter `PyDetectorConfig::new`
  are pure transcription with no mechanization.

The cost is not the raw line count (config.rs ≈ 1.5k, `_config.py` ≈ 0.5k) but
the **O(layers) manual edit per knob** and the **absence of a parity gate**.

## Options

### A. Rust macro derives serde shim + PyO3 struct from the flat struct
A proc-macro (or `macro_rules!` over a single field table) generates
representations 2 and 4 — and the `From` impls between them — from the flat
`DetectorConfig` (1) annotated with per-field group + JSON-key metadata.

- **Pros:** collapses 1 → 2 → 4 into one source of truth in Rust; the
  `From<ProfileJson>` and `PyDetectorConfig` boilerplate disappears; adding a
  knob becomes a one-line struct-field edit.
- **Cons:** a bespoke macro is a maintenance surface of its own; proc-macro
  compile-time cost; the flat struct must carry grouping metadata (attributes)
  that muddies the hot-path type; does **not** cover the Python side (3) or the
  `.pyi`.

### B. Generate the Pydantic model (and `.pyi`) from the JSON Schema
Treat `schemas/profile.schema.json` as the source for the Python side and
codegen the Pydantic model + stub, inverting today's dump-Pydantic-to-schema
flow (or making it bidirectional with a single generator).

- **Pros:** removes the hand-written Pydantic nesting and the `.pyi` drift class
  entirely (the exact bug this pass hit); the schema is already the referee.
- **Cons:** the validators that live on the Pydantic model (cross-group invariants
  like *EdLines + Erf incompatible*, radius ordering) are not expressible in
  vanilla JSON Schema and would need a hand-written overlay; codegen’d models are
  awkward to add custom methods to (`from_profile`, `_to_ffi_config`).

### C. Keep four representations, add a 4th tripwire: field-set parity
Do no codegen. Add one test that reflects over all four representations and
asserts the **field name sets are identical** (Rust struct fields via a
`const &[&str]` or a derive; Pydantic `model_fields`; PyO3 `__init__` signature;
`.pyi` parse). Fails CI the instant any layer drifts.

- **Pros:** cheapest by far; directly closes the specific failure mode (silent
  set drift) that this pass exposed; no generated code, no macro; each layer
  stays hand-written and readable.
- **Cons:** does not reduce the per-knob edit count — it only makes an omission
  loud instead of silent. Value is a safety net, not a simplification.

## Recommendation

Sequence, not a single choice:

1. **Do C first** (small, high-value, low-risk). A parity tripwire would have
   turned every seam this pass hit into an immediate red build instead of a
   latent bug. It is worth landing regardless of whether A/B ever happen.
2. **Then evaluate B** for the Python side specifically — the `.pyi`/Pydantic
   drift is where the sync is both most fragile (no runtime coverage) and most
   mechanical. The validator overlay is the open question to prototype.
3. **Treat A as optional.** The Rust `From<ProfileJson>` / PyO3 boilerplate is
   verbose but compiler-checked (drift there is a build error today, not a silent
   bug), so it is the *least* leveraged place to spend a proc-macro.

Decide after this maintainability pass has landed and settled. None of A/B/C
should be bundled with behavior changes — they are refactors that must keep the
render-tag / ICRA snapshots byte-identical, and should be reviewed as such.
