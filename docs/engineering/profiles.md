# Detector Profiles

Detector settings in Locus are authored once, as JSON, and consumed by both
the Rust core and the Python wrapper. This document explains why.

## Design invariants

1. **JSON wins.** If a Rust constant and the shipped JSON disagree, the JSON
   is authoritative. The `profile_loading.rs` integration test fails loudly
   on drift — two engineers diff-review any change to a shipped profile.
2. **Flat Rust, nested JSON.** The `DetectorConfig` struct in
   `locus-core` stays flat (30+ scalar fields, touched on the hot path).
   Nesting exists only at two boundaries: the serde shim that accepts JSON
   input, and the Pydantic model that Python consumers edit.
3. **Build-time embedded profiles.** The three shipped profiles
   (`standard`, `grid`, `high_accuracy`) live in
   `crates/locus-core/profiles/*.json` and are `include_str!`'d into the
   Rust crate at compile time. The Python wheel does not ship the JSON
   files separately — `DetectorConfig.from_profile` reads the exact
   embedded bytes via the `_shipped_profile_json` FFI hook, so a user
   cannot accidentally break their detector by deleting a data file.
4. **Custom profiles are a first-class path.** Teams with tuned
   configurations author a JSON file, version-control it, and load it via
   `DetectorConfig.from_profile_json(text)` (Python) or
   `DetectorConfig::from_profile_json(text)` (Rust).
5. **Validation lives in Python.** The Pydantic model is the primary
   front-line — cross-group invariants like `EdLines + Erf is incompatible`
   and `min_fill_ratio < max_fill_ratio` surface there. Rust's
   `DetectorConfig::validate()` stays as a defence-in-depth gate.

## Why not YAML or TOML?

JSON Schema exists as a mature draft (2020-12) with editor integrations in
every major language. YAML's duplicate-key and implicit-type behaviour adds
surface for silent config errors in a safety-critical perception pipeline;
TOML's flat section model fights the nested grouping we want for humans.
JSON + JSON Schema wins because it is boring and works everywhere.

## Why Pydantic on the Python side?

The Python consumer base uses Pydantic ubiquitously for config management;
reusing the model keeps editor validation, IDE autocomplete, and our own
validators on one toolchain. `model_dump_json` round-trips to the same
string the Rust serde shim consumes *and produces* — the two are independent
reader **and writer** of one shared file format, which is exactly what lets the
config cross the FFI as JSON (see "Crossing the Rust ↔ Python boundary").

## Why build-time embedding?

A runtime profile load that fails produces an obscure
`FileNotFoundError` at detector construction — typically long after the
user expected a working detector. Embedding the three shipped profiles
makes `Detector(profile="standard")` infallible for the default path, and
moves file I/O to only the explicit custom-profile case
(`DetectorConfig.from_profile_json(...)`), where failure modes are the
caller's to handle.

## Where the files live

| Purpose | Path |
| --- | --- |
| Source of truth (embedded into Rust) | `crates/locus-core/profiles/*.json` |
| Python accessor (reads embedded bytes via FFI) | `locus._config.DetectorConfig.from_profile` |
| Referee schema | `schemas/profile.schema.json` |
| Serde shim (JSON ↔ flat, both directions) | `crates/locus-core/src/config.rs::ProfileJson` |
| Pydantic model | `locus._config.DetectorConfig` |
| Flat Rust struct | `crates/locus-core/src/config.rs::DetectorConfig` |
| Generated Python type stub | `crates/locus-py/locus/locus.pyi` (regenerate: `cargo run --bin stub_gen --no-default-features --features profiles,stub-gen`) |

## Crossing the Rust ↔ Python boundary

Config crosses the FFI as **JSON text, not a typed struct**. Constructing a
detector serializes the Pydantic model with `model_dump_json()` and hands the
string to Rust's `from_profile_json`; `Detector.config()` calls Rust's
`to_profile_json` and Python re-parses the result. The shipped JSON profile
format is therefore the single contract on both sides — Rust and Python are
independent *reader and writer* of one format, and the round-trip is total over
every field (no hand-maintained field-by-field FFI struct to drift).

One consequence: `f64::INFINITY` (the documented value that disables the
pose-consistency escape clause) has no JSON number form, so it round-trips as
`null` — the serde shim maps `∞ ↔ null` and Pydantic maps `math.inf ↔ null`.

The Python type stub (`locus.pyi`) is **generated** from the annotated pyo3
surface by the `stub_gen` binary; do not hand-edit it.

## Drift tripwires

Five failure modes are instrumented, closing the loop
**Rust flat struct ↔ serde shim ↔ Pydantic ↔ schema ↔ `.pyi`**:

1. `profile_loading.rs` — Rust test; loads each shipped profile, asserts
   hand-transcribed values, and round-trips a config through
   `to_profile_json`/`from_profile_json`. Catches profile drift and value
   mis-wiring in the flat↔nested mapping.
2. `config::schema_parity_tests` — Rust unit test; asserts the serde shim's
   JSON key set equals `schemas/profile.schema.json`. Catches Rust ↔ schema
   field-set drift.
3. `schemas/profile.schema.json` — CI job dumps Pydantic's
   `model_json_schema()` and diffs against this file. Catches Pydantic ↔ schema
   drift.
4. `test_profiles.py` — Python test; builds a `Detector` from each shipped
   profile and asserts `Detector.config()` equals the source config over every
   field. Catches Rust ↔ Python FFI drift.
5. `stub_gen --check` — CI step; regenerates `locus.pyi` and fails on drift.
   Catches pyo3 surface ↔ stub drift.

A regression in any one should fail CI. A regression in several simultaneously
is a design-level issue — escalate.
