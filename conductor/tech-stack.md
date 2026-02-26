# Locus Technology Stack

## 1. Core Architecture
- **Language:** Rust (2024 Edition)
- **Rust Build System:** `cargo` for managing crates, dependencies, and core compilation.
- **Math Backend:** `nalgebra` (specifically `SMatrix` for stack-allocated geometric algebra)
- **SIMD & Optimization:** `portable-simd` and the `multiversion` crate for runtime dispatch (compiling optimal code paths for AVX2, AVX-512, and NEON).
- **Memory Management:** `bumpalo` (Arena allocation for ephemeral per-frame structures to ensure zero heap allocations in the hot loop).

## 2. Python Integration
- **Bindings:** `pyo3` combined with `numpy` (Strict Zero-Copy via the Buffer Protocol).
- **Python Build System:** `maturin` (v1.0+) for PyPI distribution, creating the extension module, and seamless cross-compilation.

## 3. Observability & Debugging
- **Profiling:** `tracing` and `tracing-tracy` for low-overhead, span-based profiling.
- **Visual Debugging:** Rerun SDK (`rerun`) for logging and visualizing intermediate stages like threshold images, quad candidates, and pose axes.

## 4. Benchmarking Data Format
- **Format:** Synced Hugging Face Hub datasets use a native, extensible JSON Lines format (`annotations.jsonl`) accompanied by an `images/` directory, rather than the legacy ICRA2020 flattened CSV structure. This preserves rich metadata (e.g., distance, angle of incidence, 6DoF pose) provided by modern benchmarks. *(Note added 2026-02-25)*