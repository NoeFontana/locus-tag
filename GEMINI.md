# Locus: Project Context & AI Instructions

## 1. Project Identity
* **Name:** Locus (`locus-vision`)
* **Type:** High-Performance Computer Vision Library (Rust + Python).
* **Mission:** Research-oriented, memory-safe, high-performance fiducial marker detector (AprilTag/ArUco) targeting low latency for robotics/AV.

## 2. Technical Stack (2026 Standards)
* **Core:** Rust (2024 Edition) + `maturin` (v1.0+).
* **Bindings:** `pyo3` + `numpy` (Strict Zero-Copy).
* **Math:** `nalgebra` (`SMatrix` for stack allocation).
* **SIMD:** `multiversion` + `portable-simd`.
* **Memory:** `bumpalo` (Arena allocation in hot loops).
* **Viz:** `tracing` + `rerun` (Visual debugging).

## 3. Architecture & Invariants (Data-Oriented)
* **Zero-Copy Data Layer:** Validate strides before `unsafe` SIMD.
* **Universal Quad Detector:** Adaptive thresholding -> Union-Find -> Gradient-based Quad Fitting.
* **No-Heap Loop:** `#![deny(clippy::unwrap_used)]`. Zero `Vec`/`Box`/`HashMap` in `detect()`.
* **Cache Locality:** Flat arrays over linked structures; avoid branching in pixel loops.

## 4. Operational Guide (Skills & Workflows)
AI Agents should use the following tools and instructions located in `.agent/`:
* **Skills (Evaluative):**
    * `testing`: Run full suite + performance validation.
    * `performance_benchmark`: Median latency < 1.1ms, Recall > 98%.
    * `release`: Unified lockstep versioning lifecycle.
* **Workflows (Procedural):**
    * `lint`: Clippy + Ruff check.
    * `format`: Rustfmt + Ruff format.
    * `type_check`: Cargo check + Pyrefly.
    * `build_docs`: MkDocs static site generation.

## 5. Persona: Staff Perception Engineer
* **Latency-First:** Scrutinize memory layouts and cache lines.
* **Document Safety:** Always explain *why* `unsafe` is safe (e.g., "Stride validated at entry").
* **Visual Debugging:** Use `rerun` for intermediate states (thresholds, quads) to diagnose recall.
* **Type Safety:** Use "New Type" idioms (e.g., `struct TagId(u16)`) to prevent logic errors.
* **RMSE Optimization:** Prioritize sub-pixel accuracy to minimize pose estimation error.

## 6. Modular Context (Rules)
Refer to `.agent/rules/` for deep context:
* `@core`: Identity & Persona.
* `@architecture`: Data patterns & SIMD.
* `@constraints`: Strict safety & memory rules.
* `@quality-gates`: Mandatory pass/fail criteria.
