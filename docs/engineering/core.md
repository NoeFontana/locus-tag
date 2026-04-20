# Core Identity & Engineering Principles

## 1. Project Identity
* **Name:** Locus (`locus-tag`)
* **Mission:** Deliver a production-grade, memory-safe, state-of-the-art fiducial marker detector.
* **Target Audience:** Robotics, Autonomous Vehicles (AV), and Perception Engineers who require bounded latency and high reliability.

## 2. Engineering Directives
* **Latency Obsession:** Every microsecond counts. Scrutinize cache lines, memory access patterns, and branching behavior. Prioritize Data-Oriented Design (DOD) over classical Object-Oriented patterns.
* **Concurrency First:** All heavy pipeline stages must release the Python Global Interpreter Lock (GIL) to enable high-throughput multi-threaded perception systems.
* **Safety First:** Rust's safety guarantees are a feature, not a hurdle. When bypassing them via `unsafe` (e.g., for SIMD or raw buffer access), the burden of proof is on the author to document soundness.
* **OpenCV Ecosystem Parity:** We strictly adhere to modern OpenCV (`cv2.aruco`) conventions for tag layout, bit ordering (row-major), and canonical orientation. This ensures seamless interoperability with the broader computer vision ecosystem.
* **Visual Verifiability:** Perception algorithms are notoriously difficult to debug via text. We mandate the use of the `rerun` SDK to emit rich, interactive visualizations of intermediate pipeline stages.
* **Modern Toolchain:** We embrace the bleeding edge of tooling to improve developer velocity: `uv` for Python environments, `cargo nextest` for parallel testing, and `maturin` for seamless cross-language builds.
*   **Performance Observability:** We maintain nanosecond-level visibility into the pipeline without sacrificing production speed. By leveraging static `tracing` spans and compile-time erasure, we ensure that performance regressions are immediately detectable during profiling while maintaining zero runtime overhead in deployment.
*   **Mathematical Rigor:** Sub-pixel localization is a science of offsets. We strictly adhere to the **+0.5 pixel center rule** and center-aware decimation mapping to ensure that our synthetic benchmarks and real-world optimizations are numerically consistent with standard vision libraries (OpenCV/Blender).
*   **Zero-Overhead Debugging:** Our commitment to performance extends to our diagnostic tools. The Rerun-based debug pipeline follows a "pay-only-when-using" model: zero runtime impact on the production hot-path, arena-allocated ephemeral telemetry, and zero-copy data views.
*   **Configuration is nested for humans, flat for machines, authored in JSON, embedded at build.** Detector settings live in one place: the shipped JSON profiles (`crates/locus-py/locus/profiles/*.json`). Rust keeps its flat `DetectorConfig` struct for hot-path access; a serde shim nests at the JSON boundary. Python users interact with a nested Pydantic `DetectorConfig`. The three shipped profiles are `include_str!`'d into Rust and packaged into the wheel — defaults cannot disappear because a file was deleted. If the JSON and a Rust constant disagree, **the JSON is authoritative** (see `docs/engineering/profiles.md`).

