# Technology Stack: Locus

## 1. Core Systems
*   **Rust (2024 Edition):** The primary language for the high-performance detection engine, chosen for its memory safety, zero-cost abstractions, and robust SIMD support.
*   **Python 3.10+:** The integration language for user-facing APIs, providing an accessible interface for robotics and computer vision pipelines.

## 2. Frameworks & Bindings
*   **PyO3:** High-performance Rust bindings for Python, enabling seamless integration between the two languages.
*   **Maturin:** Build system for Python packages with Rust extensions, ensuring efficient compilation and distribution.
*   **NumPy:** Strict zero-copy integration via the Buffer Protocol, ensuring images and detection results are passed between Rust and Python without memory overhead.

## 3. Performance & Mathematics
*   **nalgebra:** Used for all geometric algebra and linear transformations. Specifically utilizes `SMatrix` and `SVector` for fixed-size, stack-allocated calculations.
*   **multiversion:** Provides runtime SIMD dispatch, allowing the binary to automatically select the most efficient code paths for the host CPU (AVX2, AVX-512, NEON).
*   **portable-simd:** Explicit vectorization for pixel-processing kernels.
*   **bumpalo:** Arena-based memory allocator used to eliminate heap allocations in the detection hot loop.

## 4. Observability & Tooling
*   **rerun:** Native SDK for high-performance visual debugging, used to log intermediate pipeline states like thresholds and quad candidates.
*   **tracing + tracing-tracy:** Low-overhead instrumentation for performance profiling and bottleneck identification.
*   **uv:** Fast Python package and project manager for dependency resolution and environment synchronization.

## 5. Development & Testing
*   **cargo-nextest:** Parallel test runner for fast Rust integration and unit testing.
*   **pytest:** Standard Python testing framework for API validation.
*   **ruff:** High-speed Python linter and formatter.
