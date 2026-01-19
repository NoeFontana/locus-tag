# Initial Concept
A high-performance AprilTag and ArUco detector in Rust with Python bindings, targeting sub-millisecond latencies.

## Key Features
- **Sub-millisecond Latency:** Optimized for real-time robotics and AV applications.
- **Zero-Allocation Hot Loop:** Utilizes arena allocation to eliminate heap overhead during detection.
- **Configurable Decimation:** Efficiently handles high-resolution (1080p, 4K) streams by downsampling early stages while maintaining sub-pixel accuracy via high-res refinement.
- **SIMD-Accelerated:** Runtime dispatch for AVX2, NEON, and more.
- **Zero-Copy Python Bindings:** Seamless integration with NumPy.
