# Locus (`locus-vision`) Product Guide

## Vision
Locus is a memory-safe, high-performance fiducial marker detector (AprilTag & ArUco) designed to provide ultra-low latency, high-precision pose estimation, and exceptional robustness. It bridges the gap between high-speed Rust execution and seamless Python integration for the robotics ecosystem.

## Target Audience
- **Robotics & AV Engineers:** Building SLAM, autonomous navigation, and high-frequency control loops.
- **Academic Researchers:** Requiring precise tag detection and ground truth for lab setups and experiments.

## Primary Goal
To deliver the industry's most reliable sub-pixel 6DoF pose estimation with minimal processing latency, ensuring stable tracking even under challenging conditions (e.g., occlusion, motion blur, varying lighting).

## Key Features & Differentiators
1.  **Strict Zero-Copy Python API:** Effortless integration with the scientific Python ecosystem (`pyo3` + `numpy`) without the overhead of memory copying.
2.  **Runtime SIMD Dispatch:** Automatically compiled and optimized code paths for AVX2, AVX-512, and NEON architectures to maximize host CPU performance.
3.  **Visual Debugging (Rerun SDK):** Rich, interactive debugging capabilities allowing developers to visualize intermediate processing steps (threshold images, quad candidates, pose axes) for rapid iteration and troubleshooting.