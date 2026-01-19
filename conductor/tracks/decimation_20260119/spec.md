# Specification: Decimation Strategy for Locus

## Overview
To handle high-resolution video streams (e.g., 1080p, 4K) within the sub-millisecond target latency, Locus will implement a "decimation" strategy. This feature allows the detector to operate on a downsampled version of the input image for the computationally expensive thresholding and segmentation stages, while maintaining high precision by performing final quad refinement on the original high-resolution image.

## Functional Requirements
*   **Configurable Decimation ($):** The `detect` and `detect_tags` functions in both Rust and Python must accept an optional `decimation` parameter (integer  \ge 1$). =1$ represents no decimation.
*   **Simple Subsampling:** For  > 1$, the internal detection pipeline (Thresholding, Segmentation, and Quad Detection) will operate on a decimated image created by taking every hBcth pixel from the source.
*   **High-Resolution Refinement:** Once quad candidates are identified in the decimated image, their corner coordinates must be scaled by $ to serve as the initial guess for the sub-pixel gradient refinement stage, which will operate on the **original** high-resolution image.
*   **Coordinate Space Consistency:** All output corner coordinates and pose estimations must be returned in the coordinate space of the original input image.

## Non-Functional Requirements
*   **Performance:** For =2$, the thresholding and segmentation stages should exhibit a theoretical speedup approaching 4x.
*   **Precision (RMSE):** The Root Mean Square Error (RMSE) of corner localization should remain within a narrow tolerance (e.g., < 10% increase) compared to the non-decimated baseline, thanks to the high-res refinement step.
*   **Memory Efficiency:** The decimation process should avoid creating unnecessary full-image copies. If possible, utilize views or strides to perform subsampling on-the-fly or in a single pass.

## Acceptance Criteria
*   [ ] The `decimation` parameter is correctly exposed in the Python `locus.detect_tags()` and Rust `locus_core::detect()` APIs.
*   [ ] Detection works correctly for =1, 2, 4$ on standard datasets.
*   [ ] Performance benchmarks show a significant latency reduction in the preprocessing and segmentation phases when  > 1$.
*   [ ] Corner localization accuracy (RMSE) on the UMich AprilTag dataset shows no significant degradation between =1$ and =2$.

## Out of Scope
*   Advanced anti-aliasing or box-averaging downsampling methods.
*   Automatic decimation (auto-tuning $ based on latency targets).
