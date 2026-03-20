# SIMD CCL Fusion Benchmarks

**Track:** Segmentation (CCL): Defeating the Memory Wall (~40 ms)
**Date:** March 19, 2026
**Target:** Replace scalar pixel-based Union-Find with a SIMD-accelerated Fused Run-Length Encoding (RLE) and Light-Speed Labeling (LSL) pipeline.

## Results

Executed via Divan benchmarking framework on the `segmentation_bench` suite.

```text
Timer precision: 59 ns
segmentation_bench                                     fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ bench_segmentation_real_icra_threshold_model        8.458 ms      │ 43.41 ms      │ 8.92 ms       │ 9.484 ms      │ 100     │ 100
├─ bench_segmentation_real_icra_threshold_model_4k     12.52 ms      │ 34.27 ms      │ 13.26 ms      │ 13.6 ms       │ 100     │ 100
╰─ bench_segmentation_real_icra_threshold_model_1080p  3.372 ms      │ 3.971 ms      │ 3.539 ms      │ 3.55 ms       │ 100     │ 100
```

## Analysis

- **1080p Performance:** Segmentation takes just ~3.5 ms.
- **4K Performance:** Segmentation takes ~13.2 ms.
- **Impact:** The pipeline completely defeated the memory wall. By processing vectors (up to 64 pixels at a time with AVX-512, 32 with AVX2) and extracting 1D segments directly from the threshold pass, the 2D connected components problem was reduced to a 1D run-merging problem. Latency dropped well below the original 40 ms target.
