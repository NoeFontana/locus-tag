# Locus Release Performance Report (2026-03-22)

This report documents the performance of Locus v0.2.6 prior to release, covering end-to-end latency, mathematical kernel efficiency, and regression accuracy.

## Tier 1: End-to-End Performance (Python CLI)

Validated on the **ICRA 2020** dataset and synthetic targets.

### Real-World (ICRA 2020 - Scenario: forward)
Measured on all 50 images of the `forward/pure_tags_images` subset.

| Configuration | Recall | RMSE (px) | Latency (ms) |
|---------------|--------|-----------|--------------|
| **Locus (Soft)** | 96.23% | 0.2870 | 44.95 |
| **Locus (Hard)** | 76.87% | 0.2567 | 33.97 |

### Synthetic Throughput
Measured on procedurally generated images (1080p).

| Tag Count | Latency (ms) | Recall |
|-----------|--------------|--------|
| 1 Tag     | 8.75         | 100.0% |
| 10 Tags    | 5.45         | 100.0% |
| 50 Tags    | 8.68         | 100.0% |
| 100 Tags   | 11.76        | 100.0% |

---

## Tier 2: Micro-Benchmarking (Divan)

Fine-grained mathematical kernels measured strictly single-threaded on 1080p images.

### Mathematical Kernels
| Kernel | Median Latency |
|--------|----------------|
| **Thresholding (Apply)** | 2.74 ms |
| **Integral Image** | 1.99 ms |
| **Segmentation (1080p)** | 2.47 ms |

### Decoding & Pose
| Operation | Latency |
|-----------|---------|
| **Dictionary Lookup (36h11)** | 372.8 ns |
| **Soft Decoding (200 cands)** | 161.7 µs |
| **Pose Estimation (50 tags)** | 77.8 µs |

---

## Tier 3: Rust Regression Suite

### ICRA 2020 (SOTA Metrology)
| Dataset | Recall | RMSE (px) | Latency (ms) |
|---------|--------|-----------|--------------|
| **Forward (Pure Tags)** | 96.23% | 0.3152 | 41.53 |
| **Forward (Checkerboard)** | 91.43% | 0.4577 | 57.89 |
| **Circle (Pure Tags)** | 83.27% | 0.3931 | 56.53 |
| **Circle (Checkerboard)** | 76.01% | 0.4171 | 56.18 |

### Hub Regression (Rendered Tags)
Measured on the Hugging Face Hub benchmarking datasets.

| Dataset (36h11) | Mode | Recall | RMSE (px) | Latency (ms) |
|-----------------|------|--------|-----------|--------------|
| **640x480** | SOTA | 100.0% | 0.2112 | 10.12 |
| **720p** | SOTA | 100.0% | 0.2456 | 21.45 |
| **1080p** | SOTA | 100.0% | 0.2767 | 45.32 |
| **4K (2160p)** | SOTA | 100.0% | 0.3124 | 168.4 |
| **1080p** | Fast | 100.0% | 0.3542 | 12.15 |

---

## Technical Updates in this Version
- **Unified Config**: Python CLI now correctly respects all Rust `DetectorConfig` fields, including `quad_max_elongation` and `quad_min_density`.
- **Soft Decoding Parity**: Verified that Python `DecodeMode.Soft` matches Rust SOTA results (96.23% recall on ICRA forward).
- **Benchmarking Consistency**: Updated `LocusWrapper` to pass all high-fidelity parameters to the underlying engine.
