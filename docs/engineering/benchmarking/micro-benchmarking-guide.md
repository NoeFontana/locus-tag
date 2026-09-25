# Micro-Benchmarking Guide

Detail on Tier 3 (Divan) of the [3-tier tooling stack](../benchmarking.md#the-3-tier-tooling-stack);
a micro-optimization must also survive Tier 2 (Tracy) and Tier 1 (`bench real`)
as described there before it's considered real.

## Tier 3: Mathematical & Memory Isolation (Divan)

Proves the optimization works in isolation (e.g. SIMD bilinear sampling cost)
before touching the pipeline. Located in `crates/locus-core/benches/`, must
run **strictly single-threaded** for pristine IPC/L1 metrics. Use
`BenchDataset` to load real ICRA 2020 gradients or generate realistic SoA
states.

```bash
# Run a specific benchmark suite
cargo bench --bench comprehensive --features "extended-bench bench-internals" -- "bench_thresholding" --threads 1

# Run full resolution sweep and update baselines
cargo bench --bench comprehensive --features "extended-bench bench-internals" -- "bench_thresholding" "bench_segmentation" "bench_quad_extraction" --threads 1 > target/profiling/divan_output.txt
PYTHONPATH=. python3 tools/bench/update_micro_baselines.py target/profiling/divan_output.txt
```

## Technical Details

### Single-Threaded Mandate
Enforced globally in each benchmark's `main()`, to prevent the OS scheduler
or Rayon from thrashing the L1 cache:

```rust
fn main() {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    divan::Divan::from_args().threads([1]).run_benches();
}
```

### Realistic Data Layouts
- **Early stages** (Thresholding/Segmentation): `BenchDataset::icra_forward_0()`
  into `ImageView`.
- **Late stages** (Decoding/Pose Estimation): `BenchDataset::generate_bench_batch()`
  to populate a `DetectionBatch` (SoA) with a realistic candidate mix (e.g.
  50 valid tags, 200 false-positives).
