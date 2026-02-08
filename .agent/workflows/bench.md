---
description: Run comprehensive performance benchmarks (Python & Rust).
---

1. Run Rust micro-benchmarks (criterion)
```bash
cargo bench --workspace
```

2. Run Python end-to-end benchmark on ICRA 2020 dataset
```bash
uv run python -m scripts.bench.run real --compare
```

3. Run Rust regression suite (Full evaluation)
```bash
cargo test --release --test regression_icra2020 -- --test-threads=1
```
