"""Phase 0 rotation-tail failure-mode diagnostic harness.

Three-stage pipeline driven by the `bench rotation-tail-diag` typer command:

    extract.py  → scenes.json + corners.parquet + recordings/*.rrd
    classify.py → failure_modes.json
    report.py   → docs/engineering/rotation_tail_diagnostic_phase0_<DATE>.md

The harness is built against the `bench-internals` Cargo feature in
`locus-py`. The Rust `bench_*` helpers it consumes are explicitly throwaway
and revert before the Phase 0 PR merges; the harness scripts here may live
on as a regression gate for subsequent rotation-tail work.
"""
