# Convenience targets for Locus development workflows.
# Always opt-in: nothing here runs unless invoked explicitly.

.PHONY: help diagnostics-rotation-tail-phase0

help:
	@echo "Available targets:"
	@echo "  diagnostics-rotation-tail-phase0   Run the Phase 0 rotation-tail harness end-to-end."

# Phase 0 rotation-tail diagnostic harness. Requires:
#   uv run maturin develop --release \
#     --manifest-path crates/locus-py/Cargo.toml --features bench-internals
# (the bench harness imports the bench-internals build of locus-py).
#
# Outputs:
#   diagnostics/<YYYY-MM-DD>/scenes.json
#   diagnostics/<YYYY-MM-DD>/corners.parquet
#   diagnostics/<YYYY-MM-DD>/failure_modes.json
#   diagnostics/<YYYY-MM-DD>/recordings/scene_NNNN.rrd
#   docs/engineering/rotation_tail_diagnostic_phase0_<YYYYMMDD>.md
diagnostics-rotation-tail-phase0:
	PYTHONPATH=. uv run --group bench tools/cli.py bench rotation-tail-diag \
		--hub-config locus_v1_tag36h11_1920x1080 \
		--profile render_tag_hub \
		--pose-mode Accurate
