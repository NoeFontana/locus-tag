"""Plot modules — narrow functions over a Tier-1-records DataFrame.

Each module is a single ``plot(df, out_path, …)`` entry point so the
:mod:`tools.bench.report` orchestrator can call them uniformly.
"""

from tools.bench.plots._io import load_records_df

__all__ = ["load_records_df"]
