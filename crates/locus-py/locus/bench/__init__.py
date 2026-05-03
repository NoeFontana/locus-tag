"""Phase 0 rotation-tail diagnostic harness — bench-internals only.

Importable only when the wheel was built with `--features bench-internals`.
A clean ImportError below points at the build flag if the symbols are missing.

THROWAWAY: revert this package alongside the bench helpers in `pose.rs` /
`pose_weighted.rs` once the Phase 0 memo lands. The permanent foundation
is the `bench-internals` Cargo feature itself, the noise-floor estimator,
and the `branch_chosen` field on the production `PoseDiagnostics`.
"""

from __future__ import annotations

from .. import locus as _native

_REQUIRED = (
    "_bench_estimate_image_noise",
    "_bench_estimate_both_branches",
    "_bench_refine_pose_lm_weighted_with_telemetry",
    "_bench_refit_pose_drop_corner",
    "_bench_corner_structure_tensor_eigenvalues",
    "_bench_compute_corner_covariance",
    "_bench_estimate_tag_pose",
)
_missing = [name for name in _REQUIRED if not hasattr(_native, name)]
if _missing:
    raise ImportError(
        "locus.bench requires the wheel to be built with the `bench-internals` "
        "Cargo feature. Rebuild with:\n"
        "  uv run maturin develop --release "
        "--manifest-path crates/locus-py/Cargo.toml "
        "--features bench-internals\n"
        f"Missing symbols: {_missing}"
    )

# `bench-internals` symbols are conditionally compiled, so the static type
# checker can't see them on the module — the runtime guard above raises
# ImportError before any of these assignments execute when the feature is off.
estimate_image_noise = _native._bench_estimate_image_noise  # pyright: ignore[reportAttributeAccessIssue]
estimate_both_branches = _native._bench_estimate_both_branches  # pyright: ignore[reportAttributeAccessIssue]
refine_pose_lm_weighted_with_telemetry = _native._bench_refine_pose_lm_weighted_with_telemetry  # pyright: ignore[reportAttributeAccessIssue]
refit_pose_drop_corner = _native._bench_refit_pose_drop_corner  # pyright: ignore[reportAttributeAccessIssue]
corner_structure_tensor_eigenvalues = _native._bench_corner_structure_tensor_eigenvalues  # pyright: ignore[reportAttributeAccessIssue]
compute_corner_covariance = _native._bench_compute_corner_covariance  # pyright: ignore[reportAttributeAccessIssue]
estimate_tag_pose = _native._bench_estimate_tag_pose  # pyright: ignore[reportAttributeAccessIssue]

__all__ = [
    "compute_corner_covariance",
    "corner_structure_tensor_eigenvalues",
    "estimate_both_branches",
    "estimate_image_noise",
    "estimate_tag_pose",
    "refine_pose_lm_weighted_with_telemetry",
    "refit_pose_drop_corner",
]
