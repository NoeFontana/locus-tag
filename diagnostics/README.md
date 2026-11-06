# diagnostics/

Raw artifacts and memos from one-off investigations, kept for
reproducibility and citation from source doc-comments. Prose write-ups
(`MEMO.md`) are agent-readable; bare `.json`/`.parquet` dumps are raw data
only — open them directly if you need the numbers, they carry no inline
narrative.

| Directory | Date | Contents | What it investigated |
| :--- | :--- | :--- | :--- |
| `pool_c_only_seed_2026-05-14/` | 2026-05-14 | `MEMO.md` | Ablation proving Pool C (DLT-homography + IPPE-Square decomposition) is optimal for board seeding. |
| `2026-05-02/` | 2026-05-02 | `corners.parquet`, `failure_modes.json`, `scenes.json` (raw) | Rotation-tail scene classifier run (`rotation_tail_diag/v1`) — per-scene mode (healthy/failure) + rotation error, 50-scene population. |
| `pose_cov_audit_2026-05-03/` | 2026-05-03 | `report.json`, `samples.json` (raw) | Pose-covariance audit on `high_accuracy`/`tag36h11_1920x1080`, 50 samples — global Mahalanobis `d²` stats. |
| `pathb_gt_corners/` | 2026-05-03 | `report.json`, `samples.json` (raw) | Same audit re-run with ground-truth (not detected) corners fed to the pose solver — isolates solver-covariance error from corner-localization error. |
