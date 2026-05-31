# Architectural Contract: DetectionBatch (SoA)

## 1. The Core Invariant (The Identity Rule)
The concept of a discrete Candidate or Quad object is officially deprecated in the Rust hot-path.
The identity of a fiducial marker is now strictly defined by its Index (i).
If a quad exists at index 7, then corners[7], homographies[7], ids[7], and poses[7] are guaranteed to belong to that exact same physical marker.

## 2. Memory & Capacity Constraints
To guarantee zero allocations and prevent L1 cache fragmentation, the batch arena must obey the following physical constraints:
*   **Fixed Pre-Allocation**: The DetectionBatch must be initialized with a rigid maximum capacity (e.g., `MAX_CANDIDATES = 1024`).
*   **SIMD Alignment**: The underlying arrays (especially homographies and corners) must be explicitly aligned in memory to 32-byte boundaries to support unaligned-penalty-free AVX2 vector loads.
*   **Zero-Heap Hot Loop**: Once the DetectionBatch is instantiated during the detector's setup phase, calling `detect()` may not trigger a single call to the OS allocator. The arena is strictly reset (cursor moved to 0) at the start of each frame.

## 3. The Data Layout (The Columns)
The DetectionBatch struct encapsulates the following parallel arrays (slices):
*   `corners`: `[[Point2f; 4]; MAX_CANDIDATES]` (Sub-pixel quad vertices, 4 per candidate).
*   `homographies`: `[Matrix3x3; MAX_CANDIDATES]` (The $3\times3$ projection matrices, padded to 64 bytes for cache line alignment).
*   `ids`: `[u32; MAX_CANDIDATES]` (The decoded tag IDs).
*   `payloads`: `[u64; MAX_CANDIDATES]` (The extracted bitstrings).
*   `error_rates`: `[f32; MAX_CANDIDATES]` (The Hamming distance or confidence scores).
*   `poses`: `[Pose6D; MAX_CANDIDATES]` (Translation vectors and unit quaternions, padded to 32 bytes).
*   `status_mask`: `[CandidateState; MAX_CANDIDATES]` (A dense byte-array tracking the lifecycle. e.g., Empty, Active, FailedDecode, Valid).
*   `funnel_status`: `[FunnelStatus; MAX_CANDIDATES]` (Detailed status from the fast-path funnel for rejected quads).
*   `corner_covariances`: `[[f32; 16]; MAX_CANDIDATES]` (Four 2×2 per-corner covariance matrices, packed row-major as 16 floats; populated by GWLF refinement and consumed by the weighted pose solver).
*   `corner_empirical_noise`: `[[f32; 4]; MAX_CANDIDATES]` (Phase 4 per-corner empirical noise variance σ_n² in px², drawn from the ERF edge-fit residual MSE; **Phase-D internal input**, inert when `pose.use_empirical_corner_noise = false`. `0.0` sentinel means "no ERF measurement on either adjacent edge" and reduces to today's behaviour. See `docs/engineering/pose_covariance_followup_2026-05-22.md` §4).
*   `outlier_corner_idx`: `[u8; MAX_CANDIDATES]` (Phase D telemetry only, `bench-internals`-gated. Sentinel `u8::MAX` ⇒ no corner was dropped for this candidate. Values `0..=3` identify which corner the outlier-aware LM masked when the 3-corner pose was kept; in that case the stored pose covariance reflects 6 observations instead of 8. Inert when `pose.outlier_drop_d2_threshold = 0.0`).

## 4. Phase-Isolated Execution Privileges
To prevent data contention and enable lock-free parallelization (Rayon), engineers must adhere to strict Read/Write privileges for each phase of the pipeline. These privileges are enforced by the `contract_detection_batch` integration test (see `crates/locus-core/tests/contract_detection_batch.rs`), which seeds every column with sentinel values and asserts that a single phase only mutates its declared write set.

### Phase A: Contour Extraction
*   **Privileges**: Write to `corners`, `status_mask`, `corner_covariances`, and `corner_empirical_noise`.
*   **Contract**: The extractor sequentially writes quad vertices into memory, zeroes the covariance blocks (or fills them with Structure-Tensor estimates when GWLF is enabled), populates `corner_empirical_noise` with per-corner ERF residual MSEs (or the `0.0` sentinel when the route does not run ERF), and marks each populated slot `Active`. It returns a single integer N representing the total active candidates found in the frame.

### Phase B: Homography Computation
*   **Privileges**: Read-Only on `corners[0..N]` and `status_mask[0..N]`; Write-Only on `homographies[0..N]`.
*   **Contract**: A purely mathematical loop. It calculates the perspective warps. Because it has no side effects, it can be trivially parallelized using `corners[0..N].par_iter()`.

### Phase B.5: Fast-Path Funnel
*   **Privileges**: Read-Only on the Image Tensor and `corners[0..N]`. Write-Only to `status_mask[0..N]` and `funnel_status[0..N]`.
*   **Contract**: This phase performs $O(1)$ edge contrast rejection. If a candidate lacks photometric evidence of an edge, its `status_mask` is flipped to `FailedDecode` and `funnel_status` is updated to `RejectedContrast`.

### Phase C: Batched Sampling & Decoding
*   **Privileges**: Read-Only on the Image Tensor. Write to `ids[0..N]`, `payloads[0..N]`, `error_rates[0..N]`, `status_mask[0..N]`, `corners[0..N]` (rotation-permutation + ERF refinement — see below), and `homographies[0..N]` (recomputed iff corners changed — see below).
*   **Contract**: This phase executes the SIMD bilinear interpolation. If a candidate fails the Hamming distance check, its `status_mask` at index i is flipped to `FailedDecode`. On successful decode with non-zero rotation, the four corners are cyclically permuted in place to match the decoded rotation so that downstream consumers see canonical orientation. When soft-decoding / ERF refinement is active, refined sub-pixel corners are also written back. This is the single exception to the "corners is read-only after Phase A" principle: the rotation permutation preserves the identity invariant (index `i` still refers to the same marker) but renames the four corner slots. **Whenever Phase C writes corners** (rotation, ERF refinement, or both), it MUST also recompute and write `homographies[i]` so the (corners, H) pair stays consistent — downstream `CharucoRefiner` projects saddle predictions through `batch.homographies[i]`, and a stale `h_slot` against refined corners is the same hazard documented in `memory/project_refine_saddle_noop.md`.

### Phase D: Pose Refinement
*   **Privileges**: Read-Only on `corners`, `corner_covariances`, and `corner_empirical_noise`. Write-Only on `poses`. Under `--features bench-internals`, Phase D ALSO writes the per-candidate diagnostic SoA columns `outlier_corner_idx`, `pose_consistency_d2`, `pose_consistency_d2_max_corner`, and `ippe_branch_d2_ratio` — these are surfaced for offline benchmark / regression analysis and are inert in default release builds. (Note: `status_mask` is implicitly read via the upstream `partition(v)` invariant which guarantees `[0..v]` is Valid; Phase D itself does not re-check the mask.)
*   **Contract**: Before this phase runs, the arrays must be Partitioned. Candidates marked Valid are swapped to the front of the arrays `[0..V]`. The heavy Anisotropic Levenberg-Marquardt solver then strictly iterates over `[0..V]`, calculating 6D poses only for mathematically verified markers. When `corner_covariances` contains non-zero entries from GWLF, the weighted path consumes them as Fisher-information priors. When `pose.use_empirical_corner_noise = true`, `corner_empirical_noise[i]` inflates the per-corner structure-tensor variance via `max(σ_n², min(empirical, 16·σ_n²))` (Phase 4 — `docs/engineering/pose_covariance_followup_2026-05-22.md` §4). When `pose.outlier_drop_d2_threshold > 0.0`, after the weighted LM converges Phase D checks the worst per-corner d²; if it exceeds the threshold and dominates the second-worst by ≥ 2×, that corner is masked and the LM is re-run on the remaining 3. The 3-corner pose is kept only if its aggregate d² over the **three kept corners** is strictly lower than the 4-corner pose's aggregate d² over those same three corners (self-rejection invariant: dropping must demonstrably improve the fit on the others). The dropped corner is recorded in `outlier_corner_idx[i]`.

## 5. The FFI Boundary (Object Reassembly)
The Python environment and downstream consumers expect discrete objects. The SoA architecture must not leak across the FFI boundary.
*   **The Reassembly Step**: Inside the PyO3 wrapper, at the very end of the `detect()` call, a single loop iterates over the Valid indices `[0..V]`. It reads horizontally across the parallel arrays at index i, constructs the Detection Python dataclasses, and returns them.
