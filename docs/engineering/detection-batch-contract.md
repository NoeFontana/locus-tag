# Architectural Contract: DetectionBatch (SoA)

## 1. The Core Invariant (The Identity Rule)
The concept of a discrete Candidate or Quad object is officially deprecated in the Rust hot-path.
The identity of a fiducial marker is now strictly defined by its Index (i).
If a quad exists at index 7, then corners[7], homographies[7], and poses[7] are guaranteed to belong to that exact same physical marker.

## 2. Memory & Capacity Constraints
To guarantee zero allocations and prevent L1 cache fragmentation, the batch arena must obey the following physical constraints:
*   **Fixed Pre-Allocation**: The DetectionBatch must be initialized with a rigid maximum capacity (e.g., `MAX_CANDIDATES = 256`).
*   **SIMD Alignment**: The underlying arrays (especially homographies and corners) must be explicitly aligned in memory to 32-byte boundaries to support unaligned-penalty-free AVX2 vector loads.
*   **Zero-Heap Hot Loop**: Once the DetectionBatch is instantiated during the detector's setup phase, calling `detect()` may not trigger a single call to the OS allocator. The arena is strictly reset (cursor moved to 0) at the start of each frame.

## 3. The Data Layout (The Columns)
The DetectionBatch struct encapsulates the following parallel arrays (slices):
*   `corners`: `[Point2f; MAX_CANDIDATES * 4]` (Flattened array of sub-pixel quad vertices).
*   `homographies`: `[Matrix3x3; MAX_CANDIDATES]` (The $3	imes3$ projection matrices).
*   `payloads`: `[u64; MAX_CANDIDATES]` (The extracted bitstrings).
*   `error_rates`: `[f32; MAX_CANDIDATES]` (The MSE or Log-Likelihood Ratio confidence scores).
*   `poses`: `[Pose6D; MAX_CANDIDATES]` (Translation vectors and unit quaternions).
*   `status_mask`: `[CandidateState; MAX_CANDIDATES]` (A dense byte-array tracking the lifecycle. e.g., Empty, Active, FailedDecode, Valid).

## 4. Phase-Isolated Execution Privileges
To prevent data contention and enable lock-free parallelization (Rayon), engineers must adhere to strict Read/Write privileges for each phase of the pipeline.

### Phase A: Contour Extraction
*   **Privileges**: Write-Only to `corners` and `status_mask`.
*   **Contract**: The extractor sequentially writes quad vertices into memory. It returns a single integer N representing the total active candidates found in the frame.

### Phase B: Homography Computation
*   **Privileges**: Read-Only on `corners[0..N]`, Write-Only on `homographies[0..N]`.
*   **Contract**: A purely mathematical loop. It calculates the perspective warps. Because it has no side effects, it can be trivially parallelized using `corners[0..N].par_iter()`.

### Phase C: Batched Sampling & Decoding
*   **Privileges**: Read-Only on the Image Tensor and `homographies[0..N]`. Write-Only to `payloads[0..N]`, `error_rates[0..N]`, and `status_mask[0..N]`.
*   **Contract**: This phase executes the SIMD bilinear interpolation. If a candidate fails the Hamming distance check, its `status_mask` at index i is flipped to `FailedDecode`.

### Phase D: Pose Refinement
*   **Privileges**: Read-Only on `corners` and `status_mask`. Write-Only on `poses`.
*   **Contract**: Before this phase runs, the arrays must be Partitioned. Candidates marked Valid are swapped to the front of the arrays `[0..V]`. The heavy Anisotropic Levenberg-Marquardt solver then strictly iterates over `[0..V]`, calculating 6D poses only for mathematically verified markers.

## 5. The FFI Boundary (Object Reassembly)
The Python environment and downstream consumers expect discrete objects. The SoA architecture must not leak across the FFI boundary.
*   **The Reassembly Step**: Inside the PyO3 wrapper, at the very end of the `detect()` call, a single loop iterates over the Valid indices `[0..V]`. It reads horizontally across the parallel arrays at index i, constructs the Detection Python dataclasses, and returns them.
