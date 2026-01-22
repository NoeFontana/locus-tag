# Specification: Parallelize RLE Pass in Segmentation

## 1. Overview
The current "Fast" segmentation path in `segmentation.rs` (`label_components_with_stats`) utilizes a purely sequential Run-Length Encoding (RLE) pass to extract candidates from the binary image. On high-resolution images (e.g., 4K), this sequential processing becomes a significant bottleneck, especially as other stages of the pipeline (thresholding, decoding) are already parallelized. This track aims to parallelize the RLE extraction using Rayon to improve segmentation throughput and reduce latency.

## 2. Goals
- **Reduce Segmentation Latency:** Parallelize the RLE extraction pass in `label_components_with_stats`.
- **Maintain Correctness:** Ensure the parallel implementation produces identical connected components and statistics as the sequential version.
- **Efficient Merging:** Implement an efficient parallel-to-serial transition for merging runs collected from different threads.

## 3. Functional Requirements
- **Parallel RLE Extraction:**
    - Refactor the row-wise loop in `label_components_with_stats` to use Rayon's `par_iter` or `par_chunks`.
    - Implement a thread-safe collection mechanism for `Run` objects. Since `BumpVec` is not thread-safe, each thread/task will collect its own subset of runs.
    - Merge the collected runs into the final `BumpVec` or equivalent structure for the subsequent Union-Find pass.
- **Performance:**
    - The overhead of merging parallel results must be significantly less than the time saved by parallel processing.
    - Aim for high cache locality during the RLE pass.

## 4. Verification & Acceptance Criteria
- **Regression Testing:**
    - Run `crates/locus-core/tests/regression_icra2020.rs` to verify that Recall and RMSE remain identical to the baseline.
- **Performance Validation:**
    - Compare `mean_total_ms` and specific segmentation latency in `regression_icra2020.rs` before and after the change.
    - Verify improvement on high-resolution targets (e.g., 4K synthetic or ICRA datasets).
- **Code Standards:**
    - Adhere to the project's "Zero-Allocation Hot Loop" principle by utilizing the `Bump` arena for temporary run collections where possible, or minimizing heap allocations during the merge.

## 5. Out of Scope
- Parallelizing the Union-Find logic itself (Pass 2).
- Optimizing the threshold-model variant of segmentation (already parallelized).
