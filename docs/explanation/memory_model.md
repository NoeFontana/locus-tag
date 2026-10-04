# Memory Model

This document details Locus's memory architecture: the Data-Oriented Design (DOD) philosophy, the Structure of Arrays (SoA) batch layout, zero-copy FFI boundaries, and the arena allocation lifecycle. For the pipeline data flow, see [Pipeline](pipeline.md). For the formal SoA contract, see [DetectionBatch Contract](../engineering/detection-batch-contract.md).

## Design Philosophy

Locus keeps latency low and bounded by treating memory as an explicit resource rather than delegating it to the system allocator. The hot path (`detect()`) executes **zero heap allocations** after initialization. All ephemeral per-frame data lives in a bump arena that is reset in $O(1)$ time at frame boundaries.

## Memory Hierarchy

```mermaid
flowchart LR
    subgraph Python ["Python Heap"]
        PyArr["NumPy Array<br/>(u8 Pixels)"]
    end

    subgraph Interface ["FFI Boundary"]
        View["ImageView<br/>(Ptr + Stride)"]
    end

    subgraph Rust ["Rust Internal Memory"]
        subgraph State ["DetectorState (Persistent)"]
            Arena["Bump Arena<br/>(Reset Per Frame)"]
            Batch["DetectionBatch<br/>(SoA, Pre-allocated)"]
        end

        subgraph Static ["Pooled Buffers (Persistent)"]
            Upscale["Upscale Buffer"]
        end
    end

    PyArr -.->|"Zero-Copy Read<br/>(Buffer Protocol)"| View
    View -->|Process| Upscale
    Upscale -->|Write| Arena
    Arena -->|Store| Batch
```

### Allocation Classes

| Class | Lifetime | Strategy | Examples |
| :--- | :--- | :--- | :--- |
| **Persistent** | Detector lifetime | Pre-allocated at `Detector::new()` | `DetectionBatch` (boxed), upscale buffer |
| **Per-Frame** | Single `detect()` call | Bump arena (`bumpalo::Bump`) | Sharpened / decimated image, threshold map, binarized image, runs, contours, intermediate SoA slices |
| **Stack** | Function scope | Fixed-size arrays, `SmallVec`, `arrayvec` | Homography matrices, sample buffers, ROI caches |

### Forbidden in the Hot Path

The following are **strictly forbidden** inside `detect()` (enforced by code review and CI):

- `Vec::new()`, `Box::new()`, `HashMap::new()` or any implicit heap allocation
- `String` formatting or dynamic dispatch that triggers allocation
- Growing any collection beyond its pre-allocated capacity

## The DetectionBatch (SoA Layout)

The `DetectionBatch` is the central data structure of the pipeline. It replaces discrete `Candidate` or `Quad` objects with a set of parallel arrays indexed by candidate ID. This eliminates pointer chasing and ensures SIMD-friendly memory access patterns.

### Data Layout

```
Index i    0         1         2         ...       N-1
           ┌─────────┬─────────┬─────────┬─────────┐
corners    │ [4×P2f] │ [4×P2f] │ [4×P2f] │   ...   │  32-byte aligned
           ├─────────┼─────────┼─────────┼─────────┤
homographies│ [3×3+pad]│ [3×3+pad]│ [3×3+pad]│   ...   │  64-byte aligned
           ├─────────┼─────────┼─────────┼─────────┤
ids        │  u32    │  u32    │  u32    │   ...   │
           ├─────────┼─────────┼─────────┼─────────┤
payloads   │  u64    │  u64    │  u64    │   ...   │
           ├─────────┼─────────┼─────────┼─────────┤
error_rates│  f32    │  f32    │  f32    │   ...   │
           ├─────────┼─────────┼─────────┼─────────┤
poses      │ Pose6D  │ Pose6D  │ Pose6D  │   ...   │  32-byte aligned
           ├─────────┼─────────┼─────────┼─────────┤
status_mask│  u8     │  u8     │  u8     │   ...   │
           ├─────────┼─────────┼─────────┼─────────┤
funnel_status│ u8    │  u8     │  u8     │   ...   │
           ├─────────┼─────────┼─────────┼─────────┤
corner_covariances│ [16×f32]│ [16×f32]│ ...  │   ...   │  four 2×2 blocks
           ├─────────┼─────────┼─────────┼─────────┤
corner_refined│ u8   │  u8     │  u8     │   ...   │  bit j: corner j from the junction pass
           ├─────────┼─────────┼─────────┼─────────┤
routed_to  │  u8     │  u8     │  u8     │   ...   │  AdaptivePpb route (telemetry)
           ├─────────┼─────────┼─────────┼─────────┤
ppb_estimate│ f32    │  f32    │  f32    │   ...   │  pixels per bit (telemetry)
           └─────────┴─────────┴─────────┴─────────┘
```

`bench-internals` builds add Phase D diagnostic columns (`outlier_corner_idx`, `pose_consistency_d2`, `pose_consistency_d2_max_corner`, `ippe_branch_d2_ratio`). The batch is far larger than a thread stack, so it is only ever constructed on the heap (`DetectionBatch::new_boxed`) and owned through a `Box` (see [Constraints §1](../engineering/constraints.md)).

### The Identity Rule

The identity of a fiducial marker is defined by its **index**. If a quad exists at index 7, then `corners[7]`, `homographies[7]`, `ids[7]`, and `poses[7]` are guaranteed to belong to the same physical marker. There is no separate `Candidate` struct in the hot path.

### Capacity & Alignment

- **Fixed Pre-Allocation:** `MAX_CANDIDATES = 1024`. No runtime growth.
- **SIMD Alignment:** `corners` and `homographies` arrays are aligned to 32-byte boundaries for penalty-free AVX2 loads.
- **Cache Line Alignment:** `homographies` entries are padded to 64 bytes to prevent false sharing during parallel computation.

## Arena Lifecycle

The `bumpalo::Bump` arena provides $O(1)$ reset semantics. At the start of each frame, a single pointer reset frees all prior allocations without calling destructors or returning memory to the OS.

```mermaid
sequenceDiagram
    participant Frame as detect() Call
    participant Arena as Bump Arena
    participant Allocs as Ephemeral Data

    Frame->>Arena: arena.reset()
    Note over Arena: All prior allocations freed (O(1))

    Frame->>Arena: alloc(sharpened_image, threshold_map)
    Frame->>Arena: alloc(runs, contours)
    Arena->>Allocs: Pointer bumps only

    Note over Frame: Pipeline runs...
    Frame->>Frame: Return via DetectionBatch
```

### Why Not Standard Allocation?

| Approach | Cost Per Frame | Fragmentation | Cache Behavior |
| :--- | :--- | :--- | :--- |
| `malloc`/`free` per object | $O(K)$ syscalls | Unbounded | Cold, scattered |
| Arena (bump) | $O(1)$ reset | Zero | Hot, sequential |

For a 50-tag frame producing ~200 intermediate allocations, the arena saves ~200 allocator round-trips per frame.

## Zero-Copy FFI Boundary

### Input Path (Python to Rust)

NumPy arrays are accessed via the Python Buffer Protocol (`PyReadonlyArray2<u8>`). The Rust side receives a raw pointer and stride, avoiding any copy of the pixel data.

**Validation (performed once at the FFI boundary):**

1. Array must be 2D with `dtype=uint8`.
2. `stride_x` must equal 1 (contiguous rows). Non-contiguous arrays raise `ValueError`.
3. SIMD kernels require **3 bytes of end-padding** for safe 32-bit gather operations.

### Output Path (Rust to Python)

At the end of `detect()`, a single reassembly loop iterates over valid indices `[0..V]` and reads horizontally across the SoA arrays to construct Python `Detection` dataclass instances. The `DetectionBatch` Python wrapper exposes zero-copy NumPy views of the internal SoA columns for vectorized downstream processing.

```mermaid
flowchart TD
    subgraph Rust ["Rust (SoA)"]
        Corners["corners[0..V]"]
        IDs["ids[0..V]"]
        Poses["poses[0..V]"]
    end

    subgraph FFI ["PyO3 Reassembly"]
        Loop["for i in 0..V"]
    end

    subgraph Python ["Python (AoS + Views)"]
        Dets["List[Detection]"]
        Batch["DetectionBatch<br/>(NumPy views)"]
    end

    Corners --> Loop
    IDs --> Loop
    Poses --> Loop
    Loop --> Dets
    Loop --> Batch
```

## Phase-Isolated Write Privileges

To enable lock-free parallelization, each pipeline phase has strict read/write privileges over the SoA columns. See the full contract in [DetectionBatch Contract](../engineering/detection-batch-contract.md).

Phase labels are stable identifiers, not the execution order, which is A → B.5 → B → (B.7 → B) → C → partition → D.

| Phase | Reads | Writes |
| :--- | :--- | :--- |
| **A: Contour Extraction** | Image | `corners`, `status_mask`, `corner_covariances`, `corner_refined` (cleared) |
| **B.5: Funnel** | Image, `corners` | `status_mask`, `funnel_status` |
| **B: Homography** | `corners`, `status_mask` | `homographies` |
| **C: Decoding** | Image, `corners`, `homographies` | `ids`, `payloads`, `error_rates`, `status_mask`, `corners`¹, `corner_refined`, `homographies`² |
| **D: Pose** | `corners`, `corner_covariances` (on the partitioned `[0..V]`) | `poses` (+ `bench-internals` diagnostic columns) |

¹ Phase C writes `corners` only for the rotation permutation (the four slots of one index are cyclically re-labelled; the identity invariant holds), the decode-first / ERF refinement of a decoded candidate, and the `decoder.corner_subpix` corner stage.
² Whenever Phase C writes corners it recomputes `homographies[i]`, so the (corners, H) pair stays consistent for downstream consumers such as `CharucoRefiner`.

The full contract is in [DetectionBatch Contract §4](../engineering/detection-batch-contract.md); it is enforced by `crates/locus-core/tests/contract_detection_batch.rs`.

This isolation guarantees that phases B and C can be parallelized via `rayon` without synchronization.

## Hybrid ROI Caching

During decoding, each tag candidate's image region is copied into a contiguous buffer before sampling to maximize L1 cache utilization:

- **Small tags** (ROI fits in ~4 KB): Stack-allocated buffer (`[u8; N]`).
- **Large tags** (ROI exceeds stack budget): Arena-allocated buffer.

This ensures that the SIMD bilinear interpolation kernel operates on sequential memory regardless of the original image stride.
