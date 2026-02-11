# System Architecture

This document provides a high-level overview of the Locus system architecture, designed for high-performance fiducial marker detection.

## High-Level Overview

Locus is built as a hybrid Rust/Python system. The core logic resides in a high-performance Rust crate (`locus-core`), which is exposed to Python via `pyo3` bindings (`locus-py`).

```mermaid
flowchart TD
    User[User / Application] -->|Images| PyBindings["Python Bindings<br/>(locus-py)"]
    PyBindings -->|PyReadonlyArray2| RustCore["Rust Core<br/>(locus-core)"]
    
    subgraph RustCore
        Pipeline["Detection Pipeline"]
        Memory["Arena Memory<br/>(Bumpalo)"]
        SIMD["SIMD Kernels<br/>(Multiversion)"]
    end
    
    RustCore -->|Detections| PyBindings
    PyBindings -->|"List[Detection]"| User
```

## Detection Pipeline

The detection pipeline follows a Data-Oriented Design (DOD) approach to minimize cache misses and allocation overhead. The entire hot path runs without heap allocations, using a pre-allocated arena that is reset per frame.

```mermaid
sequenceDiagram
    participant App as Application
    participant Det as Detector
    participant Thresh as ThresholdEngine
    participant Seg as Segmentation
    participant Quad as QuadExtraction
    participant Decode as Decoder

    App->>Det: detect(image)
    activate Det
    
    Note over Det: 0. Pre-allocation & Upscaling
    Det->>Det: Arena Reset
    
    Note over Det: 1. Preprocessing
    Det->>Thresh: compute_integral_image()
    Det->>Thresh: adaptive_threshold()
    Thresh-->>Det: Binarized Image
    
    Note over Det: 2. Segmentation
    Det->>Seg: label_components()
    Note right of Seg: Union-Find (Flat Array)
    Seg-->>Det: Component Labels
    
    Note over Det: 3. Quad Extraction
    loop For each component
        Det->>Quad: extract_quad()
        Quad->>Quad: Contour Tracing
        Quad->>Quad: Polygon Approx (Douglas-Peucker)
        Quad->>Quad: Sub-pixel Refinement (Gradient)
    end
    Quad-->>Det: Quad Candidates
    
    Note over Det: 4. Decoding
    loop For each candidate
        Det->>Decode: Homography Sampling
        Note right of Decode: Strategy: Hard (Bit) vs Soft (LLR)
        Decode->>Decode: Bit/LLR Extraction (Bilinear)
        Decode->>Decode: Error Correction (Hamming/Soft-ML)
    end
    
    Note over Det: 5. Pose Estimation (Optional)
    opt If Intrinsics Provided
        Det->>Pose: estimate_tag_pose()
        alt Mode = Fast
            Pose->>Pose: IPPE + LM (Geometric Error)
        else Mode = Accurate
            Pose->>Pose: Structure Tensor (Corner Uncertainty)
            Pose->>Pose: Weighted LM (Mahalanobis Distance)
        end
    end

    Det-->>App: Final Detections
    deactivate Det
```

## Component Diagram

The system is structured around the `Detector` struct, which manages configuration and state.

```mermaid
classDiagram
    class Detector {
        -DetectorConfig config
        -Bump arena
        -Vec~Box~TagDecoder~~ decoders
        +detect(image) Vec~Detection~
    }
    
    class DetectorConfig {
        +usize threshold_tile_size
        +bool enable_bilateral
        +f64 quad_min_edge_score
        +...
    }
    
    class TagDecoder {
        <<interface>>
        +decode(bits) Option~id, hamming~
        +sample_points()
        +rotated_codes()
    }

    class DecodingStrategy {
        <<interface>>
        +Code from_intensities(intensities, thresholds)
        +u32 distance(code, target)
        +decode(code, decoder)
    }

    class HardStrategy {
        +Code = u64
    }

    class SoftStrategy {
        +Code = SoftCode (stack-allocated)
    }
    
    class AprilTag36h11 {
        +decode()
    }
    
    class ArUco4x4 {
        +decode()
    }
    
    class Detection {
        +u32 id
        +Point center
        +Point[4] corners
        +Pose pose
    }

    Detector *-- DetectorConfig
    Detector o-- TagDecoder
    TagDecoder <|-- AprilTag36h11
    TagDecoder <|-- ArUco4x4
    Detector ..> Detection : Produces
```

## Design Principles

1.  **Zero-Copy Integration**: Utilizes the Python Buffer Protocol to access NumPy arrays directly, avoiding pixel data duplication.
2.  **Arena Memory**: Per-frame scratchpad (`bumpalo`) eliminates `malloc`/`free` overhead in the hot path.
3.  **Cache Locality**: Algorithms (thresholding, CCL) process data in linear, cache-friendly passes.
4.  **Runtime SIMD Dispatch**: Uses `multiversion` to target AVX2, AVX-512, or NEON based on host CPU capabilities.
5.  **Hybrid Parallelism**: Scales via `rayon` for data-parallel tasks while maintaining sequential cache-coherence for state-heavy stages.

## Memory Architecture

Locus minimizes latency through explicit memory management, almost entirely avoiding the system allocator during detection.

```mermaid
flowchart LR
    subgraph Python ["Python Heap"]
        PyArr["NumPy Array<br/>(u8 Pixels)"]
    end
    
    subgraph Interface ["FFI Boundary"]
        View["ImageView<br/>(Ptr + Stride)"]
    end
    
    subgraph Rust ["Rust Internal Memory"]
        Arena["Bump Arena<br/>(Reset Per Frame)"]
        
        subgraph Static ["Pooled Buffers"]
            Upscale["Upscale Buffer"]
        end
        
        subgraph Ephemeral ["Arena Allocated"]
            Contours
            QuadCandidates
            Homographies
        end
    end

    PyArr -.->|Zero-Copy Read| View
    View -->|Process| Upscale
    Upscale -->|Write| Arena
    Arena -->|Store| Contours
    Arena -->|Store| QuadCandidates
    QuadCandidates -->|Refine| Homographies
```

### Arena Lifecycle

The `Bump` arena is reset at the start of `detect()`, freeing all ephemeral data in $O(1)$ time.

```mermaid
sequenceDiagram
    participant Frame as detect() Call
    participant Arena as Bump Arena
    participant Allocs as Ephemeral Data

    Frame->>Arena: arena.reset()
    Note over Arena: All prior allocations freed (O(1))
    
    Frame->>Arena: alloc(binarized_image)
    Frame->>Arena: alloc(integral_image)
    Frame->>Arena: alloc(contours)
    Arena->>Allocs: Pointer bumps only
    
    Note over Frame: Pipeline runs...
    Frame->>Frame: Return Vec<Detection>
```

## Observability & Debugging

Locus includes built-in instrumentation for performance profiling and visual debugging.

1.  **Tracing**: Uses the `tracing` crate to emit spans for every pipeline stage, allowing integration with `tracy` or `perfetto`.
2.  **Visual Debugging (Rerun)**: When enabled, Locus logs intermediate processing artifacts (threshold images, candidate quads, geometric fits) to the Rerun SDK for real-time inspection.

## Performance Characteristics

Targets a **low latency** budget for 1080p frames on modern CPUs.

| Stage | Complexity | Latency | Notes |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | $O(N)$ | ~7.5 ms | Bandwidth-bound; SIMD-accelerated. |
| **Segmentation** | $O(N)$ | ~2.7 ms | Single-pass Union-Find. |
| **Quad Extraction** | $O(K \cdot M)$ | ~1.5 ms | $K$ components, $M$ perimeter pixels. |
| **Decoding (Hard)** | $O(Q)$ | ~0.5 ms | $Q$ candidates, bit-LUT based. |
| **Decoding (Soft)** | $O(Q \cdot D)$ | ~50 ms | $D$ dictionary entries, LLR search. |

*Note: Latencies are approximate for a single core on a modern CPU (e.g., Zen 4).*

## Decoding Strategies

The `DecodingStrategy` trait enables static dispatch between throughput-optimized and recall-optimized paths.

| Mode | Mechanism | Strength | Cost |
| :--- | :--- | :--- | :--- |
| **Hard-Decision** | Direct intensity thresholding. | Highest throughput; $O(1)$ lookup. | Requires stable SNR/contrast. |
| **Soft-Decision** | ML search using LLRs. | Recovers tags with blur or noise. | Latency scales with dictionary size. |

### Hard-Decision (High Throughput)
The default mode. It samples pixel intensities at grid points and compares them against the local adaptive threshold. This path is extremely fast and ideal for industrial applications with consistent lighting.

### Soft-Decision (Maximum Recall)
Designed for challenging conditions. Instead of binarizing, it computes the **Log-Likelihood Ratio (LLR)** for each bit and performs a Maximum Likelihood search across the dictionary. The implementation is zero-allocation and uses early-exit pruning to minimize search overhead.

## Pose Estimation Strategies

Locus provides two algorithms for 6-DOF recovery, allowing users to prioritize either geometric speed or probabilistic precision.

### Fast Mode: IPPE
*   **Target**: High-speed tracking and mobile robotics.
*   **Method**: Uses the **Infinitesimal Plane-Based Pose Estimation** algorithm for an analytic solution.
*   **Refinement**: Levenberg-Marquardt (LM) on geometric reprojection error.
*   **Latency**: ~50µs per tag.

### Accurate Mode: Probabilistic
*   **Target**: Metrology, calibration, and long-range precision landing.
*   **Method**: Estimates sub-pixel corner uncertainty via the **Structure Tensor** ($J^T J$).
*   **Refinement**: **Anisotropic Weighted LM** minimizing the Mahalanobis distance.
*   **Output**: Provides a full $6 \times 6$ `pose_covariance` matrix.
*   **Latency**: ~200µs per tag.

## Extensibility

Locus is designed to support new fiducial marker systems without modifying the core pipeline.

### Adding a New Tag Family

The `TagDecoder` trait serves as the extension point. To add a new family (e.g., `STag` or a custom ArUco dictionary):

1.  **Implement `TagDecoder`**: Define the grid dimension and bit extraction logic.
2.  **Define `TagDictionary`**: Provide the hamming distance lookup table.
3.  **Register**: Pass the new decoder to `Detector::add_decoder`.

```rust
struct MyCustomDecoder;

impl TagDecoder for MyCustomDecoder {
    fn name(&self) -> &str { "CustomTags" }
    fn dimension(&self) -> usize { 4 } // 4x4 grid
    
    // ... implementation ...
}

// Usage
let mut detector = Detector::new();
detector.add_decoder(Box::new(MyCustomDecoder));
```

## Packaging & Distribution

Locus uses `maturin` to bridge the Rust and Python worlds, creating a native Python extension module.

```mermaid
flowchart LR
    subgraph Build ["Build Process"]
        RustSrc["Rust Source<br/>(locus-core)"] -->|Cargo| CoreLib["Static Lib"]
        CoreLib -->|Maturin| PyMod["Python Module<br/>(locus.abi3.so)"]
        PyStub["Type Stubs<br/>(.pyi)"] -->|Maturin| Wheel
    end
    
    subgraph Dist ["Distribution"]
        PyMod --> Wheel[".whl File"]
        Wheel -->|pip install| Env["User Environment"]
    end
```

## Source Code Organization

The `locus-core` crate is organized into logical modules mirroring the pipeline stages.

| Module | Description | Key Structs |
| :--- | :--- | :--- |
| `image` | Zero-copy image views and pixel access. | `ImageView` |
| `threshold` | Adaptive thresholding and integral images. | `ThresholdEngine` |
| `segmentation` | Connected components labeling. | `UnionFind` |
| `quad` | Contour tracing and quad fitting. | `extract_quads` |
| `decoder` | Bit extraction and hamming decoding. | `TagDecoder`, `Homography` |
| `pose` | 3D pose estimation (PnP). | `Pose`, `CameraIntrinsics` |
| `pose_weighted` | Structure Tensor & Weighted LM. | `refine_pose_lm_weighted` |
| `gradient` | Image gradients & Sub-pixel windows. | `compute_structure_tensor` |
| `filter` | Pre-processing filters (Bilateral, Sharpen). | `bilateral_filter` |

