---
description: Architecture overview and data patterns.
---

# Architecture & Patterns

## 1. Data-Oriented Design
* **Data Layer:** Zero-copy `PyReadonlyArray2<u8>`. Validate strides before unsafe SIMD.
* **Preprocessing:** Tile-based adaptive thresholding with `multiversion` kernels.
* **Segmentation:** Flat array Union-Find (Cache Locality).
* **Quad Fitting:** Gradient-based refinement using stack-allocated `nalgebra::SMatrix`.

## 2. System Overview

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

## 3. Key Invariants
* **Zero Alloc in Hot Loop:** Use `bumpalo::Bump` arenas.
* **No Branching:** Use masks/CMOVs in pixel loops.

## 4. Memory Architecture

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

## 5. Packaging & Distribution

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
