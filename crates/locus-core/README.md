# locus-core

**locus-core** is the high-performance engine powering the [locus-tag](https://github.com/NoeFontana/locus-tag) fiducial marker detection system.

Designed for robotics and autonomous systems, it provides a low-latency pipeline for detecting and decoding fiducial markers (tags) in real-time video streams.

## Key Features

*   **High Performance:** Optimized with SIMD (SSE/AVX/NEON) for rapid image processing.
*   **Architecture Support:** Verified for x86_64 and aarch64.
*   **Modular Design:** Clean separation between thresholding, segmentation, and quad extraction.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
locus-core = "0.2.5"
```

## Cargo features

| Feature | Default | Description |
| :--- | :---: | :--- |
| `non_rectified` | off | Enables the `BrownConradyModel` and `KannalaBrandtModel` camera types and the corresponding `DistortionCoeffs` variants. Required for detecting markers in imagery with significant lens distortion. |
| `tracy` | off | Enables the [Tracy](https://github.com/wolfpld/tracy) profiler backend via the `tracing-tracy` bridge. |
| `rerun` | off | Enables [Rerun](https://rerun.io) SDK logging of intermediate pipeline artifacts for visual debugging. |
| `bench-internals` | off | Exposes internal modules (`bench_api`) for benchmarking and integration testing. Do not enable in production. |

Opt in with:

```toml
[dependencies]
locus-core = { version = "0.3", features = ["non_rectified"] }
```

### Basic Example

```rust
use locus_core::{Detector, Config, Image};

fn main() {
    let config = Config::default();
    let detector = Detector::new(config);
    
    // Load your grayscale image data
    let image = Image::from_raw(width, height, pixels);
    
    let detections = detector.detect(&image);
    for detection in detections {
        println!("Detected tag {} at {:?}", detection.id, detection.corners);
    }
}
```

## Related Projects

*   [locus-tag](https://github.com/NoeFontana/locus-tag): The main repository containing Python bindings and CLI tools.

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/NoeFontana/locus-tag/blob/main/LICENSE-APACHE))
* MIT license ([LICENSE-MIT](https://github.com/NoeFontana/locus-tag/blob/main/LICENSE-MIT))

at your option.
