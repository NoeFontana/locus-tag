# How-to Guide: Debugging with Rerun

Locus provides a high-fidelity debugging pipeline integrated with the [Rerun SDK](https://rerun.io). This guide explains how to use these tools to pinpoint convergence limits, such as subpixel jitter, reprojection errors, and decoding failures.

## 1. Prerequisites

To use the visual debugging features, ensure you have the `rerun-sdk` installed:

```bash
uv add rerun-sdk
# OR install with development groups
uv sync --group dev --group bench
```

## 2. Using the CLI Visualizer

The `tools/cli.py` script is the primary entry point for visual debugging.

### Basic Visualization
Launch the visualizer on a standard dataset:

```bash
uv run tools/cli.py visualize --scenario forward --limit 5
```

This will initialize a Rerun session and stream diagnostic data for the first 5 images of the "forward" scenario.

### Custom Data Directories
If your datasets are stored outside the default cache (e.g., a local `render-tag` export), use the `--data-dir` flag:

```bash
uv run tools/cli.py visualize \
    --data-dir /path/to/custom/datasets \
    --scenario my_scenario_name
```

1. **On your local machine**: Start the Rerun viewer.
   ```bash
   rerun
   ```
2. **On the remote device**: Run the CLI with the `--rerun-addr` flag.
   ```bash
   uv run tools/cli.py visualize --rerun-addr <LOCAL_IP>:9876
   ```

Alternatively, you can use `--rerun-serve` to host a web-based viewer on the remote device:
```bash
uv run tools/cli.py visualize --rerun-serve
```

## 3. Interpreting Diagnostic Layers

When Rerun is enabled, the following layers are available in the viewport:

### Pipeline Intermediate Stages
- **`pipeline/0_input`**: The raw grayscale image.
- **`pipeline/1_threshold`**: The adaptive threshold map.
- **`pipeline/2_binarized`**: The binarized image used for segmentation.

### Detection & Convergence Layers
- **`pipeline/3_detections` (Green)**: Successfully detected and decoded tags.
- **`pipeline/rejected` (Red/Orange)**:
  - **Red**: Quads rejected due to geometric constraints (area, edge score, etc.).
  - **Orange**: Quads that passed geometric checks but failed decoding (High Hamming distance).
- **`pipeline/detections/subpixel_jitter` (Yellow Arrows)**: Vectors showing the shift from the initial quad corners to the refined subpixel positions. Large or erratic arrows indicate unstable lighting or blur.
- **`pipeline/repro_err/tag_ID` (Scalar Plot)**: Real-time plot of the PnP reprojection RMSE. This helps identify tags with poor geometric fit.

## 4. Programmatic Debugging

You can also enable telemetry in your own Python scripts by passing `debug_telemetry=True` to the `detect()` method.

```python
import locus
import rerun as rr

# Initialize Rerun
rr.init("my_app")
rr.connect() # Connect to local viewer

detector = locus.Detector()
# Telemetry is ONLY computed when debug_telemetry is True
batch = detector.detect(img, debug_telemetry=True)

if batch.telemetry:
    # Access raw telemetry arrays
    bin_img = batch.telemetry.binarized
    jitter = batch.telemetry.subpixel_jitter # Shape: (N, 4, 2)
    
    # Log to Rerun
    rr.log("debug/binarized", rr.Image(bin_img))
```

## 5. Performance Note

The debugging pipeline follows a **Zero Production Overhead** principle:
- Telemetry data is only computed if `debug_telemetry=True` is passed.
- Ephemeral debug data is allocated in the detector's internal `arena`, ensuring zero heap fragmentation.
- When disabled, there is **zero impact** on the detection hot-path.
