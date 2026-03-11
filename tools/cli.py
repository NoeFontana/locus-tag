import json
import time
from pathlib import Path

import typer

try:
    import rerun as rr

    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False

app = typer.Typer(help="Locus Developer CLI")
bench_app = typer.Typer(help="Locus Unified Benchmarking")
app.add_typer(bench_app, name="bench")


@app.command()
def validate_dicts(
    schema: Path = typer.Option(..., help="Path to JSON schema file"),
    files: list[Path] = typer.Argument(..., help="JSON files to validate"),
):
    """
    Validate dictionary JSON files against a schema.
    """
    import jsonschema

    try:
        with open(schema) as f:
            schema_data = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading schema {schema}: {e}", err=True)
        raise typer.Exit(code=1) from e

    has_errors = False

    for file_path in files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            jsonschema.validate(instance=data, schema=schema_data)

            # Mathematical consistency checks
            if len(data["canonical_sampling_points"]) != data["payload_length"]:
                typer.echo(
                    f"FAILED: {file_path} - length of canonical_sampling_points ({len(data['canonical_sampling_points'])}) does not match payload_length ({data['payload_length']})",
                    err=True,
                )
                has_errors = True
            elif len(data["base_codes"]) != data["dictionary_size"]:
                typer.echo(
                    f"FAILED: {file_path} - length of base_codes ({len(data['base_codes'])}) does not match dictionary_size ({data['dictionary_size']})",
                    err=True,
                )
                has_errors = True
            else:
                typer.echo(f"PASS: {file_path}")
        except jsonschema.exceptions.ValidationError as e:
            typer.echo(f"FAILED: {file_path} - Schema validation error: {e.message}", err=True)
            has_errors = True
        except Exception as e:
            typer.echo(f"FAILED: {file_path} - {e}", err=True)
            has_errors = True

    if has_errors:
        raise typer.Exit(code=1)
    else:
        typer.echo("All dictionaries passed validation.")


@app.command()
def visualize(
    ctx: typer.Context,
    limit: int | None = typer.Option(10, help="Limit number of images"),
    scenario: str = typer.Option("forward", help="Scenario to visualize"),
    tile_size: int = typer.Option(8, help="Threshold tile size"),
    min_area: int = typer.Option(16, help="Min quad area"),
    bilateral: bool = typer.Option(False, help="Enable bilateral filter"),
    upscale: int = typer.Option(1, help="Upscale factor"),
):
    """
    Launch the Rerun-based visualizer for the detection pipeline.
    """
    import cv2
    import locus
    import numpy as np
    from tqdm import tqdm

    from tools.bench.utils import DatasetLoader

    if not RERUN_AVAILABLE:
        typer.echo(
            "Error: Rerun SDK not installed. Run 'uv add rerun-sdk' or install with [bench] group.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Initialize Rerun with extra arguments from context if any
    # Note: Typer doesn't automatically handle unknown args like argparse,
    # but we can pass common ones or just initialize.
    rr.init("locus_debug_pipeline", spawn=True)

    loader = DatasetLoader()
    if not loader.prepare_icra(scenario):
        typer.echo(f"Failed to prepare scenario {scenario}", err=True)
        raise typer.Exit(code=1)

    datasets = loader.find_datasets(scenario, ["tags"])

    # Initialize Detector using new keyword arguments
    detector = locus.Detector(
        threshold_tile_size=tile_size,
        quad_min_area=min_area,
        enable_bilateral=bilateral,
        upscale_factor=upscale,
    )

    for ds_name, img_dir, gt_map in datasets:
        typer.echo(f"\nVisualizing {ds_name}...")

        img_names = sorted(gt_map.keys())
        if limit:
            img_names = img_names[:limit]

        for i, img_name in enumerate(tqdm(img_names)):
            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            rr.set_time_sequence("frame_idx", i)

            # Perform Detection (Vectorized API with debug telemetry)
            batch = detector.detect(img, debug_telemetry=True)

            # 1. Input & Ground Truth
            rr.log("pipeline/0_input", rr.Image(img))

            gt_tags = gt_map.get(img_name, [])
            if gt_tags:
                gt_strips = []
                gt_labels = []
                for gt in gt_tags:
                    c = np.vstack([gt.corners, gt.corners[0]])
                    gt_strips.append(c)
                    gt_labels.append(f"GT:{gt.tag_id}")

                rr.log(
                    "pipeline/0_input/ground_truth",
                    rr.LineStrips2D(gt_strips, colors=[0, 255, 0], radii=2.0, labels=gt_labels),
                )

            # 2. Intermediate Telemetry
            if batch.telemetry is not None:
                rr.log("pipeline/1_threshold", rr.Image(batch.telemetry.threshold_map))
                rr.log("pipeline/2_binarized", rr.Image(batch.telemetry.binarized))

                # Overlay GT and Detections on intermediate stages for alignment check
                if gt_tags:
                    rr.log(
                        "pipeline/2_binarized/ground_truth",
                        rr.LineStrips2D(gt_strips, colors=[0, 255, 0], radii=1.0),
                    )

            # Note: Internal pipeline artifacts (binarized, labels, candidates)
            # are NOT currently exposed in the vectorized high-level API.
            # They will be re-added if/when the Rust side exposes them via DetectorState.

            # 5. Final Detections
            if len(batch) > 0:
                det_strips = []
                det_labels = []
                for j in range(len(batch)):
                    c = batch.corners[j]
                    c = np.vstack([c, c[0]])
                    det_strips.append(c)
                    det_labels.append(f"ID:{batch.ids[j]}")

                rr.log(
                    "pipeline/4_detections",
                    rr.LineStrips2D(det_strips, colors=[255, 50, 50], radii=1.2, labels=det_labels),
                )

                # Also log detections on binarized for alignment check
                if batch.telemetry is not None:
                    rr.log(
                        "pipeline/2_binarized/detections",
                        rr.LineStrips2D(det_strips, colors=[255, 50, 50], radii=0.8),
                    )


# --- Benchmarking Commands ---


@bench_app.command("real")
def bench_real(
    scenarios: list[str] = typer.Option(["forward"], help="Scenarios to run"),
    types: list[str] = typer.Option(["tags"], help="Dataset types (tags, checkerboard)"),
    limit: int | None = typer.Option(None, help="Limit number of images"),
    skip: int = typer.Option(0, help="Skip first N images"),
    compare: bool = typer.Option(False, help="Compare against other libraries"),
    decimation: int = typer.Option(1, help="Image decimation factor"),
    baseline: Path | None = typer.Option(None, help="Path to baseline JSON"),
    save_baseline: Path | None = typer.Option(None, help="Path to save results as baseline"),
    profile: bool = typer.Option(False, help="Enable Tracy profiling"),
    family: str = typer.Option("AprilTag36h11", help="Tag family to detect"),
    refinement: str = typer.Option("Erf", help="Refinement mode (None, Edge, GridFit, Erf)"),
    tile_size: int = typer.Option(8, help="Threshold tile size"),
    constant: int = typer.Option(0, help="Adaptive threshold constant"),
    min_fill: float = typer.Option(0.10, help="Min quad fill ratio"),
    min_range: int = typer.Option(10, help="Threshold min range"),
    max_hamming: int = typer.Option(2, help="Max hamming error"),
    min_edge_score: float = typer.Option(4.0, help="Min edge alignment score"),
):
    """Run benchmarks on real-world datasets (ICRA)."""
    import cv2
    import locus
    import numpy as np
    from tqdm import tqdm

    from tools.bench.utils import (
        AprilTagWrapper,
        DatasetLoader,
        LibraryWrapper,
        LocusWrapper,
        Metrics,
        OpenCVWrapper,
    )

    if profile:
        locus.init_tracy()

    # Map string to locus.TagFamily
    family_mapping = {
        "AprilTag16h5": int(locus.TagFamily.AprilTag16h5),
        "AprilTag36h11": int(locus.TagFamily.AprilTag36h11),
        "ArUco4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "ArUco4x4_100": int(locus.TagFamily.ArUco4x4_100),
        "16h5": int(locus.TagFamily.AprilTag16h5),
        "36h11": int(locus.TagFamily.AprilTag36h11),
        "4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "4x4_100": int(locus.TagFamily.ArUco4x4_100),
    }

    tag_family_int = family_mapping.get(family)
    if tag_family_int is None:
        typer.echo(f"Error: Unknown tag family '{family}'", err=True)
        raise typer.Exit(code=1)

    refinement_mapping = {
        "None": getattr(locus.CornerRefinementMode, "None"),
        "Edge": locus.CornerRefinementMode.Edge,
        "GridFit": locus.CornerRefinementMode.GridFit,
        "Erf": locus.CornerRefinementMode.Erf,
    }
    refinement_mode = refinement_mapping.get(refinement, locus.CornerRefinementMode.Edge)

    loader = DatasetLoader()
    wrappers: list[LibraryWrapper] = []

    # Soft mode (Production Default + Soft Override + CLI Overrides)
    soft_detector = locus.Detector.production_config()
    soft_detector.set_families([tag_family_int])
    soft_config = soft_detector.config()
    soft_config.decode_mode = locus.DecodeMode.Soft
    soft_config.refinement_mode = refinement_mode
    soft_config.threshold_tile_size = tile_size
    soft_config.adaptive_threshold_constant = constant
    soft_config.quad_min_fill_ratio = min_fill
    soft_config.threshold_min_range = min_range
    soft_config.max_hamming_error = max_hamming
    soft_config.quad_min_edge_score = min_edge_score

    wrappers.append(
        LocusWrapper(
            name="Locus (Soft)", config=soft_config, decimation=decimation, family=tag_family_int
        )
    )

    # Hard mode (Production Default + Hard Override + CLI Overrides)
    hard_detector = locus.Detector.production_config()
    hard_detector.set_families([tag_family_int])
    hard_config = hard_detector.config()
    hard_config.decode_mode = locus.DecodeMode.Hard
    hard_config.refinement_mode = refinement_mode
    hard_config.threshold_tile_size = tile_size
    hard_config.adaptive_threshold_constant = constant
    hard_config.quad_min_fill_ratio = min_fill
    hard_config.threshold_min_range = min_range
    hard_config.max_hamming_error = max_hamming
    hard_config.quad_min_edge_score = min_edge_score

    wrappers.append(
        LocusWrapper(
            name="Locus (Hard)", config=hard_config, decimation=decimation, family=tag_family_int
        )
    )

    if compare:
        # Map locus.TagFamily to library specific names
        # ICRA 2020 dataset is AprilTag 36h11
        wrappers.append(OpenCVWrapper(family=tag_family_int))
        wrappers.append(AprilTagWrapper(nthreads=8, family=tag_family_int))

    baseline_data = {}
    if baseline and baseline.exists():
        with open(baseline) as f:
            baseline_data = json.load(f)
        typer.echo(f"Loaded baseline from {baseline}")

    current_results = {}

    for scenario in scenarios:
        if not loader.prepare_icra(scenario):
            continue

        datasets = loader.find_datasets(scenario, types)
        for ds_name, img_dir, gt_map in datasets:
            typer.echo(f"\nEvaluating {ds_name}...")
            current_results[ds_name] = {}

            img_names = sorted(gt_map.keys())
            if skip:
                img_names = img_names[skip:]
            if limit:
                img_names = img_names[:limit]

            for wrapper in wrappers:
                stats = {"gt": 0, "det": 0, "err_sum": 0.0, "latency": []}

                for img_name in tqdm(img_names, desc=f"{wrapper.name:<10}"):
                    img_path = img_dir / img_name
                    if not img_path.exists():
                        continue

                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    gt_tags = gt_map[img_name]
                    start = time.perf_counter()
                    detections, _ = wrapper.detect(img)
                    stats["latency"].append((time.perf_counter() - start) * 1000)

                    correct, err_sum, _ = Metrics.match_detections(detections, gt_tags)
                    stats["gt"] += len(gt_tags)
                    stats["det"] += correct
                    stats["err_sum"] += err_sum

                recall = (stats["det"] / stats["gt"] * 100) if stats["gt"] > 0 else 0
                avg_err = (stats["err_sum"] / stats["det"]) if stats["det"] > 0 else 0
                avg_lat = np.mean(stats["latency"])

                res = {"recall": recall, "rmse": avg_err, "latency": avg_lat}
                current_results[ds_name][wrapper.name] = res

                typer.echo(
                    f"  {wrapper.name:<10} | Recall: {recall:>6.2f}% | RMSE: {avg_err:>6.4f} px | Latency: {avg_lat:>6.2f} ms"
                )

                if ds_name in baseline_data and wrapper.name in baseline_data[ds_name]:
                    base = baseline_data[ds_name][wrapper.name]
                    lat_diff = avg_lat - base["latency"]
                    recall_diff = recall - base["recall"]
                    typer.echo(
                        f"    [Baseline] Latency: {lat_diff:+.2f}ms ({lat_diff / base['latency'] * 100:+.1f}%) | "
                        f"Recall: {recall_diff:+.2f}%"
                    )

    if save_baseline:
        with open(save_baseline, "w") as f:
            json.dump(current_results, f, indent=2)
        typer.echo(f"\nSaved baseline to {save_baseline}")


@bench_app.command("synthetic")
def bench_synthetic(
    targets: str = typer.Option("1,10,50,100", help="Comma-separated list of tag counts"),
    noise: float = typer.Option(0.0, help="Noise sigma"),
    iterations: int = typer.Option(10, help="Number of iterations per count"),
    compare: bool = typer.Option(False, help="Compare against other libraries"),
    decimation: int = typer.Option(1, help="Image decimation factor"),
    family: str = typer.Option("AprilTag36h11", help="Tag family to detect"),
):
    """Run benchmarks on procedurally generated synthetic images."""
    import locus
    import numpy as np

    from tools.bench.utils import (
        AprilTagWrapper,
        LibraryWrapper,
        LocusWrapper,
        Metrics,
        OpenCVWrapper,
        generate_synthetic_image,
    )

    # Map string to locus.TagFamily
    family_mapping = {
        "AprilTag16h5": int(locus.TagFamily.AprilTag16h5),
        "AprilTag36h11": int(locus.TagFamily.AprilTag36h11),
        "ArUco4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "ArUco4x4_100": int(locus.TagFamily.ArUco4x4_100),
        "16h5": int(locus.TagFamily.AprilTag16h5),
        "36h11": int(locus.TagFamily.AprilTag36h11),
        "4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "4x4_100": int(locus.TagFamily.ArUco4x4_100),
    }

    tag_family_int = family_mapping.get(family)
    if tag_family_int is None:
        typer.echo(f"Error: Unknown tag family '{family}'", err=True)
        raise typer.Exit(code=1)

    wrappers: list[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=decimation, family=tag_family_int))

    if compare:
        wrappers.append(OpenCVWrapper(family=tag_family_int))
        wrappers.append(AprilTagWrapper(family=tag_family_int))

    counts = [int(x) for x in targets.split(",")]
    res = (1280, 720)

    typer.echo(
        f"{'Targets':<8} | {'Lib':<10} | {'Latency (ms)':<12} | {'Recall':<8} | {'CornerErr':<10}"
    )
    typer.echo("-" * 60)

    for count in counts:
        img, gt_tags = generate_synthetic_image(
            count, res, noise_sigma=noise, family=tag_family_int
        )

        for wrapper in wrappers:
            latencies = []
            detections = []
            for _ in range(iterations):
                start = time.perf_counter()
                detections, _ = wrapper.detect(img)
                latencies.append((time.perf_counter() - start) * 1000)

            correct, err_sum, _ = Metrics.match_detections(detections, gt_tags)
            recall = (correct / len(gt_tags) * 100) if gt_tags else 0
            avg_err = (err_sum / correct) if correct > 0 else 0

            typer.echo(
                f"{count:<8} | {wrapper.name:<10} | {np.mean(latencies):<12.3f} | {recall:<7.1f}% | {avg_err:<10.3f}"
            )


@bench_app.command("hosted")
def bench_hosted(
    configs: list[str] = typer.Option(..., help="HF Hub dataset configurations"),
    limit: int | None = typer.Option(None, help="Limit number of images"),
    skip: int = typer.Option(0, help="Skip first N images"),
    compare: bool = typer.Option(False, help="Compare against other libraries"),
    decimation: int = typer.Option(1, help="Image decimation factor"),
    family: str = typer.Option("AprilTag36h11", help="Tag family to detect"),
):
    """Evaluate against datasets hosted on Hugging Face Hub."""
    import locus
    import numpy as np
    from tqdm import tqdm

    from tools.bench.utils import (
        AprilTagWrapper,
        HubBenchmarkLoader,
        LibraryWrapper,
        LocusWrapper,
        Metrics,
        OpenCVWrapper,
    )

    # Map string to locus.TagFamily
    family_mapping = {
        "AprilTag16h5": int(locus.TagFamily.AprilTag16h5),
        "AprilTag36h11": int(locus.TagFamily.AprilTag36h11),
        "ArUco4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "ArUco4x4_100": int(locus.TagFamily.ArUco4x4_100),
        "16h5": int(locus.TagFamily.AprilTag16h5),
        "36h11": int(locus.TagFamily.AprilTag36h11),
        "4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "4x4_100": int(locus.TagFamily.ArUco4x4_100),
    }

    tag_family_int = family_mapping.get(family)
    if tag_family_int is None:
        typer.echo(f"Error: Unknown tag family '{family}'", err=True)
        raise typer.Exit(code=1)

    loader = HubBenchmarkLoader()
    wrappers: list[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=decimation, family=tag_family_int))

    if compare:
        wrappers.append(OpenCVWrapper(family=tag_family_int))
        wrappers.append(AprilTagWrapper(nthreads=8, family=tag_family_int))

    for config in configs:
        typer.echo(f"\nEvaluating {config} (from Hugging Face Hub)...")
        wrapper_stats = {
            w.name: {
                "gt": 0,
                "det": 0,
                "err_sum": 0.0,
                "latency": [],
            }
            for w in wrappers
        }

        subset_stream = loader.stream_subset(config)
        for idx, (_name, img, gt_tags) in enumerate(tqdm(subset_stream, desc="Processing Images")):
            if skip and idx < skip:
                continue
            if limit and (idx - skip) >= limit:
                break

            for wrapper in wrappers:
                start = time.perf_counter()
                detections, _ = wrapper.detect(img)
                latency = (time.perf_counter() - start) * 1000

                correct, err_sum, _ = Metrics.match_detections(detections, gt_tags)
                stats = wrapper_stats[wrapper.name]
                stats["latency"].append(latency)
                stats["gt"] += len(gt_tags)
                stats["det"] += correct
                stats["err_sum"] += err_sum

        for wrapper in wrappers:
            stats = wrapper_stats[wrapper.name]
            recall = (stats["det"] / stats["gt"] * 100) if stats["gt"] > 0 else 0
            avg_err = (stats["err_sum"] / stats["det"]) if stats["det"] > 0 else 0
            avg_lat = np.mean(stats["latency"])
            typer.echo(
                f"  {wrapper.name:<10} | Recall: {recall:>6.2f}% | RMSE: {avg_err:>6.4f} px | Latency: {avg_lat:>6.2f} ms"
            )


@bench_app.command("analyze")
def bench_analyze(
    scenarios: list[str] = typer.Option(["forward", "circle"], help="Scenarios to analyze"),
):
    """Analyze tag size distribution in datasets."""
    import numpy as np

    from tools.bench.utils import DatasetLoader

    loader = DatasetLoader()
    for scenario in scenarios:
        if not loader.prepare_icra(scenario):
            continue
        datasets = loader.find_datasets(scenario, ["tags"])
        for ds_name, _, gt_map in datasets:
            typer.echo(f"\nAnalyzing Tag Sizes in {ds_name}...")
            sizes = []
            for _, tags in gt_map.items():
                for gt in tags:
                    edges = [
                        np.linalg.norm(gt.corners[i] - gt.corners[(i + 1) % 4]) for i in range(4)
                    ]
                    sizes.append(np.mean(edges))
            if not sizes:
                continue
            sizes_arr = np.array(sizes)
            typer.echo(f"  Total tags: {len(sizes_arr)}")
            typer.echo(f"  Mean size: {np.mean(sizes_arr):.1f} px")
            typer.echo(f"  Min size: {np.min(sizes_arr):.1f} px")
            typer.echo(
                f"  Small (<10px): {np.sum(sizes_arr < 10)} ({np.mean(sizes_arr < 10) * 100:.1f}%)"
            )


@bench_app.command("profile")
def bench_profile(
    targets: int = typer.Option(50, help="Number of tags to generate"),
    noise: float = typer.Option(0.0, help="Noise sigma"),
    iterations: int = typer.Option(50, help="Number of iterations"),
    family: str = typer.Option("AprilTag36h11", help="Tag family to detect"),
):
    """Profile pipeline bottlenecks using synthetic images."""
    import locus
    import numpy as np

    from tools.bench.utils import LocusWrapper, generate_synthetic_image

    # Map string to locus.TagFamily
    family_mapping = {
        "AprilTag16h5": int(locus.TagFamily.AprilTag16h5),
        "AprilTag36h11": int(locus.TagFamily.AprilTag36h11),
        "ArUco4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "ArUco4x4_100": int(locus.TagFamily.ArUco4x4_100),
        "16h5": int(locus.TagFamily.AprilTag16h5),
        "36h11": int(locus.TagFamily.AprilTag36h11),
        "4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "4x4_100": int(locus.TagFamily.ArUco4x4_100),
    }

    tag_family_int = family_mapping.get(family)
    if tag_family_int is None:
        typer.echo(f"Error: Unknown tag family '{family}'", err=True)
        raise typer.Exit(code=1)

    wrapper = LocusWrapper(family=tag_family_int)
    res = (1280, 720)
    img, _ = generate_synthetic_image(targets, res, noise_sigma=noise, family=tag_family_int)
    typer.echo(f"\nProfiling {targets} tags (noise={noise}, family={family})...")

    # In the new API, we just measure total time.
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        wrapper.detect(img)
        latencies.append((time.perf_counter() - start) * 1000)

    typer.echo(f"  Total (avg): {np.mean(latencies):.2f} ms")


@app.command()
def extract_bits(
    tag_id: int = typer.Argument(..., help="Tag ID to extract bits from"),
    family: str = typer.Option("AprilTag36h11", help="Tag family (OpenCV constant name)"),
):
    """
    Extract bits from an AprilTag or ArUco marker using OpenCV.
    Useful for verifying low-level encoding.
    """
    import cv2

    family_map = {
        "AprilTag16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "AprilTag36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "ArUco4x4_50": cv2.aruco.DICT_4X4_50,
        "ArUco4x4_100": cv2.aruco.DICT_4X4_100,
        "ArUco4x4_250": cv2.aruco.DICT_4X4_250,
        "ArUco4x4_1000": cv2.aruco.DICT_4X4_1000,
    }

    cv_family = family_map.get(family)
    if cv_family is None:
        typer.echo(f"Error: Unsupported family '{family}'", err=True)
        raise typer.Exit(code=1)

    dictionary = cv2.aruco.getPredefinedDictionary(cv_family)
    # Generate a large enough image to see bits clearly
    tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, 200)

    # For AprilTag 36h11, it is 8x8 (6x6 data + 1 cell border)
    # For ArUco 4x4, it's 6x6 (4x4 data + 1 cell border)
    # We can infer grid size from the dictionary
    grid_size = dictionary.markerSize + 2
    cell_size = 200 // grid_size
    bits = 0
    data_size = dictionary.markerSize

    # Sample the inner grid
    for row in range(data_size):
        for col in range(data_size):
            # Inner grid starts at index 1,1
            cy = (row + 1) * cell_size + cell_size // 2
            cx = (col + 1) * cell_size + cell_size // 2
            val = tag_img[cy, cx]
            if val > 128:
                bit_idx = row * data_size + col
                bits |= 1 << bit_idx

    typer.echo(f"ID {tag_id} ({family})")
    typer.echo(f"Bits (hex): {hex(bits)}")
    typer.echo(f"Bits (bin): {bin(bits)}")


class LocalHubLoader:
    def __init__(self, root: Path = Path("tests/data/hub_cache")):
        self.root = root

    def stream_subset(self, subset_name: str):
        import cv2
        import numpy as np

        from tools.bench.utils import TagGroundTruth

        subset_dir = self.root / subset_name
        jsonl_path = subset_dir / "annotations.jsonl"
        img_dir = subset_dir / "images"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"Metadata not found at {jsonl_path}")

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                img_name = item["image_filename"]
                img_path = img_dir / img_name

                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                tag_ids = item["tag_id"]
                corners_list = item["corners"]
                if not isinstance(tag_ids, list):
                    tag_ids = [tag_ids]
                    corners_list = [corners_list]

                gt_tags = []
                for tid, corners in zip(tag_ids, corners_list, strict=True):
                    gt_tags.append(
                        TagGroundTruth(tag_id=int(tid), corners=np.array(corners, dtype=np.float32))
                    )

                yield img_name, img, gt_tags


@app.command()
def debug_report(
    configs: list[str] = typer.Option(..., help="HF Hub dataset configurations"),
    limit: int | None = typer.Option(5, help="Limit number of images per config"),
    output: Path = typer.Option(Path("debug_report"), help="Output directory"),
    refinement_mode: str = typer.Option("Erf", help="Sub-pixel refinement mode (None, Edge, Erf)"),
):
    """
    Generate a high-fidelity HTML debug report for dataset subsets.
    """
    import locus
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir(exist_ok=True)

    loader = LocalHubLoader()
    ref_map = {
        "None": 0,
        "Edge": 1,
        "Erf": 3,
    }
    sel_ref_mode = ref_map.get(refinement_mode, 3)  # Default to Erf (3)

    detector = locus.Detector(
        families=[locus.TagFamily.AprilTag36h11], refinement_mode=sel_ref_mode
    )

    report_data = []

    for config in configs:
        typer.echo(f"Processing {config}...")
        subset_stream = loader.stream_subset(config)

        for idx, (img_name, img_np, gt_tags) in enumerate(tqdm(subset_stream, desc=config)):
            if limit and idx >= limit:
                break

            # Run detection with telemetry
            batch = detector.detect(img_np, debug_telemetry=True)

            # Find best match for first GT tag (for illustration)
            best_det = None
            rmse = 0.0
            if gt_tags and len(batch) > 0:
                gt = gt_tags[0]
                gt_arr = gt.corners
                best_err = float("inf")
                for j in range(len(batch)):
                    det = batch.corners[j]
                    for rot in range(4):
                        rotated = np.roll(det, rot, axis=0)
                        err = np.sqrt(np.mean(np.sum((rotated - gt_arr) ** 2, axis=1)))
                        if err < best_err:
                            best_err = err
                            best_det = rotated
                rmse = best_err

            # Generate Overlay Images
            # 1. Original with GT & Det
            img_pil = Image.fromarray(img_np).convert("RGB")
            draw = ImageDraw.Draw(img_pil)

            if gt_tags:
                for gt in gt_tags:
                    pts = [tuple(p) for p in gt.corners]
                    draw.polygon(pts, outline=(0, 255, 0), width=2)

            if best_det is not None:
                pts = [tuple(p) for p in best_det]
                draw.polygon(pts, outline=(255, 0, 0), width=1)

            orig_rel_path = f"images/{config}_{img_name}_orig.png"
            img_pil.save(output / orig_rel_path)

            # 2. Binarized with GT & Det
            bin_rel_path = "n/a"
            if batch.telemetry:
                bin_pil = Image.fromarray(batch.telemetry.binarized).convert("RGB")
                draw_bin = ImageDraw.Draw(bin_pil)
                if gt_tags:
                    for gt in gt_tags:
                        pts = [tuple(p) for p in gt.corners]
                        draw_bin.polygon(pts, outline=(0, 255, 0), width=1)

                if best_det is not None:
                    pts = [tuple(p) for p in best_det]
                    draw_bin.polygon(pts, outline=(255, 0, 0), width=1)

                bin_rel_path = f"images/{config}_{img_name}_bin.png"
                bin_pil.save(output / bin_rel_path)

            report_data.append(
                {
                    "config": config,
                    "name": img_name,
                    "rmse": rmse,
                    "orig": orig_rel_path,
                    "bin": bin_rel_path,
                }
            )

    # Generate HTML
    html = """
    <html>
    <head>
        <style>
            body { font-family: sans-serif; background: #1a1a1a; color: #eee; padding: 20px; }
            .card { background: #2a2a2a; border: 1px solid #444; padding: 15px; margin-bottom: 20px; border_radius: 8px; }
            .img-row { display: flex; gap: 10px; margin-top: 10px; }
            img { border: 1px solid #555; max-width: 48%; }
            .meta { color: #aaa; margin-bottom: 5px; }
            .rmse { color: #f0ad4e; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Locus Debug Report</h1>
    """
    for d in report_data:
        html += f"""
        <div class="card">
            <div class="meta">{d["config"]} / {d["name"]}</div>
            <div class="rmse">Best Corner RMSE: {d["rmse"]:.4f} px</div>
            <div class="img-row">
                <img src="{d["orig"]}">
                <img src="{d["bin"]}">
            </div>
        </div>
        """
    html += "</body></html>"

    with open(output / "report.html", "w") as f:
        f.write(html)

    typer.echo(f"\nReport generated at {output / 'report.html'}")


@bench_app.command("prepare")
def bench_prepare():
    """Download and prepare all benchmarking datasets."""
    from tools.bench.utils import DatasetLoader

    loader = DatasetLoader()
    typer.echo("Preparing datasets...")
    loader.prepare_all()
    typer.echo("Done.")


if __name__ == "__main__":
    app()
