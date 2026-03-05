import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
import jsonschema
import cv2
import numpy as np
from tqdm import tqdm

import locus

# We assume scripts is in the python path
try:
    from scripts.bench.utils import (
        AprilTagWrapper,
        DatasetLoader,
        HubBenchmarkLoader,
        LibraryWrapper,
        LocusWrapper,
        Metrics,
        OpenCVWrapper,
        generate_synthetic_image,
    )
except ImportError:
    # Fallback for when running from root without proper PYTHONPATH
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.bench.utils import (
        AprilTagWrapper,
        DatasetLoader,
        HubBenchmarkLoader,
        LibraryWrapper,
        LocusWrapper,
        Metrics,
        OpenCVWrapper,
        generate_synthetic_image,
    )

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
    files: List[Path] = typer.Argument(..., help="JSON files to validate"),
):
    """
    Validate dictionary JSON files against a schema.
    """
    try:
        with open(schema) as f:
            schema_data = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading schema {schema}: {e}", err=True)
        raise typer.Exit(code=1)

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
                    err=True
                )
                has_errors = True
            elif len(data["base_codes"]) != data["dictionary_size"]:
                typer.echo(
                    f"FAILED: {file_path} - length of base_codes ({len(data['base_codes'])}) does not match dictionary_size ({data['dictionary_size']})",
                    err=True
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
    limit: Optional[int] = typer.Option(10, help="Limit number of images"),
    scenario: str = typer.Option("forward", help="Scenario to visualize"),
    tile_size: int = typer.Option(8, help="Threshold tile size"),
    min_area: int = typer.Option(16, help="Min quad area"),
    bilateral: bool = typer.Option(False, help="Enable bilateral filter"),
    upscale: int = typer.Option(1, help="Upscale factor"),
):
    """
    Launch the Rerun-based visualizer for the detection pipeline.
    """
    if not RERUN_AVAILABLE:
        typer.echo("Error: Rerun SDK not installed. Run 'uv add rerun-sdk' or install with [bench] group.", err=True)
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

            # Perform Detection (Vectorized API)
            batch = detector.detect(img)

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

# --- Benchmarking Commands ---

@bench_app.command("real")
def bench_real(
    scenarios: List[str] = typer.Option(["forward"], help="Scenarios to run"),
    types: List[str] = typer.Option(["tags"], help="Dataset types (tags, checkerboard)"),
    limit: Optional[int] = typer.Option(None, help="Limit number of images"),
    skip: int = typer.Option(0, help="Skip first N images"),
    compare: bool = typer.Option(False, help="Compare against other libraries"),
    decimation: int = typer.Option(1, help="Image decimation factor"),
    baseline: Optional[Path] = typer.Option(None, help="Path to baseline JSON"),
    save_baseline: Optional[Path] = typer.Option(None, help="Path to save results as baseline"),
    profile: bool = typer.Option(False, help="Enable Tracy profiling"),
    family: str = typer.Option("AprilTag36h11", help="Tag family to detect"),
    refinement: str = typer.Option("Edge", help="Refinement mode (None, Edge, GridFit, Erf)"),
    tile_size: int = typer.Option(4, help="Threshold tile size"),
    constant: int = typer.Option(3, help="Adaptive threshold constant"),
    min_fill: float = typer.Option(0.30, help="Min quad fill ratio"),
):
    """Run benchmarks on real-world datasets (ICRA)."""
    if profile:
        locus.init_tracy()

    # Map string to locus.TagFamily
    family_mapping = {
        "AprilTag36h11": int(locus.TagFamily.AprilTag36h11),
        "AprilTag41h12": int(locus.TagFamily.AprilTag41h12),
        "ArUco4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "ArUco4x4_100": int(locus.TagFamily.ArUco4x4_100),
        "36h11": int(locus.TagFamily.AprilTag36h11),
        "41h12": int(locus.TagFamily.AprilTag41h12),
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
    wrappers: List[LibraryWrapper] = []

    # Soft mode
    soft_config = locus.DetectorConfig(
        decode_mode=locus.DecodeMode.Soft,
        enable_sharpening=True,
        upscale_factor=1,
        refinement_mode=refinement_mode,
        threshold_tile_size=tile_size,
        adaptive_threshold_constant=constant,
        quad_min_fill_ratio=min_fill,
    )
    wrappers.append(LocusWrapper(name="Locus (Soft)", config=soft_config, decimation=decimation, family=tag_family_int))

    # Hard mode
    hard_config = locus.DetectorConfig(
        decode_mode=locus.DecodeMode.Hard,
        enable_sharpening=True,
        upscale_factor=1,
        refinement_mode=refinement_mode,
        threshold_tile_size=tile_size,
        adaptive_threshold_constant=constant,
        quad_min_fill_ratio=min_fill,
    )
    wrappers.append(LocusWrapper(name="Locus (Hard)", config=hard_config, decimation=decimation, family=tag_family_int))

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
                stage_stats = {
                    "threshold": [],
                    "segmentation": [],
                    "quad": [],
                    "decoding": [],
                }
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
                    detections, lib_stats = wrapper.detect(img)
                    stats["latency"].append((time.perf_counter() - start) * 1000)

                    if lib_stats:
                        stage_stats["threshold"].append(lib_stats.threshold_ms)
                        stage_stats["segmentation"].append(lib_stats.segmentation_ms)
                        stage_stats["quad"].append(lib_stats.quad_extraction_ms)
                        stage_stats["decoding"].append(lib_stats.decoding_ms)

                    correct, err_sum, _ = Metrics.match_detections(detections, gt_tags)
                    stats["gt"] += len(gt_tags)
                    stats["det"] += correct
                    stats["err_sum"] += err_sum

                recall = (stats["det"] / stats["gt"] * 100) if stats["gt"] > 0 else 0
                avg_err = (stats["err_sum"] / stats["det"]) if stats["det"] > 0 else 0
                avg_lat = np.mean(stats["latency"])

                res = {"recall": recall, "rmse": avg_err, "latency": avg_lat}
                if stage_stats["threshold"]:
                    res["stages"] = {k: float(np.mean(v)) for k, v in stage_stats.items()}
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
    # Map string to locus.TagFamily
    family_mapping = {
        "AprilTag36h11": int(locus.TagFamily.AprilTag36h11),
        "AprilTag41h12": int(locus.TagFamily.AprilTag41h12),
        "ArUco4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "ArUco4x4_100": int(locus.TagFamily.ArUco4x4_100),
        "36h11": int(locus.TagFamily.AprilTag36h11),
        "41h12": int(locus.TagFamily.AprilTag41h12),
        "4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "4x4_100": int(locus.TagFamily.ArUco4x4_100),
    }
    tag_family_int = family_mapping.get(family)
    if tag_family_int is None:
        typer.echo(f"Error: Unknown tag family '{family}'", err=True)
        raise typer.Exit(code=1)

    wrappers: List[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=decimation, family=tag_family_int))

    if compare:
        wrappers.append(OpenCVWrapper(family=tag_family_int))
        wrappers.append(AprilTagWrapper(family=tag_family_int))

    counts = [int(x) for x in targets.split(",")]
    res = (1280, 720)

    typer.echo(f"{'Targets':<8} | {'Lib':<10} | {'Latency (ms)':<12} | {'Recall':<8} | {'CornerErr':<10}")
    typer.echo("-" * 60)

    for count in counts:
        img, gt_tags = generate_synthetic_image(count, res, noise_sigma=noise, family=tag_family_int)

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

            typer.echo(f"{count:<8} | {wrapper.name:<10} | {np.mean(latencies):<12.3f} | {recall:<7.1f}% | {avg_err:<10.3f}")

@bench_app.command("hosted")
def bench_hosted(
    configs: List[str] = typer.Option(..., help="HF Hub dataset configurations"),
    limit: Optional[int] = typer.Option(None, help="Limit number of images"),
    skip: int = typer.Option(0, help="Skip first N images"),
    compare: bool = typer.Option(False, help="Compare against other libraries"),
    decimation: int = typer.Option(1, help="Image decimation factor"),
    family: str = typer.Option("AprilTag36h11", help="Tag family to detect"),
):
    """Evaluate against datasets hosted on Hugging Face Hub."""
    # Map string to locus.TagFamily
    family_mapping = {
        "AprilTag36h11": int(locus.TagFamily.AprilTag36h11),
        "AprilTag41h12": int(locus.TagFamily.AprilTag41h12),
        "ArUco4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "ArUco4x4_100": int(locus.TagFamily.ArUco4x4_100),
        "36h11": int(locus.TagFamily.AprilTag36h11),
        "41h12": int(locus.TagFamily.AprilTag41h12),
        "4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "4x4_100": int(locus.TagFamily.ArUco4x4_100),
    }
    tag_family_int = family_mapping.get(family)
    if tag_family_int is None:
        typer.echo(f"Error: Unknown tag family '{family}'", err=True)
        raise typer.Exit(code=1)

    loader = HubBenchmarkLoader()
    wrappers: List[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=decimation, family=tag_family_int))

    if compare:
        wrappers.append(OpenCVWrapper(family=tag_family_int))
        wrappers.append(AprilTagWrapper(nthreads=8, family=tag_family_int))

    for config in configs:
        typer.echo(f"\nEvaluating {config} (from Hugging Face Hub)...")
        wrapper_stats = {
            w.name: {
                "gt": 0, "det": 0, "err_sum": 0.0, "latency": [],
                "stages": {"threshold": [], "segmentation": [], "quad": [], "decoding": []},
            } for w in wrappers
        }

        subset_stream = loader.stream_subset(config)
        for idx, (_name, img, gt_tags) in enumerate(tqdm(subset_stream, desc="Processing Images")):
            if skip and idx < skip: continue
            if limit and (idx - skip) >= limit: break

            for wrapper in wrappers:
                start = time.perf_counter()
                detections, lib_stats = wrapper.detect(img)
                latency = (time.perf_counter() - start) * 1000

                correct, err_sum, _ = Metrics.match_detections(detections, gt_tags)
                stats = wrapper_stats[wrapper.name]
                stats["latency"].append(latency)
                stats["gt"] += len(gt_tags)
                stats["det"] += correct
                stats["err_sum"] += err_sum

                if lib_stats:
                    stats["stages"]["threshold"].append(lib_stats.threshold_ms)
                    stats["stages"]["segmentation"].append(lib_stats.segmentation_ms)
                    stats["stages"]["quad"].append(lib_stats.quad_extraction_ms)
                    stats["stages"]["decoding"].append(lib_stats.decoding_ms)

        for wrapper in wrappers:
            stats = wrapper_stats[wrapper.name]
            recall = (stats["det"] / stats["gt"] * 100) if stats["gt"] > 0 else 0
            avg_err = (stats["err_sum"] / stats["det"]) if stats["det"] > 0 else 0
            avg_lat = np.mean(stats["latency"])
            typer.echo(f"  {wrapper.name:<10} | Recall: {recall:>6.2f}% | RMSE: {avg_err:>6.4f} px | Latency: {avg_lat:>6.2f} ms")

@bench_app.command("analyze")
def bench_analyze(
    scenarios: List[str] = typer.Option(["forward", "circle"], help="Scenarios to analyze"),
):
    """Analyze tag size distribution in datasets."""
    loader = DatasetLoader()
    for scenario in scenarios:
        if not loader.prepare_icra(scenario): continue
        datasets = loader.find_datasets(scenario, ["tags"])
        for ds_name, _, gt_map in datasets:
            typer.echo(f"\nAnalyzing Tag Sizes in {ds_name}...")
            sizes = []
            for _, tags in gt_map.items():
                for gt in tags:
                    edges = [np.linalg.norm(gt.corners[i] - gt.corners[(i + 1) % 4]) for i in range(4)]
                    sizes.append(np.mean(edges))
            if not sizes: continue
            sizes_arr = np.array(sizes)
            typer.echo(f"  Total tags: {len(sizes_arr)}")
            typer.echo(f"  Mean size: {np.mean(sizes_arr):.1f} px")
            typer.echo(f"  Min size: {np.min(sizes_arr):.1f} px")
            typer.echo(f"  Small (<10px): {np.sum(sizes_arr < 10)} ({np.mean(sizes_arr < 10) * 100:.1f}%)")

@bench_app.command("profile")
def bench_profile(
    targets: int = typer.Option(50, help="Number of tags to generate"),
    noise: float = typer.Option(0.0, help="Noise sigma"),
    iterations: int = typer.Option(50, help="Number of iterations"),
    family: str = typer.Option("AprilTag36h11", help="Tag family to detect"),
):
    """Profile pipeline bottlenecks using synthetic images."""
    # Map string to locus.TagFamily
    family_mapping = {
        "AprilTag36h11": int(locus.TagFamily.AprilTag36h11),
        "AprilTag41h12": int(locus.TagFamily.AprilTag41h12),
        "ArUco4x4_50": int(locus.TagFamily.ArUco4x4_50),
        "ArUco4x4_100": int(locus.TagFamily.ArUco4x4_100),
        "36h11": int(locus.TagFamily.AprilTag36h11),
        "41h12": int(locus.TagFamily.AprilTag41h12),
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
    
    # In the new API, we don't have per-stage stats exposed yet in the high-level API
    # We just measure total time for now.
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        wrapper.detect(img)
        latencies.append((time.perf_counter() - start) * 1000)
    
    typer.echo(f"  Total (avg): {np.mean(latencies):.2f} ms")

@bench_app.command("prepare")
def bench_prepare():
    """Download and prepare all benchmarking datasets."""
    loader = DatasetLoader()
    typer.echo("Preparing datasets...")
    loader.prepare_all()
    typer.echo("Done.")

if __name__ == "__main__":
    app()
