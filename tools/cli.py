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
):
    """Run benchmarks on real-world datasets (ICRA)."""
    if profile:
        locus.init_tracy()

    loader = DatasetLoader()
    wrappers: List[LibraryWrapper] = []

    # Soft mode
    soft_config = locus.DetectorConfig(
        decode_mode=locus.DecodeMode.Soft,
        enable_sharpening=True,
        upscale_factor=1,
    )
    wrappers.append(LocusWrapper(name="Locus (Soft)", config=soft_config, decimation=decimation))

    # Hard mode
    hard_config = locus.DetectorConfig(
        decode_mode=locus.DecodeMode.Hard,
        enable_sharpening=True,
        upscale_factor=1,
    )
    wrappers.append(LocusWrapper(name="Locus (Hard)", config=hard_config, decimation=decimation))

    if compare:
        wrappers.append(OpenCVWrapper())
        wrappers.append(AprilTagWrapper(nthreads=8))

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
):
    """Run benchmarks on procedurally generated synthetic images."""
    wrappers: List[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=decimation))
    if compare:
        wrappers.append(OpenCVWrapper())
        wrappers.append(AprilTagWrapper())

    counts = [int(x) for x in targets.split(",")]
    res = (1280, 720)

    typer.echo(f"{'Targets':<8} | {'Lib':<10} | {'Latency (ms)':<12} | {'Recall':<8} | {'CornerErr':<10}")
    typer.echo("-" * 60)

    for count in counts:
        img, gt_tags = generate_synthetic_image(count, res, noise_sigma=noise)
        for wrapper in wrappers:
            latencies = []
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
):
    """Evaluate against datasets hosted on Hugging Face Hub."""
    loader = HubBenchmarkLoader()
    wrappers: List[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=decimation))
    if compare:
        wrappers.append(OpenCVWrapper())
        wrappers.append(AprilTagWrapper(nthreads=8))

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
):
    """Profile pipeline bottlenecks using synthetic images."""
    wrapper = LocusWrapper()
    res = (1280, 720)
    img, _ = generate_synthetic_image(targets, res, noise_sigma=noise)
    typer.echo(f"\nProfiling {targets} tags (noise={noise})...")
    stats_list = []
    for _ in range(iterations):
        _, stats = wrapper.detect(img)
        stats_list.append(stats)
    
    typer.echo(f"  Thresholding: {np.mean([s.threshold_ms for s in stats_list]):.2f} ms")
    typer.echo(f"  Segmentation: {np.mean([s.segmentation_ms for s in stats_list]):.2f} ms")
    typer.echo(f"  Quad Extraction: {np.mean([s.quad_extraction_ms for s in stats_list]):.2f} ms")
    typer.echo(f"  Decoding: {np.mean([s.decoding_ms for s in stats_list]):.2f} ms")
    typer.echo(f"  Total: {np.mean([s.total_ms for s in stats_list]):.2f} ms")

@bench_app.command("prepare")
def bench_prepare():
    """Download and prepare all benchmarking datasets."""
    loader = DatasetLoader()
    typer.echo("Preparing datasets...")
    loader.prepare_all()
    typer.echo("Done.")

if __name__ == "__main__":
    app()
