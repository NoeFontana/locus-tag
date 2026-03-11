import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Regex to match Divan's tabular output
# Example: ├─ (640, 480)          3.688 ms      │ 7.689 ms      │ 3.893 ms      │ 4.328 ms      │ 100     │ 100
LINE_REGEX = re.compile(
    r"(?:├─|╰─|│)\s*(?P<name>[\(\)\d,\s\w_]+?)\s+(?P<fastest>[\d\.]+)\s+(?P<f_unit>ms|µs|ns|s)\s+│\s+(?P<slowest>[\d\.]+)\s+(?P<s_unit>ms|µs|ns|s)\s+│\s+(?P<median>[\d\.]+)\s+(?P<m_unit>ms|µs|ns|s)\s+│\s+(?P<mean>[\d\.]+)\s+(?P<mean_unit>ms|µs|ns|s)"
)

# Regex to match the section header
SECTION_REGEX = re.compile(r"(?:├─|╰─|│)\s*(?P<section>bench_[\w_]+)\s+│\s+│\s+│\s+│")

UNIT_CONVERSION = {
    "ns": 1e-6,
    "µs": 1e-3,
    "ms": 1.0,
    "s": 1000.0,
}


def parse_time(value: str, unit: str) -> float:
    return float(value) * UNIT_CONVERSION.get(unit, 1.0)


def get_cpu_info() -> str:
    try:
        output = subprocess.check_output("lscpu", shell=True).decode()
        for line in output.splitlines():
            if "Model name" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return "Unknown CPU"


def parse_divan_output(content: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    current_section = None

    for line in content.splitlines():
        # Check for section header
        section_match = SECTION_REGEX.search(line)
        if section_match:
            current_section = section_match.group("section").strip()
            results[current_section] = {}
            continue

        # Check for data line
        match = LINE_REGEX.search(line)
        if match:
            name = match.group("name").strip()
            data = {
                "fastest": parse_time(match.group("fastest"), match.group("f_unit")),
                "slowest": parse_time(match.group("slowest"), match.group("s_unit")),
                "median": parse_time(match.group("median"), match.group("m_unit")),
                "mean": parse_time(match.group("mean"), match.group("mean_unit")),
            }

            if current_section and (name.startswith("(") or name.isdigit()):
                results[current_section][name] = data
            else:
                results[name] = data

    return results


def generate_markdown_report(results: dict[str, Any], output_path: Path):
    cpu = get_cpu_info()
    # Extract date from filename if possible, otherwise use today
    date_part = output_path.stem.split("_")[-1]

    content = f"# Micro-Benchmark Baseline ({date_part})\n\n"
    content += f"**Environment:**\n- CPU: {cpu}\n- OS: Linux\n- Build: `--release` (Profile: bench)\n- Mode: Single-threaded (`--threads 1`)\n\n"

    for section, data in results.items():
        if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
            title = section.replace("bench_", "").replace("_", " ").title()
            content += f"## {title} (Median Latency)\n\n"
            content += "| Resolution | Latency (ms) |\n| :--- | :--- |\n"
            for res, metrics in sorted(data.items()):
                content += f"| {res} | {metrics['median']:.2f} ms |\n"
            content += "\n"

    content += "---\n\n*Note: Raw metrics are stored in `divan_baseline.json`.*\n"

    with open(output_path, "w") as f:
        f.write(content)
    print(f"Generated markdown report at {output_path}")


def update_baseline(output_file: Path, new_results: dict[str, Any]):
    if output_file.exists():
        with open(output_file) as f:
            try:
                baseline = json.load(f)
            except json.JSONDecodeError:
                baseline = {}
    else:
        baseline = {}

    baseline.update(new_results)

    with open(output_file, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Updated baseline at {output_file}")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python update_micro_baselines.py <divan_output_file> [baseline_json] [report_md]"
        )
        sys.exit(1)

    input_file = Path(sys.argv[1])
    baseline_file = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/benchmarking/divan_baseline.json")
    )
    report_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)

    with open(input_file) as f:
        content = f.read()

    results = parse_divan_output(content)
    if not results:
        print("Warning: No benchmarks found in output")

    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    update_baseline(baseline_file, results)

    if report_file:
        generate_markdown_report(results, report_file)


if __name__ == "__main__":
    main()
