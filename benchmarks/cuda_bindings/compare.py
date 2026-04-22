# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare Python and C++ benchmark results in a summary table."""

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = PROJECT_ROOT / "results-python.json"
DEFAULT_CPP = PROJECT_ROOT / "results-cpp.json"


def load_benchmarks(path: Path) -> dict[str, list[float]]:
    """Load a pyperf JSON file and return {name: [values]}."""
    with open(path) as f:
        data = json.load(f)

    results: dict[str, list[float]] = {}
    for bench in data.get("benchmarks", []):
        name = bench.get("metadata", {}).get("name", "")
        if not name:
            # Try to find name in run metadata
            for run in bench.get("runs", []):
                name = run.get("metadata", {}).get("name", "")
                if name:
                    break
        values: list[float] = []
        for run in bench.get("runs", []):
            values.extend(run.get("values", []))
        if name and values:
            results[name] = values
    return results


def stats(values: list[float]) -> tuple[float, float, float, int]:
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    rsd = (stdev / mean) if mean else 0.0
    return mean, stdev, rsd, len(values)


def fmt_rsd(rsd: float | None) -> str:
    if rsd is None:
        return "-"
    return f"{rsd * 100:.1f}%"


def fmt_ns(seconds: float) -> str:
    ns = seconds * 1e9
    if ns >= 1000:
        return f"{ns / 1000:.2f} us"
    return f"{ns:.0f} ns"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python vs C++ benchmark results")
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON,
        help=f"Python results JSON (default: {DEFAULT_PYTHON.name})",
    )
    parser.add_argument(
        "--cpp",
        type=Path,
        default=DEFAULT_CPP,
        help=f"C++ results JSON (default: {DEFAULT_CPP.name})",
    )
    args = parser.parse_args()

    if not args.python.exists():
        print(f"Python results not found: {args.python}", file=sys.stderr)
        print("Run: pixi run -e wheel bench", file=sys.stderr)
        sys.exit(1)

    py_benchmarks = load_benchmarks(args.python)
    cpp_benchmarks = load_benchmarks(args.cpp) if args.cpp.exists() else {}

    if not py_benchmarks:
        print("No benchmarks found in Python results.", file=sys.stderr)
        sys.exit(1)

    # Column widths
    all_names = sorted(set(py_benchmarks) | set(cpp_benchmarks))
    name_width = max(len(n) for n in all_names)
    name_width = max(name_width, len("Benchmark"))

    # Header
    if cpp_benchmarks:
        header = (
            f"{'Benchmark':<{name_width}}  {'C++ (mean)':>12}  {'C++ RSD':>8}  "
            f"{'Python (mean)':>14}  {'Py RSD':>7}  {'Overhead':>10}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
    else:
        header = f"{'Benchmark':<{name_width}}  {'Python (mean)':>14}  {'Py RSD':>7}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)

    for name in all_names:
        py_vals = py_benchmarks.get(name)
        cpp_vals = cpp_benchmarks.get(name)

        py_stats = stats(py_vals) if py_vals else None
        cpp_stats = stats(cpp_vals) if cpp_vals else None

        py_str = fmt_ns(py_stats[0]) if py_stats else "-"
        cpp_str = fmt_ns(cpp_stats[0]) if cpp_stats else "-"
        py_rsd = fmt_rsd(py_stats[2]) if py_stats else "-"
        cpp_rsd = fmt_rsd(cpp_stats[2]) if cpp_stats else "-"

        if py_stats and cpp_stats:
            py_mean = py_stats[0]
            cpp_mean = cpp_stats[0]
            overhead_ns = (py_mean - cpp_mean) * 1e9
            overhead_str = f"+{overhead_ns:.0f} ns"
        else:
            overhead_str = "-"

        if cpp_benchmarks:
            print(
                f"{name:<{name_width}}  {cpp_str:>12}  {cpp_rsd:>8}  "
                f"{py_str:>14}  {py_rsd:>7}  {overhead_str:>10}"
            )
        else:
            print(f"{name:<{name_width}}  {py_str:>14}  {py_rsd:>7}")

    print(sep)


if __name__ == "__main__":
    main()
