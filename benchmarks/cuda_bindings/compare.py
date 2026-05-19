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
    """Format a duration in nanoseconds with a thousands separator.

    Using a single unit across the whole table makes side-by-side comparison
    easier, even when some entries get into the multi-microsecond range.
    """
    return f"{seconds * 1e9:,.0f}"


def fmt_overhead_ns(py_mean: float, cpp_mean: float) -> str:
    return f"{(py_mean - cpp_mean) * 1e9:+,.0f}"


def fmt_overhead_pct(py_mean: float, cpp_mean: float) -> str:
    if cpp_mean <= 0.0:
        return "-"
    pct = (py_mean - cpp_mean) / cpp_mean * 100
    return f"{pct:+,.0f}%"


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

    # Right-aligned numeric columns. Widths chosen so header text fits and
    # multi-microsecond ns values with thousands separators still align.
    cpp_w = 12
    py_w = 12
    rsd_w = 8
    oh_ns_w = 12
    oh_pct_w = 10

    # Header
    if cpp_benchmarks:
        header = (
            f"{'Benchmark':<{name_width}}  "
            f"{'C++ (ns)':>{cpp_w}}  {'C++ RSD':>{rsd_w}}  "
            f"{'Python (ns)':>{py_w}}  {'Py RSD':>{rsd_w}}  "
            f"{'Overhead ns':>{oh_ns_w}}  {'Overhead %':>{oh_pct_w}}"
        )
    else:
        header = f"{'Benchmark':<{name_width}}  {'Python (ns)':>{py_w}}  {'Py RSD':>{rsd_w}}"

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
            overhead_ns_str = fmt_overhead_ns(py_stats[0], cpp_stats[0])
            overhead_pct_str = fmt_overhead_pct(py_stats[0], cpp_stats[0])
        else:
            overhead_ns_str = "-"
            overhead_pct_str = "-"

        if cpp_benchmarks:
            print(
                f"{name:<{name_width}}  "
                f"{cpp_str:>{cpp_w}}  {cpp_rsd:>{rsd_w}}  "
                f"{py_str:>{py_w}}  {py_rsd:>{rsd_w}}  "
                f"{overhead_ns_str:>{oh_ns_w}}  {overhead_pct_str:>{oh_pct_w}}"
            )
        else:
            print(f"{name:<{name_width}}  {py_str:>{py_w}}  {py_rsd:>{rsd_w}}")

    print(sep)


if __name__ == "__main__":
    main()
