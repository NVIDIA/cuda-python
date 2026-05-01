# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare cuda.core and cuda.bindings benchmark results side by side.

Benchmark IDs are kept intentionally identical across the two suites
(e.g. ``stream.stream_create_destroy``) so a result present in both
files can be diffed directly. IDs that exist in only one suite are
rendered with ``-`` in the missing column.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CORE = HERE / "results-python.json"
DEFAULT_BINDINGS = HERE.parent / "cuda_bindings" / "results-python.json"


def load_benchmarks(path: Path) -> dict[str, list[float]]:
    """Load a pyperf JSON file and return {name: [values]}."""
    with open(path) as f:
        data = json.load(f)

    results: dict[str, list[float]] = {}
    for bench in data.get("benchmarks", []):
        name = bench.get("metadata", {}).get("name", "")
        if not name:
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
    return f"{seconds * 1e9:,.0f}"


def fmt_overhead_ns(core_mean: float, bindings_mean: float) -> str:
    return f"{(core_mean - bindings_mean) * 1e9:+,.0f}"


def fmt_overhead_pct(core_mean: float, bindings_mean: float) -> str:
    if bindings_mean <= 0.0:
        return "-"
    pct = (core_mean - bindings_mean) / bindings_mean * 100
    return f"{pct:+,.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare cuda.core vs cuda.bindings benchmark results")
    parser.add_argument(
        "--core",
        type=Path,
        default=DEFAULT_CORE,
        help=f"cuda.core results JSON (default: {DEFAULT_CORE.name})",
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=DEFAULT_BINDINGS,
        help=f"cuda.bindings results JSON (default: {DEFAULT_BINDINGS})",
    )
    args = parser.parse_args()

    if not args.core.exists():
        print(f"cuda.core results not found: {args.core}", file=sys.stderr)
        print("Run: pixi run -e source bench", file=sys.stderr)
        sys.exit(1)

    core_benchmarks = load_benchmarks(args.core)
    bindings_benchmarks = load_benchmarks(args.bindings) if args.bindings.exists() else {}

    if not core_benchmarks:
        print("No benchmarks found in cuda.core results.", file=sys.stderr)
        sys.exit(1)

    all_names = sorted(set(core_benchmarks) | set(bindings_benchmarks))
    name_width = max(len(n) for n in all_names)
    name_width = max(name_width, len("Benchmark"))

    bind_w = 14
    core_w = 14
    rsd_w = 8
    oh_ns_w = 12
    oh_pct_w = 10

    if bindings_benchmarks:
        header = (
            f"{'Benchmark':<{name_width}}  "
            f"{'bindings (ns)':>{bind_w}}  {'RSD':>{rsd_w}}  "
            f"{'core (ns)':>{core_w}}  {'RSD':>{rsd_w}}  "
            f"{'Overhead ns':>{oh_ns_w}}  {'Overhead %':>{oh_pct_w}}"
        )
    else:
        header = f"{'Benchmark':<{name_width}}  {'core (ns)':>{core_w}}  {'RSD':>{rsd_w}}"

    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for name in all_names:
        core_vals = core_benchmarks.get(name)
        bind_vals = bindings_benchmarks.get(name)

        core_stats = stats(core_vals) if core_vals else None
        bind_stats = stats(bind_vals) if bind_vals else None

        core_str = fmt_ns(core_stats[0]) if core_stats else "-"
        bind_str = fmt_ns(bind_stats[0]) if bind_stats else "-"
        core_rsd = fmt_rsd(core_stats[2]) if core_stats else "-"
        bind_rsd = fmt_rsd(bind_stats[2]) if bind_stats else "-"

        if core_stats and bind_stats:
            overhead_ns_str = fmt_overhead_ns(core_stats[0], bind_stats[0])
            overhead_pct_str = fmt_overhead_pct(core_stats[0], bind_stats[0])
        else:
            overhead_ns_str = "-"
            overhead_pct_str = "-"

        if bindings_benchmarks:
            print(
                f"{name:<{name_width}}  "
                f"{bind_str:>{bind_w}}  {bind_rsd:>{rsd_w}}  "
                f"{core_str:>{core_w}}  {core_rsd:>{rsd_w}}  "
                f"{overhead_ns_str:>{oh_ns_w}}  {overhead_pct_str:>{oh_pct_w}}"
            )
        else:
            print(f"{name:<{name_width}}  {core_str:>{core_w}}  {core_rsd:>{rsd_w}}")

    print(sep)


if __name__ == "__main__":
    main()
