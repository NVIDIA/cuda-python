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

# Benchmark IDs where cuda.core and cuda.bindings exercise *different*
# underlying driver calls or hit a cuda.core-side cache, so the "Delta"
# column is NOT pure Python wrapper overhead. See BENCHMARK_PLAN.md's
# "Audit notes" section for a full explanation of each entry.
DIFFERENT_CODEPATH_BENCHMARKS: frozenset[str] = frozenset(
    {
        # cuCtxGetDevice (core) vs cuCtxGetCurrent (bindings).
        "ctx_device.ctx_get_current",
        # TLS list lookup (core) vs cuDeviceGet (bindings).
        "ctx_device.device_get",
        # DeviceProperties dict cache hit (core) vs cuDeviceGetAttribute
        # (bindings) on every iteration.
        "ctx_device.device_get_attribute",
        # cuMemAllocFromPoolAsync on default stream (core) vs synchronous
        # cuMemAlloc (bindings).
        "memory.mem_alloc_free",
        # cuLaunchKernelEx + per-call ParamHolder (core) vs cuLaunchKernel
        # with pre-built arg tuple (bindings).
        "launch.launch_empty_kernel",
        "launch.launch_small_kernel",
        "launch.launch_16_args",
        "launch.launch_256_args",
        "launch.launch_512_args",
    }
)
DIFFERENT_CODEPATH_MARKER = "*"


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


def fmt_delta_ns(core_mean: float, bindings_mean: float) -> str:
    return f"{(core_mean - bindings_mean) * 1e9:+,.0f}"


def fmt_delta_pct(core_mean: float, bindings_mean: float) -> str:
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

    # Reserve a trailing column of space for the "different codepath" marker
    # so it does not collide visually with the benchmark ID.
    display_names = {
        name: f"{name} {DIFFERENT_CODEPATH_MARKER}" if name in DIFFERENT_CODEPATH_BENCHMARKS else name
        for name in all_names
    }
    name_width = max(len(display_names[n]) for n in all_names)
    name_width = max(name_width, len("Benchmark"))

    bind_w = 14
    core_w = 14
    rsd_w = 8
    delta_ns_w = 12
    delta_pct_w = 10

    if bindings_benchmarks:
        header = (
            f"{'Benchmark':<{name_width}}  "
            f"{'bindings (ns)':>{bind_w}}  {'RSD':>{rsd_w}}  "
            f"{'core (ns)':>{core_w}}  {'RSD':>{rsd_w}}  "
            f"{'Delta ns':>{delta_ns_w}}  {'Delta %':>{delta_pct_w}}"
        )
    else:
        header = f"{'Benchmark':<{name_width}}  {'core (ns)':>{core_w}}  {'RSD':>{rsd_w}}"

    sep = "-" * len(header)

    if bindings_benchmarks:
        # Keep legend lines shorter than the table so they don't overflow.
        print("Delta = core mean - bindings mean (positive = cuda.core slower).")
        print(f"{DIFFERENT_CODEPATH_MARKER} marks benchmarks where core and bindings exercise different")
        print("  underlying driver calls or hit a cuda.core cache — see BENCHMARK_PLAN.md")
        print("  (Audit notes) for details on each row.")
        print()

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
            delta_ns_str = fmt_delta_ns(core_stats[0], bind_stats[0])
            delta_pct_str = fmt_delta_pct(core_stats[0], bind_stats[0])
        else:
            delta_ns_str = "-"
            delta_pct_str = "-"

        display_name = display_names[name]

        if bindings_benchmarks:
            print(
                f"{display_name:<{name_width}}  "
                f"{bind_str:>{bind_w}}  {bind_rsd:>{rsd_w}}  "
                f"{core_str:>{core_w}}  {core_rsd:>{rsd_w}}  "
                f"{delta_ns_str:>{delta_ns_w}}  {delta_pct_str:>{delta_pct_w}}"
            )
        else:
            print(f"{display_name:<{name_width}}  {core_str:>{core_w}}  {core_rsd:>{rsd_w}}")

    print(sep)


if __name__ == "__main__":
    main()
