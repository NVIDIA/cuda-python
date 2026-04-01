# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pyperf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "benchmarks"
DEFAULT_OUTPUT = PROJECT_ROOT / "results-python.json"


def load_module(module_path: Path) -> ModuleType:
    module_name = f"cuda_bindings_bench_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_id(module_name: str, function_name: str) -> str:
    module_suffix = module_name.removeprefix("bench_")
    suffix = function_name.removeprefix("bench_")
    return f"{module_suffix}.{suffix}"


def discover_benchmarks() -> dict[str, Callable[[int], float]]:
    """Discover bench_ functions.

    Each bench_ function must have the signature: bench_*(loops: int) -> float
    where it calls the operation `loops` times and returns the total elapsed
    time in seconds (using time.perf_counter).
    """
    registry: dict[str, Callable[[int], float]] = {}
    for module_path in sorted(BENCH_DIR.glob("bench_*.py")):
        module = load_module(module_path)
        module_name = module_path.stem
        for function_name, function in inspect.getmembers(module, inspect.isfunction):
            if not function_name.startswith("bench_"):
                continue
            if function.__module__ != module.__name__:
                continue
            bench_id = benchmark_id(module_name, function_name)
            if bench_id in registry:
                raise ValueError(f"Duplicate benchmark ID discovered: {bench_id}")
            registry[bench_id] = function
    return registry


def strip_pyperf_output_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in ("-o", "--output", "--append"):
            skip_next = True
            continue
        if arg.startswith("-o=") or arg.startswith("--output=") or arg.startswith("--append="):
            continue
        cleaned.append(arg)
    return cleaned


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Benchmark ID to run. Repeat to run multiple IDs. Defaults to all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered benchmark IDs and exit.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output file path (default: {DEFAULT_OUTPUT.name})",
    )
    parsed, remaining = parser.parse_known_args(argv)
    return parsed, remaining


def main() -> None:
    parsed, remaining_argv = parse_args(sys.argv[1:])

    registry = discover_benchmarks()
    if not registry:
        raise RuntimeError(f"No benchmark functions found in {BENCH_DIR}")

    if parsed.list:
        for bench_id in sorted(registry):
            print(bench_id)
        return

    if parsed.benchmark:
        missing = sorted(set(parsed.benchmark) - set(registry))
        if missing:
            known = ", ".join(sorted(registry))
            unknown = ", ".join(missing)
            raise ValueError(f"Unknown benchmark(s): {unknown}. Known benchmarks: {known}")
        benchmark_ids = parsed.benchmark
    else:
        benchmark_ids = sorted(registry)

    # Strip any --output args to avoid conflicts with our output handling
    output_path = parsed.output.resolve()
    remaining_argv = strip_pyperf_output_args(remaining_argv)
    is_worker = "--worker" in remaining_argv

    # Delete the file so this run starts fresh
    if not is_worker:
        output_path.unlink(missing_ok=True)

    sys.argv = [sys.argv[0], "--append", str(output_path), *remaining_argv]

    runner = pyperf.Runner()
    for bench_id in benchmark_ids:
        runner.bench_time_func(bench_id, registry[bench_id])

    if not is_worker:
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
