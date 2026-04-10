# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pyperf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "benchmarks"
DEFAULT_OUTPUT = PROJECT_ROOT / "results-python.json"
PYPERF_INHERITED_ENV_VARS = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
)
_MODULE_CACHE: dict[Path, ModuleType] = {}


def load_module(module_path: Path) -> ModuleType:
    module_path = module_path.resolve()
    cached_module = _MODULE_CACHE.get(module_path)
    if cached_module is not None:
        return cached_module

    module_name = f"cuda_bindings_bench_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[module_path] = module
    return module


def benchmark_id(module_name: str, function_name: str) -> str:
    module_suffix = module_name.removeprefix("bench_")
    suffix = function_name.removeprefix("bench_")
    return f"{module_suffix}.{suffix}"


def _discover_module_functions(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("bench_")
    ]


def _lazy_benchmark(module_path: Path, function_name: str) -> Callable[[int], float]:
    loaded_function: Callable[[int], float] | None = None

    def run(loops: int) -> float:
        nonlocal loaded_function
        if loaded_function is None:
            module = load_module(module_path)
            loaded_function = getattr(module, function_name)
        return loaded_function(loops)

    run.__name__ = function_name
    return run


def discover_benchmarks() -> dict[str, Callable[[int], float]]:
    """Discover bench_ functions.

    Each bench_ function must have the signature: bench_*(loops: int) -> float
    where it calls the operation `loops` times and returns the total elapsed
    time in seconds (using time.perf_counter).
    """
    registry: dict[str, Callable[[int], float]] = {}
    for module_path in sorted(BENCH_DIR.glob("bench_*.py")):
        module_name = module_path.stem
        for function_name in _discover_module_functions(module_path):
            bench_id = benchmark_id(module_name, function_name)
            if bench_id in registry:
                raise ValueError(f"Duplicate benchmark ID discovered: {bench_id}")
            registry[bench_id] = _lazy_benchmark(module_path, function_name)
    return registry


def strip_pyperf_output_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("-o", "--output", "--append"):
            skip_next = True
            continue
        if arg.startswith(("-o=", "--output=", "--append=")):
            continue
        cleaned.append(arg)
    return cleaned


def _split_env_vars(arg_value: str) -> list[str]:
    return [env_var for env_var in arg_value.split(",") if env_var]


def ensure_pyperf_worker_env(argv: list[str]) -> list[str]:
    if "--copy-env" in argv:
        return list(argv)

    inherited_env: list[str] = []
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            inherited_env.extend(_split_env_vars(arg))
            skip_next = False
            continue
        if arg == "--inherit-environ":
            skip_next = True
            continue
        if arg.startswith("--inherit-environ="):
            inherited_env.extend(_split_env_vars(arg.partition("=")[2]))
            continue
        cleaned.append(arg)

    if skip_next:
        raise ValueError("Missing value for --inherit-environ")

    for env_var in PYPERF_INHERITED_ENV_VARS:
        if env_var in os.environ:
            inherited_env.append(env_var)

    deduped_env: list[str] = []
    for env_var in inherited_env:
        if env_var not in deduped_env:
            deduped_env.append(env_var)

    if deduped_env:
        cleaned.extend(["--inherit-environ", ",".join(deduped_env)])

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

    # Strip any --output args to avoid conflicts with our output handling.
    output_path = parsed.output.resolve()
    remaining_argv = strip_pyperf_output_args(remaining_argv)
    remaining_argv = ensure_pyperf_worker_env(remaining_argv)
    is_worker = "--worker" in remaining_argv

    # Delete the file so this run starts fresh.
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
