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
# Env var used to propagate the --benchmark filter from the parent to pyperf
# worker subprocesses. pyperf reconstructs worker argv from scratch and drops
# custom flags like --benchmark, so without this the worker would register the
# full bench list and pyperf would run the wrong bench by task index.
BENCH_FILTER_ENV_VAR = "CUDA_BINDINGS_BENCH_FILTER"

PYPERF_INHERITED_ENV_VARS = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    BENCH_FILTER_ENV_VAR,
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
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("bench_")
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
    # Expose the binding source so the runner can introspect skip lists before
    # handing the benchmark to pyperf. Accessing these attributes does not
    # trigger module import — discovery stays lazy.
    run._bench_module_path = module_path  # type: ignore[attr-defined]
    run._bench_function_name = function_name  # type: ignore[attr-defined]
    return run


def _collect_skipped_benchmarks(
    bench_ids: list[str],
    registry: dict[str, Callable[[int], float]],
) -> set[str]:
    """Return bench IDs that the owning module has marked as unsupported.

    Benchmark modules may declare a module-level
    ``SKIPPED_BENCHMARKS: set[str]`` containing the names of ``bench_*``
    functions whose underlying API is unavailable on the current driver or
    device (e.g. TMA encoders on pre-Hopper GPUs). Loading the module runs
    its import-time probe, so this call is the same cost as the first
    real invocation would have been.
    """
    skipped: set[str] = set()
    loaded_modules: dict[Path, ModuleType] = {}
    for bench_id in bench_ids:
        fn = registry[bench_id]
        module_path = getattr(fn, "_bench_module_path", None)
        function_name = getattr(fn, "_bench_function_name", None)
        if module_path is None or function_name is None:
            continue
        module = loaded_modules.get(module_path)
        if module is None:
            module = load_module(module_path)
            loaded_modules[module_path] = module
        module_skip = getattr(module, "SKIPPED_BENCHMARKS", None)
        if module_skip and function_name in module_skip:
            skipped.add(bench_id)
    return skipped


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

    # The --benchmark filter must be the same in the parent and in any pyperf
    # worker subprocess, otherwise pyperf's task-index bookkeeping points at
    # the wrong bench. pyperf drops unknown CLI flags when spawning workers,
    # so fall back to an env var carrying the filter.
    requested = list(parsed.benchmark)
    env_filter = os.environ.get(BENCH_FILTER_ENV_VAR, "")
    if not requested and env_filter:
        requested = [bid for bid in env_filter.split(",") if bid]

    if requested:
        missing = sorted(set(requested) - set(registry))
        if missing:
            known = ", ".join(sorted(registry))
            unknown = ", ".join(missing)
            raise ValueError(f"Unknown benchmark(s): {unknown}. Known benchmarks: {known}")
        benchmark_ids = requested
        # Propagate to any pyperf worker we're about to spawn.
        os.environ[BENCH_FILTER_ENV_VAR] = ",".join(benchmark_ids)
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

    # Drop benchmarks that the owning module has marked as unavailable on
    # this driver/device. Without this step a single unsupported bench
    # (e.g. TMA on a pre-Hopper GPU) would abort the whole pyperf run,
    # since pyperf treats a raised exception as a fatal worker failure.
    skipped = _collect_skipped_benchmarks(benchmark_ids, registry)
    if skipped and not is_worker:
        for bench_id in sorted(skipped):
            print(f"Skipping {bench_id}: unsupported on this driver/device", file=sys.stderr)
    benchmark_ids = [bench_id for bench_id in benchmark_ids if bench_id not in skipped]

    runner = pyperf.Runner()
    for bench_id in benchmark_ids:
        runner.bench_time_func(bench_id, registry[bench_id])

    if not is_worker:
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
