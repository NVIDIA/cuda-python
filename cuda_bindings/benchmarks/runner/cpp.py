# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / ".build" / "cpp"
DEFAULT_OUTPUT = PROJECT_ROOT / "results-cpp.json"

BINARY_PREFIX = "bench_"
BINARY_SUFFIX = "_cpp"


def discover_binaries() -> dict[str, Path]:
    """Discover C++ benchmark binaries in the build directory"""
    if not BUILD_DIR.is_dir():
        return {}

    registry: dict[str, Path] = {}
    for path in sorted(BUILD_DIR.iterdir()):
        if not path.is_file() or not path.name.startswith(BINARY_PREFIX):
            continue
        if not path.name.endswith(BINARY_SUFFIX):
            continue
        name = path.name.removeprefix(BINARY_PREFIX).removesuffix(BINARY_SUFFIX)
        registry[name] = path
    return registry


def strip_output_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("-o", "--output"):
            skip_next = True
            continue
        if arg.startswith("-o=") or arg.startswith("--output="):
            continue
        cleaned.append(arg)
    return cleaned


def merge_pyperf_json(individual_files: list[Path], output_path: Path) -> int:
    """Merge individual pyperf JSON files into a single BenchmarkSuite file.

    Each C++ binary produces a file with structure:
        {"version": "1.0", "metadata": {...}, "benchmarks": [{...}]}

    We merge them by collecting all benchmark entries into one file.
    """
    all_benchmarks = []

    for path in individual_files:
        with open(path) as f:
            data = json.load(f)

        file_metadata = data.get("metadata", {})
        bench_name = file_metadata.get("name", "")
        loops = file_metadata.get("loops")
        unit = file_metadata.get("unit", "second")

        for bench in data.get("benchmarks", []):
            for run in bench.get("runs", []):
                run_meta = run.setdefault("metadata", {})
                if bench_name:
                    run_meta.setdefault("name", bench_name)
                if loops is not None:
                    run_meta.setdefault("loops", loops)
                run_meta.setdefault("unit", unit)

            all_benchmarks.append(bench)

    merged = {
        "version": "1.0",
        "benchmarks": all_benchmarks,
    }

    with open(output_path, "w") as f:
        json.dump(merged, f)

    return len(all_benchmarks)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run C++ CUDA benchmarks",
        add_help=False,
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Benchmark name to run (e.g. 'ctx_device'). Repeat for multiple. Defaults to all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered benchmark names and exit.",
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

    registry = discover_binaries()
    if not registry:
        print(
            f"No C++ benchmark binaries found in {BUILD_DIR}.\nRun 'pixi run bench-cpp-build' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if parsed.list:
        for name in sorted(registry):
            print(name)
        return

    if parsed.benchmark:
        missing = sorted(set(parsed.benchmark) - set(registry))
        if missing:
            known = ", ".join(sorted(registry))
            unknown = ", ".join(missing)
            print(
                f"Unknown benchmark(s): {unknown}. Known benchmarks: {known}",
                file=sys.stderr,
            )
            sys.exit(1)
        names = parsed.benchmark
    else:
        names = sorted(registry)

    # Strip any --output args to avoid conflicts with our output handling
    passthrough_argv = strip_output_args(remaining_argv)

    output_path = parsed.output.resolve()
    failed = False
    individual_files: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="cuda_bench_cpp_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for name in names:
            binary = registry[name]
            tmp_json = tmpdir_path / f"{name}.json"
            cmd = [str(binary), "-o", str(tmp_json), *passthrough_argv]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"FAILED: {name} (exit code {result.returncode})", file=sys.stderr)
                failed = True
            elif tmp_json.exists():
                individual_files.append(tmp_json)

        if individual_files:
            count = merge_pyperf_json(individual_files, output_path)
            print(f"\nResults saved to {output_path} ({count} benchmark(s))")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
