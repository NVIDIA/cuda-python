# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary PR3436 diagnostics: fresh processes, unfolded logs, and first failures.

Run from the repository root after the core test dependencies are installed.
The CI workflow owns the overall timeout and child-process cleanup. Remove this
diagnostic script before merging the PR.
"""

import argparse
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime

CASE_NAMES = (
    "isolated-watermarks",
    "pinned-kernel-access",
    "owned-reserved",
    "owned-used",
    "warmed-owned-reserved",
    "warmed-owned-used",
    "full-core-without-current-kernel",
)
OOM_FILENAME = "cuda_core_oom_diagnostics.txt"


def build_cases(root, output, cycles):
    core = root / "cuda_core"
    cases = []

    def pytest_case(name, selectors, count, selection=None):
        command = [
            sys.executable,
            "-u",
            "-m",
            "pytest",
            *selectors,
            f"--count={count}",
            "--repeat-scope=session",
            "--parallel-threads=1",
            "--randomly-dont-reorganize",
            "--maxfail=1",
            "--tb=long",
            "-vv",
            "-s",
            f"--junitxml={output / (name + '.xml')}",
        ]
        if selection is not None:
            command.extend(["-k", selection])
        cases.append({"name": name, "cwd": str(core), "command": command, "kind": "pytest"})

    pytest_case(
        "isolated-watermarks",
        ["tests/test_memory.py::test_mempool_watermark_reset"],
        cycles,
        "PinnedMR",
    )
    pytest_case("pinned-kernel-access", ["tests/test_memory.py::test_pinned_memory_resource_kernel_access"], 1)
    for mode, prefix in (("owned-capped", "owned"), ("warmed-owned-capped", "warmed-owned")):
        for watermark in ("reserved", "used"):
            name = f"{prefix}-{watermark}"
            raw_output = output / name
            command = [
                sys.executable,
                "-u",
                str(root / "ci/tools/pr3436_raw_pool_probe.py"),
                "--mode",
                mode,
                "--host-id",
                "-1",
                "--cycles",
                str(cycles),
                "--watermark",
                watermark,
                "--output-dir",
                str(raw_output),
            ]
            if mode == "warmed-owned-capped":
                command.append("--allow-warmup-allocation-oom")
            cases.append(
                {"name": name, "cwd": str(root), "command": command, "kind": "raw", "raw_output": str(raw_output)}
            )
    pytest_case(
        "full-core-without-current-kernel",
        ["tests/"],
        1,
        "not (test_pinned_memory_resource_kernel_access and current)",
    )
    return cases


def run_case(case, output, env):
    log_path = output / (case["name"] + ".log")
    started = time.monotonic()
    print(f"Starting {case['name']}: {json.dumps(case['command'])}", flush=True)
    with (
        log_path.open("w", encoding="utf-8", buffering=1) as log,
        subprocess.Popen(  # noqa: S603 - fixed diagnostic argv, passed without a shell.
            case["command"],
            cwd=case["cwd"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process,
    ):
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    result = {
        **case,
        "log_path": str(log_path),
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
    }
    result["failed"] = returncode != 0
    if case["kind"] == "pytest":
        oom = pathlib.Path(case["cwd"]) / OOM_FILENAME
        if oom.exists():
            destination = output / (case["name"] + "-" + OOM_FILENAME)
            shutil.copy2(oom, destination)
            oom.unlink()
            result["first_oom_path"] = str(destination)
    else:
        raw_summary = pathlib.Path(case["raw_output"]) / "summary.json"
        result["raw_summary_path"] = str(raw_summary)
        try:
            result["raw_result"] = json.loads(raw_summary.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            result["artifact_error"] = str(exc)
            result["failed"] = True
    print(f"Finished {case['name']}: returncode={returncode}, elapsed={result['elapsed_seconds']:.2f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("watermark-diagnostic"))
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--case", choices=CASE_NAMES, action="append", help="Run only these cases; may be repeated.")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")

    root = pathlib.Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTEST_ADDOPTS="", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    versions = {}
    for package in (
        "cuda-core",
        "cuda-bindings",
        "cuda-pathfinder",
        "pytest",
        "pytest-repeat",
        "pytest-run-parallel",
        "numpy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    summary = {
        "started_at": datetime.now(UTC).isoformat(),
        "completed": False,
        "interpreter": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "pytest_addopts": env["PYTEST_ADDOPTS"],
        "cycles": args.cycles,
        "requested_cases": args.case or list(CASE_NAMES),
        "cases": [],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Keep an older artifact separate so it cannot masquerade as this run's OOM.
    existing_oom = root / "cuda_core" / OOM_FILENAME
    if existing_oom.exists():
        shutil.move(existing_oom, output / ("preexisting-" + OOM_FILENAME))
    for case in build_cases(root, output, args.cycles):
        if args.case is not None and case["name"] not in args.case:
            continue
        summary["cases"].append(run_case(case, output, env))
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["completed"] = True
    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["returncode"] = int(any(case["failed"] for case in summary["cases"]))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}; overall returncode={summary['returncode']}", flush=True)
    return summary["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
