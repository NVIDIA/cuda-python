# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned-memory diagnostics: fresh processes, unfolded logs, and first failures.

Run after installing the core wheel and test dependencies. The CI workflow owns
the overall timeout and child-process cleanup. Each pytest child uses isolated
Python startup so the unbuilt cuda_core checkout cannot shadow the wheel.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

LIFECYCLE_CASES = tuple(
    (f"lifecycle-{stream}-{access}", stream, access == "revoke")
    for stream in ("legacy", "per-thread", "nonblocking")
    for access in ("revoke", "keep-access")
)
NATIVE_CASES = (
    *(
        (f"native-{stream}-{access}", stream, access == "revoke", "baseline")
        for stream in ("legacy", "per-thread", "nonblocking")
        for access in ("revoke", "keep-access")
    ),
    *(
        (f"native-legacy-keep-access-{variant}", "legacy", False, variant)
        for variant in ("no-trim", "no-reset", "reuse")
    ),
)
CASE_NAMES = (
    *(name for name, _, _, _ in NATIVE_CASES),
    "isolated-watermarks",
    "pinned-kernel-access",
    *(name for name, _, _ in LIFECYCLE_CASES),
    "owned-reserved",
    "owned-used",
    "warmed-owned-reserved",
    "warmed-owned-used",
    "full-core-without-current-kernel",
)
OOM_FILENAME = "cuda_core_oom_diagnostics.txt"


def build_cases(root, output, cycles, native_executable=None):
    core = root / "cuda_core"
    cases = []
    if native_executable is not None:
        for name, stream, revoke, variant in NATIVE_CASES:
            raw_output = output / name
            command = [
                str(native_executable),
                "--stream",
                stream,
                "--variant",
                variant,
                "--cycles",
                str(2 * cycles),
                "--output-dir",
                str(raw_output),
            ]
            if revoke:
                command.append("--revoke-access")
            cases.append(
                {
                    "name": name,
                    "cwd": str(root),
                    "command": command,
                    "kind": "native",
                    "raw_output": str(raw_output),
                    "timeout_seconds": 120,
                }
            )

    def pytest_case(name, selectors, count, selection=None):
        command = [
            sys.executable,
            "-I",
            "-u",
            str(pathlib.Path(__file__).resolve()),
            "--pytest-child",
            str(output / (name + "-pytest.json")),
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
    for name, stream, revoke in LIFECYCLE_CASES:
        raw_output = output / name
        command = [
            sys.executable,
            "-I",
            "-u",
            str(pathlib.Path(__file__).with_name("pinned_pool_probe.py").resolve()),
            "--mode",
            "owned-capped",
            "--host-id",
            "-1",
            # Each pytest repetition runs both watermark kinds in this order.
            "--cycles",
            str(2 * cycles),
            "--watermark",
            "alternating",
            "--stream",
            stream,
            "--low-overhead",
            "--output-dir",
            str(raw_output),
        ]
        if revoke:
            command.append("--revoke-access")
        cases.append(
            {
                "name": name,
                "cwd": str(root),
                "command": command,
                "kind": "raw",
                "raw_output": str(raw_output),
                "timeout_seconds": 120,
            }
        )
    for mode, prefix in (("owned-capped", "owned"), ("warmed-owned-capped", "warmed-owned")):
        for watermark in ("reserved", "used"):
            name = f"{prefix}-{watermark}"
            raw_output = output / name
            command = [
                sys.executable,
                "-I",
                "-u",
                str(pathlib.Path(__file__).with_name("pinned_pool_probe.py").resolve()),
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
    report_path = (
        output / (case["name"] + "-pytest.json")
        if case["kind"] == "pytest"
        else pathlib.Path(case["raw_output"]) / "summary.json"
    )
    report_path.unlink(missing_ok=True)
    timed_out = threading.Event()
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

        def stop_hung_probe():
            if process.poll() is None:
                timed_out.set()
                process.kill()

        # Only the standalone lifecycle probes get this watchdog. A driver
        # hang must leave time to run the other variants and upload progress.
        timer = None
        if "timeout_seconds" in case:
            timer = threading.Timer(case["timeout_seconds"], stop_hung_probe)
            timer.daemon = True
            timer.start()
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            returncode = process.wait()
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
    result = {
        **case,
        "log_path": str(log_path),
        "returncode": returncode,
        "timed_out": timed_out.is_set(),
        "elapsed_seconds": time.monotonic() - started,
    }
    result["failed"] = returncode != 0 or timed_out.is_set()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result["report"] = report
        if case["kind"] == "pytest":
            result["failed"] |= not report["completed"] or report["counts"]["passed"] == 0
        else:
            result["failed"] |= (
                not report["required_path_completed"] or report["completed_cycles"] != report["requested_cycles"]
            )
    except (OSError, ValueError, KeyError) as exc:
        result["artifact_error"] = str(exc)
        result["failed"] = True
    if case["kind"] == "pytest":
        oom = pathlib.Path(case["cwd"]) / OOM_FILENAME
        if oom.exists():
            destination = output / (case["name"] + "-" + OOM_FILENAME)
            shutil.copy2(oom, destination)
            oom.unlink()
            result["first_oom_path"] = str(destination)
    print(f"Finished {case['name']}: returncode={returncode}, elapsed={result['elapsed_seconds']:.2f}s", flush=True)
    return result


def run_pytest_child(report_path, pytest_args):
    """Import the installed wheel before pytest adds the test helper directories."""
    report = {"completed": False, "phase": "wheel-import", "counts": {"passed": 0, "failed": 0, "skipped": 0}}
    report_path = pathlib.Path(report_path)
    try:
        if not sys.flags.isolated:
            raise RuntimeError("The pytest child must run with python -I")
        distribution = importlib.metadata.distribution("cuda-core")
        installed_core = pathlib.Path(distribution.locate_file("cuda/core")).resolve()
        origins = {}
        for name in ("cuda.core", "cuda.core._version"):
            module = importlib.import_module(name)
            origin = pathlib.Path(module.__file__).resolve()
            if not origin.is_relative_to(installed_core):
                raise RuntimeError(f"{name} resolved to {origin}, outside the installed wheel at {installed_core}")
            origins[name] = str(origin)
        report["wheel"] = {"version": distribution.version, "origins": origins}
        print(f"Installed wheel preflight: {json.dumps(report['wheel'])}", flush=True)

        import pytest

        class Outcomes:
            def pytest_runtest_logreport(self, report):
                if report.failed:
                    counts["failed"] += 1
                elif report.skipped:
                    counts["skipped"] += 1
                elif report.when == "call" and report.passed:
                    counts["passed"] += 1

        counts = report["counts"]
        report["phase"] = "pytest"
        returncode = int(pytest.main(pytest_args, plugins=[Outcomes()]))
        if returncode == 0 and counts["passed"] == 0:
            print("Diagnostic failed: no test call passed (all tests skipped or collection only).", flush=True)
            returncode = 1
        report.update(completed=True, returncode=returncode)
        return returncode
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("pinned-memory-diagnostics"))
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument(
        "--native-executable",
        type=pathlib.Path,
        help="Also run native driver probes using this prebuilt executable (no Python in the child).",
    )
    parser.add_argument("--case", choices=CASE_NAMES, action="append", help="Run only these cases; may be repeated.")
    args = parser.parse_args()
    if not 1 <= args.cycles <= 100:
        parser.error("--cycles must be between 1 and 100")

    if args.case and any(name.startswith("native-") for name in args.case) and args.native_executable is None:
        parser.error("Native cases require --native-executable")
    native_executable = args.native_executable.resolve() if args.native_executable is not None else None
    root = args.repo_root.resolve()
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
    cases = [
        case
        for case in build_cases(root, output, args.cycles, native_executable)
        if args.case is None or case["name"] in args.case
    ]
    summary = {
        "started_at": datetime.now(UTC).isoformat(),
        "completed": False,
        "interpreter": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "pytest_addopts": env["PYTEST_ADDOPTS"],
        "cycles": args.cycles,
        "requested_cases": [case["name"] for case in cases],
        "cases": [],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Keep an older artifact separate so it cannot masquerade as this run's OOM.
    existing_oom = root / "cuda_core" / OOM_FILENAME
    if existing_oom.exists():
        shutil.move(existing_oom, output / ("preexisting-" + OOM_FILENAME))
    for case in cases:
        # A missing executable/build failure is a failed experiment, not a skip.
        # Continue collecting independent cases and preserve the build log.
        try:
            result = run_case(case, output, env)
        except OSError as exc:
            result = {**case, "failed": True, "launch_error": str(exc)}
            print(f"Could not run {case['name']}: {exc}", flush=True)
        summary["cases"].append(result)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["completed"] = True
    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["returncode"] = int(any(case["failed"] for case in summary["cases"]))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}; overall returncode={summary['returncode']}", flush=True)
    return summary["returncode"]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--pytest-child"]:
        raise SystemExit(run_pytest_child(sys.argv[2], sys.argv[3:]))
    raise SystemExit(main())
