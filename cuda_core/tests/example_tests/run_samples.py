#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sample test orchestrator for the standalone samples under the repo-root
``samples/`` directory.

Discovers ``samples/<name>/<name>.py``, applies per-sample overrides from
``test_args.json`` (same schema used in cuda-samples, plus a ``python``
sub-object for Python-specific CLI args / launcher), and executes each sample
in its own subprocess.

This module lives in the cuda.core test suite so the samples are exercised as
part of ``pytest cuda_core/tests`` (see ``test_samples.py``), but it can also
be invoked directly as a standalone runner:
    python cuda_core/tests/example_tests/run_samples.py [--samples-dir samples] [--config .../test_args.json]

Exit-code contract (matches cuda-samples):
    0  -> sample passed
    2  -> sample waived (missing dependency / unmet hardware requirement)
    *  -> sample failed
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion

# Default timeout per sample run (seconds). Match cuda-samples.
DEFAULT_TIMEOUT = 300
EXIT_WAIVED = 2
# This file lives at ``cuda_core/tests/example_tests/run_samples.py``; the
# repo root (which holds the top-level ``samples/`` directory) is four levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SAMPLES_DIR = REPO_ROOT / "samples"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "test_args.json"

_print_lock = threading.Lock()


def _safe_print(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_samples(samples_dir: Path) -> list[Path]:
    """Return every ``<name>/<name>.py`` sample entrypoint under ``samples_dir``.

    Samples can either sit directly under ``samples_dir``
    (``samples/<name>/<name>.py``) or in a category subdirectory such as
    ``samples/0_Introduction/<name>/<name>.py``. The Utilities directory is
    excluded at any level. Matching the cuda-samples convention, exactly one
    Python entrypoint per sample directory is recognised (the one whose stem
    matches the directory name).
    """
    samples: list[Path] = []

    def walk(current: Path) -> None:
        for child in sorted(current.iterdir()):
            if not child.is_dir() or child.name == "Utilities":
                continue
            entry = child / f"{child.name}.py"
            if entry.is_file():
                samples.append(entry)
            else:
                # Not a sample dir; treat it as a category and recurse.
                walk(child)

    walk(samples_dir)
    return samples


# ---------------------------------------------------------------------------
# Config + GPU detection
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {}
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        _safe_print(f"Warning: failed to parse {config_path}: {exc}")
        return {}
    if not isinstance(data, dict):
        _safe_print(f"Warning: {config_path} must contain a JSON object")
        return {}
    # Drop any keys starting with '_' (used for comments).
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_gpu_count() -> int:
    """Return the visible CUDA GPU count, conservatively 0 on error.

    Matches cuda-samples/run_tests.py::get_gpu_count(): uses ``nvidia-smi -L``
    first and falls back to ``CUDA_VISIBLE_DEVICES``.
    """
    try:
        smi = subprocess.run(
            ["nvidia-smi", "-L"],  # noqa: S607
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if smi.returncode == 0:
            return sum(1 for line in smi.stdout.splitlines() if line.strip().lower().startswith("gpu "))
    except FileNotFoundError:
        pass
    except OSError:
        pass

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"no", "none"}:
        return len([v for v in visible.split(",") if v])
    return 0


# ---------------------------------------------------------------------------
# PEP 723 dependency gating
# ---------------------------------------------------------------------------

_DISTRIBUTION_PROVIDERS: dict[str, tuple[str, ...]] = {
    # cuda-python is a metapackage; cuda-bindings carries the matching CUDA API
    # version and is the package samples actually import.
    "cuda-python": ("cuda-bindings",),
    # Conda installs CuPy as ``cupy`` while PyPI uses CUDA-major-specific names.
    "cupy-cuda11x": ("cupy-cuda11x", "cupy"),
    "cupy-cuda12x": ("cupy-cuda12x", "cupy"),
    "cupy-cuda13x": ("cupy-cuda13x", "cupy"),
}


class DependencyMetadataError(ValueError):
    """A sample contains invalid PEP 723 dependency metadata."""


def _metadata_error(example: Path, detail: str) -> DependencyMetadataError:
    return DependencyMetadataError(f"{example}: invalid PEP 723 metadata: {detail}")


def _extract_pep723_dependencies(example: Path) -> list[str] | None:
    """Return the dependency list declared via PEP 723, or ``None`` if absent."""
    lines = example.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if line == "# /// script"]
    if not starts:
        return None
    if len(starts) != 1:
        raise _metadata_error(example, "expected exactly one script block")

    metadata_lines: list[str] = []
    for line_number, line in enumerate(lines[starts[0] + 1 :], start=starts[0] + 2):
        if line == "# ///":
            break
        if line == "#":
            metadata_lines.append("")
        elif line.startswith("# "):
            metadata_lines.append(line[2:])
        else:
            raise _metadata_error(example, f"line {line_number} inside the script block is not a comment")
    else:
        raise _metadata_error(example, "script block is missing its closing '# ///'")

    try:
        metadata = tomllib.loads("\n".join(metadata_lines))
    except tomllib.TOMLDecodeError as exc:
        raise _metadata_error(example, str(exc)) from exc

    dependencies = metadata.get("dependencies")
    if dependencies is None:
        return None
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise _metadata_error(example, "'dependencies' must be an array of strings")
    return dependencies


def _distribution_providers(requirement: Requirement) -> tuple[str, ...]:
    name = canonicalize_name(requirement.name)
    return _DISTRIBUTION_PROVIDERS.get(name, (name,))


def _provided_extras(dist: importlib.metadata.Distribution) -> set[str]:
    return {canonicalize_name(extra) for extra in (dist.metadata.get_all("Provides-Extra") or ())}


def _distribution_mismatches(requirement: Requirement, dist: importlib.metadata.Distribution) -> list[str]:
    mismatches: list[str] = []
    if requirement.specifier:
        try:
            version_matches = requirement.specifier.contains(dist.version)
        except InvalidVersion:
            mismatches.append(f"has invalid version {dist.version!r}")
        else:
            if not version_matches:
                mismatches.append(f"version {dist.version} does not satisfy {requirement.specifier}")

    missing_extras = {canonicalize_name(extra) for extra in requirement.extras} - _provided_extras(dist)
    if missing_extras:
        mismatches.append(f"does not provide extra(s): {', '.join(sorted(missing_extras))}")
    return mismatches


def missing_dependencies(example: Path) -> list[str]:
    """Return useful diagnostics for unmet PEP 723 dependency requirements.

    Returns an empty list if all declared deps are present, or if no PEP 723
    block exists (no gating to perform).
    """
    deps = _extract_pep723_dependencies(example)
    if not deps:
        return []

    missing: list[str] = []
    for spec in deps:
        try:
            requirement = Requirement(spec)
        except InvalidRequirement as exc:
            raise _metadata_error(example, f"invalid dependency requirement {spec!r}: {exc}") from exc

        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue

        providers = _distribution_providers(requirement)
        installed: list[str] = []
        for provider in providers:
            try:
                dist = importlib.metadata.distribution(provider)
            except importlib.metadata.PackageNotFoundError:
                continue
            mismatches = _distribution_mismatches(requirement, dist)
            if not mismatches:
                break
            installed.append(f"{provider} {dist.version}: {', '.join(mismatches)}")
        else:
            if installed:
                detail = "; ".join(installed)
            else:
                detail = f"no provider distribution installed (checked: {', '.join(providers)})"
            missing.append(f"{requirement} ({detail})")
    return missing


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPlan:
    sample: Path
    args: list[str]
    launcher: list[str]
    timeout: int
    skip_reason: str | None = None


def _expand_env(value: str) -> str:
    return os.path.expandvars(value)


def build_run_plan(
    sample: Path,
    config: dict[str, Any],
    gpu_count: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> RunPlan:
    """Combine config overrides + GPU availability into a concrete run plan.

    The returned plan carries either a ``skip_reason`` (sample must be
    waived) or the command components to invoke.
    """
    sample_cfg = config.get(sample.parent.name, {})

    if sample_cfg.get("skip"):
        return RunPlan(sample, [], [], timeout, skip_reason="skipped in test_args.json")

    required_gpus = int(sample_cfg.get("min_gpus", 1))
    if required_gpus > gpu_count:
        return RunPlan(
            sample,
            [],
            [],
            timeout,
            skip_reason=(f"requires {required_gpus} GPU(s), only {gpu_count} available"),
        )

    python_cfg = sample_cfg.get("python", {})
    raw_args = python_cfg.get("args", []) or []
    raw_launcher = python_cfg.get("launcher", []) or []
    if not isinstance(raw_args, list) or not isinstance(raw_launcher, list):
        return RunPlan(
            sample,
            [],
            [],
            timeout,
            skip_reason="invalid config: 'args' and 'launcher' must be lists",
        )

    return RunPlan(
        sample=sample,
        args=[_expand_env(str(a)) for a in raw_args],
        launcher=[_expand_env(str(a)) for a in raw_launcher],
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    sample: Path
    status: str  # "PASS", "FAIL", "WAIVED", "TIMEOUT", "ERROR"
    return_code: int
    detail: str = ""


def run_sample(plan: RunPlan) -> RunResult:
    sample = plan.sample
    name = sample.parent.name

    if plan.skip_reason is not None:
        _safe_print(f"  [WAIVED] {name}: {plan.skip_reason}")
        return RunResult(sample, "WAIVED", EXIT_WAIVED, plan.skip_reason)

    try:
        missing = missing_dependencies(sample)
    except DependencyMetadataError as exc:
        reason = str(exc)
        _safe_print(f"  [ERROR] {name}: {reason}")
        return RunResult(sample, "ERROR", -1, reason)
    if missing:
        reason = f"unmet package requirement(s): {'; '.join(missing)}"
        _safe_print(f"  [WAIVED] {name}: {reason}")
        return RunResult(sample, "WAIVED", EXIT_WAIVED, reason)

    cmd = list(plan.launcher) + [sys.executable, str(sample)] + list(plan.args)
    _safe_print(f"  [RUN ] {name}: {' '.join(cmd)}")

    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(sample.parent),
            capture_output=True,
            text=True,
            timeout=plan.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _safe_print(f"  [TIMEOUT] {name}: exceeded {plan.timeout}s")
        return RunResult(sample, "TIMEOUT", -1, f"timed out after {plan.timeout}s")
    except OSError as exc:
        _safe_print(f"  [ERROR] {name}: {exc}")
        return RunResult(sample, "ERROR", -1, str(exc))

    if proc.returncode == 0:
        _safe_print(f"  [PASS ] {name}")
        return RunResult(sample, "PASS", 0)
    if proc.returncode == EXIT_WAIVED:
        _safe_print(f"  [WAIVED] {name}: sample reported waived")
        return RunResult(sample, "WAIVED", EXIT_WAIVED, "sample-reported")

    # Fail. Surface output so CI logs are diagnosable.
    msg = f"return code {proc.returncode}"
    _safe_print(f"  [FAIL ] {name}: {msg}")
    if proc.stdout:
        _safe_print(f"---- stdout ({name}) ----\n{proc.stdout.rstrip()}")
    if proc.stderr:
        _safe_print(f"---- stderr ({name}) ----\n{proc.stderr.rstrip()}")
    return RunResult(sample, "FAIL", proc.returncode, msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cuda-python samples")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory containing one subdir per sample (default: ./samples)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to test_args.json (default: alongside this script)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum number of samples to run concurrently (default: 1)",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help=("Run only samples whose directory name contains the given substring (may be repeated)"),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-sample timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args(argv)

    samples_dir: Path = args.samples_dir.resolve()
    if not samples_dir.is_dir():
        _safe_print(f"Error: samples directory not found: {samples_dir}")
        return 1

    samples = discover_samples(samples_dir)
    if args.filter:
        keep = []
        for sample in samples:
            if any(token in sample.parent.name for token in args.filter):
                keep.append(sample)
        samples = keep
    if not samples:
        _safe_print("No samples found.")
        return 1

    config = load_config(args.config.resolve())
    gpu_count = get_gpu_count()
    _safe_print(f"Detected {gpu_count} GPU(s).")
    _safe_print(f"Running {len(samples)} sample(s) with parallelism={args.parallel}\n")

    plans = [build_run_plan(s, config, gpu_count, args.timeout) for s in samples]

    if args.parallel <= 1:
        results = [run_sample(plan) for plan in plans]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(run_sample, plans))

    failed = [r for r in results if r.status in {"FAIL", "TIMEOUT", "ERROR"}]
    waived = [r for r in results if r.status == "WAIVED"]
    passed = [r for r in results if r.status == "PASS"]

    _safe_print("\nSummary")
    _safe_print(f"  passed : {len(passed)}")
    _safe_print(f"  waived : {len(waived)}")
    _safe_print(f"  failed : {len(failed)}")
    if failed:
        for r in failed:
            _safe_print(f"  - {r.sample.parent.name}: {r.status} ({r.detail})")
        first = next((r.return_code for r in failed if r.return_code > 0), 1)
        return first
    return 0


if __name__ == "__main__":
    sys.exit(main())
