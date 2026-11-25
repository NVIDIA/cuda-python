#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared discovery and execution support for cuda-python sample test suites.

Package-owned wrappers provide their sample root, namespace, and
``test_args.json`` path. Samples are executed in isolated subprocesses.

Exit-code contract:
    0  -> sample passed
    77 -> sample waived when negotiated through the runner environment
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
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion

# Default timeout per sample run (seconds). Match cuda-samples.
DEFAULT_TIMEOUT = 300
EXIT_WAIVED = 77
WAIVER_EXIT_CODE_ENV = "CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE"
# Prefix of the RunResult detail for a sample waived for unmet PEP 723 dependencies.
UNMET_DEPENDENCIES = "unmet package requirement(s):"
_print_lock = threading.Lock()


def _safe_print(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        print(*args, **kwargs)  # noqa: T201 - this module is a CLI runner


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_samples(samples_dir: Path) -> list[Path]:
    """Return every ``<name>/<name>.py`` sample entrypoint under ``samples_dir``.

    Samples can either sit directly under ``samples_dir``
    (``<samples_dir>/<name>/<name>.py``) or in a category subdirectory such as
    ``<samples_dir>/0_Introduction/<name>/<name>.py``. The Utilities directory is
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


def get_sample_id(sample: Path, samples_dir: Path, namespace: str | None = None) -> str:
    """Return a stable, POSIX-style ID for an entrypoint below ``samples_dir``."""
    try:
        relative_dir = sample.parent.relative_to(samples_dir)
    except ValueError as exc:
        raise ValueError(f"sample {sample} is not below sample root {samples_dir}") from exc

    parts = ((namespace,) if namespace else ()) + relative_dir.parts
    return "/".join(parts)


def collect_sample_entries(samples_dir: Path, namespace: str | None = None) -> dict[str, Path]:
    """Return discovered entrypoints keyed by path-aware sample IDs."""
    if not samples_dir.is_dir():
        return {}

    entries: dict[str, Path] = {}
    for entry in discover_samples(samples_dir):
        key = get_sample_id(entry, samples_dir, namespace)
        if key in entries:
            raise ValueError(f"duplicate sample ID {key!r}: {entries[key]} and {entry}")
        entries[key] = entry
    return entries


# ---------------------------------------------------------------------------
# Config + GPU detection
# ---------------------------------------------------------------------------


class InvalidSampleConfig(ValueError):  # noqa: N818 - requested public exception name
    """The sample runner configuration does not match its expected schema."""


_ENTRY_FIELDS = frozenset({"skip", "min_gpus", "python"})
_PYTHON_FIELDS = frozenset({"args", "launcher"})


def _validate_string_list(value: Any, *, field: str, location: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InvalidSampleConfig(f"{location}: {field!r} must be a list of strings")
    return value


def _validate_config_entry(key: str, value: Any, *, source: str) -> dict[str, Any]:
    location = f"{source}: entry {key!r}"
    if not isinstance(value, dict):
        raise InvalidSampleConfig(f"{location} must be an object")

    unknown_fields = set(value) - _ENTRY_FIELDS
    if unknown_fields:
        fields = ", ".join(repr(field) for field in sorted(unknown_fields))
        raise InvalidSampleConfig(f"{location} has unknown field(s): {fields}")

    if "skip" in value and not isinstance(value["skip"], bool):
        raise InvalidSampleConfig(f"{location}: 'skip' must be a boolean")

    if "min_gpus" in value:
        min_gpus = value["min_gpus"]
        if isinstance(min_gpus, bool) or not isinstance(min_gpus, int) or min_gpus < 1:
            raise InvalidSampleConfig(f"{location}: 'min_gpus' must be a positive integer")

    if "python" in value:
        python_cfg = value["python"]
        if not isinstance(python_cfg, dict):
            raise InvalidSampleConfig(f"{location}: 'python' must be an object")

        unknown_python_fields = set(python_cfg) - _PYTHON_FIELDS
        if unknown_python_fields:
            fields = ", ".join(repr(field) for field in sorted(unknown_python_fields))
            raise InvalidSampleConfig(f"{location}: 'python' has unknown field(s): {fields}")

        for field in _PYTHON_FIELDS:
            if field in python_cfg:
                _validate_string_list(python_cfg[field], field=field, location=f"{location}: 'python'")

    return value


def _validate_config(data: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise InvalidSampleConfig(f"{source} must contain a JSON object")

    config: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise InvalidSampleConfig(f"{source} contains a non-string entry name")
        if key.startswith("_"):
            continue
        config[key] = _validate_config_entry(key, value, source=source)
    return config


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise InvalidSampleConfig(f"{config_path}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise InvalidSampleConfig(f"{config_path}: cannot read configuration: {exc}") from exc
    return _validate_config(data, source=str(config_path))


def _runtime_gpu_count() -> int | None:
    """Return the CUDA Runtime's visible device count, or ``None`` if unavailable."""
    try:
        runtime = importlib.import_module("cuda.bindings.runtime")
    except (ImportError, OSError):
        return None

    try:
        error, count = runtime.cudaGetDeviceCount()
    except (OSError, RuntimeError):
        return None

    if int(error) != 0:
        return 0
    return int(count)


def _visible_devices_gpu_count() -> int | None:
    """Return the count implied by ``CUDA_VISIBLE_DEVICES``, if it is set."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None

    count = 0
    for raw_token in visible.split(","):
        token = raw_token.strip()
        if not token or token.lower() in {"-1", "no", "none"}:
            break
        if token.isascii() and token.isdecimal():
            count += 1
            continue
        if token.startswith(("GPU-", "MIG-")) and len(token) > 4:
            count += 1
            continue
        # CUDA stops enumerating at the first invalid device identifier.
        break
    return count


def get_gpu_count() -> int:
    """Return the number of GPUs visible to CUDA, conservatively 0 on error.

    Prefer the CUDA Runtime because it applies the driver's complete visibility
    rules, including UUIDs and MIG devices. When the runtime bindings are not
    available, an explicitly set ``CUDA_VISIBLE_DEVICES`` remains authoritative;
    ``nvidia-smi`` is only a last-resort estimate when visibility is unspecified.
    """
    runtime_count = _runtime_gpu_count()
    if runtime_count is not None:
        return runtime_count

    visible_count = _visible_devices_gpu_count()
    if visible_count is not None:
        return visible_count

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


def _extra_requirements(dist: importlib.metadata.Distribution, extra: str) -> list[Requirement]:
    """Return the requirements that ``extra`` adds to ``dist``'s base requirements."""
    requirements: list[Requirement] = []
    for spec in dist.requires or ():
        requirement = Requirement(spec)
        marker = requirement.marker
        if marker is not None and marker.evaluate({"extra": extra}) and not marker.evaluate({"extra": ""}):
            requirements.append(requirement)
    return requirements


def _requirement_without_marker(requirement: Requirement) -> str:
    text = requirement.name
    if requirement.extras:
        text += f"[{','.join(sorted(requirement.extras))}]"
    return text + str(requirement.specifier)


def _extra_mismatches(
    name: str, dist: importlib.metadata.Distribution, extra: str, seen: set[tuple[str, str]]
) -> list[str]:
    """Check the installed distributions that ``extra`` of ``dist`` requires.

    A distribution the extra requires but that is not installed is accepted:
    conda packages and a local CUDA Toolkit provide many of these libraries
    without pip metadata. An installed one must satisfy the requirement, so
    that, for example, ``cuda-core[cu13]`` is unmet when cuda-bindings 12.x is
    installed.
    """
    key = (name, extra)
    if key in seen:
        return []
    seen.add(key)

    mismatches: list[str] = []
    for requirement in _extra_requirements(dist, extra):
        for provider in _distribution_providers(requirement):
            try:
                provider_dist = importlib.metadata.distribution(provider)
            except importlib.metadata.PackageNotFoundError:
                continue
            for detail in _distribution_mismatches(requirement, provider_dist, provider, seen):
                mismatches.append(
                    f"extra {extra} requires {_requirement_without_marker(requirement)} "
                    f"({provider} {provider_dist.version}: {detail})"
                )
            break
    return mismatches


def _distribution_mismatches(
    requirement: Requirement,
    dist: importlib.metadata.Distribution,
    name: str,
    seen: set[tuple[str, str]] | None = None,
) -> list[str]:
    mismatches: list[str] = []
    if requirement.specifier:
        try:
            version_matches = requirement.specifier.contains(dist.version)
        except InvalidVersion:
            mismatches.append(f"has invalid version {dist.version!r}")
        else:
            if not version_matches:
                mismatches.append(f"version {dist.version} does not satisfy {requirement.specifier}")

    requested_extras = {canonicalize_name(extra) for extra in requirement.extras}
    missing_extras = requested_extras - _provided_extras(dist)
    if missing_extras:
        mismatches.append(f"does not provide extra(s): {', '.join(sorted(missing_extras))}")

    if seen is None:
        seen = set()
    for extra in sorted(requested_extras - missing_extras):
        mismatches.extend(_extra_mismatches(canonicalize_name(name), dist, extra, seen))
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
            mismatches = _distribution_mismatches(requirement, dist, provider)
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
    sample_key: str | None = None


def _expand_env(value: str) -> str:
    return os.path.expandvars(value)


def build_run_plan(
    sample: Path,
    config: dict[str, Any],
    gpu_count: int,
    timeout: int = DEFAULT_TIMEOUT,
    *,
    sample_key: str | None = None,
) -> RunPlan:
    """Combine config overrides + GPU availability into a concrete run plan.

    The returned plan carries either a ``skip_reason`` (sample must be
    waived) or the command components to invoke.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise InvalidSampleConfig("sample timeout must be a positive integer")
    if not isinstance(config, dict):
        raise InvalidSampleConfig("sample configuration must be an object")

    # Namespaced keys prevent collisions. The leaf fallback keeps existing
    # cuda-samples-style configuration files compatible.
    if sample_key is not None and sample_key in config:
        config_key = sample_key
        sample_cfg = config[sample_key]
    else:
        config_key = sample.parent.name
        sample_cfg = config.get(config_key, {})
    sample_cfg = _validate_config_entry(config_key, sample_cfg, source="sample configuration")

    if sample_cfg.get("skip"):
        return RunPlan(
            sample,
            [],
            [],
            timeout,
            skip_reason="skipped in test_args.json",
            sample_key=sample_key,
        )

    required_gpus = sample_cfg.get("min_gpus", 1)
    if required_gpus > gpu_count:
        return RunPlan(
            sample,
            [],
            [],
            timeout,
            skip_reason=(f"requires {required_gpus} GPU(s), only {gpu_count} available"),
            sample_key=sample_key,
        )

    python_cfg = sample_cfg.get("python", {})
    raw_args = python_cfg.get("args", [])
    raw_launcher = python_cfg.get("launcher", [])

    return RunPlan(
        sample=sample,
        args=[_expand_env(arg) for arg in raw_args],
        launcher=[_expand_env(arg) for arg in raw_launcher],
        timeout=timeout,
        sample_key=sample_key,
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
    sample_key: str | None = None


def run_sample(plan: RunPlan) -> RunResult:
    sample = plan.sample
    name = plan.sample_key or sample.parent.name

    if plan.skip_reason is not None:
        _safe_print(f"  [WAIVED] {name}: {plan.skip_reason}")
        return RunResult(sample, "WAIVED", EXIT_WAIVED, plan.skip_reason, plan.sample_key)

    try:
        missing = missing_dependencies(sample)
    except DependencyMetadataError as exc:
        reason = str(exc)
        _safe_print(f"  [ERROR] {name}: {reason}")
        return RunResult(sample, "ERROR", -1, reason, plan.sample_key)
    if missing:
        reason = f"{UNMET_DEPENDENCIES} {'; '.join(missing)}"
        _safe_print(f"  [WAIVED] {name}: {reason}")
        return RunResult(sample, "WAIVED", EXIT_WAIVED, reason, plan.sample_key)

    cmd = list(plan.launcher) + [sys.executable, str(sample)] + list(plan.args)
    _safe_print(f"  [RUN ] {name}: {' '.join(cmd)}")
    child_env = os.environ.copy()
    child_env[WAIVER_EXIT_CODE_ENV] = str(EXIT_WAIVED)

    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(sample.parent),
            capture_output=True,
            text=True,
            timeout=plan.timeout,
            check=False,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        _safe_print(f"  [TIMEOUT] {name}: exceeded {plan.timeout}s")
        return RunResult(sample, "TIMEOUT", -1, f"timed out after {plan.timeout}s", plan.sample_key)
    except OSError as exc:
        _safe_print(f"  [ERROR] {name}: {exc}")
        return RunResult(sample, "ERROR", -1, str(exc), plan.sample_key)

    if proc.returncode == 0:
        _safe_print(f"  [PASS ] {name}")
        return RunResult(sample, "PASS", 0, sample_key=plan.sample_key)
    if proc.returncode == EXIT_WAIVED:
        _safe_print(f"  [WAIVED] {name}: sample reported waived")
        return RunResult(sample, "WAIVED", EXIT_WAIVED, "sample-reported", plan.sample_key)

    # Fail. Surface output so CI logs are diagnosable.
    msg = f"return code {proc.returncode}"
    _safe_print(f"  [FAIL ] {name}: {msg}")
    if proc.stdout:
        _safe_print(f"---- stdout ({name}) ----\n{proc.stdout.rstrip()}")
    if proc.stderr:
        _safe_print(f"---- stderr ({name}) ----\n{proc.stderr.rstrip()}")
    return RunResult(sample, "FAIL", proc.returncode, msg, plan.sample_key)


# ---------------------------------------------------------------------------
# Pytest integration
# ---------------------------------------------------------------------------


class SampleSuite:
    """The samples one package owns, resolved once at collection time."""

    def __init__(self, samples_dir: Path, namespace: str, config_path: Path) -> None:
        self.entries = collect_sample_entries(samples_dir, namespace)
        self.keys = sorted(self.entries)
        self.config = load_config(config_path)
        # Resolve the GPU count once, so every parametrized test id reports the
        # same skip reason.
        if self.keys:
            self.gpu_count = get_gpu_count()
        else:
            self.gpu_count = 0

    def check(self, sample_key: str, *, fail_on_unmet_dependencies: bool = False) -> None:
        """Run one sample and turn its result into a pytest outcome.

        A waived sample skips. With ``fail_on_unmet_dependencies``, a sample
        waived because its PEP 723 dependencies are not installed fails instead.
        """
        if self.gpu_count == 0:
            pytest.skip("No CUDA GPU detected on the test runner")

        entry = self.entries.get(sample_key)
        if entry is None or not entry.is_file():
            pytest.fail(f"Sample entrypoint missing: {sample_key}")

        plan = build_run_plan(entry, self.config, self.gpu_count, sample_key=sample_key)
        result = run_sample(plan)

        if result.status == "WAIVED":
            if fail_on_unmet_dependencies and result.detail.startswith(UNMET_DEPENDENCIES):
                pytest.fail(f"sample dependencies are not satisfied: {result.detail}")
            pytest.skip(result.detail or "sample waived")
        if result.status == "PASS":
            return

        # run_sample printed the sample's output; flush it so pytest's failure
        # capture surfaces it in the report.
        sys.stdout.flush()
        pytest.fail(f"sample {sample_key} returned status={result.status} (rc={result.return_code}): {result.detail}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def main(
    argv: list[str] | None = None,
    *,
    default_samples_dir: Path,
    default_config: Path,
    namespace: str | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run cuda-python samples")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=default_samples_dir,
        help=f"Directory containing package-owned samples (default: {default_samples_dir})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to test_args.json (default: {default_config})",
    )
    parser.add_argument(
        "--parallel",
        type=_positive_int,
        default=1,
        help="Maximum number of samples to run concurrently (default: 1)",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help=("Run only samples whose ID contains the given substring (may be repeated)"),
    )
    parser.add_argument(
        "--timeout",
        type=_positive_int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-sample timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args(argv)

    samples_dir: Path = args.samples_dir.resolve()
    if not samples_dir.is_dir():
        _safe_print(f"Error: samples directory not found: {samples_dir}")
        return 1

    entries = collect_sample_entries(samples_dir, namespace)
    if args.filter:
        entries = {key: sample for key, sample in entries.items() if any(token in key for token in args.filter)}
    if not entries:
        _safe_print("No samples found.")
        return 1

    try:
        config = load_config(args.config.resolve())
    except InvalidSampleConfig as exc:
        _safe_print(f"Error: invalid sample configuration: {exc}")
        return 1
    gpu_count = get_gpu_count()
    _safe_print(f"Detected {gpu_count} GPU(s).")
    _safe_print(f"Running {len(entries)} sample(s) with parallelism={args.parallel}\n")

    try:
        plans = [
            build_run_plan(sample, config, gpu_count, args.timeout, sample_key=key) for key, sample in entries.items()
        ]
    except InvalidSampleConfig as exc:
        _safe_print(f"Error: invalid sample configuration: {exc}")
        return 1

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
            name = r.sample_key or r.sample.parent.name
            _safe_print(f"  - {name}: {r.status} ({r.detail})")
        first = next((r.return_code for r in failed if r.return_code > 0), 1)
        return first
    return 0
