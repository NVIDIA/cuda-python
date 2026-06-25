#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sample test orchestrator for samples under ./samples/.

``samples/<name>/<name>.py``, applies per-sample overrides from
``tests/samples/test_args.json`` (same schema used in cuda-samples, plus a
``python`` sub-object for Python-specific CLI args / launcher), and executes
each sample in its own subprocess.

Exit-code contract (matches cuda-samples):
    0  -> sample passed
    2  -> sample waived (missing dependency / unmet hardware requirement)
    *  -> sample failed

The script can be invoked directly:
    python tests/samples/run_samples.py [--samples-dir samples] [--config tests/samples/test_args.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default timeout per sample run (seconds). Match cuda-samples.
DEFAULT_TIMEOUT = 300
EXIT_WAIVED = 2
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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
    """Return ``samples/<name>/<name>.py`` for every sample directory.

    Only one Python entrypoint per sample is recognised, matching the
    cuda-samples convention. The Utilities directory is excluded.
    """
    samples: list[Path] = []
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name == "Utilities":
            continue
        entry = sample_dir / f"{sample_dir.name}.py"
        if entry.is_file():
            samples.append(entry)
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
# PEP 723 dep gating (reuse the helper that ships with cuda-bindings test
# infrastructure when available; otherwise fall back to a local parser so the
# runner stays usable without cuda-bindings installed).
# ---------------------------------------------------------------------------

_DEP_NAME_RE = re.compile(r"[a-zA-Z0-9_-]+")
_PEP723_RE = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")

# Aliases bridging PyPI distribution names declared in sample PEP 723 blocks
# and the install-name a conda/pixi environment provides. CI uses wheels where
# the names match exactly, so this map only fires in local pixi runs. Each
# entry maps a PyPI name to a list of alternative import names to try with
# ``importlib.import_module`` before declaring the dep missing.
_DEP_FALLBACK_IMPORTS: dict[str, tuple[str, ...]] = {
    "cuda-python": ("cuda.bindings",),
    "cuda-bindings": ("cuda.bindings",),
    "cuda-core": ("cuda.core",),
    "cuda-pathfinder": ("cuda.pathfinder",),
    "cuda-cccl": ("cuda.cccl", "cccl"),
    "cupy-cuda11x": ("cupy",),
    "cupy-cuda12x": ("cupy",),
    "cupy-cuda13x": ("cupy",),
    "nvidia-nvjitlink": ("nvjitlink",),
    "nvmath-python": ("nvmath",),
    "cugraph-cu12": ("cugraph",),
    "cugraph-cu13": ("cugraph",),
    "cudf-cu12": ("cudf",),
    "cudf-cu13": ("cudf",),
}


def _extract_pep723_dependencies(example: Path) -> list[str] | None:
    """Return the dependency list declared via PEP 723, or ``None`` if absent."""
    content = example.read_text(encoding="utf-8")
    match = _PEP723_RE.search(content)
    if not match:
        return None
    metadata: dict[str, str] = {}
    for raw in match.group("content").splitlines():
        line = raw.lstrip("# ").rstrip()
        if not line:
            continue
        key, _, value = line.partition("=")
        if not _:
            continue
        metadata[key.strip()] = value.strip()
    deps_literal = metadata.get("dependencies")
    if not deps_literal:
        return None
    try:
        # The PEP 723 spec uses TOML semantics, but in practice the values
        # are simple list-of-strings literals; eval keeps the runner aligned
        # with the cuda-bindings helper without taking a TOML dependency.
        result = eval(deps_literal, {"__builtins__": {}})  # noqa: S307
    except Exception:
        return None
    if not isinstance(result, list):
        return None
    return [str(item) for item in result]


def missing_dependencies(example: Path) -> list[str]:
    """Return the subset of declared deps that are not importable as distributions.

    Returns an empty list if all declared deps are present, or if no PEP 723
    block exists (no gating to perform).
    """
    deps = _extract_pep723_dependencies(example)
    if not deps:
        return []
    # Local imports keep top-level import cost down.
    import importlib
    import importlib.metadata

    missing: list[str] = []
    for spec in deps:
        match = _DEP_NAME_RE.match(spec)
        if match is None:
            continue
        name = match.group(0)
        try:
            importlib.metadata.distribution(name)
            continue
        except importlib.metadata.PackageNotFoundError:
            pass

        # Strict distribution check missed it. Try the known alias imports so
        # conda/pixi environments (which install under different distribution
        # names than the PyPI wheels) don't waive every sample.
        for module_name in _DEP_FALLBACK_IMPORTS.get(name, ()):
            try:
                importlib.import_module(module_name)
                break
            except ImportError:
                continue
        else:
            missing.append(name)
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

    missing = missing_dependencies(sample)
    if missing:
        reason = f"missing package(s): {', '.join(missing)}"
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
        help="Path to test_args.json (default: tests/samples/test_args.json)",
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
