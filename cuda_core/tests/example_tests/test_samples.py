# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pytest wrapper that runs the standalone samples under the repo-root
``samples/`` directory as part of the cuda.core test suite.

The samples themselves are plain runnable scripts (they are periodically synced
to NVIDIA/cuda-samples). Each discovered sample is parametrized into its own
test id and executed in a subprocess via the ``run_samples`` orchestrator, so
this runs alongside the other cuda.core example tests under
``pytest cuda_core/tests``.
"""

from __future__ import annotations

import sys

import pytest

from .run_samples import (
    DEFAULT_CONFIG,
    DEFAULT_SAMPLES_DIR,
    build_run_plan,
    discover_samples,
    get_gpu_count,
    load_config,
    run_sample,
)


def _collect_samples() -> dict[str, object]:
    """Return an ordered mapping of ``sample_name -> entry_path``.

    Sample names come from the leaf directory (e.g. ``clockNvrtc``); samples
    may live either at the top of ``samples/`` or under a category
    subdirectory such as ``samples/0_Introduction/``.
    """
    if not DEFAULT_SAMPLES_DIR.is_dir():
        return {}
    return {entry.parent.name: entry for entry in discover_samples(DEFAULT_SAMPLES_DIR)}


_ENTRIES = _collect_samples()
_SAMPLES = sorted(_ENTRIES)
_CONFIG = load_config(DEFAULT_CONFIG)
# Resolve GPU count once at collection time so we report the same skip reason
# consistently across the parametrized test ids.
_GPU_COUNT = get_gpu_count() if _SAMPLES else 0


# Samples launch full GPU workloads in their own subprocess, so keep them
# serialized when the suite is run under pytest-run-parallel.
@pytest.mark.parallel_threads_limit(1)
@pytest.mark.parametrize("sample_name", _SAMPLES)
def test_sample(sample_name: str) -> None:
    if _GPU_COUNT == 0:
        pytest.skip("No CUDA GPU detected on the test runner")

    entry = _ENTRIES.get(sample_name)
    if entry is None or not entry.is_file():
        pytest.fail(f"Sample entrypoint missing: {sample_name}")

    plan = build_run_plan(entry, _CONFIG, _GPU_COUNT)
    result = run_sample(plan)

    if result.status == "WAIVED":
        pytest.skip(result.detail or "sample waived")
    if result.status == "PASS":
        return

    # Re-print captured output through stdout/stderr so pytest's failure
    # capture surfaces it in the report.
    sys.stdout.flush()
    pytest.fail(f"sample {sample_name} returned status={result.status} (rc={result.return_code}): {result.detail}")
