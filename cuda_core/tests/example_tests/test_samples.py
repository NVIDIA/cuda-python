# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the samples owned by cuda.core as part of its test suite."""

from __future__ import annotations

import sys

import pytest

from .run_samples import (
    DEFAULT_CONFIG,
    DEFAULT_SAMPLES_DIR,
    SAMPLE_NAMESPACE,
    build_run_plan,
    collect_sample_entries,
    get_gpu_count,
    load_config,
    run_sample,
)

_ENTRIES = collect_sample_entries(DEFAULT_SAMPLES_DIR, SAMPLE_NAMESPACE)
_SAMPLES = sorted(_ENTRIES)
_CONFIG = load_config(DEFAULT_CONFIG)
# Resolve GPU count once at collection time so we report the same skip reason
# consistently across the parametrized test ids.
_GPU_COUNT = get_gpu_count() if _SAMPLES else 0


# Samples launch full GPU workloads in their own subprocess, so keep them
# serialized when the suite is run under pytest-run-parallel.
@pytest.mark.parallel_threads_limit(1)
@pytest.mark.parametrize("sample_key", _SAMPLES)
@pytest.mark.samples
@pytest.mark.agent_authored(model="gpt-5")
def test_sample(sample_key: str) -> None:
    if _GPU_COUNT == 0:
        pytest.skip("No CUDA GPU detected on the test runner")

    entry = _ENTRIES.get(sample_key)
    if entry is None or not entry.is_file():
        pytest.fail(f"Sample entrypoint missing: {sample_key}")

    plan = build_run_plan(entry, _CONFIG, _GPU_COUNT, sample_key=sample_key)
    result = run_sample(plan)

    if result.status == "WAIVED":
        pytest.skip(result.detail or "sample waived")
    if result.status == "PASS":
        return

    # Re-print captured output through stdout/stderr so pytest's failure
    # capture surfaces it in the report.
    sys.stdout.flush()
    pytest.fail(f"sample {sample_key} returned status={result.status} (rc={result.return_code}): {result.detail}")
