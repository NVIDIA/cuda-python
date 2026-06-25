# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pytest wrapper for samples under ./samples/.

The samples themselves should be plain runnable scripts.

This module uses ``run_samples.py`` to run the samples, which is a
convenience wrapper around the ``cuda_bindings.samples.run_samples`` module.
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


def _collect_samples() -> list[str]:
    if not DEFAULT_SAMPLES_DIR.is_dir():
        return []
    return [s.parent.name for s in discover_samples(DEFAULT_SAMPLES_DIR)]


_SAMPLES = _collect_samples()
_CONFIG = load_config(DEFAULT_CONFIG)
# Resolve GPU count once at collection time so we report the same skip reason
# consistently across the parametrized test ids.
_GPU_COUNT = get_gpu_count() if _SAMPLES else 0


@pytest.mark.parametrize("sample_name", _SAMPLES)
def test_sample(sample_name: str) -> None:
    if _GPU_COUNT == 0:
        pytest.skip("No CUDA GPU detected on the test runner")

    entry = DEFAULT_SAMPLES_DIR / sample_name / f"{sample_name}.py"
    if not entry.is_file():
        pytest.fail(f"Sample entrypoint missing: {entry}")

    plan = build_run_plan(entry, _CONFIG, _GPU_COUNT)
    result = run_sample(plan)

    if result.status == "WAIVED":
        pytest.skip(result.detail or "sample waived")
    if result.status == "PASS":
        return

    # Re-print captured output through stdout/stderr so pytest's failure
    # capture surfaces it in the report.
    sys.stdout.flush()
    pytest.fail(
        f"sample {sample_name} returned status={result.status} (rc={result.return_code}): {result.detail}"
    )
