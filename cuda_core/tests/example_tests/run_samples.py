#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuda.core defaults for the shared sample runner."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_HELPERS_ROOT = REPO_ROOT / "cuda_python_test_helpers"
if str(_TEST_HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_HELPERS_ROOT))

from cuda_python_test_helpers import sample_runner as _runner

DEFAULT_CONFIG = Path(__file__).resolve().parent / "test_args.json"
DEFAULT_SAMPLES_DIR = REPO_ROOT / "samples" / "cuda_core"
SAMPLE_NAMESPACE = "cuda_core"

DEFAULT_TIMEOUT = _runner.DEFAULT_TIMEOUT
DependencyMetadataError = _runner.DependencyMetadataError
RunPlan = _runner.RunPlan
RunResult = _runner.RunResult
build_run_plan = _runner.build_run_plan
collect_sample_entries = _runner.collect_sample_entries
discover_samples = _runner.discover_samples
get_gpu_count = _runner.get_gpu_count
get_sample_id = _runner.get_sample_id
load_config = _runner.load_config
missing_dependencies = _runner.missing_dependencies
run_sample = _runner.run_sample

# Kept available for runner unit tests that patch distribution metadata.
importlib = _runner.importlib


def main(argv: list[str] | None = None) -> int:
    return _runner.main(
        argv,
        default_samples_dir=DEFAULT_SAMPLES_DIR,
        default_config=DEFAULT_CONFIG,
        namespace=SAMPLE_NAMESPACE,
    )


if __name__ == "__main__":
    sys.exit(main())
