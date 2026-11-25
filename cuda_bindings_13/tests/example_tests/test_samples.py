# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the samples owned by cuda.bindings as part of its test suite."""

from __future__ import annotations

import os

import pytest
from cuda_python_test_helpers.sample_runner import SampleSuite
from run_samples import DEFAULT_CONFIG, DEFAULT_SAMPLES_DIR, SAMPLE_NAMESPACE

_SUITE = SampleSuite(DEFAULT_SAMPLES_DIR, SAMPLE_NAMESPACE, DEFAULT_CONFIG)
_CUDA_12_PIXI_ENV = os.environ.get("PIXI_ENVIRONMENT_NAME") == "cu12"


@pytest.mark.parallel_threads_limit(1)
@pytest.mark.parametrize("sample_key", _SUITE.keys)
@pytest.mark.samples
@pytest.mark.skipif(
    _CUDA_12_PIXI_ENV,
    reason="cuda.bindings samples require cuda-python 13 or newer",
)
@pytest.mark.agent_authored(model="gpt-5")
def test_sample(sample_key: str) -> None:
    _SUITE.check(sample_key, fail_on_unmet_dependencies=True)
